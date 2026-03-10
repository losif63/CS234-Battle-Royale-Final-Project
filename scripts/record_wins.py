"""Record wins of latest agent vs opponent, first-person, into one mp4.

Uses scan+replay pattern:
  1. GPU-batched scan: run many games in parallel, find wins with 4+ kills
  2. Replay winning games one at a time with full frame capture
"""
import os, sys, subprocess, torch, pickle

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYTHONUNBUFFERED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
from battle_royale.sim import BatchedBRSim
from battle_royale.obs import ObservationBuilder
from battle_royale.network import (
    AttentionActorCritic, pack_actor_obs, _actions_to_sim, _apply_action_masks, _sample_actions,
    MAX_AGENTS, LSTM_HIDDEN, N_ENTITIES, ENTITY_DIM, SELF_DIM, NUM_DISCRETE_ACTIONS,
)
from battle_royale.config import FPS, ENTITY_FOV_RADIUS, MAX_EPISODE_FRAMES
from battle_royale.train import _load_checkpoint, ACTION_REPEAT
from battle_royale.renderer import Renderer

CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "battle_royale", "runs", "apex_rp", "checkpoints")
LATEST = os.path.join(CKPT_DIR, "br_ppo_16900.pt")
OPPONENT = os.path.join(CKPT_DIR, "br_ppo_8000.pt")
OUTPUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wins_16k_vs_8k.mp4")
NUM_WINS = 3
MIN_KILLS = 4
SCAN_BATCH = 500  # games per scan batch
TEMP = 0.5

device = torch.device("cuda")


def save_rng_state():
    return {
        "cpu": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device),
    }

def restore_rng_state(state):
    torch.random.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"], device)


# ---------------------------------------------------------------------------
# Phase 1: Fast GPU scan (no rendering)
# ---------------------------------------------------------------------------

@torch.no_grad()
def scan_batch(net_latest, net_opp, num_envs):
    """Run num_envs games, return list of (env_idx, kills, place) for agent 0 wins with 4+ kills."""
    B = num_envs
    A = MAX_AGENTS

    rng_state = save_rng_state()

    sim = BatchedBRSim(num_envs=B, max_agents=A, device="cuda")
    obs_builder = ObservationBuilder(sim)
    sim.reset()

    # LSTM states: agent 0 uses net_latest, agents 1..A-1 use net_opp
    hx_0 = torch.zeros(B, 1, LSTM_HIDDEN, device=device)
    cx_0 = torch.zeros(B, 1, LSTM_HIDDEN, device=device)
    hx_opp = torch.zeros(B, A - 1, LSTM_HIDDEN, device=device)
    cx_opp = torch.zeros(B, A - 1, LSTM_HIDDEN, device=device)

    step = 0
    while not sim.episode_done.all():
        step += 1
        if step > (MAX_EPISODE_FRAMES // ACTION_REPEAT) + 100:
            break

        ao = obs_builder.actor_obs()
        sf, ent, emask = pack_actor_obs(ao)

        # Agent 0 (latest) - batched across all envs
        sf0 = sf[:, 0:1].reshape(B, SELF_DIM)
        ent0 = ent[:, 0:1].reshape(B, N_ENTITIES, ENTITY_DIM)
        em0 = emask[:, 0:1].reshape(B, N_ENTITIES)
        hx0_f = hx_0.reshape(B, LSTM_HIDDEN)
        cx0_f = cx_0.reshape(B, LSTM_HIDDEN)

        log0, a0, b0, (hx0_f, cx0_f) = net_latest.forward_actor(
            sf0, ent0, em0, hx=hx0_f, cx=cx0_f)
        log0 = _apply_action_masks(log0, sf0)
        log0 = tuple(l / TEMP for l in log0)
        a0_t = 1.0 + (a0 - 1.0) / TEMP
        b0_t = 1.0 + (b0 - 1.0) / TEMP
        d0, c0 = _sample_actions(log0, a0_t, b0_t)

        hx_0 = hx0_f.reshape(B, 1, LSTM_HIDDEN)
        cx_0 = cx0_f.reshape(B, 1, LSTM_HIDDEN)

        # Opponents (8k) - batched across all envs x (A-1) agents
        sf_opp = sf[:, 1:].reshape(B * (A - 1), SELF_DIM)
        ent_opp = ent[:, 1:].reshape(B * (A - 1), N_ENTITIES, ENTITY_DIM)
        em_opp = emask[:, 1:].reshape(B * (A - 1), N_ENTITIES)
        hx_opp_f = hx_opp.reshape(B * (A - 1), LSTM_HIDDEN)
        cx_opp_f = cx_opp.reshape(B * (A - 1), LSTM_HIDDEN)

        log_opp, a_opp, b_opp, (hx_opp_f, cx_opp_f) = net_opp.forward_actor(
            sf_opp, ent_opp, em_opp, hx=hx_opp_f, cx=cx_opp_f)
        log_opp = _apply_action_masks(log_opp, sf_opp)
        log_opp = tuple(l / TEMP for l in log_opp)
        a_opp_t = 1.0 + (a_opp - 1.0) / TEMP
        b_opp_t = 1.0 + (b_opp - 1.0) / TEMP
        d_opp, c_opp = _sample_actions(log_opp, a_opp_t, b_opp_t)

        hx_opp = hx_opp_f.reshape(B, A - 1, LSTM_HIDDEN)
        cx_opp = cx_opp_f.reshape(B, A - 1, LSTM_HIDDEN)

        # Combine actions
        d0_ba = d0.reshape(B, 1, NUM_DISCRETE_ACTIONS)
        d_opp_ba = d_opp.reshape(B, A - 1, NUM_DISCRETE_ACTIONS)
        disc = torch.cat([d0_ba, d_opp_ba], dim=1)
        cont = torch.cat([c0.reshape(B, 1), c_opp.reshape(B, A - 1)], dim=1)

        mx, my, aim, fire, heal = _actions_to_sim(disc, cont, sim.agent_dir)

        for _ in range(ACTION_REPEAT):
            alive = sim.agent_alive
            _, done = sim.step(
                mx * alive.float(), my * alive.float(),
                aim, fire & alive, heal & alive)
            if sim.episode_done.all():
                break

        # Zero LSTM for dead agents
        alive_0 = sim.agent_alive[:, 0:1].unsqueeze(-1).float()
        hx_0 = hx_0 * alive_0
        cx_0 = cx_0 * alive_0
        alive_opp = sim.agent_alive[:, 1:].unsqueeze(-1).float()
        hx_opp = hx_opp * alive_opp
        cx_opp = cx_opp * alive_opp

    # Collect results
    kills = sim.agent_kills[:, 0].cpu()
    places = sim.agent_place[:, 0].cpu()
    done = sim.episode_done.cpu()

    winners = []
    for b in range(B):
        if not done[b]:
            continue
        k = int(kills[b].item())
        p = int(places[b].item())
        if p == 1 and k >= MIN_KILLS:
            winners.append((b, k, p))

    del sim, obs_builder
    return winners, rng_state


# ---------------------------------------------------------------------------
# Phase 2: Replay a single game with frame capture
# ---------------------------------------------------------------------------

@torch.no_grad()
def replay_game(net_latest, net_opp, rng_state, num_envs, env_idx):
    """Re-run the batch with restored RNG, render only env_idx, return RGB bytes."""
    restore_rng_state(rng_state)

    B = num_envs
    A = MAX_AGENTS

    sim = BatchedBRSim(num_envs=B, max_agents=A, device="cuda")
    obs_builder = ObservationBuilder(sim)
    sim.reset()

    hx_0 = torch.zeros(B, 1, LSTM_HIDDEN, device=device)
    cx_0 = torch.zeros(B, 1, LSTM_HIDDEN, device=device)
    hx_opp = torch.zeros(B, A - 1, LSTM_HIDDEN, device=device)
    cx_opp = torch.zeros(B, A - 1, LSTM_HIDDEN, device=device)

    # Renderer for the one game we care about
    renderer = Renderer(instant_restart=True, camera_follow=0, fov_mask=False)
    vp_size = int(ENTITY_FOV_RADIUS * 2)
    out_w = vp_size // 2 * 2
    out_h = vp_size // 2 * 2
    capture_surf = pygame.Surface((out_w, out_h))

    frame_bufs = []
    step = 0

    while not sim.episode_done.all():
        step += 1
        if step > (MAX_EPISODE_FRAMES // ACTION_REPEAT) + 100:
            break

        ao = obs_builder.actor_obs()
        sf, ent, emask = pack_actor_obs(ao)

        # Agent 0
        sf0 = sf[:, 0:1].reshape(B, SELF_DIM)
        ent0 = ent[:, 0:1].reshape(B, N_ENTITIES, ENTITY_DIM)
        em0 = emask[:, 0:1].reshape(B, N_ENTITIES)
        hx0_f = hx_0.reshape(B, LSTM_HIDDEN)
        cx0_f = cx_0.reshape(B, LSTM_HIDDEN)

        log0, a0, b0, (hx0_f, cx0_f) = net_latest.forward_actor(
            sf0, ent0, em0, hx=hx0_f, cx=cx0_f)
        log0 = _apply_action_masks(log0, sf0)
        log0 = tuple(l / TEMP for l in log0)
        a0_t = 1.0 + (a0 - 1.0) / TEMP
        b0_t = 1.0 + (b0 - 1.0) / TEMP
        d0, c0 = _sample_actions(log0, a0_t, b0_t)

        hx_0 = hx0_f.reshape(B, 1, LSTM_HIDDEN)
        cx_0 = cx0_f.reshape(B, 1, LSTM_HIDDEN)

        # Opponents
        sf_opp = sf[:, 1:].reshape(B * (A - 1), SELF_DIM)
        ent_opp = ent[:, 1:].reshape(B * (A - 1), N_ENTITIES, ENTITY_DIM)
        em_opp = emask[:, 1:].reshape(B * (A - 1), N_ENTITIES)
        hx_opp_f = hx_opp.reshape(B * (A - 1), LSTM_HIDDEN)
        cx_opp_f = cx_opp.reshape(B * (A - 1), LSTM_HIDDEN)

        log_opp, a_opp, b_opp, (hx_opp_f, cx_opp_f) = net_opp.forward_actor(
            sf_opp, ent_opp, em_opp, hx=hx_opp_f, cx=cx_opp_f)
        log_opp = _apply_action_masks(log_opp, sf_opp)
        log_opp = tuple(l / TEMP for l in log_opp)
        a_opp_t = 1.0 + (a_opp - 1.0) / TEMP
        b_opp_t = 1.0 + (b_opp - 1.0) / TEMP
        d_opp, c_opp = _sample_actions(log_opp, a_opp_t, b_opp_t)

        hx_opp = hx_opp_f.reshape(B, A - 1, LSTM_HIDDEN)
        cx_opp = cx_opp_f.reshape(B, A - 1, LSTM_HIDDEN)

        # Combine
        d0_ba = d0.reshape(B, 1, NUM_DISCRETE_ACTIONS)
        d_opp_ba = d_opp.reshape(B, A - 1, NUM_DISCRETE_ACTIONS)
        disc = torch.cat([d0_ba, d_opp_ba], dim=1)
        cont = torch.cat([c0.reshape(B, 1), c_opp.reshape(B, A - 1)], dim=1)

        mx, my, aim, fire, heal = _actions_to_sim(disc, cont, sim.agent_dir)

        # Win probs for HUD (only for the env we care about)
        sf_all = sf[env_idx]
        ent_all = ent[env_idx]
        em_all = emask[env_idx]
        hx_crit = torch.cat([hx_0[env_idx], hx_opp[env_idx]], dim=0)
        values = net_latest.forward_critic(sf_all, ent_all, em_all, hx=hx_crit)
        alive_mask = sim.agent_alive[env_idx].bool()
        v = values.clone()
        v[~alive_mask] = float('-inf')
        wp = torch.softmax(v, dim=0)
        wp[~alive_mask] = 0.0
        win_probs = wp.tolist()

        for _ in range(ACTION_REPEAT):
            alive = sim.agent_alive
            _, done = sim.step(
                mx * alive.float(), my * alive.float(),
                aim, fire & alive, heal & alive)

            # Capture frame for our env
            if not sim.episode_done[env_idx]:
                state = sim.get_state(env_idx)
                state["win_probs"] = win_probs
                renderer.render(state)
                pygame.transform.scale(renderer.screen, (out_w, out_h), capture_surf)
                frame_bufs.append(pygame.image.tobytes(capture_surf, "RGB"))

            if sim.episode_done.all():
                break

        # Zero LSTM for dead
        alive_0 = sim.agent_alive[:, 0:1].unsqueeze(-1).float()
        hx_0 = hx_0 * alive_0
        cx_0 = cx_0 * alive_0
        alive_opp = sim.agent_alive[:, 1:].unsqueeze(-1).float()
        hx_opp = hx_opp * alive_opp
        cx_opp = cx_opp * alive_opp

    del sim, obs_builder
    return frame_bufs, out_w, out_h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load networks
    net_latest = AttentionActorCritic().to(device)
    sd, ckpt = _load_checkpoint(LATEST, device)
    net_latest.load_state_dict(sd, strict=False)
    net_latest.eval()
    print(f"Agent 0 (latest): {LATEST} (update {ckpt.get('update_count', '?')})", flush=True)

    net_opp = AttentionActorCritic().to(device)
    sd_opp, ckpt_opp = _load_checkpoint(OPPONENT, device)
    net_opp.load_state_dict(sd_opp, strict=False)
    net_opp.eval()
    print(f"Opponents: {OPPONENT} (update {ckpt_opp.get('update_count', '?')})", flush=True)

    pygame.init()

    # Scan until we have enough wins
    winning_replays = []  # list of (rng_state, env_idx, kills)
    total_games = 0
    batch_num = 0

    print(f"\nScanning for wins with {MIN_KILLS}+ kills (batch size {SCAN_BATCH})...", flush=True)

    while len(winning_replays) < NUM_WINS:
        batch_num += 1
        winners, rng_state = scan_batch(net_latest, net_opp, SCAN_BATCH)
        total_games += SCAN_BATCH

        for env_idx, kills, place in winners:
            if len(winning_replays) < NUM_WINS:
                winning_replays.append((rng_state, env_idx, kills))

        found = len(winners)
        have = len(winning_replays)
        print(f"  Batch {batch_num}: {SCAN_BATCH} games, {found} qualifying wins found | {have}/{NUM_WINS} collected | {total_games} total games", flush=True)

    # Sort by kills descending for best games first
    winning_replays.sort(key=lambda x: x[2], reverse=True)
    winning_replays = winning_replays[:NUM_WINS]

    print(f"\nFound {NUM_WINS} wins in {total_games} games. Replaying and recording...", flush=True)
    for i, (_, env_idx, kills) in enumerate(winning_replays):
        print(f"  Win {i+1}: env {env_idx}, {kills} kills", flush=True)

    # Setup ffmpeg
    vp_size = int(ENTITY_FOV_RADIUS * 2)
    out_w = vp_size // 2 * 2
    out_h = vp_size // 2 * 2

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "rgb24",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", "fast",
        OUTPUT,
    ]
    print(f"Recording to {OUTPUT} ({out_w}x{out_h} @ {FPS}fps)", flush=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    info_font = pygame.font.SysFont("monospace", 14)
    written = 0

    for i, (rng_state, env_idx, kills) in enumerate(winning_replays):
        print(f"  Replaying win {i+1}/{NUM_WINS} (env {env_idx}, {kills} kills)...", flush=True)
        frame_bufs, ow, oh = replay_game(net_latest, net_opp, rng_state, SCAN_BATCH, env_idx)
        print(f"    {len(frame_bufs)} frames captured", flush=True)

        for buf in frame_bufs:
            proc.stdin.write(buf)
            written += 1

        # Add 2 seconds of black between games (except after last)
        if i < NUM_WINS - 1:
            black = pygame.Surface((out_w, out_h))
            black.fill((0, 0, 0))
            sep_text = info_font.render(f"Win {i+1}/{NUM_WINS} complete  ({kills} kills)", True, (255, 255, 100))
            black.blit(sep_text, (out_w // 2 - sep_text.get_width() // 2, out_h // 2))
            for _ in range(FPS * 2):
                proc.stdin.write(pygame.image.tobytes(black, "RGB"))
                written += 1

    proc.stdin.close()
    proc.wait()
    pygame.quit()

    if proc.returncode != 0:
        print(f"ffmpeg error: {proc.stderr.read().decode()}", flush=True)
    else:
        size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
        print(f"\nSaved: {OUTPUT} ({size_mb:.1f} MB, {written} frames)", flush=True)


if __name__ == "__main__":
    main()
