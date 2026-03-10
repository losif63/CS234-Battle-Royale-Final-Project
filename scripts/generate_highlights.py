"""
Highlight Detection & Replay System — Generator

Scan+Replay approach:
  1. Scan thousands of envs on GPU (no frame capture, ~100 games/s)
  2. Save RNG state before each batch so interesting games can be replayed
  3. Re-run only the best env indices with full frame capture

Usage:
    uv run python scripts/generate_highlights.py                              # 10k scan + auto record
    uv run python scripts/generate_highlights.py --scan-envs 30000            # 30k scan
    uv run python scripts/generate_highlights.py --scan-only --scan-envs 50000
"""

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass, field

import torch
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from battle_royale.sim import BatchedBRSim
from battle_royale.obs import ObservationBuilder
from battle_royale.network import (
    AttentionActorCritic, pack_actor_obs,
    _actions_to_sim, _apply_action_masks, _sample_actions,
    MAX_AGENTS, LSTM_HIDDEN, N_ENTITIES, ENTITY_DIM, SELF_DIM,
)
from battle_royale.train import _load_checkpoint, ACTION_REPEAT
from battle_royale.config import (
    AGENT_MAX_HP, ARENA_W, ARENA_H,
    ZONE_SHRINK_START, ZONE_SHRINK_END, ZONE_MAX_RADIUS, ZONE_MIN_RADIUS,
    MAX_EPISODE_FRAMES,
)

CATEGORIES = [
    "dominant_victory",
    "underdog_victory",
    "multi_kill",
    "ace",
    "sniper_shot",
    "speed_kill",
    "low_hp_clutch",
    "bullet_matrix",
    "comeback",
    "zone_surfer",
]


# ---------------------------------------------------------------------------
# RNG state save/restore
# ---------------------------------------------------------------------------

def save_rng_state(device):
    state = {"cpu": torch.random.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state

def restore_rng_state(state, device):
    torch.random.set_rng_state(state["cpu"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"], device)


# ---------------------------------------------------------------------------
# Phase 1: Fast GPU-only scan
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """Result for a single env from the scan phase."""
    env_idx: int
    winner_id: int
    winner_kills: int
    max_kills: int  # by any agent
    max_kills_agent: int
    min_hp_winner: float
    max_hp_after_min_winner: float
    ep_length: int

    @property
    def dominant_rp(self):
        if self.winner_id < 0:
            return 0.0
        k = self.winner_kills
        return 1.0 * (1.0 + 0.2 * k) + 0.15 * k


@torch.no_grad()
def scan_batch(network, num_envs, device):
    """Run one batch of envs, return scan results + RNG state for replay."""
    B = num_envs
    A = MAX_AGENTS

    # Save RNG before reset so we can replay
    rng_state = save_rng_state(device)

    sim = BatchedBRSim(num_envs=B, max_agents=A, device=str(device))
    obs_builder = ObservationBuilder(sim)
    sim.reset()

    lstm_hx = torch.zeros(B * A, LSTM_HIDDEN, device=device)
    lstm_cx = torch.zeros(B * A, LSTM_HIDDEN, device=device)

    # GPU tracking
    min_hp = torch.full((B, A), AGENT_MAX_HP, device=device)
    max_hp_after_min = torch.full((B, A), AGENT_MAX_HP, device=device)
    prev_kills = torch.zeros(B, A, dtype=torch.long, device=device)
    env_done = torch.zeros(B, dtype=torch.bool, device=device)

    step_count = 0
    while not env_done.all().item():
        step_count += 1
        if step_count > (MAX_EPISODE_FRAMES // ACTION_REPEAT) + 100:
            break

        actor_obs = obs_builder.actor_obs()
        self_feat, entities, entity_mask = pack_actor_obs(actor_obs)
        sf = self_feat.reshape(B * A, SELF_DIM)
        ent = entities.reshape(B * A, N_ENTITIES, ENTITY_DIM)
        emask = entity_mask.reshape(B * A, N_ENTITIES)

        logits, alpha, beta_param, (lstm_hx, lstm_cx) = network.forward_actor(
            sf, ent, emask, hx=lstm_hx, cx=lstm_cx)
        logits = _apply_action_masks(logits, sf)
        disc, cont = _sample_actions(logits, alpha, beta_param)

        disc_ba = disc.reshape(B, A, -1)
        cont_ba = cont.reshape(B, A)
        mx, my, aim, fire, heal = _actions_to_sim(disc_ba, cont_ba, sim.agent_dir)

        for _rep in range(ACTION_REPEAT):
            cur_alive = sim.agent_alive.clone()
            _, done = sim.step(
                mx * cur_alive.float(), my * cur_alive.float(),
                aim, fire & cur_alive, heal & cur_alive)

            alive = sim.agent_alive
            health = sim.agent_health
            active = ~env_done.unsqueeze(1)

            new_min = (health < min_hp) & alive & active
            min_hp = torch.where(new_min, health, min_hp)
            max_hp_after_min = torch.where(new_min, health, max_hp_after_min)
            new_max = (health > max_hp_after_min) & alive & active
            max_hp_after_min = torch.where(new_max, health, max_hp_after_min)

            prev_kills = sim.agent_kills.clone()
            env_done |= done
            if env_done.all():
                break

        alive_flat = sim.agent_alive.reshape(B * A)
        lstm_hx = lstm_hx * alive_flat.unsqueeze(1).float()
        lstm_cx = lstm_cx * alive_flat.unsqueeze(1).float()

    # Collect results
    kills_cpu = sim.agent_kills.cpu()
    min_hp_cpu = min_hp.cpu()
    max_hp_cpu = max_hp_after_min.cpu()
    alive_cpu = sim.agent_alive.cpu()
    done_cpu = sim.episode_done.cpu()
    frames_cpu = sim.frame.cpu()

    results = []
    for b in range(B):
        if not done_cpu[b]:
            continue
        winner = -1
        if alive_cpu[b].sum() == 1:
            winner = alive_cpu[b].nonzero(as_tuple=True)[0][0].item()
        max_k = int(kills_cpu[b].max().item())
        max_k_agent = int(kills_cpu[b].argmax().item())
        wk = int(kills_cpu[b, winner].item()) if winner >= 0 else 0
        mhw = float(min_hp_cpu[b, winner]) if winner >= 0 else AGENT_MAX_HP
        mhaw = float(max_hp_cpu[b, winner]) if winner >= 0 else AGENT_MAX_HP

        results.append(ScanResult(
            env_idx=b, winner_id=winner, winner_kills=wk,
            max_kills=max_k, max_kills_agent=max_k_agent,
            min_hp_winner=mhw, max_hp_after_min_winner=mhaw,
            ep_length=int(frames_cpu[b].item()),
        ))

    del sim, obs_builder
    return results, rng_state


# ---------------------------------------------------------------------------
# Phase 2: Re-run specific envs with frame capture
# ---------------------------------------------------------------------------

def get_states_for(sim: BatchedBRSim, env_indices: list[int]) -> dict[int, dict]:
    """Extract states only for specific env indices. One indexed GPU->CPU transfer per tensor."""
    idx = torch.tensor(env_indices, dtype=torch.long, device=sim.device)

    agent_x = sim.agent_x[idx].cpu().numpy()
    agent_y = sim.agent_y[idx].cpu().numpy()
    agent_dir = sim.agent_dir[idx].cpu().numpy()
    agent_health = sim.agent_health[idx].cpu().numpy()
    agent_alive = sim.agent_alive[idx].cpu().numpy()
    agent_ammo = sim.agent_ammo[idx].cpu().numpy()
    agent_medkits = sim.agent_medkits[idx].cpu().numpy()
    agent_heal_progress = sim.agent_heal_progress[idx].cpu().numpy()
    walls = sim.walls[idx].cpu().numpy()
    frames = sim.frame[idx].cpu().numpy()
    episode_done = sim.episode_done[idx].cpu().numpy()
    agent_kills = sim.agent_kills[idx].cpu().numpy()
    agent_last_hit_by = sim.agent_last_hit_by[idx].cpu().numpy()
    bullet_active = sim.bullet_active[idx].cpu().numpy()
    bullet_x = sim.bullet_x[idx].cpu().numpy()
    bullet_y = sim.bullet_y[idx].cpu().numpy()
    bullet_vx = sim.bullet_vx[idx].cpu().numpy()
    bullet_vy = sim.bullet_vy[idx].cpu().numpy()
    bullet_spawn_x = sim.bullet_spawn_x[idx].cpu().numpy()
    bullet_spawn_y = sim.bullet_spawn_y[idx].cpu().numpy()
    deposit_x = sim.deposit_x[idx].cpu().numpy()
    deposit_y = sim.deposit_y[idx].cpu().numpy()
    deposit_alive = sim.deposit_alive[idx].cpu().numpy()
    health_pickup_x = sim.health_pickup_x[idx].cpu().numpy()
    health_pickup_y = sim.health_pickup_y[idx].cpu().numpy()
    health_pickup_alive = sim.health_pickup_alive[idx].cpu().numpy()

    states = {}
    for i, b in enumerate(env_indices):
        frame_val = int(frames[i])
        zp = max(0.0, min(1.0, (frame_val - ZONE_SHRINK_START) / (ZONE_SHRINK_END - ZONE_SHRINK_START)))
        zone_radius = ZONE_MAX_RADIUS + (ZONE_MIN_RADIUS - ZONE_MAX_RADIUS) * zp
        active_mask = bullet_active[i]
        num_alive = int(agent_alive[i].sum())
        done = bool(episode_done[i])
        winner_id = -1
        if done and num_alive >= 1:
            alive_idx = np.where(agent_alive[i])[0]
            if len(alive_idx) == 1:
                winner_id = int(alive_idx[0])
        states[b] = {
            "agent_x": agent_x[i].copy(), "agent_y": agent_y[i].copy(),
            "agent_dir": agent_dir[i].copy(), "agent_health": agent_health[i].copy(),
            "agent_alive": agent_alive[i].copy(), "agent_ammo": agent_ammo[i].copy(),
            "agent_medkits": agent_medkits[i].copy(),
            "agent_heal_progress": agent_heal_progress[i].copy(),
            "walls": walls[i].copy(), "frame": frame_val, "num_agents": sim.A,
            "bullet_x": bullet_x[i][active_mask].copy(),
            "bullet_y": bullet_y[i][active_mask].copy(),
            "bullet_vx": bullet_vx[i][active_mask].copy(),
            "bullet_vy": bullet_vy[i][active_mask].copy(),
            "bullet_spawn_x": bullet_spawn_x[i][active_mask].copy(),
            "bullet_spawn_y": bullet_spawn_y[i][active_mask].copy(),
            "episode_done": done, "winner_id": winner_id,
            "deposit_x": deposit_x[i].copy(), "deposit_y": deposit_y[i].copy(),
            "deposit_alive": deposit_alive[i].copy(),
            "health_pickup_x": health_pickup_x[i].copy(),
            "health_pickup_y": health_pickup_y[i].copy(),
            "health_pickup_alive": health_pickup_alive[i].copy(),
            "zone_radius": zone_radius,
            "_agent_kills": agent_kills[i].copy(),
            "_agent_last_hit_by": agent_last_hit_by[i].copy(),
        }
    return states


@dataclass
class AgentTracker:
    kill_frames: list[int] = field(default_factory=list)
    min_hp: float = AGENT_MAX_HP
    min_hp_frame: int = 0
    max_hp_after_min: float = AGENT_MAX_HP
    first_damage_dealt_frame: int = -1
    near_miss_bullets: list[tuple[int, int]] = field(default_factory=list)
    zone_kills: int = 0
    total_kills: int = 0

@dataclass
class EpisodeTracker:
    num_agents: int
    agent_trackers: list[AgentTracker] = field(default_factory=list)
    critic_history: list[tuple[int, np.ndarray]] = field(default_factory=list)
    prev_kills: np.ndarray = field(default_factory=lambda: np.zeros(MAX_AGENTS, dtype=np.int64))

    def __post_init__(self):
        self.agent_trackers = [AgentTracker() for _ in range(self.num_agents)]
        self.prev_kills = np.zeros(self.num_agents, dtype=np.int64)

    def update(self, state: dict, frame: int):
        kills = state["_agent_kills"]
        alive = state["agent_alive"]
        health = state["agent_health"]
        for a in range(self.num_agents):
            new_kills = int(kills[a]) - int(self.prev_kills[a])
            if new_kills > 0:
                for _ in range(new_kills):
                    self.agent_trackers[a].kill_frames.append(frame)
                self.agent_trackers[a].total_kills += new_kills
            if alive[a]:
                hp = float(health[a])
                if hp < self.agent_trackers[a].min_hp:
                    self.agent_trackers[a].min_hp = hp
                    self.agent_trackers[a].min_hp_frame = frame
                    self.agent_trackers[a].max_hp_after_min = hp
                elif hp > self.agent_trackers[a].max_hp_after_min:
                    self.agent_trackers[a].max_hp_after_min = hp
            if self.agent_trackers[a].first_damage_dealt_frame < 0 and new_kills > 0:
                self.agent_trackers[a].first_damage_dealt_frame = frame
        if len(state["bullet_x"]) > 0:
            bx, by = state["bullet_x"], state["bullet_y"]
            for a in range(self.num_agents):
                if not alive[a]:
                    continue
                dx = bx - float(state["agent_x"][a])
                dy = by - float(state["agent_y"][a])
                near_count = int(np.sum(dx**2 + dy**2 < 900.0))  # 30px
                if near_count >= 3:
                    self.agent_trackers[a].near_miss_bullets.append((frame, near_count))
        for a in range(self.num_agents):
            nk = int(kills[a]) - int(self.prev_kills[a])
            if nk > 0:
                zone_r = state["zone_radius"]
                ax, ay = float(state["agent_x"][a]), float(state["agent_y"][a])
                if np.sqrt((ax - ARENA_W/2)**2 + (ay - ARENA_H/2)**2) > zone_r:
                    self.agent_trackers[a].zone_kills += nk
        self.prev_kills = kills.copy()

    def add_critic_values(self, frame: int, values: np.ndarray):
        self.critic_history.append((frame, values.copy()))


def score_episode(tracker: EpisodeTracker, frames: list[dict]) -> dict[str, tuple[float, dict]]:
    results = {}
    if not frames:
        return results
    last_state = frames[-1]
    winner_id = last_state.get("winner_id", -1)
    A = tracker.num_agents
    ep_length = last_state["frame"]

    if winner_id >= 0:
        kills = tracker.agent_trackers[winner_id].total_kills
        rp = 1.0 * (1.0 + 0.2 * kills) + 0.15 * kills
        results["dominant_victory"] = (rp, {
            "winner": winner_id, "kills": kills, "total_rp": round(rp, 3), "frames": ep_length})

    if winner_id >= 0 and tracker.critic_history:
        min_gap, min_gap_frame = 0.0, 0
        for cf, vals in tracker.critic_history:
            if winner_id < len(vals):
                gap = np.max(np.concatenate([vals[:winner_id], vals[winner_id+1:]])) - vals[winner_id]
                if gap > min_gap:
                    min_gap, min_gap_frame = gap, cf
        if min_gap > 0.05:
            results["underdog_victory"] = (min_gap, {
                "winner": winner_id, "value_gap": round(float(min_gap), 3), "lowest_frame": min_gap_frame})

    for a in range(A):
        kf = tracker.agent_trackers[a].kill_frames
        if len(kf) < 3:
            continue
        best = max((sum(1 for j in range(i, len(kf)) if kf[j] - kf[i] <= 300)
                     for i in range(len(kf))), default=0)
        if best >= 3:
            prev = results.get("multi_kill", (0, {}))
            if best > prev[0]:
                results["multi_kill"] = (float(best), {"agent": a, "kill_count": best})

    for a in range(A):
        if tracker.agent_trackers[a].total_kills >= A - 1:
            results["ace"] = (1.0 / max(1, ep_length), {
                "agent": a, "kills": tracker.agent_trackers[a].total_kills, "frames": ep_length})

    for f_idx in range(1, len(frames)):
        pf, cf = frames[f_idx-1], frames[f_idx]
        for a in range(A):
            if pf["agent_alive"][a] and not cf["agent_alive"][a]:
                lh = cf["_agent_last_hit_by"][a]
                if lh < 0 or not (cf["agent_alive"][lh] or pf["agent_alive"][lh]):
                    continue
                dist = np.sqrt((float(pf["agent_x"][lh]) - float(pf["agent_x"][a]))**2 +
                               (float(pf["agent_y"][lh]) - float(pf["agent_y"][a]))**2)
                if dist > 400 and dist > results.get("sniper_shot", (0, {}))[0]:
                    results["sniper_shot"] = (dist, {
                        "killer": int(lh), "victim": a, "distance": round(dist, 1), "frame": cf["frame"]})

    for a in range(A):
        at = tracker.agent_trackers[a]
        if at.first_damage_dealt_frame >= 0 and at.kill_frames:
            delta = at.kill_frames[0] - at.first_damage_dealt_frame
            if 0 <= delta <= 60:
                score = 1.0 / max(1, delta + 1)
                if score > results.get("speed_kill", (0, {}))[0]:
                    results["speed_kill"] = (score, {"agent": a, "frames_to_kill": delta})

    if winner_id >= 0:
        at = tracker.agent_trackers[winner_id]
        if 0 < at.min_hp < 15:
            results["low_hp_clutch"] = (1.0 / max(0.1, at.min_hp), {
                "winner": winner_id, "lowest_hp": round(at.min_hp, 1), "lowest_hp_frame": at.min_hp_frame})

    for a in range(A):
        at = tracker.agent_trackers[a]
        if not at.near_miss_bullets:
            continue
        near = at.near_miss_bullets
        best = 0
        for i in range(len(near)):
            t = sum(near[j][1] for j in range(i, len(near)) if near[j][0] - near[i][0] <= 30)
            if t > best:
                best = t
        if best >= 3 and last_state["agent_alive"][a]:
            if best > results.get("bullet_matrix", (0, {}))[0]:
                results["bullet_matrix"] = (float(best), {"agent": a, "bullet_count": best})

    if winner_id >= 0:
        at = tracker.agent_trackers[winner_id]
        if at.min_hp < AGENT_MAX_HP * 0.15 and at.max_hp_after_min > AGENT_MAX_HP * 0.5:
            results["comeback"] = (1.0 / max(0.1, at.min_hp), {
                "winner": winner_id, "lowest_hp": round(at.min_hp, 1),
                "recovered_to": round(at.max_hp_after_min, 1)})

    for a in range(A):
        zk = tracker.agent_trackers[a].zone_kills
        if zk > 0 and zk > results.get("zone_surfer", (0, {}))[0]:
            results["zone_surfer"] = (float(zk), {"agent": a, "zone_kills": zk})

    return results


class HighlightKeeper:
    def __init__(self, top_n=5):
        self.top_n = top_n
        self.highlights: dict[str, list[tuple[float, dict, list[dict]]]] = {c: [] for c in CATEGORIES}

    def submit(self, category, score, metadata, frames):
        heap = self.highlights[category]
        clean = [{k: v for k, v in f.items() if not k.startswith("_")} for f in frames]
        heap.append((score, metadata, clean))
        heap.sort(key=lambda x: x[0], reverse=True)
        if len(heap) > self.top_n:
            heap.pop()

    def save(self, output_dir):
        for cat in CATEGORIES:
            for i, (score, metadata, frames) in enumerate(self.highlights[cat]):
                game_dir = os.path.join(output_dir, cat, f"game_{i+1:03d}")
                os.makedirs(game_dir, exist_ok=True)
                metadata["score"] = round(score, 4)
                metadata["category"] = cat
                metadata["num_frames"] = len(frames)
                def _j(o):
                    if isinstance(o, np.integer): return int(o)
                    if isinstance(o, np.floating): return float(o)
                    if isinstance(o, np.ndarray): return o.tolist()
                    return o
                with open(os.path.join(game_dir, "metadata.json"), "w") as f:
                    json.dump({k: _j(v) for k, v in metadata.items()}, f, indent=2)
                with open(os.path.join(game_dir, "frames.pkl"), "wb") as f:
                    pickle.dump(frames, f)

    def summary(self):
        lines = []
        for cat in CATEGORIES:
            e = self.highlights[cat]
            if e:
                lines.append(f"  {cat}: {len(e)} (scores: {', '.join(f'{x[0]:.3f}' for x in e)})")
            else:
                lines.append(f"  {cat}: (none)")
        return "\n".join(lines)


@torch.no_grad()
def replay_batch(network, rng_state, num_envs, env_indices, device):
    """Re-run a batch with restored RNG, capture frames only for env_indices."""
    restore_rng_state(rng_state, device)

    B = num_envs
    A = MAX_AGENTS
    record_set = set(env_indices)

    sim = BatchedBRSim(num_envs=B, max_agents=A, device=str(device))
    obs_builder = ObservationBuilder(sim)
    sim.reset()

    lstm_hx = torch.zeros(B * A, LSTM_HIDDEN, device=device)
    lstm_cx = torch.zeros(B * A, LSTM_HIDDEN, device=device)

    frame_buffers = {b: [] for b in record_set}
    trackers = {b: EpisodeTracker(num_agents=A) for b in record_set}
    env_done = torch.zeros(B, dtype=torch.bool, device=device)

    step_count = 0
    action_step = 0

    while not env_done.all().item():
        step_count += 1
        if step_count > (MAX_EPISODE_FRAMES // ACTION_REPEAT) + 100:
            break

        actor_obs = obs_builder.actor_obs()
        self_feat, entities, entity_mask = pack_actor_obs(actor_obs)
        sf = self_feat.reshape(B * A, SELF_DIM)
        ent = entities.reshape(B * A, N_ENTITIES, ENTITY_DIM)
        emask = entity_mask.reshape(B * A, N_ENTITIES)

        logits, alpha, beta_param, (lstm_hx, lstm_cx) = network.forward_actor(
            sf, ent, emask, hx=lstm_hx, cx=lstm_cx)
        logits = _apply_action_masks(logits, sf)
        disc, cont = _sample_actions(logits, alpha, beta_param)

        disc_ba = disc.reshape(B, A, -1)
        cont_ba = cont.reshape(B, A)
        mx, my, aim, fire, heal = _actions_to_sim(disc_ba, cont_ba, sim.agent_dir)

        # Critic values for recorded envs
        if action_step % 10 == 0:
            values = network.forward_critic(sf, ent, emask, hx=lstm_hx)
            values_np = values.reshape(B, A).cpu().numpy()
            for b in record_set:
                if not env_done[b].item():
                    trackers[b].add_critic_values(int(sim.frame[b].item()), values_np[b])
        action_step += 1

        for _rep in range(ACTION_REPEAT):
            cur_alive = sim.agent_alive.clone()
            _, done = sim.step(
                mx * cur_alive.float(), my * cur_alive.float(),
                aim, fire & cur_alive, heal & cur_alive)

            # Capture frames only for recorded envs (targeted GPU->CPU)
            active_records = [b for b in record_set if not env_done[b].item()]
            if active_records:
                extracted = get_states_for(sim, active_records)
                for b, state in extracted.items():
                    frame_buffers[b].append(state)
                    trackers[b].update(state, state["frame"])

            env_done |= done
            if env_done.all():
                break

        alive_flat = sim.agent_alive.reshape(B * A)
        lstm_hx = lstm_hx * alive_flat.unsqueeze(1).float()
        lstm_cx = lstm_cx * alive_flat.unsqueeze(1).float()

    del sim, obs_builder
    return frame_buffers, trackers


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def find_default_checkpoint():
    ckpt_dir = os.path.join(PROJECT_ROOT, "battle_royale", "runs", "apex_rp", "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return None
    files = [f for f in os.listdir(ckpt_dir)
             if f.startswith("br_ppo_") and f.endswith(".pt") and f != "br_ppo_final.pt"]
    if not files:
        return None
    files.sort(key=lambda f: int(f.replace("br_ppo_", "").replace(".pt", "")))
    return os.path.join(ckpt_dir, files[-1])


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Generate highlight reels")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--scan-envs", type=int, default=10000)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--output", type=str, default="highlights")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()

    ckpt = args.checkpoint or find_default_checkpoint()
    if not ckpt or not os.path.exists(ckpt):
        print("No checkpoint found.")
        sys.exit(1)

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    network = AttentionActorCritic()
    sd, ckpt_data = _load_checkpoint(ckpt, device)
    network.load_state_dict(sd, strict=False)
    network.eval()
    network.to(device)
    print(f"Loaded: {ckpt} (update {ckpt_data.get('update_count', '?')})")

    B = args.scan_envs
    A = MAX_AGENTS

    # --- Phase 1: Scan ---
    print(f"\n=== SCAN: {B} games on {device} ===")
    t0 = time.time()
    scan_results, rng_state = scan_batch(network, B, device)
    elapsed = time.time() - t0
    print(f"Scanned {B} games in {elapsed:.1f}s ({B/elapsed:.0f} games/s)")

    # Stats
    winner_kills = [r.winner_kills for r in scan_results if r.winner_id >= 0]
    all_max_kills = [r.max_kills for r in scan_results]
    if winner_kills:
        from collections import Counter
        kc = Counter(winner_kills)
        print(f"\nWinner kill distribution (n={len(winner_kills)}):")
        for k in sorted(kc.keys()):
            print(f"  {k} kills: {kc[k]:>4d} {'#' * min(kc[k], 80)}")
        print(f"  Max winner kills: {max(winner_kills)}")
    if all_max_kills:
        print(f"  Max kills by any agent: {max(all_max_kills)}")

    if args.scan_only:
        return

    # --- Phase 2: Pick best env indices and re-run with frame capture ---
    # Score each scan result, pick top envs to record
    candidates: list[tuple[float, str, ScanResult]] = []
    for r in scan_results:
        if r.winner_id >= 0:
            candidates.append((r.dominant_rp, "dominant_victory", r))
        if r.winner_id >= 0 and 0 < r.min_hp_winner < 15:
            candidates.append((1.0 / max(0.1, r.min_hp_winner), "low_hp_clutch", r))
        if r.winner_id >= 0 and r.min_hp_winner < AGENT_MAX_HP * 0.15 and r.max_hp_after_min_winner > AGENT_MAX_HP * 0.5:
            candidates.append((1.0 / max(0.1, r.min_hp_winner), "comeback", r))
        if r.max_kills >= A - 1:
            candidates.append((1.0 / max(1, r.ep_length), "ace", r))

    # Sort by score, pick unique env indices (top 50 most interesting)
    candidates.sort(key=lambda x: x[0], reverse=True)
    env_indices = []
    seen = set()
    for score, cat, r in candidates:
        if r.env_idx not in seen:
            env_indices.append(r.env_idx)
            seen.add(r.env_idx)
        if len(env_indices) >= 50:
            break
    # Always include top dominant victories
    for r in sorted(scan_results, key=lambda r: r.dominant_rp, reverse=True)[:20]:
        if r.env_idx not in seen:
            env_indices.append(r.env_idx)
            seen.add(r.env_idx)

    if not env_indices:
        print("No interesting games found to record.")
        return

    print(f"\n=== RECORD: re-running batch, capturing {len(env_indices)} envs ===")
    t1 = time.time()
    frame_buffers, trackers = replay_batch(network, rng_state, B, env_indices, device)
    elapsed = time.time() - t1
    print(f"Recorded in {elapsed:.1f}s")

    # Score and save
    keeper = HighlightKeeper(top_n=args.top_n)
    for b in env_indices:
        frames = frame_buffers.get(b, [])
        if not frames:
            continue
        scores = score_episode(trackers[b], frames)
        for cat, (score, meta) in scores.items():
            keeper.submit(cat, score, meta, frames)

    print(f"\nSaving to {args.output}/")
    keeper.save(args.output)
    print(f"\nHighlight summary:")
    print(keeper.summary())


if __name__ == "__main__":
    main()
