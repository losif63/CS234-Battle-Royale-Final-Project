"""
PPO training for batched 1v1 battle royale (shared policy, LSTM).
All agents use the same policy. Placement rewards: +1 (1st) to -1 (last).

Usage:
    uv run python -m battle_royale.train
    uv run python -m battle_royale.train --updates 2000
    uv run python -m battle_royale.train --resume battle_royale/checkpoints/br_ppo_500.pt
"""

import argparse
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

from .sim import BatchedBRSim
from .obs import ObservationBuilder

# Re-export network symbols for backward compatibility (analysis scripts import from here)
from .network import (  # noqa: F401
    AttentionActorCritic, pack_actor_obs,
    _actions_to_sim, _apply_action_masks, _sample_actions, _greedy_actions,
    MAX_AGENTS, LSTM_HIDDEN, SELF_DIM, ENTITY_DIM, N_ENTITIES,
    MAX_VISIBLE_BULLETS, NUM_DISCRETE_ACTIONS, DISCRETE_ACTION_HEADS,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_ENVS = 30000
STEPS_PER_ROLLOUT = 32
MINI_BATCH_SIZE = 131072
PPO_EPOCHS = 3
GAMMA = 0.999
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEFF = 0.5
ENT_COEFF = 0.003
HEAL_ENT_COEFF = 0.03  # 10x base entropy for heal head to prevent exploration collapse
LR = 3e-4
MAX_GRAD_NORM = 0.5
DEFAULT_NUM_UPDATES = 1000
LOG_EVERY = 10
ACTION_REPEAT = 3
SAVE_EVERY = 100

RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")

POOL_MAX_SIZE = 20
POOL_SNAPSHOT_EVERY = 50


# ---------------------------------------------------------------------------
# Policy Pool
# ---------------------------------------------------------------------------

class PolicyPool:
    def __init__(self, max_size=POOL_MAX_SIZE):
        self.max_size = max_size
        self.snapshots: list[dict] = []
        self.elos: list[float] = []
        self.anchor: dict | None = None  # fixed reference at ELO 1000

    def add_snapshot(self, network, elo: float = 1000.0):
        sd = {k.removeprefix("_orig_mod."): v.cpu().clone()
              for k, v in network.state_dict().items()}
        if self.anchor is None:
            self.anchor = sd  # first snapshot becomes permanent anchor
        self.snapshots.append(sd)
        self.elos.append(elo)
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)
            self.elos.pop(0)

    def sample(self) -> tuple[dict | None, float, int | None]:
        """Returns (snapshot, elo, pool_index). pool_index=None for anchor."""
        if not self.snapshots:
            return None, 1000.0, None
        # 20% chance to play against the anchor (fixed ELO 1000)
        if self.anchor is not None and torch.rand(1).item() < 0.2:
            return self.anchor, 1000.0, None
        idx = torch.randint(0, len(self.snapshots), (1,)).item()
        return self.snapshots[idx], self.elos[idx], idx

    def elo_stats(self) -> tuple[float, float, float]:
        if not self.elos:
            return 1000.0, 1000.0, 1000.0
        return min(self.elos), sum(self.elos) / len(self.elos), max(self.elos)

    def __len__(self):
        return len(self.snapshots)


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(search_dir=None) -> str | None:
    import glob as g
    if search_dir is None:
        # Search all runs for the latest checkpoint
        search_dir = os.path.join(RUNS_DIR, "**", "checkpoints")
    pattern = os.path.join(search_dir, "br_ppo_*.pt")
    files = g.glob(pattern, recursive=True)
    if not files:
        final = os.path.join(search_dir, "br_ppo_final.pt")
        return final if os.path.exists(final) else None
    def _num(f):
        s = os.path.basename(f).replace("br_ppo_", "").replace(".pt", "")
        try:
            return int(s)
        except ValueError:
            return -1
    return max(files, key=_num)


def _load_checkpoint(path, device):
    """Load checkpoint, stripping _orig_mod. prefix for uncompiled networks."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["network"].items()}
    return sd, ckpt


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(run_name, num_updates=DEFAULT_NUM_UPDATES, resume=None, resume_weights_only=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    B, A, T = NUM_ENVS, MAX_AGENTS, STEPS_PER_ROLLOUT

    sim = BatchedBRSim(num_envs=B, max_agents=A, device=str(device))
    sim.step = torch.compile(sim.step, fullgraph=False)
    sim.reset = torch.compile(sim.reset, fullgraph=False)
    obs_builder = ObservationBuilder(sim)
    obs_builder.actor_obs = torch.compile(obs_builder.actor_obs, fullgraph=False)
    network = AttentionActorCritic().to(device)
    network = torch.compile(network)
    optimizer = optim.Adam(network.parameters(), lr=LR, eps=1e-5)

    # Opponent network (compiled — weight changes don't trigger recompilation)
    opp_network = AttentionActorCritic().to(device)
    opp_network = torch.compile(opp_network)
    opp_network.eval()

    pool = PolicyPool()
    learner_elo = 1000.0
    pool.add_snapshot(network, elo=learner_elo)  # seed with initial weights

    start_update = 0
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["network"].items()}
        # Filter out keys with shape mismatches (e.g. critic_mlp after architecture change)
        model_sd = network._orig_mod.state_dict()
        sd = {k: v for k, v in sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        network._orig_mod.load_state_dict(sd, strict=False)
        if resume_weights_only:
            pool.add_snapshot(network, elo=learner_elo)
            print(f"Loaded weights from {resume} (weights only, fresh optimizer/pool)")
        else:
            optimizer.load_state_dict(ckpt["optimizer"])
            start_update = ckpt["update_count"]
            if "pool_snapshots" in ckpt:
                pool.snapshots = ckpt["pool_snapshots"]
            if "pool_elos" in ckpt:
                pool.elos = ckpt["pool_elos"]
            else:
                pool.elos = [1000.0] * len(pool.snapshots)
            if "pool_anchor" in ckpt:
                pool.anchor = ckpt["pool_anchor"]
            learner_elo = ckpt.get("learner_elo", 1000.0)
            print(f"Resumed from {resume} (update {start_update}, elo {learner_elo:.0f})")

    print(f"Envs: {B}  Agents: {A}  Rollout: {T}  "
          f"Trans/update: {B*T:,}  Updates: {num_updates}")
    print(f"Self dim: {SELF_DIM}  Entity dim: {ENTITY_DIM}  N entities: {N_ENTITIES}  "
          f"LSTM: {LSTM_HIDDEN}  Pool max: {POOL_MAX_SIZE}")
    print()

    # Per-run directories
    run_dir = os.path.join(RUNS_DIR, run_name)
    save_dir = os.path.join(run_dir, "checkpoints")
    writer = SummaryWriter(log_dir=run_dir)

    # Rollout buffers — learner (agent 0) only, shape (T, B)
    buf_self     = torch.zeros(T, B, SELF_DIM, device=device)
    buf_entities = torch.zeros(T, B, N_ENTITIES, ENTITY_DIM, device=device)
    buf_mask     = torch.zeros(T, B, N_ENTITIES, dtype=torch.bool, device=device)
    buf_disc_actions = torch.zeros(T, B, NUM_DISCRETE_ACTIONS, dtype=torch.long, device=device)
    buf_cont_actions = torch.zeros(T, B, device=device)
    buf_log_probs = torch.zeros(T, B, device=device)
    buf_values   = torch.zeros(T, B, device=device)
    buf_rewards  = torch.zeros(T, B, device=device)
    buf_dones    = torch.zeros(T, B, device=device)
    buf_alive    = torch.zeros(T, B, dtype=torch.bool, device=device)
    buf_hx       = torch.zeros(T, B, LSTM_HIDDEN, device=device)
    buf_cx       = torch.zeros(T, B, LSTM_HIDDEN, device=device)

    total_frames = 0
    wall_start = time.time()

    # Rolling metrics (GPU-side, synced only at log time)
    rollout_ep_len_sum = torch.zeros(1, device=device)
    rollout_ep_count = torch.zeros(1, device=device)
    rollout_win_sum = torch.zeros(1, device=device)
    rollout_win_count = torch.zeros(1, device=device)

    use_cuda = device.type == "cuda"
    def _sync_time():
        if use_cuda:
            torch.cuda.synchronize()
        return time.time()

    t_rollout_acc = 0.0
    t_bootstrap_acc = 0.0
    t_gae_acc = 0.0
    t_ppo_acc = 0.0
    t_total_acc = 0.0

    # Persistent LSTM states
    learner_hx = torch.zeros(B, LSTM_HIDDEN, device=device)
    learner_cx = torch.zeros(B, LSTM_HIDDEN, device=device)
    opp_hx = torch.zeros(B * (A - 1), LSTM_HIDDEN, device=device)
    opp_cx = torch.zeros(B * (A - 1), LSTM_HIDDEN, device=device)

    for update in range(start_update, start_update + num_updates):
        update_start = _sync_time()

        # LR annealing
        frac = 1.0 - (update - start_update) / max(1, num_updates)
        for pg in optimizer.param_groups:
            pg["lr"] = LR * frac

        # Sample opponent snapshot for this rollout
        snapshot, opp_elo, opp_pool_idx = pool.sample()
        if snapshot is not None:
            opp_network._orig_mod.load_state_dict(snapshot, strict=False)

        rollout_ep_len_sum.zero_()
        rollout_ep_count.zero_()
        rollout_win_sum.zero_()
        rollout_win_count.zero_()

        # ----- Collect rollout -----
        for t in range(T):
            actor_obs = obs_builder.actor_obs()
            alive = sim.agent_alive.clone()  # (B, A)

            self_feat, entities, entity_mask = pack_actor_obs(actor_obs)

            # Learner (agent 0)
            sf_l    = self_feat[:, 0]        # (B, SELF_DIM)
            ent_l   = entities[:, 0]          # (B, N, ENTITY_DIM)
            emask_l = entity_mask[:, 0]       # (B, N)

            # Opponents (agents 1..A-1) — reshape to (B*(A-1), ...)
            sf_o    = self_feat[:, 1:].reshape(B * (A - 1), SELF_DIM)
            ent_o   = entities[:, 1:].reshape(B * (A - 1), N_ENTITIES, ENTITY_DIM)
            emask_o = entity_mask[:, 1:].reshape(B * (A - 1), N_ENTITIES)

            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
                # Learner: get action + value + lstm update
                l_disc, l_cont, l_lp, _, _, l_val, l_hx_out, l_cx_out = network.get_action_and_value(
                    sf_l, ent_l, emask_l, hx=learner_hx, cx=learner_cx)

                # Opponents: actions only (no grad, no buffer)
                o_logits, o_alpha, o_beta, (o_hx_out, o_cx_out) = opp_network.forward_actor(
                    sf_o, ent_o, emask_o, hx=opp_hx, cx=opp_cx)
                o_logits = _apply_action_masks(o_logits, sf_o)
                o_disc, o_cont = _sample_actions(o_logits, o_alpha, o_beta)

            # Combine discrete actions: (B, A, 4)
            all_disc = torch.empty(B, A, NUM_DISCRETE_ACTIONS, dtype=torch.long, device=device)
            all_disc[:, 0] = l_disc
            all_disc[:, 1:] = o_disc.reshape(B, A - 1, NUM_DISCRETE_ACTIONS)

            # Combine continuous rotation: (B, A)
            all_cont = torch.empty(B, A, device=device)
            all_cont[:, 0] = l_cont
            all_cont[:, 1:] = o_cont.reshape(B, A - 1)

            # Convert to sim inputs
            mx, my, aim, fire, heal = _actions_to_sim(all_disc, all_cont, sim.agent_dir)

            # Step sim with ACTION_REPEAT
            step_rewards = torch.zeros(B, A, device=device)
            agent_done = torch.zeros(B, A, dtype=torch.bool, device=device)

            for _rep in range(ACTION_REPEAT):
                cur_alive = sim.agent_alive.clone()
                move_x = mx * cur_alive
                move_y = my * cur_alive
                fire_bool = fire & cur_alive
                heal_bool = heal & cur_alive

                rewards, episode_done = sim.step(move_x, move_y, aim, fire_bool, heal_bool)
                step_rewards += rewards
                agent_done |= cur_alive & (~sim.agent_alive | episode_done[:, None])

                # Track episode lengths before reset clears frame counter
                rollout_ep_len_sum += (episode_done.float() * sim.frame.float()).sum()

                # Always reset done envs (branchless — no-op when mask is all False)
                sim.reset(mask=episode_done)

            # Accumulate stats on GPU (no CPU sync)
            done_count = agent_done[:, 0].sum()
            rollout_ep_count += done_count
            rollout_win_sum += (agent_done[:, 0] & sim.agent_alive[:, 0]).float().sum()
            rollout_win_count += done_count

            # Update LSTM states
            learner_done = agent_done[:, 0]  # (B,)
            learner_hx = torch.where(
                learner_done[:, None], torch.zeros_like(l_hx_out), l_hx_out).detach()
            learner_cx = torch.where(
                learner_done[:, None], torch.zeros_like(l_cx_out), l_cx_out).detach()

            opp_done = agent_done[:, 1:].reshape(B * (A - 1))  # (B*(A-1),)
            opp_hx = torch.where(
                opp_done[:, None], torch.zeros_like(o_hx_out), o_hx_out).detach()
            opp_cx = torch.where(
                opp_done[:, None], torch.zeros_like(o_cx_out), o_cx_out).detach()

            # Store in buffers (learner only)
            buf_self[t]      = sf_l
            buf_entities[t]  = ent_l
            buf_mask[t]      = emask_l
            buf_disc_actions[t] = l_disc
            buf_cont_actions[t] = l_cont
            buf_log_probs[t] = l_lp
            buf_values[t]    = l_val
            buf_rewards[t]   = step_rewards[:, 0]
            buf_dones[t]     = agent_done[:, 0].float()
            buf_alive[t]     = alive[:, 0]
            buf_hx[t]        = learner_hx
            buf_cx[t]        = learner_cx

        t_after_rollout = _sync_time()

        total_frames += B * T * A * ACTION_REPEAT

        # ----- Bootstrap (agent 0 only) -----
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
            boot_obs = obs_builder.actor_obs()
            boot_sf, boot_ent, boot_emask = pack_actor_obs(boot_obs)
            bootstrap = network.forward_critic(
                boot_sf[:, 0],
                boot_ent[:, 0],
                boot_emask[:, 0],
                hx=learner_hx).float()
            bootstrap = bootstrap * sim.agent_alive[:, 0].float()

        t_after_bootstrap = _sync_time()

        # ----- GAE -----
        advantages = torch.zeros_like(buf_rewards)  # (T, B)
        last_gae = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            next_val = bootstrap if t == T - 1 else buf_values[t + 1]
            not_done = 1.0 - buf_dones[t]
            delta = buf_rewards[t] + GAMMA * next_val * not_done - buf_values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * not_done * last_gae
            advantages[t] = last_gae
        returns = advantages + buf_values

        t_after_gae = _sync_time()

        # ----- PPO update -----
        alive_mask = buf_alive.reshape(-1)  # (T*B,)
        N = int(alive_mask.sum().item())

        flat_self     = buf_self.reshape(-1, SELF_DIM)[alive_mask]
        flat_ent      = buf_entities.reshape(-1, N_ENTITIES, ENTITY_DIM)[alive_mask]
        flat_emask    = buf_mask.reshape(-1, N_ENTITIES)[alive_mask]
        flat_disc_act = buf_disc_actions.reshape(-1, NUM_DISCRETE_ACTIONS)[alive_mask]
        flat_cont_act = buf_cont_actions.reshape(-1)[alive_mask]
        flat_lp       = buf_log_probs.reshape(-1)[alive_mask]
        flat_adv      = advantages.reshape(-1)[alive_mask]
        flat_ret      = returns.reshape(-1)[alive_mask]
        flat_hx       = buf_hx.reshape(-1, LSTM_HIDDEN)[alive_mask]
        flat_cx       = buf_cx.reshape(-1, LSTM_HIDDEN)[alive_mask]

        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

        total_ploss = total_vloss = total_ent = 0.0
        n_mbs = 0
        scaler = torch.amp.GradScaler('cuda')

        for _ in range(PPO_EPOCHS):
            idx = torch.randperm(N, device=device)
            for start in range(0, N, MINI_BATCH_SIZE):
                mb = idx[start:start + MINI_BATCH_SIZE]
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    _, _, new_lp, base_ent, heal_ent, new_val, _, _ = network.get_action_and_value(
                        flat_self[mb], flat_ent[mb], flat_emask[mb],
                        hx=flat_hx[mb], cx=flat_cx[mb],
                        discrete_actions=flat_disc_act[mb],
                        continuous_actions=flat_cont_act[mb])

                    ratio = torch.exp(new_lp - flat_lp[mb])
                    surr1 = ratio * flat_adv[mb]
                    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * flat_adv[mb]
                    ploss = -torch.min(surr1, surr2).mean()
                    vloss = nn.functional.mse_loss(new_val, flat_ret[mb])
                    eloss = -base_ent.mean()
                    heal_eloss = -heal_ent.mean()

                    loss = ploss + VF_COEFF * vloss + ENT_COEFF * eloss + HEAL_ENT_COEFF * heal_eloss

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(network.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()

                total_ploss += ploss.item()
                total_vloss += vloss.item()
                total_ent += -eloss.item() + -heal_eloss.item()
                n_mbs += 1

        t_after_ppo = _sync_time()

        # ELO update (zero-sum: learner gains = opponent loses)
        # Single sync point for stats
        rollout_games = int(rollout_win_count.item())
        rollout_wins = rollout_win_sum.item()
        if rollout_games > 0:
            win_rate = rollout_wins / rollout_games
            baseline = 1.0 / A  # 0.5 for 1v1, 0.25 for 4 players
            # Map win_rate so that baseline -> 0.5 for ELO formula
            if win_rate <= baseline:
                score = 0.5 * (win_rate / baseline)
            else:
                score = 0.5 + 0.5 * ((win_rate - baseline) / (1.0 - baseline))
            expected = 1.0 / (1.0 + 10 ** ((opp_elo - learner_elo) / 400))
            elo_delta = 32 * (score - expected)
            learner_elo += elo_delta
            # Zero-sum: opponent loses what learner gains
            if opp_pool_idx is not None:
                pool.elos[opp_pool_idx] -= elo_delta

        # Snapshot
        if (update + 1) % POOL_SNAPSHOT_EVERY == 0:
            pool.add_snapshot(network, elo=learner_elo)

        # Accumulate phase times
        t_rollout_acc += t_after_rollout - update_start
        t_bootstrap_acc += t_after_bootstrap - t_after_rollout
        t_gae_acc += t_after_gae - t_after_bootstrap
        t_ppo_acc += t_after_ppo - t_after_gae
        t_total_acc += t_after_ppo - update_start

        # ----- Logging -----
        l_win_rate = win_rate if rollout_games > 0 else 0.0
        update_time = time.time() - update_start
        fps = (B * T * A) / update_time

        if (update + 1) % LOG_EVERY == 0 or update == start_update:
            ep_count = rollout_ep_count.item()
            avg_ep_len = rollout_ep_len_sum.item() / max(1, ep_count)
            avg_rew = buf_rewards.mean().item()
            pool_min, pool_avg, pool_max = pool.elo_stats()
            print(
                f"Update {update+1:>5d}/{start_update+num_updates} | "
                f"FPS {fps:>10,.0f} | "
                f"ep_len {avg_ep_len:>3.0f} | "
                f"L_win {l_win_rate:.2f} | "
                f"elo {learner_elo:>6.0f} | "
                f"pool {len(pool)} [{pool_min:.0f}/{pool_avg:.0f}/{pool_max:.0f}] | "
                f"avg_rew {avg_rew:.3f} | "
                f"pl {total_ploss/n_mbs:.2f} vl {total_vloss/n_mbs:.2f} "
                f"ent {total_ent/n_mbs:.1f}")
            n_log = LOG_EVERY if (update + 1) % LOG_EVERY == 0 else 1
            print(
                f"  timing: rollout {t_rollout_acc/n_log:.2f}s "
                f" bootstrap {t_bootstrap_acc/n_log:.2f}s "
                f" gae {t_gae_acc/n_log:.2f}s "
                f" ppo {t_ppo_acc/n_log:.2f}s "
                f" total {t_total_acc/n_log:.2f}s")
            t_rollout_acc = t_bootstrap_acc = t_gae_acc = t_ppo_acc = t_total_acc = 0.0

            # TensorBoard
            step = update + 1
            writer.add_scalar("perf/fps", fps, step)
            writer.add_scalar("episode/length", avg_ep_len, step)
            writer.add_scalar("episode/reward", avg_rew, step)
            writer.add_scalar("episode/win_rate", l_win_rate, step)
            writer.add_scalar("elo/learner", learner_elo, step)
            writer.add_scalar("elo/pool_min", pool_min, step)
            writer.add_scalar("elo/pool_avg", pool_avg, step)
            writer.add_scalar("elo/pool_max", pool_max, step)
            writer.add_scalar("pool/size", len(pool), step)
            writer.add_scalar("loss/policy", total_ploss / n_mbs, step)
            writer.add_scalar("loss/value", total_vloss / n_mbs, step)
            writer.add_scalar("loss/entropy", total_ent / n_mbs, step)

        # ----- Save -----
        if (update + 1) % SAVE_EVERY == 0:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"br_ppo_{update+1}.pt")
            torch.save({
                "network": network.state_dict(),
                "optimizer": optimizer.state_dict(),
                "update_count": update + 1,
                "pool_snapshots": pool.snapshots,
                "pool_elos": pool.elos,
                "pool_anchor": pool.anchor,
                "learner_elo": learner_elo,
            }, path)
            print(f"  -> saved {path}")

    # Final save
    os.makedirs(save_dir, exist_ok=True)
    final_path = os.path.join(save_dir, "br_ppo_final.pt")
    torch.save({
        "network": network.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update_count": start_update + num_updates,
        "pool_snapshots": pool.snapshots,
        "pool_elos": pool.elos,
        "pool_anchor": pool.anchor,
        "learner_elo": learner_elo,
    }, final_path)

    writer.close()

    total_time = time.time() - wall_start
    print(f"\nDone. {num_updates} updates in {total_time:.1f}s")
    print(f"Total frames: {total_frames:,}")
    print(f"Average FPS: {total_frames/total_time:,.0f}")
    print(f"Saved to {final_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None,
                        help="Run name (required for training). Checkpoints and logs saved to runs/<name>/")
    parser.add_argument("--updates", type=int, default=DEFAULT_NUM_UPDATES)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-weights-only", action="store_true",
                        help="Only load network weights from --resume (fresh optimizer/pool)")
    parser.add_argument("--watch", action="store_true", help="Watch trained agents fight")
    parser.add_argument("--play", action="store_true",
                        help="Play as Agent 0 against the trained AI (Agent 1)")
    parser.add_argument("--debug-lidar", action="store_true",
                        help="Debug lidar visualization with WASD control")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path for --watch / --play mode")
    parser.add_argument("--fp", action="store_true",
                        help="First-person camera (--play mode only): follow player with FOV fog of war")
    args = parser.parse_args()

    if args.watch:
        from .watch import watch
        watch(checkpoint=args.checkpoint, run_name=args.run)
    elif args.play:
        from .watch import play
        play(checkpoint=args.checkpoint, run_name=args.run, first_person=args.fp)
    elif args.debug_lidar:
        from .watch import debug_lidar
        debug_lidar()
    else:
        if not args.run:
            parser.error("--run is required for training (e.g. --run 1v2_heal_entropy)")
        train(args.run, num_updates=args.updates, resume=args.resume,
              resume_weights_only=args.resume_weights_only)


if __name__ == "__main__":
    main()
