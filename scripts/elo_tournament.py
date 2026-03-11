"""
ELO Tournament: Measure skill progression across training checkpoints.

Loads checkpoints from battle_royale/runs/apex_rp/checkpoints/, runs FFA matches
with 10 agents per game (each from a different checkpoint), and computes ELO ratings
from placement results.

GPU-batched: runs BATCH_GAMES games in parallel, grouping network forward passes
by checkpoint for efficient GPU utilization.

Usage:
    uv run python scripts/elo_tournament.py
"""

import os
import sys
import csv
import random
import time

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from battle_royale.sim import BatchedBRSim
from battle_royale.obs import ObservationBuilder
from battle_royale.network import (
    AttentionActorCritic, pack_actor_obs,
    _actions_to_sim, _apply_action_masks, _sample_actions,
    MAX_AGENTS, LSTM_HIDDEN, N_ENTITIES, ENTITY_DIM, SELF_DIM,
    NUM_DISCRETE_ACTIONS,
)
from battle_royale.train import _load_checkpoint, ACTION_REPEAT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "battle_royale", "runs", "apex_rp", "checkpoints")
NUM_GAMES = 500
K_FACTOR = 24
NUM_AGENTS_PER_GAME = MAX_AGENTS  # 10
TARGET_CHECKPOINTS = 35  # subsample to ~35 checkpoints for speed
BATCH_GAMES = 200  # number of games to run in parallel
RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elo_results.csv")
RESULTS_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "elo_progression.png")


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------
def discover_checkpoints(ckpt_dir, target_count=TARGET_CHECKPOINTS):
    """Find all checkpoints and sample at regular intervals."""
    files = [f for f in os.listdir(ckpt_dir) if f.startswith("br_ppo_") and f.endswith(".pt") and f != "br_ppo_final.pt"]

    def extract_update(fname):
        return int(fname.replace("br_ppo_", "").replace(".pt", ""))

    files_with_num = [(f, extract_update(f)) for f in files]
    files_with_num.sort(key=lambda x: x[1])

    if len(files_with_num) <= target_count:
        return files_with_num

    # Sample at regular intervals
    step = max(1, len(files_with_num) // target_count)
    sampled = files_with_num[::step]
    # Always include first and last
    if files_with_num[0] not in sampled:
        sampled.insert(0, files_with_num[0])
    if files_with_num[-1] not in sampled:
        sampled.append(files_with_num[-1])

    return sampled


# ---------------------------------------------------------------------------
# Network cache (load each checkpoint once, on CPU then move to device)
# ---------------------------------------------------------------------------
def load_networks(ckpt_dir, checkpoints, device):
    """Load all checkpoint networks into memory (on device)."""
    networks = {}
    for fname, update_num in checkpoints:
        path = os.path.join(ckpt_dir, fname)
        net = AttentionActorCritic().to(device)
        # Load checkpoint to CPU first to avoid loading optimizer state to GPU
        sd, _ = _load_checkpoint(path, "cpu")
        net.load_state_dict(sd, strict=False)
        net.eval()
        networks[update_num] = net
        print(f"  Loaded {fname} (update {update_num})")
    return networks


# ---------------------------------------------------------------------------
# Run a batch of FFA games in parallel (GPU-batched)
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_batch(sim, obs_builder, networks, all_participant_updates, device):
    """Run B games in parallel. Returns list of (placements, participant_updates) per game.

    all_participant_updates: list of B lists, each of length A (update_num per agent slot)
    """
    B = len(all_participant_updates)
    A = NUM_AGENTS_PER_GAME

    # Build mapping: for each (env, agent), which network index to use
    # We'll assign each unique update_num an integer index for fast grouping
    update_to_idx = {}
    for participants in all_participant_updates:
        for u in participants:
            if u not in update_to_idx:
                update_to_idx[u] = len(update_to_idx)
    idx_to_update = {v: k for k, v in update_to_idx.items()}
    num_nets = len(update_to_idx)

    # net_assignment[b, a] = network index for (env b, agent a)
    net_assignment = torch.zeros(B, A, dtype=torch.long, device=device)
    for b, participants in enumerate(all_participant_updates):
        for a, u in enumerate(participants):
            net_assignment[b, a] = update_to_idx[u]

    # Ordered list of networks
    net_list = [networks[idx_to_update[i]] for i in range(num_nets)]

    sim.reset()

    # LSTM states: (B, A, LSTM_HIDDEN)
    lstm_hx = torch.zeros(B, A, LSTM_HIDDEN, device=device)
    lstm_cx = torch.zeros(B, A, LSTM_HIDDEN, device=device)

    max_steps = 5000  # safety limit
    step_count = 0

    while not sim.episode_done.all().item() and step_count < max_steps:
        step_count += 1

        actor_obs = obs_builder.actor_obs()
        self_feat, entities, entity_mask = pack_actor_obs(actor_obs)
        # self_feat: (B, A, SELF_DIM), entities: (B, A, N, ENTITY_DIM), entity_mask: (B, A, N)

        # Flatten (B, A) -> (B*A,) for batched inference
        BA = B * A
        sf_flat = self_feat.reshape(BA, SELF_DIM)
        ent_flat = entities.reshape(BA, N_ENTITIES, ENTITY_DIM)
        emask_flat = entity_mask.reshape(BA, N_ENTITIES)
        hx_flat = lstm_hx.reshape(BA, LSTM_HIDDEN)
        cx_flat = lstm_cx.reshape(BA, LSTM_HIDDEN)
        net_flat = net_assignment.reshape(BA)  # (B*A,)

        # Which agents are alive? Dead agents don't need inference
        alive_flat = sim.agent_alive.reshape(BA)
        # Also skip agents in finished episodes
        active_env = ~sim.episode_done  # (B,)
        active_flat = active_env.unsqueeze(1).expand(B, A).reshape(BA) & alive_flat

        # Output tensors
        all_disc = torch.zeros(BA, NUM_DISCRETE_ACTIONS, dtype=torch.long, device=device)
        all_cont = torch.zeros(BA, device=device)
        new_hx = hx_flat.clone()
        new_cx = cx_flat.clone()

        # Group by network and batch forward passes
        for net_idx in range(num_nets):
            mask = (net_flat == net_idx) & active_flat  # (BA,)
            if not mask.any():
                continue

            indices = mask.nonzero(as_tuple=True)[0]  # (K,)
            sf_batch = sf_flat[indices]       # (K, SELF_DIM)
            ent_batch = ent_flat[indices]     # (K, N, ENTITY_DIM)
            emask_batch = emask_flat[indices] # (K, N)
            hx_batch = hx_flat[indices]       # (K, LSTM_HIDDEN)
            cx_batch = cx_flat[indices]       # (K, LSTM_HIDDEN)

            logits, alpha, beta_param, (hx_out, cx_out) = net_list[net_idx].forward_actor(
                sf_batch, ent_batch, emask_batch, hx=hx_batch, cx=cx_batch)

            logits = _apply_action_masks(logits, sf_batch)
            disc, cont = _sample_actions(logits, alpha, beta_param)

            all_disc[indices] = disc
            all_cont[indices] = cont
            new_hx[indices] = hx_out
            new_cx[indices] = cx_out

        # Unflatten back to (B, A, ...)
        lstm_hx = new_hx.reshape(B, A, LSTM_HIDDEN)
        lstm_cx = new_cx.reshape(B, A, LSTM_HIDDEN)
        all_disc = all_disc.reshape(B, A, NUM_DISCRETE_ACTIONS)
        all_cont = all_cont.reshape(B, A)

        # Convert to sim inputs
        mx, my, aim, fire, heal = _actions_to_sim(all_disc, all_cont, sim.agent_dir)

        # Step with action repeat
        for _rep in range(ACTION_REPEAT):
            cur_alive = sim.agent_alive.clone()
            move_x = mx * cur_alive.float()
            move_y = my * cur_alive.float()
            fire_bool = fire & cur_alive
            heal_bool = heal & cur_alive

            _, done = sim.step(move_x, move_y, aim, fire_bool, heal_bool)
            if done.all().item():
                break

        # Reset LSTM for dead agents
        dead = ~sim.agent_alive  # (B, A)
        lstm_hx = lstm_hx * (~dead).unsqueeze(-1).float()
        lstm_cx = lstm_cx * (~dead).unsqueeze(-1).float()

    # Read placements for each game
    results = []
    for b in range(B):
        placements = sim.agent_place[b].tolist()
        results.append((placements, all_participant_updates[b]))

    return results


# ---------------------------------------------------------------------------
# Multi-player ELO update
# ---------------------------------------------------------------------------
def update_elo(elo_ratings, placements, participant_updates, k_factor=K_FACTOR):
    """Update ELO based on pairwise comparisons from placement.

    For each pair (i, j) where i placed better (lower number) than j:
      expected_i = 1 / (1 + 10^((elo_j - elo_i)/400))
      elo_i += K * (1 - expected_i) / num_opponents
      elo_j += K * (0 - expected_j) / num_opponents
    """
    n = len(placements)
    num_opponents = n - 1

    # Collect deltas
    deltas = {u: 0.0 for u in participant_updates}

    for i in range(n):
        for j in range(i + 1, n):
            u_i = participant_updates[i]
            u_j = participant_updates[j]
            elo_i = elo_ratings[u_i]
            elo_j = elo_ratings[u_j]

            expected_i = 1.0 / (1.0 + 10.0 ** ((elo_j - elo_i) / 400.0))
            expected_j = 1.0 - expected_i

            # Lower placement number = better
            if placements[i] < placements[j]:
                # i beat j
                deltas[u_i] += k_factor * (1.0 - expected_i) / num_opponents
                deltas[u_j] += k_factor * (0.0 - expected_j) / num_opponents
            elif placements[i] > placements[j]:
                # j beat i
                deltas[u_i] += k_factor * (0.0 - expected_i) / num_opponents
                deltas[u_j] += k_factor * (1.0 - expected_j) / num_opponents
            else:
                # tie — 0.5
                deltas[u_i] += k_factor * (0.5 - expected_i) / num_opponents
                deltas[u_j] += k_factor * (0.5 - expected_j) / num_opponents

    for u in participant_updates:
        elo_ratings[u] += deltas[u]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Batch size: {BATCH_GAMES} games in parallel")
    print()

    # Discover checkpoints
    checkpoints = discover_checkpoints(CHECKPOINT_DIR, TARGET_CHECKPOINTS)
    print(f"Found {len(checkpoints)} checkpoints:")
    for fname, update_num in checkpoints:
        print(f"  {fname} (update {update_num})")
    print()

    if len(checkpoints) < 2:
        print("Need at least 2 checkpoints to run a tournament.")
        return

    # Load all networks
    print("Loading networks...")
    networks = load_networks(CHECKPOINT_DIR, checkpoints, device)
    print(f"Loaded {len(networks)} networks.\n")

    # Initialize ELO
    update_nums = sorted(networks.keys())
    elo_ratings = {u: 1000.0 for u in update_nums}
    games_played = {u: 0 for u in update_nums}

    # Initialize sim with BATCH_GAMES parallel envs on GPU
    sim = BatchedBRSim(num_envs=BATCH_GAMES, max_agents=NUM_AGENTS_PER_GAME, device=str(device))
    obs_builder = ObservationBuilder(sim)

    num_batches = (NUM_GAMES + BATCH_GAMES - 1) // BATCH_GAMES
    total_games = num_batches * BATCH_GAMES  # may be slightly more than NUM_GAMES

    print(f"Running {total_games} games ({num_batches} batches of {BATCH_GAMES}) "
          f"with {NUM_AGENTS_PER_GAME} agents each...")
    print()

    total_start = time.time()
    games_completed = 0

    for batch_idx in range(num_batches):
        batch_start = time.time()

        # Sample participants for each game in this batch
        all_participants = []
        for _ in range(BATCH_GAMES):
            if len(update_nums) >= NUM_AGENTS_PER_GAME:
                participants = random.sample(update_nums, NUM_AGENTS_PER_GAME)
            else:
                participants = random.choices(update_nums, k=NUM_AGENTS_PER_GAME)
            all_participants.append(participants)

        # Run batch of games
        results = run_batch(sim, obs_builder, networks, all_participants, device)

        # Update ELO for each game
        for placements, participants in results:
            update_elo(elo_ratings, placements, participants)
            for u in participants:
                games_played[u] += 1

        games_completed += BATCH_GAMES
        batch_time = time.time() - batch_start
        elapsed = time.time() - total_start
        eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)

        # Show current top/bottom
        sorted_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_elo[:3]
        bot3 = sorted_elo[-3:]

        print(f"Batch {batch_idx+1:>2d}/{num_batches} ({games_completed:>3d} games) | "
              f"{batch_time:.1f}s ({batch_time/BATCH_GAMES:.2f}s/game) | "
              f"elapsed {elapsed:.0f}s | "
              f"ETA {eta:.0f}s")
        print(f"  Top 3: {', '.join(f'u{u}={elo:.0f}' for u, elo in top3)}")
        print(f"  Bot 3: {', '.join(f'u{u}={elo:.0f}' for u, elo in bot3)}")

    total_time = time.time() - total_start
    print(f"\nTournament complete in {total_time:.1f}s ({total_time/games_completed:.2f}s/game)")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    print(f"\nSaving results to {RESULTS_CSV}")
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["checkpoint_name", "update_number", "elo_rating", "games_played"])
        for u in sorted(update_nums):
            writer.writerow([f"br_ppo_{u}.pt", u, f"{elo_ratings[u]:.1f}", games_played[u]])

    # Print final rankings
    print("\n" + "=" * 60)
    print("Final ELO Rankings")
    print("=" * 60)
    sorted_elo = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)
    for rank, (u, elo) in enumerate(sorted_elo, 1):
        print(f"  #{rank:>2d}  update {u:>5d}  ELO {elo:>7.1f}  ({games_played[u]} games)")

    # ---------------------------------------------------------------------------
    # Plot
    # ---------------------------------------------------------------------------
    print(f"\nSaving plot to {RESULTS_PNG}")
    fig, ax = plt.subplots(figsize=(12, 6))

    updates_sorted = sorted(update_nums)
    elos_sorted = [elo_ratings[u] for u in updates_sorted]

    ax.plot(updates_sorted, elos_sorted, "o-", color="#2196F3", linewidth=2, markersize=6)
    ax.axhline(y=1000, color="gray", linestyle="--", alpha=0.5, label="Starting ELO (1000)")

    ax.set_xlabel("Training Update", fontsize=13)
    ax.set_ylabel("ELO Rating", fontsize=13)
    ax.set_title("ELO Progression Across Training Checkpoints", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate min and max
    max_u = max(elo_ratings, key=elo_ratings.get)
    min_u = min(elo_ratings, key=elo_ratings.get)
    ax.annotate(f"Peak: {elo_ratings[max_u]:.0f}\n(update {max_u})",
                xy=(max_u, elo_ratings[max_u]),
                xytext=(max_u, elo_ratings[max_u] + 30),
                fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="green"),
                color="green")
    ax.annotate(f"Min: {elo_ratings[min_u]:.0f}\n(update {min_u})",
                xy=(min_u, elo_ratings[min_u]),
                xytext=(min_u, elo_ratings[min_u] - 40),
                fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red")

    plt.tight_layout()
    plt.savefig(RESULTS_PNG, dpi=150)
    plt.close()

    print(f"\nDone! Results: {RESULTS_CSV}")
    print(f"Plot: {RESULTS_PNG}")


if __name__ == "__main__":
    main()
