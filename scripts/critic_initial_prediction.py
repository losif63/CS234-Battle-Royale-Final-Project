"""
Test how well the critic predicts the winner from the INITIAL game state (frame 0).

Measures how much random starting positions/wall layouts determine the outcome.
"""

import os
import sys
import math
import torch
import numpy as np
from scipy import stats

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.dirname(__file__))

from battle_royale.sim import BatchedBRSim
from battle_royale.obs import ObservationBuilder
from battle_royale.network import (
    AttentionActorCritic, pack_actor_obs,
    _actions_to_sim, _apply_action_masks, _sample_actions,
    MAX_AGENTS, LSTM_HIDDEN, N_ENTITIES, ENTITY_DIM, SELF_DIM,
)
from battle_royale.train import _find_latest_checkpoint, _load_checkpoint, ACTION_REPEAT

# ── Config ──
NUM_ENVS = 200
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "battle_royale", "runs", "apex_rp", "checkpoints")
A = MAX_AGENTS  # 10

def main():
    device = torch.device("cpu")

    # Load latest checkpoint
    ckpt_path = _find_latest_checkpoint(CHECKPOINT_DIR)
    if not ckpt_path:
        print(f"No checkpoint found in {CHECKPOINT_DIR}")
        return
    print(f"Loading checkpoint: {ckpt_path}")
    sd, ckpt = _load_checkpoint(ckpt_path, device)
    network = AttentionActorCritic()
    network.load_state_dict(sd, strict=False)
    network.eval()
    print(f"  Update: {ckpt.get('update_count', '?')}")

    # Create sim
    print(f"\nCreating sim with {NUM_ENVS} envs, {A} agents each...")
    sim = BatchedBRSim(num_envs=NUM_ENVS, max_agents=A, device="cpu")
    obs_builder = ObservationBuilder(sim)

    # ── Step 1: Get initial observations and critic values ──
    print("Computing initial critic values...")
    sim.reset()

    with torch.no_grad():
        actor_obs = obs_builder.actor_obs()
        self_feat, entities, entity_mask = pack_actor_obs(actor_obs)
        # self_feat: (B, A, SELF_DIM), entities: (B, A, N, ENTITY_DIM), entity_mask: (B, A, N)

        # Run critic on all agents
        # Reshape to (B*A, ...) for the critic
        B = NUM_ENVS
        sf_flat = self_feat.reshape(B * A, SELF_DIM)
        ent_flat = entities.reshape(B * A, N_ENTITIES, ENTITY_DIM)
        emask_flat = entity_mask.reshape(B * A, N_ENTITIES)

        initial_values = network.forward_critic(sf_flat, ent_flat, emask_flat, hx=None)
        initial_values = initial_values.reshape(B, A)  # (B, A)

    # Record critic's predicted winner (highest value)
    critic_predicted_winner = initial_values.argmax(dim=1)  # (B,)
    _, critic_top3 = initial_values.topk(3, dim=1)  # (B, 3)

    # Stats on initial value spread
    val_mean = initial_values.mean(dim=1)  # (B,)
    val_std = initial_values.std(dim=1)    # (B,)
    val_range = initial_values.max(dim=1).values - initial_values.min(dim=1).values

    print(f"\n── Initial Value Statistics ──")
    print(f"  Mean value across agents:  {val_mean.mean():.4f} (std across envs: {val_mean.std():.4f})")
    print(f"  Mean within-env std:       {val_std.mean():.4f}")
    print(f"  Mean within-env range:     {val_range.mean():.4f}")
    print(f"  Min/Max value seen:        {initial_values.min():.4f} / {initial_values.max():.4f}")

    # ── Step 2: Run episodes to completion ──
    print(f"\nRunning {NUM_ENVS} episodes to completion...")

    # Initialize LSTM states for all agents across all envs
    lstm_hx = torch.zeros(B * A, LSTM_HIDDEN)
    lstm_cx = torch.zeros(B * A, LSTM_HIDDEN)

    frame_count = 0
    max_frames = 5000  # safety limit
    done_count = 0

    while not sim.episode_done.all() and frame_count < max_frames:
        with torch.no_grad():
            actor_obs = obs_builder.actor_obs()
            self_feat, entities, entity_mask = pack_actor_obs(actor_obs)

            sf_flat = self_feat.reshape(B * A, SELF_DIM)
            ent_flat = entities.reshape(B * A, N_ENTITIES, ENTITY_DIM)
            emask_flat = entity_mask.reshape(B * A, N_ENTITIES)

            logits, alpha, beta_param, (lstm_hx, lstm_cx) = network.forward_actor(
                sf_flat, ent_flat, emask_flat, hx=lstm_hx, cx=lstm_cx)
            logits = _apply_action_masks(logits, sf_flat)
            disc, cont = _sample_actions(logits, alpha, beta_param)

            # Convert to sim actions
            agent_dir_flat = sim.agent_dir.reshape(B * A)
            mx, my, aim, fire, heal = _actions_to_sim(disc, cont, agent_dir_flat)

            # Reshape back to (B, A)
            mx = mx.reshape(B, A)
            my = my.reshape(B, A)
            aim = aim.reshape(B, A)
            fire = fire.reshape(B, A)
            heal = heal.reshape(B, A)

        # Step with action repeat
        for _rep in range(ACTION_REPEAT):
            cur_alive = sim.agent_alive.clone()
            rewards, done = sim.step(
                mx * cur_alive, my * cur_alive,
                aim, fire & cur_alive, heal & cur_alive)

        frame_count += 1
        new_done = sim.episode_done.sum().item()
        if new_done > done_count:
            done_count = new_done
            if done_count % 50 == 0 or done_count == NUM_ENVS:
                print(f"  {done_count}/{NUM_ENVS} episodes complete (step {frame_count})")

        # Reset LSTM for done envs
        done_mask = sim.episode_done.unsqueeze(1).expand(B, A).reshape(B * A)
        lstm_hx[done_mask] = 0.0
        lstm_cx[done_mask] = 0.0

    print(f"  All episodes complete after {frame_count} decision steps ({frame_count * ACTION_REPEAT} frames)")

    # ── Step 3: Determine actual winners ──
    # agent_place: (B, A) — 1 = winner, 2 = 2nd, etc.
    actual_placements = sim.agent_place.clone()  # (B, A)
    actual_winner = actual_placements.argmin(dim=1)  # agent with place=1

    # Verify all envs have a place-1 agent
    has_winner = (actual_placements == 1).any(dim=1)
    valid_envs = has_winner
    n_valid = valid_envs.sum().item()
    print(f"\n  Valid episodes (with a clear winner): {n_valid}/{NUM_ENVS}")

    # Handle ties (timeout): for tied envs, pick the one the critic liked most
    # Actually, let's just check if any env has multiple agents with place=1
    multi_winner = (actual_placements == 1).sum(dim=1) > 1
    n_tied = multi_winner.sum().item()
    if n_tied > 0:
        print(f"  Episodes with tied winners (timeout): {n_tied}")

    # For environments where there's a unique winner
    unique_winner_mask = (actual_placements == 1).sum(dim=1) == 1
    n_unique = unique_winner_mask.sum().item()

    # ── Step 4: Compute accuracy metrics ──
    print(f"\n{'='*60}")
    print(f"RESULTS ({n_unique} episodes with unique winner)")
    print(f"{'='*60}")

    if n_unique == 0:
        print("No valid episodes with unique winners!")
        return

    # Filter to unique-winner episodes
    mask = unique_winner_mask
    pred = critic_predicted_winner[mask]
    actual = actual_winner[mask]
    top3 = critic_top3[mask]  # (n_unique, 3)
    placements = actual_placements[mask]  # (n_unique, A)
    init_vals = initial_values[mask]  # (n_unique, A)

    # Top-1 accuracy
    correct = (pred == actual).sum().item()
    top1_acc = correct / n_unique
    print(f"\n  Top-1 accuracy (critic's pick wins):  {correct}/{n_unique} = {top1_acc:.1%}")
    print(f"  Random baseline:                       {1/A:.1%}")
    improvement = top1_acc / (1/A)
    print(f"  Improvement over random:               {improvement:.2f}x")

    # Top-3 accuracy
    actual_expanded = actual.unsqueeze(1).expand_as(top3)
    top3_correct = (top3 == actual_expanded).any(dim=1).sum().item()
    top3_acc = top3_correct / n_unique
    print(f"\n  Top-3 accuracy (winner in top 3):      {top3_correct}/{n_unique} = {top3_acc:.1%}")
    print(f"  Random baseline (top-3 of 10):          {3/A:.1%}")

    # ── Spearman rank correlation: initial values vs final placement ──
    # For each env, compute Spearman between initial values and placements
    spearman_rs = []
    for i in range(n_unique):
        vals = init_vals[i].numpy()
        places = placements[i].numpy()
        # Higher value should correlate with lower (better) placement
        # So we expect negative correlation
        r, p = stats.spearmanr(vals, places)
        if not np.isnan(r):
            spearman_rs.append(r)

    spearman_rs = np.array(spearman_rs)
    mean_r = spearman_rs.mean()
    std_r = spearman_rs.std()
    median_r = np.median(spearman_rs)

    print(f"\n  Spearman rank correlation (value vs placement):")
    print(f"    Mean:   {mean_r:.4f} (negative = higher value -> better placement)")
    print(f"    Median: {median_r:.4f}")
    print(f"    Std:    {std_r:.4f}")
    print(f"    Range:  [{spearman_rs.min():.4f}, {spearman_rs.max():.4f}]")

    # ── Predicted winner's actual placement ──
    pred_placement = []
    for i in range(n_unique):
        pred_placement.append(placements[i, pred[i]].item())
    pred_placement = np.array(pred_placement)
    print(f"\n  Critic's predicted winner actual placement:")
    print(f"    Mean:   {pred_placement.mean():.2f} (1=best, {A}=worst)")
    print(f"    Median: {np.median(pred_placement):.1f}")
    print(f"    Distribution:")
    for place in range(1, A + 1):
        count = (pred_placement == place).sum()
        pct = count / n_unique * 100
        bar = '#' * int(pct / 2)
        print(f"      Place {place:2d}: {count:4d} ({pct:5.1f}%) {bar}")

    # ── Value distribution analysis ──
    print(f"\n  Value spread within envs:")
    env_stds = init_vals.std(dim=1).numpy()
    print(f"    Mean std:   {env_stds.mean():.4f}")
    print(f"    Min std:    {env_stds.min():.4f}")
    print(f"    Max std:    {env_stds.max():.4f}")

    # Value of actual winner vs others
    winner_vals = []
    loser_vals = []
    for i in range(n_unique):
        w = actual[i].item()
        winner_vals.append(init_vals[i, w].item())
        for a in range(A):
            if a != w:
                loser_vals.append(init_vals[i, a].item())

    winner_vals = np.array(winner_vals)
    loser_vals = np.array(loser_vals)
    print(f"\n  Initial value of actual winners vs non-winners:")
    print(f"    Winners mean:     {winner_vals.mean():.4f} +/- {winner_vals.std():.4f}")
    print(f"    Non-winners mean: {loser_vals.mean():.4f} +/- {loser_vals.std():.4f}")
    print(f"    Difference:       {winner_vals.mean() - loser_vals.mean():.4f}")

    # What rank did the winner have in the initial value ordering?
    winner_rank = []
    for i in range(n_unique):
        w = actual[i].item()
        # Rank: how many agents had higher value than the winner?
        rank = (init_vals[i] > init_vals[i, w]).sum().item() + 1  # 1-based
        winner_rank.append(rank)
    winner_rank = np.array(winner_rank)
    print(f"\n  Actual winner's rank in initial value ordering (1=highest value):")
    print(f"    Mean rank: {winner_rank.mean():.2f}")
    print(f"    Distribution:")
    for r in range(1, A + 1):
        count = (winner_rank == r).sum()
        pct = count / n_unique * 100
        bar = '#' * int(pct / 2)
        print(f"      Rank {r:2d}: {count:4d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    main()
