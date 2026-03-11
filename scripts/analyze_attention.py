"""
Analyze attention weight distribution across entity types.

Loads a checkpoint, runs episodes, and measures how much attention
the cross-attention mechanism allocates to each entity type:
bullets, ammo deposits, health pickups, and other agents.

Usage:
    CUDA_VISIBLE_DEVICES="" uv run python analyze_attention.py
"""

import os
import sys
import torch
import numpy as np

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from battle_royale.sim import BatchedBRSim
from battle_royale.obs import ObservationBuilder
from battle_royale.network import (
    AttentionActorCritic, pack_actor_obs,
    _actions_to_sim, _apply_action_masks, _sample_actions,
    MAX_AGENTS, LSTM_HIDDEN, N_ENTITIES, MAX_VISIBLE_BULLETS,
    SELF_DIM, ENTITY_DIM, NUM_DISCRETE_ACTIONS,
)
from battle_royale.config import NUM_AMMO_DEPOSITS, NUM_HEALTH_PICKUPS, AGENT_MAX_HP, AMMO_MAX
from battle_royale.train import _find_latest_checkpoint, _load_checkpoint


# ---- Config ----
NUM_ENVS = 1
NUM_AGENTS_PER_ENV = 10
NUM_EPISODES = 5
ACTION_REPEAT = 3

# Entity slot ranges (packing order from pack_actor_obs)
K = MAX_VISIBLE_BULLETS  # 10
D = NUM_AMMO_DEPOSITS    # 12
H = NUM_HEALTH_PICKUPS   # 15
A_OTHER = MAX_AGENTS - 1 # 9
# Total = 10 + 12 + 15 + 9 = 46 = N_ENTITIES

BULLET_START = 0
BULLET_END = K
DEPOSIT_START = K
DEPOSIT_END = K + D
HEALTH_START = K + D
HEALTH_END = K + D + H
AGENT_START = K + D + H
AGENT_END = K + D + H + A_OTHER

assert AGENT_END == N_ENTITIES, f"Expected {N_ENTITIES}, got {AGENT_END}"


def main():
    device = torch.device("cpu")
    print(f"Device: {device}")

    # Find and load checkpoint
    ckpt_dir = os.path.join(os.path.dirname(__file__), "battle_royale", "runs", "apex_rp", "checkpoints")
    ckpt_path = _find_latest_checkpoint(search_dir=ckpt_dir)
    if ckpt_path is None:
        print(f"No checkpoint found in {ckpt_dir}")
        sys.exit(1)
    print(f"Loading checkpoint: {ckpt_path}")

    network = AttentionActorCritic().to(device)
    sd, ckpt = _load_checkpoint(ckpt_path, device)
    network.load_state_dict(sd)
    network.eval()
    print(f"Loaded update {ckpt.get('update_count', '?')}")

    # Setup sim
    B = NUM_ENVS
    A = NUM_AGENTS_PER_ENV
    sim = BatchedBRSim(num_envs=B, max_agents=A, device=str(device))
    obs_builder = ObservationBuilder(sim)
    sim.reset()

    # LSTM states for all agents: (B*A, LSTM_HIDDEN)
    hx = torch.zeros(B * A, LSTM_HIDDEN, device=device)
    cx = torch.zeros(B * A, LSTM_HIDDEN, device=device)

    # ---- Accumulators ----
    # Overall
    attn_sum_bullets = 0.0
    attn_sum_deposits = 0.0
    attn_sum_health = 0.0
    attn_sum_agents = 0.0
    total_samples = 0

    # By context: enemies visible
    attn_enemies_vis = {"bullets": 0.0, "deposits": 0.0, "health": 0.0, "agents": 0.0, "n": 0}
    attn_enemies_not = {"bullets": 0.0, "deposits": 0.0, "health": 0.0, "agents": 0.0, "n": 0}

    # By context: low ammo (<=20% of max)
    attn_low_ammo = {"bullets": 0.0, "deposits": 0.0, "health": 0.0, "agents": 0.0, "n": 0}
    attn_high_ammo = {"bullets": 0.0, "deposits": 0.0, "health": 0.0, "agents": 0.0, "n": 0}

    # By context: low health (<=40% of max)
    attn_low_hp = {"bullets": 0.0, "deposits": 0.0, "health": 0.0, "agents": 0.0, "n": 0}
    attn_high_hp = {"bullets": 0.0, "deposits": 0.0, "health": 0.0, "agents": 0.0, "n": 0}

    episodes_done = 0
    total_steps = 0

    print(f"\nRunning {NUM_EPISODES} episodes with {A} agents...")
    print(f"Entity slots: bullets[0:{K}], deposits[{K}:{K+D}], health[{K+D}:{K+D+H}], agents[{K+D+H}:{N_ENTITIES}]")
    print()

    while episodes_done < NUM_EPISODES:
        actor_obs = obs_builder.actor_obs()
        alive = sim.agent_alive.clone()  # (B, A)

        self_feat, entities, entity_mask = pack_actor_obs(actor_obs)

        # Reshape for all agents: (B*A, ...)
        sf_all = self_feat.reshape(B * A, SELF_DIM)
        ent_all = entities.reshape(B * A, N_ENTITIES, ENTITY_DIM)
        emask_all = entity_mask.reshape(B * A, N_ENTITIES)

        with torch.no_grad():
            logits, alpha, beta_param, (hx_out, cx_out), attn_weights = network.forward_actor(
                sf_all, ent_all, emask_all, hx=hx, cx=cx, return_attention=True)
            logits = _apply_action_masks(logits, sf_all)
            disc_actions, cont_actions = _sample_actions(logits, alpha, beta_param)

        # attn_weights: (B*A, N_ENTITIES)
        # entity_mask: (B, A, N_ENTITIES)
        alive_flat = alive.reshape(B * A)  # (B*A,)
        emask_flat = emask_all  # (B*A, N_ENTITIES)

        # Only process alive agents
        alive_idx = alive_flat.nonzero(as_tuple=True)[0]
        if len(alive_idx) > 0:
            aw = attn_weights[alive_idx]  # (n_alive, N)
            em = emask_flat[alive_idx]    # (n_alive, N)
            sf = sf_all[alive_idx]        # (n_alive, SELF_DIM)

            # Sum attention by entity type (only valid slots)
            bullet_attn = (aw[:, BULLET_START:BULLET_END] * em[:, BULLET_START:BULLET_END].float()).sum(dim=1)
            deposit_attn = (aw[:, DEPOSIT_START:DEPOSIT_END] * em[:, DEPOSIT_START:DEPOSIT_END].float()).sum(dim=1)
            health_attn = (aw[:, HEALTH_START:HEALTH_END] * em[:, HEALTH_START:HEALTH_END].float()).sum(dim=1)
            agent_attn = (aw[:, AGENT_START:AGENT_END] * em[:, AGENT_START:AGENT_END].float()).sum(dim=1)

            n_alive = len(alive_idx)

            attn_sum_bullets += bullet_attn.sum().item()
            attn_sum_deposits += deposit_attn.sum().item()
            attn_sum_health += health_attn.sum().item()
            attn_sum_agents += agent_attn.sum().item()
            total_samples += n_alive

            # Context: enemies visible (any agent entity slot is valid)
            any_enemy_visible = em[:, AGENT_START:AGENT_END].any(dim=1)  # (n_alive,)
            vis_idx = any_enemy_visible.nonzero(as_tuple=True)[0]
            not_vis_idx = (~any_enemy_visible).nonzero(as_tuple=True)[0]

            def _accum(bucket, b_attn, d_attn, h_attn, a_attn):
                bucket["bullets"] += b_attn.sum().item()
                bucket["deposits"] += d_attn.sum().item()
                bucket["health"] += h_attn.sum().item()
                bucket["agents"] += a_attn.sum().item()
                bucket["n"] += len(b_attn)

            if len(vis_idx) > 0:
                _accum(attn_enemies_vis, bullet_attn[vis_idx], deposit_attn[vis_idx],
                       health_attn[vis_idx], agent_attn[vis_idx])
            if len(not_vis_idx) > 0:
                _accum(attn_enemies_not, bullet_attn[not_vis_idx], deposit_attn[not_vis_idx],
                       health_attn[not_vis_idx], agent_attn[not_vis_idx])

            # Context: low ammo (sf[:, 6] = ammo / AMMO_MAX, low = <=0.2)
            ammo_norm = sf[:, 6]
            low_ammo_mask = ammo_norm <= 0.2
            high_ammo_mask = ~low_ammo_mask
            low_ammo_idx = low_ammo_mask.nonzero(as_tuple=True)[0]
            high_ammo_idx = high_ammo_mask.nonzero(as_tuple=True)[0]

            if len(low_ammo_idx) > 0:
                _accum(attn_low_ammo, bullet_attn[low_ammo_idx], deposit_attn[low_ammo_idx],
                       health_attn[low_ammo_idx], agent_attn[low_ammo_idx])
            if len(high_ammo_idx) > 0:
                _accum(attn_high_ammo, bullet_attn[high_ammo_idx], deposit_attn[high_ammo_idx],
                       health_attn[high_ammo_idx], agent_attn[high_ammo_idx])

            # Context: low health (sf[:, 4] = health / AGENT_MAX_HP, low = <=0.4)
            hp_norm = sf[:, 4]
            low_hp_mask = hp_norm <= 0.4
            high_hp_mask = ~low_hp_mask
            low_hp_idx = low_hp_mask.nonzero(as_tuple=True)[0]
            high_hp_idx = high_hp_mask.nonzero(as_tuple=True)[0]

            if len(low_hp_idx) > 0:
                _accum(attn_low_hp, bullet_attn[low_hp_idx], deposit_attn[low_hp_idx],
                       health_attn[low_hp_idx], agent_attn[low_hp_idx])
            if len(high_hp_idx) > 0:
                _accum(attn_high_hp, bullet_attn[high_hp_idx], deposit_attn[high_hp_idx],
                       health_attn[high_hp_idx], agent_attn[high_hp_idx])

        # Step the sim
        disc_all = disc_actions.reshape(B, A, NUM_DISCRETE_ACTIONS)
        cont_all = cont_actions.reshape(B, A)
        mx, my, aim, fire, heal = _actions_to_sim(disc_all, cont_all, sim.agent_dir)

        for _rep in range(ACTION_REPEAT):
            cur_alive = sim.agent_alive.clone()
            move_x = mx * cur_alive
            move_y = my * cur_alive
            fire_bool = fire & cur_alive
            heal_bool = heal & cur_alive
            rewards, episode_done = sim.step(move_x, move_y, aim, fire_bool, heal_bool)

            # Reset done episodes
            if episode_done.any():
                new_done = episode_done.sum().item()
                episodes_done += int(new_done)
                # Reset LSTM for agents in done envs
                done_expanded = episode_done.unsqueeze(1).expand(B, A).reshape(B * A)
                hx_out = torch.where(done_expanded[:, None], torch.zeros_like(hx_out), hx_out)
                cx_out = torch.where(done_expanded[:, None], torch.zeros_like(cx_out), cx_out)
                sim.reset(mask=episode_done)

        # Handle agents that died mid-step (reset their LSTM)
        died = alive.reshape(B * A) & ~sim.agent_alive.reshape(B * A)
        hx_out = torch.where(died[:, None], torch.zeros_like(hx_out), hx_out)
        cx_out = torch.where(died[:, None], torch.zeros_like(cx_out), cx_out)

        hx = hx_out.detach()
        cx = cx_out.detach()
        total_steps += 1

        if total_steps % 500 == 0:
            print(f"  Step {total_steps}, episodes done: {episodes_done}/{NUM_EPISODES}, "
                  f"samples: {total_samples:,}")

    # ---- Results ----
    print(f"\n{'='*60}")
    print(f"ATTENTION WEIGHT ANALYSIS")
    print(f"{'='*60}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Episodes: {episodes_done}, Steps: {total_steps}, "
          f"Agent-step samples: {total_samples:,}")

    def _pct(val, total):
        return 100.0 * val / total if total > 0 else 0.0

    total_attn = attn_sum_bullets + attn_sum_deposits + attn_sum_health + attn_sum_agents

    print(f"\n--- Overall Attention Distribution ---")
    print(f"  Bullets:        {_pct(attn_sum_bullets, total_attn):5.1f}%  "
          f"(mean {attn_sum_bullets/total_samples:.4f} per step)")
    print(f"  Ammo deposits:  {_pct(attn_sum_deposits, total_attn):5.1f}%  "
          f"(mean {attn_sum_deposits/total_samples:.4f} per step)")
    print(f"  Health pickups: {_pct(attn_sum_health, total_attn):5.1f}%  "
          f"(mean {attn_sum_health/total_samples:.4f} per step)")
    print(f"  Other agents:   {_pct(attn_sum_agents, total_attn):5.1f}%  "
          f"(mean {attn_sum_agents/total_samples:.4f} per step)")
    print(f"  (Unmasked/no-entity slots absorb remaining: "
          f"{_pct(total_samples - total_attn, total_samples):.1f}% avg)")

    def _print_context(label, bucket_a, bucket_b, label_a, label_b):
        print(f"\n--- {label} ---")
        for lbl, bkt in [(label_a, bucket_a), (label_b, bucket_b)]:
            n = bkt["n"]
            if n == 0:
                print(f"  {lbl}: (no samples)")
                continue
            t = bkt["bullets"] + bkt["deposits"] + bkt["health"] + bkt["agents"]
            if t == 0:
                print(f"  {lbl}: (no valid entities)")
                continue
            print(f"  {lbl} (n={n:,}):")
            print(f"    Bullets:        {_pct(bkt['bullets'], t):5.1f}%")
            print(f"    Ammo deposits:  {_pct(bkt['deposits'], t):5.1f}%")
            print(f"    Health pickups: {_pct(bkt['health'], t):5.1f}%")
            print(f"    Other agents:   {_pct(bkt['agents'], t):5.1f}%")

    _print_context("By Enemy Visibility", attn_enemies_vis, attn_enemies_not,
                   "Enemies visible", "No enemies visible")
    _print_context("By Ammo Level", attn_low_ammo, attn_high_ammo,
                   "Low ammo (<=20%)", "High ammo (>20%)")
    _print_context("By Health Level", attn_low_hp, attn_high_hp,
                   "Low health (<=40%)", "High health (>40%)")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
