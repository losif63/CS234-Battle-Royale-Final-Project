"""Comprehensive gameplay analysis of a battle royale PPO checkpoint.

Runs episodes with all agents sharing one policy and collects detailed
per-step statistics on combat, movement, resource management, attention,
and engagement patterns.

Usage:
    CUDA_VISIBLE_DEVICES="" uv run python analyze_gameplay.py
"""

import math
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import numpy as np

from battle_royale.sim import BatchedBRSim
from battle_royale.obs import ObservationBuilder
from battle_royale.network import (
    AttentionActorCritic, pack_actor_obs,
    _actions_to_sim, _apply_action_masks, _sample_actions,
    MAX_AGENTS, LSTM_HIDDEN, N_ENTITIES, ENTITY_DIM, SELF_DIM,
    NUM_DISCRETE_ACTIONS, MAX_VISIBLE_BULLETS,
)
from battle_royale.config import (
    FIRE_COOLDOWN, AGENT_MAX_HP, AGENT_SPEED, ARENA_W, ARENA_H,
    NUM_AMMO_DEPOSITS, NUM_HEALTH_PICKUPS,
    ZONE_MAX_RADIUS, ZONE_MIN_RADIUS, ZONE_SHRINK_START, ZONE_SHRINK_END,
    MAX_EPISODE_FRAMES,
)
from battle_royale.train import ACTION_REPEAT, _load_checkpoint

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
NUM_EPISODES = 15
NUM_ENVS = 1
NUM_AGENTS = MAX_AGENTS  # 10
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "battle_royale", "runs", "apex_rp", "checkpoints")
DEVICE = "cpu"

# Entity slot layout in packed observations
K_BULLETS = MAX_VISIBLE_BULLETS  # 10
D_DEPOSITS = NUM_AMMO_DEPOSITS   # 12
H_PICKUPS = NUM_HEALTH_PICKUPS   # 15
A_OTHER = NUM_AGENTS - 1         # 9


def find_latest_checkpoint(search_dir):
    import glob
    pattern = os.path.join(search_dir, "br_ppo_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    def _num(f):
        s = os.path.basename(f).replace("br_ppo_", "").replace(".pt", "")
        try:
            return int(s)
        except ValueError:
            return -1
    return max(files, key=_num)


def main():
    device = torch.device(DEVICE)

    # Load checkpoint
    ckpt_path = find_latest_checkpoint(CHECKPOINT_DIR)
    if not ckpt_path:
        print(f"No checkpoint found in {CHECKPOINT_DIR}")
        sys.exit(1)

    network = AttentionActorCritic()
    sd, ckpt = _load_checkpoint(ckpt_path, device)
    network.load_state_dict(sd, strict=False)
    network.eval()
    update_count = ckpt.get("update_count", "?")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Update count: {update_count}")
    print(f"Running {NUM_EPISODES} episodes, {NUM_AGENTS} agents per episode")
    print()

    # Initialize sim
    sim = BatchedBRSim(num_envs=NUM_ENVS, max_agents=NUM_AGENTS, device=DEVICE)
    obs_builder = ObservationBuilder(sim)

    # -----------------------------------------------------------------------
    # Accumulators
    # -----------------------------------------------------------------------

    # Combat
    total_bullets_fired = 0
    total_bullets_hit = 0
    aim_errors = []           # angle error in degrees when firing
    engagement_distances = [] # distance to nearest visible enemy when firing
    kill_by_bullet = 0
    kill_by_zone = 0
    kill_by_timeout = 0
    kills_per_episode = []    # per-agent kills each episode

    # Movement & positioning
    dist_to_zone_center_samples = []  # (frame, distance)
    zone_deaths = 0
    total_deaths = 0
    speed_utilizations = []  # actual_speed / AGENT_SPEED

    # Resource management
    ammo_pickups_per_episode = []
    medkit_pickups_per_episode = []
    heal_when_useful = 0        # healing with low HP + medkits
    heal_when_useful_possible = 0
    heal_with_enemies = 0
    heal_without_enemies = 0
    heal_total = 0
    fire_no_ammo = 0

    # Attention
    attn_bullets_enemy_vis = []
    attn_deposits_enemy_vis = []
    attn_health_enemy_vis = []
    attn_agents_enemy_vis = []
    attn_bullets_no_enemy = []
    attn_deposits_no_enemy = []
    attn_health_no_enemy = []
    attn_agents_no_enemy = []

    # Episode
    episode_lengths = []

    # -----------------------------------------------------------------------
    # Run episodes
    # -----------------------------------------------------------------------
    for ep in range(NUM_EPISODES):
        sim.reset()
        lstm_hx = torch.zeros(NUM_AGENTS, LSTM_HIDDEN, device=device)
        lstm_cx = torch.zeros(NUM_AGENTS, LSTM_HIDDEN, device=device)

        # Per-episode tracking
        ep_ammo_pickups = torch.zeros(NUM_AGENTS, device=device)
        ep_medkit_pickups = torch.zeros(NUM_AGENTS, device=device)
        ep_kills = torch.zeros(NUM_AGENTS, dtype=torch.long, device=device)
        prev_ammo = sim.agent_ammo[0].clone()
        prev_medkits = sim.agent_medkits[0].clone()
        prev_health = sim.agent_health[0].clone()
        prev_alive = sim.agent_alive[0].clone()

        step = 0
        while True:
            with torch.no_grad():
                actor_obs = obs_builder.actor_obs()
                self_feat, entities, entity_mask = pack_actor_obs(actor_obs)

                # All agents share the same policy
                sf = self_feat.reshape(NUM_AGENTS, SELF_DIM)
                ent = entities.reshape(NUM_AGENTS, N_ENTITIES, ENTITY_DIM)
                emask = entity_mask.reshape(NUM_AGENTS, N_ENTITIES)

                # Forward with attention
                logits, alpha, beta_param, (lstm_hx_new, lstm_cx_new), attn_weights = \
                    network.forward_actor(sf, ent, emask, hx=lstm_hx, cx=lstm_cx,
                                          return_attention=True)
                logits = _apply_action_masks(logits, sf)
                disc, cont = _sample_actions(logits, alpha, beta_param)
                mx, my, aim, fire, heal = _actions_to_sim(disc, cont, sim.agent_dir[0])

                move_x = mx.unsqueeze(0)
                move_y = my.unsqueeze(0)
                aim_angle = aim.unsqueeze(0)
                fire_bool = fire.unsqueeze(0)
                heal_bool = heal.unsqueeze(0)

            # ---------------------------------------------------------------
            # Pre-step analysis (before sim.step)
            # ---------------------------------------------------------------
            alive = sim.agent_alive[0].clone()

            # --- Attention analysis (per alive agent) ---
            for a in range(NUM_AGENTS):
                if not alive[a]:
                    continue
                w = attn_weights[a].detach()     # (N,)
                m = emask[a].detach()             # (N,)

                # Only analyze if at least one entity is visible
                if not m.any():
                    continue

                # Check if any enemy agent is visible
                agent_start = K_BULLETS + D_DEPOSITS + H_PICKUPS
                agent_end = agent_start + A_OTHER
                enemies_visible = m[agent_start:agent_end].any().item()

                # Sum attention by entity type (only over valid slots)
                bullet_mask = m[:K_BULLETS]
                deposit_mask = m[K_BULLETS:K_BULLETS + D_DEPOSITS]
                health_mask = m[K_BULLETS + D_DEPOSITS:K_BULLETS + D_DEPOSITS + H_PICKUPS]
                agent_mask = m[agent_start:agent_end]

                # Weighted sum per type
                bw = (w[:K_BULLETS] * bullet_mask.float()).sum().item()
                dw = (w[K_BULLETS:K_BULLETS + D_DEPOSITS] * deposit_mask.float()).sum().item()
                hw = (w[K_BULLETS + D_DEPOSITS:K_BULLETS + D_DEPOSITS + H_PICKUPS] * health_mask.float()).sum().item()
                aw = (w[agent_start:agent_end] * agent_mask.float()).sum().item()
                total_w = bw + dw + hw + aw
                if total_w < 1e-8:
                    continue

                # Normalize to fractions
                bw /= total_w
                dw /= total_w
                hw /= total_w
                aw /= total_w

                if enemies_visible:
                    attn_bullets_enemy_vis.append(bw)
                    attn_deposits_enemy_vis.append(dw)
                    attn_health_enemy_vis.append(hw)
                    attn_agents_enemy_vis.append(aw)
                else:
                    attn_bullets_no_enemy.append(bw)
                    attn_deposits_no_enemy.append(dw)
                    attn_health_no_enemy.append(hw)
                    attn_agents_no_enemy.append(aw)

            # --- Healing analysis ---
            for a in range(NUM_AGENTS):
                if not alive[a]:
                    continue
                is_healing = heal[a].item()
                has_medkits = sim.agent_medkits[0, a].item() > 0
                hp_fraction = sim.agent_health[0, a].item() / AGENT_MAX_HP
                low_hp = hp_fraction < 0.9

                # Check if enemies visible
                agent_start = K_BULLETS + D_DEPOSITS + H_PICKUPS
                agent_end = agent_start + A_OTHER
                enemies_vis = emask[a, agent_start:agent_end].any().item()

                # Useful heal: low HP AND has medkits
                if low_hp and has_medkits:
                    heal_when_useful_possible += 1
                    if is_healing:
                        heal_when_useful += 1

                if is_healing:
                    heal_total += 1
                    if enemies_vis:
                        heal_with_enemies += 1
                    else:
                        heal_without_enemies += 1

            # --- Speed utilization ---
            for a in range(NUM_AGENTS):
                if not alive[a]:
                    continue
                speed = math.sqrt(sim.agent_vx[0, a].item()**2 + sim.agent_vy[0, a].item()**2)
                speed_utilizations.append(speed / AGENT_SPEED)

            # --- Distance to zone center ---
            frame_val = sim.frame[0].item()
            for a in range(NUM_AGENTS):
                if not alive[a]:
                    continue
                dx = sim.agent_x[0, a].item() - ARENA_W / 2
                dy = sim.agent_y[0, a].item() - ARENA_H / 2
                dist = math.sqrt(dx**2 + dy**2)
                dist_to_zone_center_samples.append((frame_val, dist))

            # ---------------------------------------------------------------
            # Step sim with ACTION_REPEAT
            # ---------------------------------------------------------------
            for _rep in range(ACTION_REPEAT):
                pre_health = sim.agent_health[0].clone()
                pre_alive = sim.agent_alive[0].clone()
                pre_cooldown = sim.agent_cooldown[0].clone()

                cur_alive = sim.agent_alive[0].clone()
                m_x = move_x * cur_alive.unsqueeze(0).float()
                m_y = move_y * cur_alive.unsqueeze(0).float()
                f_b = fire_bool & cur_alive.unsqueeze(0)
                h_b = heal_bool & cur_alive.unsqueeze(0)

                rewards, done = sim.step(m_x, m_y, aim_angle, f_b, h_b)

                # Detect fires: agent_cooldown == FIRE_COOLDOWN means just fired
                for a in range(NUM_AGENTS):
                    if not pre_alive[a]:
                        continue
                    if sim.agent_cooldown[0, a].item() == FIRE_COOLDOWN:
                        total_bullets_fired += 1

                        # Check ammo (fire_no_ammo should be ~0 with masks)
                        if prev_ammo[a].item() <= 0:
                            fire_no_ammo += 1

                        # Aim error: angle between aim direction and nearest visible enemy
                        agent_x = sim.agent_x[0, a].item()
                        agent_y = sim.agent_y[0, a].item()
                        agent_dir = sim.agent_dir[0, a].item()

                        # Find nearest visible enemy
                        min_dist = float('inf')
                        nearest_angle = None
                        nearest_dist = None
                        for other in range(NUM_AGENTS):
                            if other == a or not pre_alive[other]:
                                continue
                            ox = sim.agent_x[0, other].item()
                            oy = sim.agent_y[0, other].item()
                            edx = ox - agent_x
                            edy = oy - agent_y
                            edist = math.sqrt(edx**2 + edy**2)
                            if edist < min_dist and edist < 550.0:  # ENTITY_FOV_RADIUS
                                min_dist = edist
                                nearest_angle = math.atan2(edy, edx)
                                nearest_dist = edist

                        if nearest_angle is not None:
                            # Aim error — properly wrap to [-pi, pi]
                            raw_diff = agent_dir - nearest_angle
                            angle_diff = abs(math.atan2(math.sin(raw_diff), math.cos(raw_diff)))
                            aim_errors.append(math.degrees(angle_diff))
                            engagement_distances.append(nearest_dist)

                # Detect hits: health decreased for any agent
                for a in range(NUM_AGENTS):
                    if not pre_alive[a]:
                        continue
                    if sim.agent_health[0, a].item() < pre_health[a].item():
                        # Determine if it was a bullet hit (not zone damage)
                        hp_drop = pre_health[a].item() - sim.agent_health[0, a].item()
                        # Bullet damage is 25, zone is 0.5/frame
                        if hp_drop >= 20:  # bullet damage (25) vs zone (0.5)
                            total_bullets_hit += 1

                # Detect deaths
                for a in range(NUM_AGENTS):
                    if pre_alive[a] and not sim.agent_alive[0, a]:
                        total_deaths += 1

                        # Zone kill check
                        agent_x = sim.agent_x[0, a].item()
                        agent_y = sim.agent_y[0, a].item()
                        frame_now = sim.frame[0].item()
                        zp = max(0.0, min(1.0, (frame_now - ZONE_SHRINK_START) / (ZONE_SHRINK_END - ZONE_SHRINK_START)))
                        zone_r = ZONE_MAX_RADIUS + (ZONE_MIN_RADIUS - ZONE_MAX_RADIUS) * zp
                        dist_to_c = math.sqrt((agent_x - ARENA_W/2)**2 + (agent_y - ARENA_H/2)**2)

                        if dist_to_c > zone_r:
                            kill_by_zone += 1
                            zone_deaths += 1
                        else:
                            kill_by_bullet += 1

                # Track ammo/medkit pickups
                for a in range(NUM_AGENTS):
                    if not sim.agent_alive[0, a]:
                        continue
                    ammo_gain = sim.agent_ammo[0, a].item() - prev_ammo[a].item()
                    if ammo_gain > 0:
                        ep_ammo_pickups[a] += 1
                    medkit_gain = sim.agent_medkits[0, a].item() - prev_medkits[a].item()
                    if medkit_gain > 0:
                        ep_medkit_pickups[a] += 1

                prev_ammo = sim.agent_ammo[0].clone()
                prev_medkits = sim.agent_medkits[0].clone()

                if done[0]:
                    break

            # Update LSTM state (reset dead agents)
            agent_died = prev_alive & ~sim.agent_alive[0]
            lstm_hx_new = torch.where(agent_died.unsqueeze(-1), torch.zeros_like(lstm_hx_new), lstm_hx_new)
            lstm_cx_new = torch.where(agent_died.unsqueeze(-1), torch.zeros_like(lstm_cx_new), lstm_cx_new)
            lstm_hx = lstm_hx_new.detach()
            lstm_cx = lstm_cx_new.detach()

            prev_alive = sim.agent_alive[0].clone()
            step += 1

            if done[0]:
                # Check for timeout kills
                timeout = sim.frame[0].item() >= MAX_EPISODE_FRAMES
                if timeout:
                    alive_count = sim.agent_alive[0].sum().item()
                    kill_by_timeout += int(alive_count)

                ep_len = sim.frame[0].item()
                episode_lengths.append(ep_len)
                ep_kills = sim.agent_kills[0].clone()
                kills_per_episode.append(ep_kills.float().mean().item())
                ammo_pickups_per_episode.append(ep_ammo_pickups.mean().item())
                medkit_pickups_per_episode.append(ep_medkit_pickups.mean().item())

                print(f"  Episode {ep+1}/{NUM_EPISODES}: length={ep_len} frames, "
                      f"kills={ep_kills.sum().item()}, "
                      f"ammo_pickups={ep_ammo_pickups.sum().item():.0f}, "
                      f"medkit_pickups={ep_medkit_pickups.sum().item():.0f}")
                break

    # ===================================================================
    # Report
    # ===================================================================
    print()
    print("=" * 70)
    print(f"GAMEPLAY ANALYSIS REPORT - Checkpoint update {update_count}")
    print(f"({NUM_EPISODES} episodes, {NUM_AGENTS} agents/episode)")
    print("=" * 70)

    # --- Combat ---
    print()
    print("--- COMBAT ---")
    print(f"  Bullets fired:       {total_bullets_fired}")
    print(f"  Bullets hit:         {total_bullets_hit}")
    if total_bullets_fired > 0:
        print(f"  Shot accuracy:       {100 * total_bullets_hit / total_bullets_fired:.1f}%")
    else:
        print(f"  Shot accuracy:       N/A (no shots)")
    if aim_errors:
        print(f"  Mean aim error:      {np.mean(aim_errors):.1f} deg (median {np.median(aim_errors):.1f})")
        print(f"    p25={np.percentile(aim_errors, 25):.1f}  p75={np.percentile(aim_errors, 75):.1f}")
    print(f"  Kill breakdown:")
    print(f"    Bullet kills:      {kill_by_bullet} ({100*kill_by_bullet/max(1,total_deaths):.1f}%)")
    print(f"    Zone kills:        {kill_by_zone} ({100*kill_by_zone/max(1,total_deaths):.1f}%)")
    print(f"    Timeout (alive):   {kill_by_timeout} ({100*kill_by_timeout/max(1,total_deaths+kill_by_timeout):.1f}%)")
    print(f"    Total deaths:      {total_deaths}")
    if kills_per_episode:
        print(f"  Mean kills/ep/agent: {np.mean(kills_per_episode):.2f}")

    # --- Movement & Positioning ---
    print()
    print("--- MOVEMENT & POSITIONING ---")
    if dist_to_zone_center_samples:
        dists = [d for _, d in dist_to_zone_center_samples]
        print(f"  Mean dist to zone center:  {np.mean(dists):.0f} px")
        # Break down by phase: early (0-600), mid (600-3600), late (3600+)
        early = [d for f, d in dist_to_zone_center_samples if f < ZONE_SHRINK_START]
        mid = [d for f, d in dist_to_zone_center_samples if ZONE_SHRINK_START <= f < ZONE_SHRINK_END]
        late = [d for f, d in dist_to_zone_center_samples if f >= ZONE_SHRINK_END]
        if early:
            print(f"    Early (<{ZONE_SHRINK_START}f):       {np.mean(early):.0f} px")
        if mid:
            print(f"    Mid ({ZONE_SHRINK_START}-{ZONE_SHRINK_END}f):   {np.mean(mid):.0f} px")
        if late:
            print(f"    Late (>{ZONE_SHRINK_END}f):     {np.mean(late):.0f} px")
    print(f"  Zone death rate:           {100*zone_deaths/max(1,total_deaths):.1f}%")
    if speed_utilizations:
        print(f"  Speed utilization:         {np.mean(speed_utilizations):.2f} (mean actual/max)")
        print(f"    p25={np.percentile(speed_utilizations, 25):.2f}  "
              f"p50={np.percentile(speed_utilizations, 50):.2f}  "
              f"p75={np.percentile(speed_utilizations, 75):.2f}")

    # --- Resource Management ---
    print()
    print("--- RESOURCE MANAGEMENT ---")
    if ammo_pickups_per_episode:
        print(f"  Ammo pickups/ep (per agent):   {np.mean(ammo_pickups_per_episode):.1f}")
    if medkit_pickups_per_episode:
        print(f"  Medkit pickups/ep (per agent):  {np.mean(medkit_pickups_per_episode):.1f}")
    if heal_when_useful_possible > 0:
        print(f"  Heal rate when useful:         {100*heal_when_useful/heal_when_useful_possible:.1f}% "
              f"({heal_when_useful}/{heal_when_useful_possible})")
    if heal_total > 0:
        print(f"  Heal with enemies visible:     {100*heal_with_enemies/heal_total:.1f}% "
              f"({heal_with_enemies}/{heal_total})")
        print(f"  Heal without enemies visible:  {100*heal_without_enemies/heal_total:.1f}% "
              f"({heal_without_enemies}/{heal_total})")
    print(f"  Fire with no ammo:             {fire_no_ammo} (should be ~0 with masks)")

    # --- Attention Analysis ---
    print()
    print("--- ATTENTION ANALYSIS ---")
    if attn_agents_enemy_vis:
        print(f"  With enemies visible ({len(attn_agents_enemy_vis)} samples):")
        print(f"    Bullets:   {100*np.mean(attn_bullets_enemy_vis):.1f}%")
        print(f"    Deposits:  {100*np.mean(attn_deposits_enemy_vis):.1f}%")
        print(f"    Health:    {100*np.mean(attn_health_enemy_vis):.1f}%")
        print(f"    Agents:    {100*np.mean(attn_agents_enemy_vis):.1f}%")
    if attn_agents_no_enemy:
        print(f"  Without enemies visible ({len(attn_agents_no_enemy)} samples):")
        print(f"    Bullets:   {100*np.mean(attn_bullets_no_enemy):.1f}%")
        print(f"    Deposits:  {100*np.mean(attn_deposits_no_enemy):.1f}%")
        print(f"    Health:    {100*np.mean(attn_health_no_enemy):.1f}%")
        print(f"    Agents:    {100*np.mean(attn_agents_no_enemy):.1f}%")

    # --- Engagement ---
    print()
    print("--- ENGAGEMENT ---")
    if engagement_distances:
        print(f"  Engagement distance (when firing at visible enemy):")
        print(f"    Mean:   {np.mean(engagement_distances):.0f} px")
        print(f"    Median: {np.median(engagement_distances):.0f} px")
        print(f"    p10={np.percentile(engagement_distances, 10):.0f}  "
              f"p25={np.percentile(engagement_distances, 25):.0f}  "
              f"p75={np.percentile(engagement_distances, 75):.0f}  "
              f"p90={np.percentile(engagement_distances, 90):.0f}")
    if episode_lengths:
        print(f"  Episode length (frames):")
        print(f"    Mean:   {np.mean(episode_lengths):.0f}")
        print(f"    Median: {np.median(episode_lengths):.0f}")
        print(f"    Min:    {np.min(episode_lengths)}")
        print(f"    Max:    {np.max(episode_lengths)}")
        print(f"    Timed out: {sum(1 for l in episode_lengths if l >= MAX_EPISODE_FRAMES)}/{len(episode_lengths)}")

    print()
    print("=" * 70)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
