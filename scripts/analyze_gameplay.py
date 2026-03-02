"""
Comprehensive gameplay analysis of trained battle royale agents.

Runs tens of thousands of parallel games and tracks deep behavioral stats:
- Death causes (zone, bullets, timeout)
- Bullet dodging behavior
- Combat stats (accuracy, kills, damage)
- Resource gathering (ammo, medkits)
- Zone awareness & positioning
- Movement patterns
- Smart decision-making metrics:
  - Fires when enemy in crosshair vs wastes ammo
  - Heals when low HP
  - Moves toward zone when outside
  - Seeks ammo when empty
  - Engages when armed vs flees when empty
- Action distributions
- Checkpoint comparison (early vs late)

Usage:
    uv run python scripts/analyze_gameplay.py
    uv run python scripts/analyze_gameplay.py --checkpoint battle_royale/runs/myrun/checkpoints/br_ppo_28000.pt
    uv run python scripts/analyze_gameplay.py --num-envs 30000 --num-episodes 50000
"""

import argparse
import time
import math
import glob
import os

import torch
import torch.nn.functional as F

from battle_royale.sim import BatchedBRSim
from battle_royale.obs import ObservationBuilder
from battle_royale.train import (
    AttentionActorCritic, pack_actor_obs, _actions_to_sim, _sample_actions,
    _greedy_actions, _apply_action_masks,
    LSTM_HIDDEN, N_ENTITIES, ENTITY_DIM, MAX_AGENTS, ACTION_REPEAT,
    SELF_DIM, NUM_DISCRETE_ACTIONS, DISCRETE_ACTION_HEADS,
)
from battle_royale.config import (
    AGENT_MAX_HP, ZONE_SHRINK_START, ZONE_SHRINK_END,
    ZONE_MAX_RADIUS, ZONE_MIN_RADIUS, ARENA_W, ARENA_H,
    BULLET_RADIUS, AGENT_RADIUS, BULLET_SPEED, ZONE_DAMAGE_PER_FRAME,
    MAX_EPISODE_FRAMES, AMMO_MAX, FIRE_COOLDOWN, BULLET_DAMAGE,
    AGENT_SPEED, ENTITY_FOV_RADIUS, MEDKIT_MAX, HEAL_CHANNEL_FRAMES,
)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["network"].items()}
    return sd, ckpt


def run_analysis(checkpoint_path, num_envs=30000, target_episodes=50000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    A = MAX_AGENTS  # 4

    # Load network
    network = AttentionActorCritic().to(device)
    sd, ckpt = load_checkpoint(checkpoint_path, device)
    network.load_state_dict(sd)
    network.eval()
    update_count = ckpt.get("update_count", "?")
    print(f"Loaded {checkpoint_path} (update {update_count})")
    print(f"Running {num_envs:,} parallel envs, targeting {target_episodes:,} completed episodes\n")

    B = num_envs
    sim = BatchedBRSim(num_envs=B, max_agents=A, device=str(device))
    obs_builder = ObservationBuilder(sim)

    # LSTM states (B*A,)
    lstm_hx = torch.zeros(B * A, LSTM_HIDDEN, device=device)
    lstm_cx = torch.zeros(B * A, LSTM_HIDDEN, device=device)

    # ===================================================================
    # Accumulators
    # ===================================================================
    total_episodes = 0
    S = {}  # stats dict for cleanliness

    def inc(key, val=1):
        S[key] = S.get(key, 0) + (val if isinstance(val, (int, float)) else val.item())

    def inc_t(key, tensor):
        """Increment by sum of a tensor."""
        S[key] = S.get(key, 0) + tensor.sum().item()

    # Pre-step tracking tensors
    prev_cooldown = sim.agent_cooldown.clone()
    health_before = sim.agent_health.clone()

    # Action distribution accumulators
    action_counts = [torch.zeros(n, device=device, dtype=torch.long) for n in DISCRETE_ACTION_HEADS]

    # Continuous rotation stats
    rotation_sum = torch.zeros(1, device=device)
    rotation_sq_sum = torch.zeros(1, device=device)
    rotation_count = torch.zeros(1, device=device, dtype=torch.long)

    # For tracking per-agent episode ammo at death
    peak_ammo_per_life = torch.zeros(B, A, device=device)

    start_time = time.time()
    total_steps = 0

    print("Running simulation...")

    while total_episodes < target_episodes:
        # ---- Observations ----
        actor_obs = obs_builder.actor_obs()
        self_feat, entities, entity_mask = pack_actor_obs(actor_obs)

        sf = self_feat.reshape(B * A, SELF_DIM)
        ent = entities.reshape(B * A, N_ENTITIES, ENTITY_DIM)
        emask = entity_mask.reshape(B * A, N_ENTITIES)

        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            logits, alpha, beta_param, (lstm_hx, lstm_cx) = network.forward_actor(
                sf, ent, emask, hx=lstm_hx, cx=lstm_cx)
            logits = _apply_action_masks(logits, sf)
            disc_actions, cont_actions = _sample_actions(logits, alpha, beta_param)

        # ---- Action distribution tracking ----
        for i, head_logits in enumerate(logits):
            chosen = disc_actions[:, i]  # (B*A,)
            action_counts[i].scatter_add_(0, chosen, torch.ones_like(chosen, dtype=torch.long))

        # Continuous rotation stats
        rotation_delta = cont_actions.float() * 2 * math.pi - math.pi  # [0,1] -> [-pi, pi]
        rotation_sum += rotation_delta.sum()
        rotation_sq_sum += (rotation_delta ** 2).sum()
        rotation_count += rotation_delta.numel()

        disc_ba = disc_actions.reshape(B, A, NUM_DISCRETE_ACTIONS)
        cont_ba = cont_actions.reshape(B, A)
        mx, my, aim, fire, heal = _actions_to_sim(disc_ba, cont_ba, sim.agent_dir)

        is_alive = sim.agent_alive  # (B, A)

        # ========== PRE-STEP SITUATIONAL ANALYSIS ==========

        # --- Zone geometry ---
        zone_progress = ((sim.frame.float() - ZONE_SHRINK_START) /
                         (ZONE_SHRINK_END - ZONE_SHRINK_START)).clamp(0, 1)
        zone_radius = ZONE_MAX_RADIUS + (ZONE_MIN_RADIUS - ZONE_MAX_RADIUS) * zone_progress
        dist_to_center = ((sim.agent_x - ARENA_W / 2)**2 +
                          (sim.agent_y - ARENA_H / 2)**2).sqrt()
        outside_zone = (dist_to_center > zone_radius.unsqueeze(1)) & is_alive

        # --- Other agent visibility ---
        # Build pairwise distances (B, A, A)
        dx_aa = sim.agent_x.unsqueeze(2) - sim.agent_x.unsqueeze(1)  # (B, A, A)
        dy_aa = sim.agent_y.unsqueeze(2) - sim.agent_y.unsqueeze(1)
        dist_aa = (dx_aa**2 + dy_aa**2).sqrt()
        eye_mask = torch.eye(A, device=device, dtype=torch.bool).unsqueeze(0)  # (1, A, A)
        other_alive = sim.agent_alive.unsqueeze(1).expand(B, A, A)  # (B, A, A)
        enemy_visible = (dist_aa < ENTITY_FOV_RADIUS) & ~eye_mask & other_alive & is_alive.unsqueeze(2)

        any_enemy_visible = enemy_visible.any(dim=2)  # (B, A)
        nearest_enemy_dist = dist_aa.masked_fill(eye_mask | ~other_alive | ~is_alive.unsqueeze(2), 1e6).min(dim=2).values  # (B, A)
        enemy_close = nearest_enemy_dist < 300  # within ~300px
        enemy_very_close = nearest_enemy_dist < 150

        # --- Aim quality: angle between aim direction and nearest enemy ---
        nearest_enemy_idx = dist_aa.masked_fill(eye_mask | ~other_alive, 1e6).argmin(dim=2)  # (B, A)
        ne_x = sim.agent_x.gather(1, nearest_enemy_idx)  # (B, A)
        ne_y = sim.agent_y.gather(1, nearest_enemy_idx)
        angle_to_nearest = torch.atan2(ne_y - sim.agent_y, ne_x - sim.agent_x)
        aim_error = torch.atan2(
            torch.sin(sim.agent_dir - angle_to_nearest),
            torch.cos(sim.agent_dir - angle_to_nearest)
        ).abs()  # (B, A) in [0, pi]
        aiming_at_enemy = aim_error < 0.15  # ~8.6 degrees
        aiming_roughly = aim_error < 0.5  # ~28 degrees

        # --- Bullet proximity ---
        bullet_nearby_sq = 150.0 ** 2
        bdx = sim.bullet_x.unsqueeze(1) - sim.agent_x.unsqueeze(-1)
        bdy = sim.bullet_y.unsqueeze(1) - sim.agent_y.unsqueeze(-1)
        bdist_sq = bdx**2 + bdy**2
        agent_idx_t = torch.arange(A, device=device).view(1, A, 1)
        not_own_bullet = sim.bullet_owner.unsqueeze(1) != agent_idx_t
        enemy_bullet_nearby = (
            (bdist_sq < bullet_nearby_sq) &
            sim.bullet_active.unsqueeze(1) &
            not_own_bullet
        ).any(dim=-1)  # (B, A)

        # Check if bullet is heading toward agent (dot product of bullet vel and relative pos)
        # Bullet vel: (B, 1, M), agent pos relative to bullet: -(bdx, bdy)
        # dot = bvx * (-bdx) + bvy * (-bdy) > 0 means approaching
        bullet_approaching = (
            (sim.bullet_vx.unsqueeze(1) * (-bdx) + sim.bullet_vy.unsqueeze(1) * (-bdy) > 0) &
            (bdist_sq < bullet_nearby_sq) &
            sim.bullet_active.unsqueeze(1) &
            not_own_bullet
        ).any(dim=-1)  # (B, A)

        # --- Movement ---
        speed = (sim.agent_vx**2 + sim.agent_vy**2).sqrt()
        is_moving = speed > 0.5
        is_healing = sim.agent_heal_progress > 0

        # --- Ammo/HP status ---
        has_ammo = sim.agent_ammo > 0
        low_ammo = sim.agent_ammo <= 5
        no_ammo = sim.agent_ammo == 0
        low_hp = sim.agent_health < 50
        very_low_hp = sim.agent_health < 25
        has_medkits = sim.agent_medkits > 0
        full_hp = sim.agent_health >= AGENT_MAX_HP - 1

        # --- Movement toward zone center ---
        vel_toward_center_x = (ARENA_W / 2 - sim.agent_x).sign()
        vel_toward_center_y = (ARENA_H / 2 - sim.agent_y).sign()
        moving_toward_center = (
            (sim.agent_vx * vel_toward_center_x > 0.5) |
            (sim.agent_vy * vel_toward_center_y > 0.5)
        )

        # ========== FRAME-LEVEL STAT TRACKING ==========
        alive_mask = is_alive  # shorthand

        # Movement
        inc_t("frames_alive", alive_mask)
        inc_t("frames_moving", is_moving & alive_mask)
        inc_t("frames_stationary", ~is_moving & alive_mask)
        inc_t("frames_healing", is_healing & alive_mask)

        # Zone
        inc_t("frames_outside_zone", outside_zone)
        inc_t("frames_outside_zone_moving_toward", outside_zone & moving_toward_center)
        inc_t("frames_outside_zone_not_moving_toward", outside_zone & ~moving_toward_center)
        # Zone awareness: when zone is actively shrinking, are they inside?
        zone_shrinking = (sim.frame >= ZONE_SHRINK_START).unsqueeze(1) & alive_mask
        inc_t("frames_zone_shrinking", zone_shrinking)
        inc_t("frames_in_zone_while_shrinking", zone_shrinking & ~outside_zone)

        # Bullet dodging
        bullets_and_alive = enemy_bullet_nearby & alive_mask
        approaching_and_alive = bullet_approaching & alive_mask
        inc_t("dodge_frames_bullet_nearby", bullets_and_alive)
        inc_t("dodge_frames_nearby_moving", bullets_and_alive & is_moving)
        inc_t("dodge_frames_nearby_stationary", bullets_and_alive & ~is_moving)
        inc_t("dodge_frames_approaching", approaching_and_alive)
        inc_t("dodge_frames_approaching_moving", approaching_and_alive & is_moving)
        inc_t("dodge_frames_approaching_stationary", approaching_and_alive & ~is_moving)

        # Aim quality
        inc_t("aim_frames_enemy_visible", any_enemy_visible & alive_mask)
        inc_t("aim_frames_on_target", aiming_at_enemy & any_enemy_visible & alive_mask)
        inc_t("aim_frames_roughly_on", aiming_roughly & any_enemy_visible & alive_mask)
        inc_t("aim_error_sum", (aim_error * any_enemy_visible.float() * alive_mask.float()))
        inc_t("aim_error_count", any_enemy_visible & alive_mask)

        # Decision quality: fire when aiming at enemy with ammo
        wants_fire = disc_ba[:, :, 2] == 1  # fire action
        inc_t("fire_when_aiming_at_enemy", wants_fire & aiming_at_enemy & has_ammo & any_enemy_visible & alive_mask)
        inc_t("fire_when_not_aiming", wants_fire & ~aiming_roughly & has_ammo & alive_mask)
        inc_t("fire_when_no_ammo", wants_fire & no_ammo & alive_mask)
        inc_t("fire_when_no_enemy_visible", wants_fire & ~any_enemy_visible & alive_mask)
        inc_t("fire_total_decisions", wants_fire & alive_mask)
        inc_t("no_fire_when_should", ~wants_fire & aiming_at_enemy & has_ammo & any_enemy_visible & alive_mask & (sim.agent_cooldown == 0))

        # Heal decisions
        wants_heal = disc_ba[:, :, 3] == 1
        inc_t("heal_when_low_hp_has_medkit", wants_heal & low_hp & has_medkits & alive_mask)
        inc_t("heal_when_full_hp", wants_heal & full_hp & alive_mask)
        inc_t("heal_opportunities_low_hp", low_hp & has_medkits & alive_mask)
        inc_t("heal_while_enemy_close", wants_heal & enemy_very_close & alive_mask)
        inc_t("heal_total_decisions", wants_heal & alive_mask)

        # Engagement decisions
        inc_t("engage_has_ammo_enemy_visible", any_enemy_visible & has_ammo & alive_mask)
        inc_t("engage_no_ammo_enemy_visible", any_enemy_visible & no_ammo & alive_mask)
        inc_t("engage_no_ammo_enemy_close_moving_away",
              enemy_close & no_ammo & alive_mask & is_moving &
              ~moving_toward_center)  # rough proxy for fleeing

        # Ammo tracking
        peak_ammo_per_life = torch.maximum(peak_ammo_per_life, sim.agent_ammo)

        # ========== STEP THE SIM ==========
        alive_before_step = sim.agent_alive.clone()
        health_before_step = sim.agent_health.clone()
        ammo_before_step = sim.agent_ammo.clone()
        medkits_before_step = sim.agent_medkits.clone()

        step_rewards = torch.zeros(B, A, device=device)
        agent_done = torch.zeros(B, A, dtype=torch.bool, device=device)

        for _rep in range(ACTION_REPEAT):
            cur_alive = sim.agent_alive.clone()
            move_x = mx * cur_alive
            move_y = my * cur_alive

            rewards, episode_done = sim.step(move_x, move_y, aim, fire & cur_alive, heal & cur_alive)
            step_rewards += rewards
            agent_done |= cur_alive & (~sim.agent_alive | episode_done[:, None])

            # Shots fired (cooldown jump to FIRE_COOLDOWN)
            just_fired = (sim.agent_cooldown == FIRE_COOLDOWN) & (prev_cooldown < FIRE_COOLDOWN) & cur_alive
            inc_t("shots_fired", just_fired)
            prev_cooldown = sim.agent_cooldown.clone()

            # Damage tracking
            health_after = sim.agent_health.clone()
            health_drop = (health_before - health_after).clamp(min=0)

            zp = ((sim.frame.float() - ZONE_SHRINK_START) /
                  (ZONE_SHRINK_END - ZONE_SHRINK_START)).clamp(0, 1)
            zr = ZONE_MAX_RADIUS + (ZONE_MIN_RADIUS - ZONE_MAX_RADIUS) * zp
            dtc = ((sim.agent_x - ARENA_W / 2)**2 + (sim.agent_y - ARENA_H / 2)**2).sqrt()
            outside = (dtc > zr.unsqueeze(1)) & cur_alive
            zone_dmg = ZONE_DAMAGE_PER_FRAME * outside.float()
            inc_t("zone_damage_taken", zone_dmg)

            bullet_dmg = (health_drop - zone_dmg).clamp(min=0)
            inc_t("bullet_hits", bullet_dmg > 0)
            inc_t("bullet_damage_dealt", bullet_dmg)

            # Deaths
            just_died = cur_alive & ~sim.agent_alive
            if just_died.any():
                died_outside = just_died & outside
                died_bullet = just_died & (bullet_dmg > 0)
                inc_t("deaths_total", just_died)
                inc_t("deaths_to_zone", died_outside & ~died_bullet)
                inc_t("deaths_to_bullets", died_bullet)
                # Death with no ammo = never picked up / ran out
                inc_t("deaths_with_no_ammo", just_died & (sim.agent_ammo <= 0))
                inc_t("deaths_with_full_ammo", just_died & (sim.agent_ammo >= AMMO_MAX - 1))
                # Death while healing
                inc_t("deaths_while_healing", just_died & (sim.agent_heal_progress > 0))
                # Track HP at death
                # (health is now <= 0, use health_before for last known HP)
                # Deaths at various HP ranges
                hp_at_death = health_before[just_died]
                inc_t("deaths_hp_was_low", (hp_at_death < 30).sum())

            health_before = health_after

            # Ammo pickups
            ammo_gained = (sim.agent_ammo - ammo_before_step).clamp(min=0)
            inc_t("ammo_pickups", ammo_gained > 0)
            ammo_before_step = sim.agent_ammo.clone()

            # Medkit pickups
            medkits_gained = (sim.agent_medkits - medkits_before_step).clamp(min=0)
            inc_t("medkit_pickups", medkits_gained)
            medkits_before_step = sim.agent_medkits.clone()

            # Heals completed
            # TODO: this is approximate

            # Episode completion
            if episode_done.any():
                done_mask = episode_done
                n_done = done_mask.sum().item()
                total_episodes += n_done

                # Placement
                for place in range(1, A + 1):
                    inc(f"placement_{place}", (sim.agent_place[done_mask] == place).sum().item())

                # Episode length
                inc_t("episode_length_sum", sim.frame[done_mask].float())

                # Timeout vs last standing
                timed_out = sim.frame[done_mask] >= MAX_EPISODE_FRAMES
                inc("episodes_timeout", timed_out.sum().item())
                inc("episodes_last_standing", (~timed_out).sum().item())

                # Survivors
                alive_end = sim.agent_alive[done_mask]
                inc_t("survivors_total", alive_end)

                # Peak ammo achieved
                inc_t("peak_ammo_sum", peak_ammo_per_life[done_mask])
                inc("peak_ammo_count", n_done * A)

                # Reset
                sim.reset(mask=done_mask)
                health_before[done_mask] = AGENT_MAX_HP
                prev_cooldown[done_mask] = 0
                ammo_before_step[done_mask] = 0
                medkits_before_step[done_mask] = 0
                peak_ammo_per_life[done_mask] = 0

                done_expanded = done_mask.unsqueeze(1).expand(B, A).reshape(B * A)
                lstm_hx = torch.where(done_expanded.unsqueeze(1), torch.zeros_like(lstm_hx), lstm_hx)
                lstm_cx = torch.where(done_expanded.unsqueeze(1), torch.zeros_like(lstm_cx), lstm_cx)

        total_steps += 1

        if total_steps % 50 == 0:
            elapsed = time.time() - start_time
            eps_per_sec = total_episodes / max(elapsed, 1)
            total_sim_frames = total_steps * ACTION_REPEAT * B * A
            pct = total_episodes * 100 / target_episodes
            print(f"  Step {total_steps:>5d} | {total_episodes:>7,d} ep "
                  f"({pct:.0f}%) | "
                  f"{total_sim_frames/1e6:>6.0f}M frames | "
                  f"{eps_per_sec:.0f} ep/s | "
                  f"{elapsed:.0f}s")

    elapsed = time.time() - start_time
    total_sim_frames = total_steps * ACTION_REPEAT * B * A
    g = lambda k: S.get(k, 0)  # getter with default 0

    # Normalize action counts
    action_total = sum(c.sum().item() for c in action_counts)

    # ===================================================================
    # PRINT REPORT
    # ===================================================================
    print("\n" + "=" * 74)
    print(f"  BATTLE ROYALE AGENT ANALYSIS — update {update_count}")
    print("=" * 74)
    print(f"  Checkpoint:    {checkpoint_path}")
    print(f"  Episodes:      {total_episodes:,}")
    print(f"  Sim frames:    {total_sim_frames:,.0f}")
    print(f"  Wall time:     {elapsed:.1f}s")
    print(f"  Throughput:    {total_episodes / elapsed:.0f} ep/s, "
          f"{total_sim_frames / elapsed / 1e6:.1f}M frames/s")

    fa = max(g("frames_alive"), 1)
    ep_a = max(total_episodes * A, 1)

    print()
    print("--- PLACEMENT DISTRIBUTION ---")
    total_p = sum(g(f"placement_{p}") for p in range(1, A + 1))
    for p in range(1, A + 1):
        c = g(f"placement_{p}")
        pct = c / max(total_p, 1) * 100
        bar = "#" * int(pct / 2)
        ordinal = f"{p}{'st' if p == 1 else 'nd' if p == 2 else 'rd' if p == 3 else 'th'}"
        print(f"  {ordinal}: {c:>8,d}  ({pct:5.1f}%)  {bar}")
    avg_ep_len = g("episode_length_sum") / max(total_episodes, 1)
    print(f"\n  Avg episode length:  {avg_ep_len:.0f} frames ({avg_ep_len/60:.1f}s)")
    print(f"  Timeouts:            {g('episodes_timeout'):,}")
    print(f"  Last-standing wins:  {g('episodes_last_standing'):,}")

    print()
    print("--- DEATH CAUSES ---")
    dt = max(g("deaths_total"), 1)
    print(f"  Total deaths:          {g('deaths_total'):>10,d}")
    print(f"  Killed by bullets:     {g('deaths_to_bullets'):>10,d}  ({g('deaths_to_bullets')/dt*100:5.1f}%)")
    print(f"  Killed by zone:        {g('deaths_to_zone'):>10,d}  ({g('deaths_to_zone')/dt*100:5.1f}%)")
    other = g("deaths_total") - g("deaths_to_bullets") - g("deaths_to_zone")
    print(f"  Mixed/other:           {other:>10,d}  ({other/dt*100:5.1f}%)")
    print(f"  Died with 0 ammo:      {g('deaths_with_no_ammo'):>10,d}  ({g('deaths_with_no_ammo')/dt*100:5.1f}%)")
    print(f"  Died with full ammo:   {g('deaths_with_full_ammo'):>10,d}  ({g('deaths_with_full_ammo')/dt*100:5.1f}%)")
    print(f"  Died while healing:    {g('deaths_while_healing'):>10,d}  ({g('deaths_while_healing')/dt*100:5.1f}%)")

    print()
    print("--- COMBAT STATS ---")
    sf = max(g("shots_fired"), 1)
    print(f"  Shots fired:           {g('shots_fired'):>10,d}  ({g('shots_fired')/ep_a:.1f}/agent/ep)")
    print(f"  Bullet hits:           {g('bullet_hits'):>10,d}  ({g('bullet_hits')/ep_a:.1f}/agent/ep)")
    print(f"  Accuracy:              {g('bullet_hits')/sf*100:>10.1f}%")
    print(f"  Bullet dmg dealt:      {g('bullet_damage_dealt'):>10,.0f} HP  ({g('bullet_damage_dealt')/ep_a:.1f}/agent/ep)")

    print()
    print("--- FIRING DECISIONS ---")
    ft = max(g("fire_total_decisions"), 1)
    print(f"  Total fire actions:    {g('fire_total_decisions'):>10,d}")
    print(f"  Fire ON target:        {g('fire_when_aiming_at_enemy'):>10,d}  ({g('fire_when_aiming_at_enemy')/ft*100:5.1f}%)")
    print(f"  Fire OFF target:       {g('fire_when_not_aiming'):>10,d}  ({g('fire_when_not_aiming')/ft*100:5.1f}%)")
    print(f"  Fire with no ammo:     {g('fire_when_no_ammo'):>10,d}  ({g('fire_when_no_ammo')/ft*100:5.1f}%)")
    print(f"  Fire with no enemy:    {g('fire_when_no_enemy_visible'):>10,d}  ({g('fire_when_no_enemy_visible')/ft*100:5.1f}%)")
    no_fire_should = g("no_fire_when_should")
    fire_on = g("fire_when_aiming_at_enemy")
    trigger_opps = max(no_fire_should + fire_on, 1)
    print(f"  Trigger discipline:    {fire_on/trigger_opps*100:>10.1f}%  "
          f"(fires {fire_on:,d} / skips {no_fire_should:,d} when on-target & ready)")

    print()
    print("--- AIM QUALITY ---")
    ae_count = max(g("aim_error_count"), 1)
    print(f"  Frames enemy visible:  {g('aim_frames_enemy_visible'):>10,d}")
    avg_aim_err = g("aim_error_sum") / ae_count
    print(f"  Mean aim error:        {avg_aim_err:>10.2f} rad  ({math.degrees(avg_aim_err):.1f} deg)")
    print(f"  On target (<8.6 deg):  {g('aim_frames_on_target')/ae_count*100:>10.1f}%")
    print(f"  Roughly on (<28 deg):  {g('aim_frames_roughly_on')/ae_count*100:>10.1f}%")

    print()
    print("--- BULLET DODGING ---")
    dn = max(g("dodge_frames_bullet_nearby"), 1)
    da = max(g("dodge_frames_approaching"), 1)
    overall_move_pct = g("frames_moving") / fa * 100
    print(f"  Enemy bullet within 150px:")
    print(f"    Total frames:        {g('dodge_frames_bullet_nearby'):>10,d}")
    print(f"    Moving:              {g('dodge_frames_nearby_moving')/dn*100:>10.1f}%")
    print(f"    Stationary:          {g('dodge_frames_nearby_stationary')/dn*100:>10.1f}%")
    print(f"  Enemy bullet APPROACHING (heading toward agent):")
    print(f"    Total frames:        {g('dodge_frames_approaching'):>10,d}")
    print(f"    Moving:              {g('dodge_frames_approaching_moving')/da*100:>10.1f}%")
    print(f"    Stationary:          {g('dodge_frames_approaching_stationary')/da*100:>10.1f}%")
    print(f"  Baseline movement:     {overall_move_pct:>10.1f}%")
    dodge_delta = g("dodge_frames_approaching_moving") / da * 100 - overall_move_pct
    print(f"  Dodge boost (approach):{dodge_delta:>+10.1f}pp")

    print()
    print("--- HEALING DECISIONS ---")
    ht = max(g("heal_total_decisions"), 1)
    ho = max(g("heal_opportunities_low_hp"), 1)
    print(f"  Heal actions (total):  {g('heal_total_decisions'):>10,d}")
    print(f"  Heal when low HP+med:  {g('heal_when_low_hp_has_medkit'):>10,d}  "
          f"({g('heal_when_low_hp_has_medkit')/ho*100:.1f}% of opportunities)")
    print(f"  Heal when full HP:     {g('heal_when_full_hp'):>10,d}  ({g('heal_when_full_hp')/ht*100:.1f}% of heal actions)")
    print(f"  Heal near enemy(<150): {g('heal_while_enemy_close'):>10,d}  ({g('heal_while_enemy_close')/ht*100:.1f}% of heal actions)")

    print()
    print("--- ZONE AWARENESS ---")
    oz = max(g("frames_outside_zone"), 1)
    zs = max(g("frames_zone_shrinking"), 1)
    print(f"  % time outside zone:   {g('frames_outside_zone')/fa*100:>10.2f}%")
    print(f"  Moving toward center:  {g('frames_outside_zone_moving_toward')/oz*100:>10.1f}%  (when outside)")
    print(f"  NOT toward center:     {g('frames_outside_zone_not_moving_toward')/oz*100:>10.1f}%  (when outside)")
    print(f"  In zone during shrink: {g('frames_in_zone_while_shrinking')/zs*100:>10.1f}%")
    print(f"  Zone dmg per agent/ep: {g('zone_damage_taken')/ep_a:>10.1f} HP")

    print()
    print("--- RESOURCES ---")
    print(f"  Ammo pickups/agent/ep: {g('ammo_pickups')/ep_a:>10.2f}")
    print(f"  Medkits picked up/a/e: {g('medkit_pickups')/ep_a:>10.2f}")
    avg_peak = g("peak_ammo_sum") / max(g("peak_ammo_count"), 1)
    print(f"  Avg peak ammo/life:    {avg_peak:>10.1f}")

    print()
    print("--- MOVEMENT ---")
    print(f"  Moving (spd>0.5):      {g('frames_moving')/fa*100:>10.1f}%")
    print(f"  Stationary:            {g('frames_stationary')/fa*100:>10.1f}%")
    print(f"  Healing (channeling):  {g('frames_healing')/fa*100:>10.1f}%")

    print()
    print("--- ACTION DISTRIBUTIONS ---")
    action_names = [
        ("Move X", ["left", "none", "right"]),
        ("Move Y", ["up", "none", "down"]),
        ("Fire", ["no", "yes"]),
        ("Heal", ["no", "yes"]),
    ]
    for i, (name, labels) in enumerate(action_names):
        counts = action_counts[i].float()
        total = counts.sum()
        pcts = counts / total * 100
        parts = "  ".join(f"{l}:{p:.1f}%" for l, p in zip(labels, pcts.tolist()))
        print(f"  {name:>8s}: {parts}")

    # Continuous rotation stats
    rc = max(rotation_count.item(), 1)
    rot_mean = rotation_sum.item() / rc
    rot_std = ((rotation_sq_sum.item() / rc) - rot_mean ** 2) ** 0.5
    print(f"  {'Rotate':>8s}: mean={math.degrees(rot_mean):.1f}°  std={math.degrees(rot_std):.1f}°  (continuous Beta)")

    print()
    print("--- INTELLIGENCE SCORECARD ---")
    # Composite scores
    trigger_score = g("fire_when_aiming_at_enemy") / max(trigger_opps, 1) * 100
    aim_score = g("aim_frames_on_target") / ae_count * 100
    zone_score = g("frames_in_zone_while_shrinking") / zs * 100
    heal_score = g("heal_when_low_hp_has_medkit") / max(ho, 1) * 100
    dodge_score = max(0, dodge_delta)
    waste_score = 100 - g("fire_when_no_ammo") / max(ft, 1) * 100  # lower waste = better

    print(f"  Trigger discipline:  {trigger_score:5.1f}%  (fires when on-target & ready)")
    print(f"  Aim tracking:        {aim_score:5.1f}%  (time aimed at nearest enemy)")
    print(f"  Zone positioning:    {zone_score:5.1f}%  (in zone during shrink)")
    print(f"  Heal decisions:      {heal_score:5.1f}%  (heals when low HP & has medkit)")
    print(f"  Dodge reflex:        {dodge_score:5.1f}pp  (extra movement when bullets approach)")
    print(f"  Ammo conservation:   {waste_score:5.1f}%  (doesn't fire with empty mag)")

    print()
    print("=" * 74)
    return S


def compare_checkpoints(ckpt_paths, num_envs=30000, target_episodes=20000):
    """Run analysis on multiple checkpoints for comparison."""
    print("=" * 74)
    print("  CHECKPOINT COMPARISON")
    print("=" * 74)
    for path in ckpt_paths:
        print(f"\n>>> Analyzing {path}...")
        run_analysis(path, num_envs=num_envs, target_episodes=target_episodes)


def main():
    parser = argparse.ArgumentParser(description="Analyze trained BR agents")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=30000)
    parser.add_argument("--num-episodes", type=int, default=50000)
    parser.add_argument("--compare", nargs="+", type=str, default=None,
                        help="Compare multiple checkpoints")
    args = parser.parse_args()

    if args.compare:
        compare_checkpoints(args.compare, num_envs=args.num_envs,
                            target_episodes=args.num_episodes)
        return

    if args.checkpoint is None:
        from battle_royale.train import _find_latest_checkpoint
        args.checkpoint = _find_latest_checkpoint()
        if not args.checkpoint:
            print("No checkpoints found!")
            return

    run_analysis(args.checkpoint, num_envs=args.num_envs, target_episodes=args.num_episodes)


if __name__ == "__main__":
    main()
