"""
Engagement range analysis: measures distances at which agents shoot, hit, and see each other.
Answers the key question: are agents fighting at ranges where dodging bullets is possible?

Usage:
    CUDA_VISIBLE_DEVICES="" uv run python analyze_engagement_range.py
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import math
import torch
import numpy as np

from battle_royale.sim import BatchedBRSim
from battle_royale.obs import ObservationBuilder
from battle_royale.config import (
    ENTITY_FOV_RADIUS, BULLET_SPEED, AGENT_SPEED, FIRE_COOLDOWN,
    AGENT_RADIUS, BULLET_RADIUS, BULLET_DAMAGE, AGENT_MAX_HP,
    ARENA_W, ARENA_H,
)
from battle_royale.network import (
    AttentionActorCritic, pack_actor_obs,
    _actions_to_sim, _apply_action_masks, _sample_actions,
    LSTM_HIDDEN, MAX_AGENTS,
)

# ---- Config ----
CHECKPOINT = "battle_royale/runs/apex_rp/checkpoints/br_ppo_1900.pt"
NUM_ENVS = 1
NUM_AGENTS = MAX_AGENTS  # 10
NUM_EPISODES = 10
DEVICE = "cpu"

# ---- Load model ----
print(f"Loading checkpoint: {CHECKPOINT}")
ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
sd = {k.removeprefix("_orig_mod."): v for k, v in ckpt["network"].items()}

network = AttentionActorCritic()
network.load_state_dict(sd, strict=False)
network.eval()

# ---- Create sim ----
sim = BatchedBRSim(num_envs=NUM_ENVS, max_agents=NUM_AGENTS, device=DEVICE)
obs_builder = ObservationBuilder(sim)

B, A = NUM_ENVS, NUM_AGENTS

# ---- Data collection ----
# When an agent fires (cooldown just became FIRE_COOLDOWN): distance to nearest visible enemy
fire_distances = []
# When a bullet hits (agent health decreased from bullet): distance shooter->target at hit time
hit_distances = []
# Every frame: distances between agents that can see each other (within FOV)
visibility_distances = []
# Per-shot tracking: bullet spawn info for matching hits
# We'll track bullet spawn positions + owner to compute hit distances

hx = torch.zeros(B * A, LSTM_HIDDEN)
cx = torch.zeros(B * A, LSTM_HIDDEN)

episodes_completed = 0
total_frames = 0
total_shots_fired = 0
total_hits = 0

print(f"Running {NUM_EPISODES} episodes with {NUM_AGENTS} agents...")
print()

while episodes_completed < NUM_EPISODES:
    sim.reset()
    hx.zero_()
    cx.zero_()

    prev_health = sim.agent_health.clone()
    episode_done = False

    while not episode_done:
        # ---- Compute observations ----
        actor_obs = obs_builder.actor_obs()
        self_feat, entities, entity_mask = pack_actor_obs(actor_obs)

        # Flatten all agents: (B, A, ...) -> (B*A, ...)
        sf_flat = self_feat.reshape(B * A, -1)
        ent_flat = entities.reshape(B * A, entities.shape[2], entities.shape[3])
        emask_flat = entity_mask.reshape(B * A, entity_mask.shape[2])

        with torch.no_grad():
            logits, alpha, beta_param, (hx_out, cx_out) = network.forward_actor(
                sf_flat, ent_flat, emask_flat, hx=hx, cx=cx)
            logits = _apply_action_masks(logits, sf_flat)
            disc, cont = _sample_actions(logits, alpha, beta_param)

        hx = hx_out.detach()
        cx = cx_out.detach()

        # Reshape back to (B, A, ...)
        all_disc = disc.reshape(B, A, -1)
        all_cont = cont.reshape(B, A)

        mx, my, aim, fire, heal = _actions_to_sim(all_disc, all_cont, sim.agent_dir)

        # ---- Record pre-step state ----
        prev_cooldown = sim.agent_cooldown.clone()
        prev_health = sim.agent_health.clone()
        prev_alive = sim.agent_alive.clone()
        prev_bullet_active = sim.bullet_active.clone()

        # Snapshot agent positions before step (for measuring distances)
        agent_x_pre = sim.agent_x.clone()
        agent_y_pre = sim.agent_y.clone()

        # ---- Measure visibility distances (every frame) ----
        for b in range(B):
            alive_mask = sim.agent_alive[b]
            alive_ids = torch.where(alive_mask)[0]
            if len(alive_ids) < 2:
                continue
            for i_idx in range(len(alive_ids)):
                i = alive_ids[i_idx].item()
                xi, yi = sim.agent_x[b, i].item(), sim.agent_y[b, i].item()
                for j_idx in range(i_idx + 1, len(alive_ids)):
                    j = alive_ids[j_idx].item()
                    xj, yj = sim.agent_x[b, j].item(), sim.agent_y[b, j].item()
                    dist = math.sqrt((xi - xj)**2 + (yi - yj)**2)
                    if dist <= ENTITY_FOV_RADIUS:
                        visibility_distances.append(dist)

        # ---- Step simulation ----
        cur_alive = sim.agent_alive.clone()
        move_x = mx * cur_alive
        move_y = my * cur_alive
        fire_bool = fire & cur_alive
        heal_bool = heal & cur_alive
        rewards, ep_done = sim.step(move_x, move_y, aim, fire_bool, heal_bool)

        total_frames += 1

        # ---- Detect shots fired ----
        # After step, agent_cooldown == FIRE_COOLDOWN means they just fired
        just_fired = (sim.agent_cooldown == FIRE_COOLDOWN) & prev_alive

        for b in range(B):
            for a in range(A):
                if just_fired[b, a].item():
                    total_shots_fired += 1
                    # Find nearest visible enemy
                    shooter_x = agent_x_pre[b, a].item()
                    shooter_y = agent_x_pre[b, a].item()
                    # Use actual positions (not the buggy line above)
                    shooter_x = agent_x_pre[b, a].item()
                    shooter_y = agent_y_pre[b, a].item()

                    min_dist = float('inf')
                    for other in range(A):
                        if other == a or not prev_alive[b, other].item():
                            continue
                        ox = agent_x_pre[b, other].item()
                        oy = agent_y_pre[b, other].item()
                        d = math.sqrt((shooter_x - ox)**2 + (shooter_y - oy)**2)
                        if d <= ENTITY_FOV_RADIUS:
                            min_dist = min(min_dist, d)
                    if min_dist < float('inf'):
                        fire_distances.append(min_dist)

        # ---- Detect bullet hits ----
        # Health decreased and agent was alive before -> bullet hit
        # (Zone damage also decreases health, but we can distinguish:
        #  bullet damage = 25.0 per hit, zone = 0.5/frame)
        health_drop = prev_health - sim.agent_health
        # A bullet hit causes exactly BULLET_DAMAGE (25) drop (or multiples)
        # Zone causes 0.5 per frame. We detect bullet hits by checking for
        # drops that are multiples of 25 (or at least >= 25)
        for b in range(B):
            for a in range(A):
                if not prev_alive[b, a].item():
                    continue
                drop = health_drop[b, a].item()
                if drop >= BULLET_DAMAGE - 0.1:  # at least one bullet hit
                    n_hits = round(drop / BULLET_DAMAGE)
                    if n_hits < 1:
                        continue
                    total_hits += n_hits
                    # Find shooter: last_hit_by tracks the attacker
                    attacker = sim.agent_last_hit_by[b, a].item()
                    if attacker >= 0 and attacker < A:
                        # Distance between victim and attacker at this frame
                        vx = agent_x_pre[b, a].item()
                        vy = agent_y_pre[b, a].item()
                        ax_pos = agent_x_pre[b, attacker].item()
                        ay_pos = agent_y_pre[b, attacker].item()
                        dist = math.sqrt((vx - ax_pos)**2 + (vy - ay_pos)**2)
                        for _ in range(n_hits):
                            hit_distances.append(dist)

        # ---- Check episode completion ----
        if ep_done.any().item():
            episodes_completed += 1
            if episodes_completed >= NUM_EPISODES:
                break
            # Reset for next episode
            sim.reset()
            hx.zero_()
            cx.zero_()

        # Safety: don't run forever
        if total_frames > 100000:
            print("WARNING: hit frame limit, stopping early")
            break

# ---- Compute statistics ----
print("=" * 70)
print("ENGAGEMENT RANGE ANALYSIS")
print("=" * 70)

# Game constants
print("\n--- Game Constants ---")
print(f"FOV radius:          {ENTITY_FOV_RADIUS:.0f} units")
print(f"Bullet speed:        {BULLET_SPEED:.0f} units/frame")
print(f"Agent speed:          {AGENT_SPEED:.0f} units/frame")
print(f"Bullet radius:        {BULLET_RADIUS:.0f} units")
print(f"Agent radius:         {AGENT_RADIUS:.0f} units")
print(f"Hit circle (sum):     {BULLET_RADIUS + AGENT_RADIUS:.0f} units")
print(f"Fire cooldown:        {FIRE_COOLDOWN} frames")
print(f"Arena:                {ARENA_W}x{ARENA_H}")
print(f"Arena diagonal:       {math.sqrt(ARENA_W**2 + ARENA_H**2):.0f}")

print(f"\n--- Data Summary ---")
print(f"Episodes completed:   {episodes_completed}")
print(f"Total frames:         {total_frames}")
print(f"Total shots fired:    {total_shots_fired}")
print(f"Total hits detected:  {total_hits}")
print(f"Fire events with visible enemy:  {len(fire_distances)}")
print(f"Hit events with known attacker:  {len(hit_distances)}")
print(f"Visibility pair-frames:          {len(visibility_distances)}")

def print_stats(name, data):
    if not data:
        print(f"\n--- {name}: NO DATA ---")
        return
    arr = np.array(data)
    print(f"\n--- {name} (n={len(arr)}) ---")
    print(f"  Mean:     {arr.mean():.1f} units")
    print(f"  Median:   {np.median(arr):.1f} units")
    print(f"  Std:      {arr.std():.1f} units")
    print(f"  P10:      {np.percentile(arr, 10):.1f} units")
    print(f"  P25:      {np.percentile(arr, 25):.1f} units")
    print(f"  P50:      {np.percentile(arr, 50):.1f} units")
    print(f"  P75:      {np.percentile(arr, 75):.1f} units")
    print(f"  P90:      {np.percentile(arr, 90):.1f} units")
    print(f"  Min:      {arr.min():.1f} units")
    print(f"  Max:      {arr.max():.1f} units")
    return arr

fire_arr = print_stats("Engagement Distance (distance to nearest enemy when firing)", fire_distances)
hit_arr = print_stats("Hit Distance (shooter-to-victim when bullet connects)", hit_distances)
vis_arr = print_stats("Visibility Distance (mutual distance when both in FOV)", visibility_distances)

# ---- Dodgeability Analysis ----
print("\n" + "=" * 70)
print("DODGEABILITY ANALYSIS")
print("=" * 70)

if fire_distances:
    typical_engage = np.median(fire_distances)
else:
    typical_engage = ENTITY_FOV_RADIUS / 2
    print(f"(Using fallback engagement distance: {typical_engage:.0f})")

if hit_distances:
    typical_hit = np.median(hit_distances)
else:
    typical_hit = typical_engage

# Time for bullet to travel engagement distance
bullet_travel_time_engage = typical_engage / BULLET_SPEED
bullet_travel_time_hit = typical_hit / BULLET_SPEED

# How far can agent move in that time?
agent_dodge_distance_engage = AGENT_SPEED * bullet_travel_time_engage
agent_dodge_distance_hit = AGENT_SPEED * bullet_travel_time_hit

# Hit circle: agent needs to move AGENT_RADIUS + BULLET_RADIUS to dodge
dodge_threshold = AGENT_RADIUS + BULLET_RADIUS

print(f"\nAt MEDIAN engagement distance ({typical_engage:.0f} units):")
print(f"  Bullet travel time:         {bullet_travel_time_engage:.1f} frames ({bullet_travel_time_engage/60*1000:.0f} ms at 60fps)")
print(f"  Agent lateral movement:     {agent_dodge_distance_engage:.1f} units")
print(f"  Required dodge distance:    {dodge_threshold:.0f} units (agent_r + bullet_r)")
print(f"  Dodge possible?             {'YES' if agent_dodge_distance_engage > dodge_threshold else 'NO'} ({agent_dodge_distance_engage:.1f} vs {dodge_threshold:.0f})")
print(f"  Dodge ratio:                {agent_dodge_distance_engage / dodge_threshold:.2f}x (>1 = dodgeable)")

if hit_distances:
    print(f"\nAt MEDIAN hit distance ({typical_hit:.0f} units):")
    print(f"  Bullet travel time:         {bullet_travel_time_hit:.1f} frames ({bullet_travel_time_hit/60*1000:.0f} ms at 60fps)")
    print(f"  Agent lateral movement:     {agent_dodge_distance_hit:.1f} units")
    print(f"  Dodge possible?             {'YES' if agent_dodge_distance_hit > dodge_threshold else 'NO'} ({agent_dodge_distance_hit:.1f} vs {dodge_threshold:.0f})")
    print(f"  Dodge ratio:                {agent_dodge_distance_hit / dodge_threshold:.2f}x")

# Minimum distance for dodging to be theoretically possible
min_dodge_distance = dodge_threshold * BULLET_SPEED / AGENT_SPEED
print(f"\nMinimum range for dodge to be possible: {min_dodge_distance:.0f} units")
print(f"  (= {min_dodge_distance / ENTITY_FOV_RADIUS * 100:.0f}% of FOV radius)")

if fire_distances:
    below_dodge = np.sum(np.array(fire_distances) < min_dodge_distance)
    pct_below = below_dodge / len(fire_distances) * 100
    print(f"  Shots fired below min dodge range: {below_dodge}/{len(fire_distances)} ({pct_below:.1f}%)")

if hit_distances:
    below_dodge_hit = np.sum(np.array(hit_distances) < min_dodge_distance)
    pct_below_hit = below_dodge_hit / len(hit_distances) * 100
    print(f"  Hits landed below min dodge range: {below_dodge_hit}/{len(hit_distances)} ({pct_below_hit:.1f}%)")

# ---- Distance bracket breakdown ----
print(f"\n--- Engagement Distance Brackets ---")
brackets = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 300), (300, 400), (400, ENTITY_FOV_RADIUS)]
if fire_distances:
    fire_arr = np.array(fire_distances)
    for lo, hi in brackets:
        count = np.sum((fire_arr >= lo) & (fire_arr < hi))
        pct = count / len(fire_arr) * 100
        travel = (lo + hi) / 2 / BULLET_SPEED
        dodgeable = "dodgeable" if (lo + hi) / 2 > min_dodge_distance else "UNDODGEABLE"
        print(f"  {lo:>4}-{hi:<4}: {count:>5} shots ({pct:>5.1f}%)  bullet_time={travel:.1f}f  [{dodgeable}]")

if hit_distances:
    print(f"\n--- Hit Distance Brackets ---")
    hit_arr = np.array(hit_distances)
    for lo, hi in brackets:
        count = np.sum((hit_arr >= lo) & (hit_arr < hi))
        pct = count / len(hit_arr) * 100
        travel = (lo + hi) / 2 / BULLET_SPEED
        dodgeable = "dodgeable" if (lo + hi) / 2 > min_dodge_distance else "UNDODGEABLE"
        print(f"  {lo:>4}-{hi:<4}: {count:>5} hits  ({pct:>5.1f}%)  bullet_time={travel:.1f}f  [{dodgeable}]")

print(f"\n--- Key Insight ---")
if fire_distances:
    median_fire = np.median(fire_distances)
    ratio = agent_dodge_distance_engage / dodge_threshold
    if ratio < 1.0:
        print(f"At the median engagement range ({median_fire:.0f} units), bullets arrive in")
        print(f"{bullet_travel_time_engage:.1f} frames. An agent can only move {agent_dodge_distance_engage:.1f} units")
        print(f"in that time, but needs {dodge_threshold:.0f} units to dodge. Dodging is NOT")
        print(f"geometrically possible at typical engagement range.")
    else:
        print(f"At the median engagement range ({median_fire:.0f} units), bullets arrive in")
        print(f"{bullet_travel_time_engage:.1f} frames. An agent can move {agent_dodge_distance_engage:.1f} units")
        print(f"in that time, needing {dodge_threshold:.0f} units to dodge. Dodging IS")
        print(f"geometrically possible (ratio {ratio:.2f}x), but requires perfect reaction.")
        if ratio < 2.0:
            print(f"However, with a ratio of only {ratio:.2f}x, the dodge window is very tight.")
            print(f"The agent would need to react within ~{1.0/ratio:.0f} frame(s) and move purely laterally.")

print()
