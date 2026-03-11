"""Analyze heal action behavior relative to health level.

Runs 1 env for ~10 episodes, collecting per-step data on CPU to measure:
1. Heal rate when HP < max AND have medkits
2. Heal rate when HP = max (wasted heals)
3. Heal rate when HP < max but NO medkits
4. Heal rate by HP percentage buckets (0-25%, 25-50%, 50-75%, 75-100%)
5. Heal rate with/without enemies visible in FOV
6. Policy heal probability (softmax of heal logits)

Usage:
    CUDA_VISIBLE_DEVICES="" uv run python scripts/analyze_heal_behavior.py
"""

import os
import sys
import time

# Ensure project root is on sys.path (scripts/ dir replaces it when run directly)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F

from battle_royale.sim import BatchedBRSim
from battle_royale.obs import ObservationBuilder
from battle_royale.train import (
    AttentionActorCritic, pack_actor_obs, _actions_to_sim,
    _sample_actions, _apply_action_masks, _load_checkpoint,
    SELF_DIM, N_ENTITIES, ENTITY_DIM, LSTM_HIDDEN,
    MAX_AGENTS, NUM_DISCRETE_ACTIONS, ACTION_REPEAT,
)
from battle_royale.config import AGENT_MAX_HP, MAX_EPISODE_FRAMES

# ── Config ──────────────────────────────────────────────────────────────
NUM_ENVS = 1
TARGET_EPISODES = 10
DEVICE = "cpu"

# ── Find latest checkpoint ──────────────────────────────────────────────
ckpt_dir = os.path.join(os.path.dirname(__file__), "..", "battle_royale", "runs", "apex_rp", "checkpoints")
ckpt_dir = os.path.normpath(ckpt_dir)

from battle_royale.train import _find_latest_checkpoint
ckpt_path = _find_latest_checkpoint(search_dir=ckpt_dir)
if not ckpt_path:
    raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
print(f"Checkpoint: {ckpt_path}")

# ── Load model ──────────────────────────────────────────────────────────
device = torch.device(DEVICE)
network = AttentionActorCritic().to(device)
sd, ckpt = _load_checkpoint(ckpt_path, device)
network.load_state_dict(sd)
network.eval()
print(f"Model loaded (update {ckpt.get('update_count', '?')}). Agents={MAX_AGENTS}")

# ── Sim + obs ───────────────────────────────────────────────────────────
B = NUM_ENVS
A = MAX_AGENTS
sim = BatchedBRSim(num_envs=B, max_agents=A, device=DEVICE)
obs_builder = ObservationBuilder(sim)

# ── Counters ────────────────────────────────────────────────────────────
# 1. HP < max AND has medkits: heal rate
heal_possible_count = 0          # frames where HP < max and medkits > 0
heal_when_possible = 0           # chose heal=1 in those frames

# 2. HP = max: wasted heals
full_hp_count = 0                # frames at full HP
heal_at_full_hp = 0              # chose heal=1 at full HP

# 3. HP < max but NO medkits
low_hp_no_med_count = 0          # frames HP < max and medkits = 0
heal_low_hp_no_med = 0           # chose heal=1 in those frames

# 4. HP buckets (0-25%, 25-50%, 50-75%, 75-100%) with medkits available
bucket_edges = [0.0, 0.25, 0.50, 0.75, 1.0]
n_buckets = len(bucket_edges) - 1
bucket_heal_count = [0] * n_buckets
bucket_total_count = [0] * n_buckets

# 5. Enemies visible vs not (when heal is possible)
enemies_vis_heal_possible = 0    # frames: heal possible + enemies visible
enemies_vis_heal_chosen = 0      # chose heal when above
no_enemies_heal_possible = 0     # frames: heal possible + no enemies visible
no_enemies_heal_chosen = 0       # chose heal when above

# 6. Heal probability tracking (softmax probability of heal=1)
#    a) When heal is possible
prob_heal_when_possible_sum = 0.0
prob_heal_when_possible_count = 0
#    b) When heal is possible + enemies visible
prob_heal_enemies_vis_sum = 0.0
prob_heal_enemies_vis_count = 0
#    c) When heal is possible + no enemies
prob_heal_no_enemies_sum = 0.0
prob_heal_no_enemies_count = 0
#    d) By HP bucket (when has medkits)
bucket_prob_sum = [0.0] * n_buckets
bucket_prob_count = [0] * n_buckets
#    e) Overall (all alive frames)
prob_heal_overall_sum = 0.0
prob_heal_overall_count = 0

# ── LSTM state ──────────────────────────────────────────────────────────
hx = torch.zeros(B * A, LSTM_HIDDEN, device=device)
cx = torch.zeros(B * A, LSTM_HIDDEN, device=device)

# ── Run episodes ────────────────────────────────────────────────────────
sim.reset()
episodes_done = 0
step_count = 0
t0 = time.time()

print(f"\nRunning {TARGET_EPISODES} episodes (1 env, {A} agents)...")
print(f"Max frames/episode: {MAX_EPISODE_FRAMES}")

while episodes_done < TARGET_EPISODES:
    step_count += 1

    # Build observations
    actor_obs = obs_builder.actor_obs()
    self_feat, entities, entity_mask = pack_actor_obs(actor_obs)

    alive = sim.agent_alive.clone()  # (B, A)

    # Agent mask tells us if enemies are visible: (B, A, A-1)
    agent_mask = actor_obs["agent_mask"]  # (B, A, A-1) bool
    enemies_visible = agent_mask.any(dim=-1)  # (B, A) — True if any enemy in FOV

    # Flatten for network: (B*A, ...)
    sf = self_feat.reshape(B * A, SELF_DIM)
    ent = entities.reshape(B * A, N_ENTITIES, ENTITY_DIM)
    emask = entity_mask.reshape(B * A, N_ENTITIES)

    with torch.no_grad():
        logits_raw, alpha, beta_param, (hx_out, cx_out) = network.forward_actor(
            sf, ent, emask, hx=hx, cx=cx
        )
        logits_masked = _apply_action_masks(logits_raw, sf)
        disc, cont = _sample_actions(logits_masked, alpha, beta_param)

    # heal action: disc[:, 3], 0=no, 1=yes
    heal_action = disc[:, 3]  # (B*A,)

    # Heal probabilities from masked logits (head index 3)
    heal_logits = logits_masked[3].float()  # (B*A, 2)
    heal_probs = F.softmax(heal_logits, dim=-1)  # (B*A, 2)
    p_heal = heal_probs[:, 1]  # probability of heal=1

    # Ground truth from sim
    alive_flat = alive.reshape(B * A)
    health_flat = sim.agent_health.reshape(B * A)
    medkits_flat = sim.agent_medkits.reshape(B * A)
    enemies_vis_flat = enemies_visible.reshape(B * A)

    hp_frac = (health_flat / AGENT_MAX_HP).clamp(0, 1)
    hp_below_max = health_flat < AGENT_MAX_HP
    has_medkit = medkits_flat > 0
    is_alive = alive_flat

    chose_heal = (heal_action == 1) & is_alive

    # ── 1. Heal possible (HP < max AND medkits > 0) ──
    heal_possible_mask = hp_below_max & has_medkit & is_alive
    heal_possible_count += heal_possible_mask.sum().item()
    heal_when_possible += (chose_heal & heal_possible_mask).sum().item()

    # ── 2. Full HP ──
    full_hp_mask = (~hp_below_max) & is_alive
    full_hp_count += full_hp_mask.sum().item()
    heal_at_full_hp += (chose_heal & full_hp_mask).sum().item()

    # ── 3. HP < max, no medkits ──
    low_no_med_mask = hp_below_max & (~has_medkit) & is_alive
    low_hp_no_med_count += low_no_med_mask.sum().item()
    heal_low_hp_no_med += (chose_heal & low_no_med_mask).sum().item()

    # ── 4. HP buckets (only when has medkits + alive) ──
    for b_idx in range(n_buckets):
        lo = bucket_edges[b_idx]
        hi = bucket_edges[b_idx + 1]
        if b_idx < n_buckets - 1:
            in_bucket = (hp_frac >= lo) & (hp_frac < hi) & has_medkit & is_alive
        else:
            # Last bucket includes upper bound but NOT full HP (which is covered by case 2)
            in_bucket = (hp_frac >= lo) & (hp_frac < 1.0) & has_medkit & is_alive
        n_in = in_bucket.sum().item()
        bucket_total_count[b_idx] += n_in
        bucket_heal_count[b_idx] += (chose_heal & in_bucket).sum().item()
        # Probabilities
        if n_in > 0:
            bucket_prob_sum[b_idx] += p_heal[in_bucket].sum().item()
            bucket_prob_count[b_idx] += n_in

    # ── 5. Enemies visible vs not (when heal is possible) ──
    heal_poss_enemy_vis = heal_possible_mask & enemies_vis_flat
    heal_poss_no_enemy = heal_possible_mask & (~enemies_vis_flat)

    enemies_vis_heal_possible += heal_poss_enemy_vis.sum().item()
    enemies_vis_heal_chosen += (chose_heal & heal_poss_enemy_vis).sum().item()
    no_enemies_heal_possible += heal_poss_no_enemy.sum().item()
    no_enemies_heal_chosen += (chose_heal & heal_poss_no_enemy).sum().item()

    # ── 6. Probability tracking ──
    # a) When heal is possible
    if heal_possible_mask.any():
        prob_heal_when_possible_sum += p_heal[heal_possible_mask].sum().item()
        prob_heal_when_possible_count += heal_possible_mask.sum().item()
    # b) When possible + enemies visible
    if heal_poss_enemy_vis.any():
        prob_heal_enemies_vis_sum += p_heal[heal_poss_enemy_vis].sum().item()
        prob_heal_enemies_vis_count += heal_poss_enemy_vis.sum().item()
    # c) When possible + no enemies
    if heal_poss_no_enemy.any():
        prob_heal_no_enemies_sum += p_heal[heal_poss_no_enemy].sum().item()
        prob_heal_no_enemies_count += heal_poss_no_enemy.sum().item()
    # e) Overall
    if is_alive.any():
        prob_heal_overall_sum += p_heal[is_alive].sum().item()
        prob_heal_overall_count += is_alive.sum().item()

    # ── Step sim ──
    disc_reshaped = disc.reshape(B, A, NUM_DISCRETE_ACTIONS)
    cont_reshaped = cont.reshape(B, A)
    mx, my, aim, fire, heal = _actions_to_sim(disc_reshaped, cont_reshaped, sim.agent_dir)

    for _rep in range(ACTION_REPEAT):
        cur_alive = sim.agent_alive.clone()
        move_x = mx * cur_alive
        move_y = my * cur_alive
        fire_bool = fire & cur_alive
        heal_bool = heal & cur_alive

        rewards, episode_done = sim.step(move_x, move_y, aim, fire_bool, heal_bool)
        if episode_done.any():
            episodes_done += episode_done.sum().item()
            sim.reset(mask=episode_done)

    # Reset LSTM for done agents
    agent_done = alive & (~sim.agent_alive | episode_done[:, None])
    done_flat = agent_done.reshape(B * A)
    hx = torch.where(done_flat[:, None], torch.zeros_like(hx_out), hx_out).detach()
    cx = torch.where(done_flat[:, None], torch.zeros_like(cx_out), cx_out).detach()

    if step_count % 500 == 0:
        print(f"  step {step_count}, episodes done: {episodes_done}/{TARGET_EPISODES}")

elapsed = time.time() - t0

# ── Results ─────────────────────────────────────────────────────────────
def pct(num, den):
    return f"{num / max(den, 1) * 100:.2f}%"

def avg_prob(s, c):
    return f"{s / max(c, 1) * 100:.2f}%" if c > 0 else "N/A"

print()
print("=" * 72)
print("HEAL BEHAVIOR ANALYSIS")
print("=" * 72)
print(f"Checkpoint:        {ckpt_path}")
print(f"Episodes:          {episodes_done}")
print(f"Decision steps:    {step_count:,}")
print(f"Wall time:         {elapsed:.1f}s")
print()

print("-" * 72)
print("1. HEAL RATE WHEN HP < MAX AND HAVE MEDKITS (useful heals)")
print("-" * 72)
print(f"   Frames where heal was possible:  {heal_possible_count:,}")
print(f"   Chose heal:                      {heal_when_possible:,}")
print(f"   Heal rate:                       {pct(heal_when_possible, heal_possible_count)}")
print()

print("-" * 72)
print("2. HEAL RATE WHEN HP = MAX (wasted heals)")
print("-" * 72)
print(f"   Frames at full HP:               {full_hp_count:,}")
print(f"   Chose heal (wasted):             {heal_at_full_hp:,}")
print(f"   Wasted heal rate:                {pct(heal_at_full_hp, full_hp_count)}")
print(f"   (Action mask should block these, so expect ~0%)")
print()

print("-" * 72)
print("3. HEAL RATE WHEN HP < MAX BUT NO MEDKITS")
print("-" * 72)
print(f"   Frames (low HP, no medkits):     {low_hp_no_med_count:,}")
print(f"   Chose heal (impossible):         {heal_low_hp_no_med:,}")
print(f"   Rate:                            {pct(heal_low_hp_no_med, low_hp_no_med_count)}")
print(f"   (Action mask should block these, so expect ~0%)")
print()

print("-" * 72)
print("4. HEAL RATE BY HP BUCKET (has medkits, HP < 100%)")
print("-" * 72)
print(f"   {'HP Range':<14} {'Frames':>10} {'Healed':>10} {'Rate':>8}  {'Avg P(heal)':>12}")
print(f"   {'-'*56}")
for b_idx in range(n_buckets):
    lo = int(bucket_edges[b_idx] * 100)
    hi = int(bucket_edges[b_idx + 1] * 100)
    label = f"{lo}-{hi}%"
    if b_idx == n_buckets - 1:
        label = f"{lo}-<100%"
    n = bucket_total_count[b_idx]
    h = bucket_heal_count[b_idx]
    rate = pct(h, n)
    ap = avg_prob(bucket_prob_sum[b_idx], bucket_prob_count[b_idx])
    print(f"   {label:<14} {n:>10,} {h:>10,} {rate:>8}  {ap:>12}")
print()

print("-" * 72)
print("5. HEAL RATE: ENEMIES VISIBLE vs NO ENEMIES (when heal is possible)")
print("-" * 72)
print(f"   {'Condition':<25} {'Frames':>10} {'Healed':>10} {'Rate':>8}")
print(f"   {'-'*55}")
print(f"   {'Enemies visible':<25} {enemies_vis_heal_possible:>10,} {enemies_vis_heal_chosen:>10,} {pct(enemies_vis_heal_chosen, enemies_vis_heal_possible):>8}")
print(f"   {'No enemies visible':<25} {no_enemies_heal_possible:>10,} {no_enemies_heal_chosen:>10,} {pct(no_enemies_heal_chosen, no_enemies_heal_possible):>8}")
print()

print("-" * 72)
print("6. POLICY HEAL PROBABILITY (softmax P(heal=1) from masked logits)")
print("-" * 72)
print(f"   Overall (all alive frames):                    {avg_prob(prob_heal_overall_sum, prob_heal_overall_count)}")
print(f"   When heal is possible (HP<max + has medkits):  {avg_prob(prob_heal_when_possible_sum, prob_heal_when_possible_count)}")
print(f"   When possible + enemies visible:               {avg_prob(prob_heal_enemies_vis_sum, prob_heal_enemies_vis_count)}")
print(f"   When possible + no enemies:                    {avg_prob(prob_heal_no_enemies_sum, prob_heal_no_enemies_count)}")
print()
print(f"   By HP bucket (has medkits, HP < 100%):")
print(f"   {'HP Range':<14} {'Avg P(heal)':>12}")
print(f"   {'-'*28}")
for b_idx in range(n_buckets):
    lo = int(bucket_edges[b_idx] * 100)
    hi = int(bucket_edges[b_idx + 1] * 100)
    label = f"{lo}-{hi}%"
    if b_idx == n_buckets - 1:
        label = f"{lo}-<100%"
    ap = avg_prob(bucket_prob_sum[b_idx], bucket_prob_count[b_idx])
    print(f"   {label:<14} {ap:>12}")

print()
print("=" * 72)
