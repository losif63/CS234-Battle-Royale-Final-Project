"""Analyze how often the trained policy chooses to heal when it is actually useful.

Loads the latest *1000* checkpoint, runs 5000 envs x 200 rollout steps x 3 action repeat
(= 3M agent-frames per agent, 9M total), and counts:
  - Total alive agent-frames
  - Total heal actions (disc[..., 3] == 1)
  - Total frames where heal was *possible* (HP < max AND medkits > 0)
  - Heal actions that were *smart* (heal when HP < max AND medkits > 0)
"""

import glob
import os
import re
import time

import torch

from battle_royale.sim import BatchedBRSim
from battle_royale.obs import ObservationBuilder
from battle_royale.train import (
    AttentionActorCritic,
    pack_actor_obs,
    _actions_to_sim,
    _sample_actions,
    _apply_action_masks,
    _load_checkpoint,
    SELF_DIM,
    N_ENTITIES,
    ENTITY_DIM,
    LSTM_HIDDEN,
    MAX_AGENTS,
    NUM_DISCRETE_ACTIONS,
)
from battle_royale.config import AGENT_MAX_HP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_ENVS = 5000
ROLLOUT_STEPS = 200
ACTION_REPEAT = 3
DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Find checkpoint
# ---------------------------------------------------------------------------
from battle_royale.train import _find_latest_checkpoint

ckpt_path = _find_latest_checkpoint()
if not ckpt_path:
    raise FileNotFoundError("No checkpoints found in battle_royale/runs/")


def _extract_num(path):
    m = re.search(r"br_ppo_(\d+)\.pt", os.path.basename(path))
    return int(m.group(1)) if m else -1


print(f"Checkpoint: {ckpt_path}  (update {_extract_num(ckpt_path)})")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
device = torch.device(DEVICE)
network = AttentionActorCritic().to(device)
sd, ckpt = _load_checkpoint(ckpt_path, device)
network.load_state_dict(sd)
network.eval()
print(f"Model loaded. LSTM hidden={LSTM_HIDDEN}, agents={MAX_AGENTS}")

# ---------------------------------------------------------------------------
# Create sim + obs builder
# ---------------------------------------------------------------------------
B = NUM_ENVS
A = MAX_AGENTS

sim = BatchedBRSim(num_envs=B, max_agents=A, device=DEVICE)
obs_builder = ObservationBuilder(sim)

# ---------------------------------------------------------------------------
# Counters (on GPU, sync once at end)
# ---------------------------------------------------------------------------
total_alive_frames = torch.zeros(1, device=device, dtype=torch.long)
total_heal_actions = torch.zeros(1, device=device, dtype=torch.long)
total_heal_possible = torch.zeros(1, device=device, dtype=torch.long)
total_smart_heals = torch.zeros(1, device=device, dtype=torch.long)
total_heal_at_full_hp = torch.zeros(1, device=device, dtype=torch.long)
total_heal_no_medkit = torch.zeros(1, device=device, dtype=torch.long)

# Also track heal by HP bucket for richer analysis
hp_buckets = 5  # [0-20%, 20-40%, 40-60%, 60-80%, 80-100%)
heal_by_bucket = torch.zeros(hp_buckets, device=device, dtype=torch.long)
possible_by_bucket = torch.zeros(hp_buckets, device=device, dtype=torch.long)

# Track raw heal logits (before and after masking) for diagnostics
logit_heal_yes_sum = torch.zeros(1, device=device, dtype=torch.float64)
logit_heal_no_sum = torch.zeros(1, device=device, dtype=torch.float64)
logit_heal_yes_masked_sum = torch.zeros(1, device=device, dtype=torch.float64)
logit_heal_no_masked_sum = torch.zeros(1, device=device, dtype=torch.float64)
logit_count = torch.zeros(1, device=device, dtype=torch.long)

# Logits specifically when heal IS possible
logit_possible_heal_yes_sum = torch.zeros(1, device=device, dtype=torch.float64)
logit_possible_heal_no_sum = torch.zeros(1, device=device, dtype=torch.float64)
logit_possible_count = torch.zeros(1, device=device, dtype=torch.long)

# ---------------------------------------------------------------------------
# LSTM state
# ---------------------------------------------------------------------------
hx = torch.zeros(B * A, LSTM_HIDDEN, device=device)
cx = torch.zeros(B * A, LSTM_HIDDEN, device=device)

# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
sim.reset()
t0 = time.time()
total_sim_frames = 0

for step in range(ROLLOUT_STEPS):
    if (step + 1) % 50 == 0:
        elapsed = time.time() - t0
        fps = total_sim_frames / max(elapsed, 1e-6)
        print(f"  step {step+1}/{ROLLOUT_STEPS}  "
              f"sim_frames={total_sim_frames:,}  "
              f"fps={fps:,.0f}")

    # Build observations
    actor_obs = obs_builder.actor_obs()
    self_feat, entities, entity_mask = pack_actor_obs(actor_obs)

    alive = sim.agent_alive.clone()  # (B, A)

    # Flatten all agents: (B*A, ...)
    sf = self_feat.reshape(B * A, SELF_DIM)
    ent = entities.reshape(B * A, N_ENTITIES, ENTITY_DIM)
    emask = entity_mask.reshape(B * A, N_ENTITIES)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        logits_raw, alpha, beta_param, (hx_out, cx_out) = network.forward_actor(
            sf, ent, emask, hx=hx, cx=cx
        )
        logits_masked = _apply_action_masks(logits_raw, sf)
        disc, cont = _sample_actions(logits_masked, alpha, beta_param)

    # disc is (B*A, 4): [move_x, move_y, fire, heal]
    heal_action = disc[:, 3]  # (B*A,) -- 0=no, 1=yes

    # --- Counting (before stepping sim, using current state) ---
    alive_flat = alive.reshape(B * A)  # (B*A,)

    # Raw sim state for ground-truth HP/medkit checks
    agent_health_flat = sim.agent_health.reshape(B * A)       # raw HP (0-100)
    agent_medkits_flat = sim.agent_medkits.reshape(B * A)     # raw medkit count

    hp_below_max = agent_health_flat < AGENT_MAX_HP
    has_medkit = agent_medkits_flat > 0

    is_alive = alive_flat
    chose_heal = (heal_action == 1) & is_alive
    heal_possible = hp_below_max & has_medkit & is_alive
    smart_heal = chose_heal & heal_possible
    heal_at_full = chose_heal & (~hp_below_max) & is_alive
    heal_no_med = chose_heal & (~has_medkit) & is_alive

    total_alive_frames += is_alive.sum()
    total_heal_actions += chose_heal.sum()
    total_heal_possible += heal_possible.sum()
    total_smart_heals += smart_heal.sum()
    total_heal_at_full_hp += heal_at_full.sum()
    total_heal_no_medkit += heal_no_med.sum()

    # HP bucket analysis (only for alive agents with medkits)
    hp_frac = (agent_health_flat / AGENT_MAX_HP).clamp(0, 1)
    bucket_idx = (hp_frac * hp_buckets).long().clamp(0, hp_buckets - 1)
    for b_idx in range(hp_buckets):
        in_bucket = (bucket_idx == b_idx) & is_alive & has_medkit
        heal_by_bucket[b_idx] += (chose_heal & in_bucket).sum()
        possible_by_bucket[b_idx] += in_bucket.sum()

    # Raw logit diagnostics (heal head: index 3 in logits tuple)
    # logits_raw[3] and logits_masked[3] are (B*A, 2) -- [no_heal, yes_heal]
    heal_logits_raw = logits_raw[3].float()     # (B*A, 2)
    heal_logits_msk = logits_masked[3].float()  # (B*A, 2)

    alive_mask = is_alive
    logit_heal_no_sum += heal_logits_raw[alive_mask, 0].sum().double()
    logit_heal_yes_sum += heal_logits_raw[alive_mask, 1].sum().double()
    logit_heal_no_masked_sum += heal_logits_msk[alive_mask, 0].sum().double()
    logit_heal_yes_masked_sum += heal_logits_msk[alive_mask, 1].sum().double()
    logit_count += alive_mask.sum()

    # When heal is possible
    possible_mask = heal_possible
    if possible_mask.any():
        logit_possible_heal_no_sum += heal_logits_msk[possible_mask, 0].sum().double()
        logit_possible_heal_yes_sum += heal_logits_msk[possible_mask, 1].sum().double()
        logit_possible_count += possible_mask.sum()

    # --- Step sim ---
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
        sim.reset(mask=episode_done)
        total_sim_frames += B * A

    # Reset LSTM for done agents
    agent_done = cur_alive & (~sim.agent_alive | episode_done[:, None])
    done_flat = agent_done.reshape(B * A)
    hx = torch.where(done_flat[:, None], torch.zeros_like(hx_out), hx_out).detach()
    cx = torch.where(done_flat[:, None], torch.zeros_like(cx_out), cx_out).detach()

elapsed = time.time() - t0

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
alive_f = total_alive_frames.item()
heal_f = total_heal_actions.item()
possible_f = total_heal_possible.item()
smart_f = total_smart_heals.item()
full_hp_f = total_heal_at_full_hp.item()
no_med_f = total_heal_no_medkit.item()

lc = logit_count.item()
lpc = logit_possible_count.item()

print()
print("=" * 70)
print("HEAL ACTION ANALYSIS")
print("=" * 70)
print(f"Checkpoint:           {ckpt_path}")
print(f"Envs:                 {B}")
print(f"Rollout steps:        {ROLLOUT_STEPS}")
print(f"Action repeat:        {ACTION_REPEAT}")
print(f"Total sim frames:     {total_sim_frames:,}")
print(f"Wall time:            {elapsed:.1f}s")
print(f"FPS:                  {total_sim_frames / elapsed:,.0f}")
print()
print(f"Total alive agent-frames (decision points):  {alive_f:,}")
print(f"Total heal actions (chose heal=1):            {heal_f:,}")
print(f"  Heal rate (of all alive frames):            {heal_f / max(alive_f, 1) * 100:.2f}%")
print()
print(f"Frames where heal was possible (HP<max + has medkit + alive):")
print(f"  Count:              {possible_f:,}")
print(f"  Fraction of alive:  {possible_f / max(alive_f, 1) * 100:.2f}%")
print()
print(f"Smart heals (chose heal=1 when HP<max AND has medkit):")
print(f"  Count:              {smart_f:,}")
if possible_f > 0:
    print(f"  Heal rate when possible:  {smart_f / possible_f * 100:.2f}%")
else:
    print(f"  Heal rate when possible:  N/A (no opportunities)")
print()
print(f"Wasted heals:")
print(f"  Heal at full HP:    {full_hp_f:,}  ({full_hp_f / max(heal_f, 1) * 100:.1f}% of heal actions)")
print(f"  Heal with no medkit:{no_med_f:,}  ({no_med_f / max(heal_f, 1) * 100:.1f}% of heal actions)")
print(f"  (Note: action mask should prevent both of these)")
print()

print("Heal rate by HP bucket (alive + has medkit):")
print(f"  {'HP range':<15} {'Heals':>10} {'Possible':>10} {'Rate':>8}")
print(f"  {'-'*45}")
for b_idx in range(hp_buckets):
    lo = b_idx * 20
    hi = (b_idx + 1) * 20
    h = heal_by_bucket[b_idx].item()
    p = possible_by_bucket[b_idx].item()
    rate = h / max(p, 1) * 100
    label = f"{lo}-{hi}%"
    if b_idx == hp_buckets - 1:
        label = f"{lo}-100%"
    print(f"  {label:<15} {h:>10,} {p:>10,} {rate:>7.2f}%")

print()
print("-" * 70)
print("HEAL LOGIT DIAGNOSTICS")
print("-" * 70)
if lc > 0:
    print(f"  Avg raw heal logit   [no_heal, yes_heal]: "
          f"[{logit_heal_no_sum.item()/lc:.4f}, {logit_heal_yes_sum.item()/lc:.4f}]")
    print(f"  Avg masked heal logit [no_heal, yes_heal]: "
          f"[{logit_heal_no_masked_sum.item()/lc:.4f}, {logit_heal_yes_masked_sum.item()/lc:.4f}]")
    diff_raw = (logit_heal_yes_sum.item() - logit_heal_no_sum.item()) / lc
    diff_msk = (logit_heal_yes_masked_sum.item() - logit_heal_no_masked_sum.item()) / lc
    print(f"  Avg logit gap (yes-no) raw={diff_raw:.4f}  masked={diff_msk:.4f}")
if lpc > 0:
    print(f"\n  When heal IS possible ({lpc:,} frames):")
    print(f"    Avg masked heal logit [no_heal, yes_heal]: "
          f"[{logit_possible_heal_no_sum.item()/lpc:.4f}, {logit_possible_heal_yes_sum.item()/lpc:.4f}]")
    diff_poss = (logit_possible_heal_yes_sum.item() - logit_possible_heal_no_sum.item()) / lpc
    import math
    prob_yes = 1.0 / (1.0 + math.exp(-diff_poss)) if abs(diff_poss) < 50 else (1.0 if diff_poss > 0 else 0.0)
    print(f"    Logit gap (yes-no): {diff_poss:.4f}  =>  P(heal) ~ {prob_yes:.4f}")
else:
    print("  (No frames where heal was possible)")

print()
print("=" * 70)
