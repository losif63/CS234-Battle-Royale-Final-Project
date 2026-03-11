"""
Behavioral analysis across training checkpoints.

Runs self-play games for ~80 checkpoints, collects per-checkpoint metrics
(zone deaths, engagement distance, medkits, attention, accuracy, etc.),
and generates plots.

GPU-batched with vectorized metric collection — no per-agent Python loops.

Usage:
    uv run python scripts/behavioral_analysis.py
"""

import os
import sys
import math
import time
import json
from collections import defaultdict

os.environ["PYTHONUNBUFFERED"] = "1"

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

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
    MAX_EPISODE_FRAMES, ENTITY_FOV_RADIUS, BULLET_DAMAGE,
)
from battle_royale.train import _load_checkpoint, ACTION_REPEAT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "battle_royale", "runs", "apex_rp", "checkpoints")
TARGET_CHECKPOINTS = 80
GAMES_PER_CHECKPOINT = 20
BATCH_GAMES = 200
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "behavioral_plots")

# Entity layout
K_BULLETS = MAX_VISIBLE_BULLETS
D_DEPOSITS = NUM_AMMO_DEPOSITS
H_PICKUPS = NUM_HEALTH_PICKUPS
A_OTHER = MAX_AGENTS - 1
AGENT_ENTITY_START = K_BULLETS + D_DEPOSITS + H_PICKUPS


def p(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ---------------------------------------------------------------------------
# Checkpoint discovery & loading
# ---------------------------------------------------------------------------
def discover_checkpoints(ckpt_dir, target_count):
    files = [f for f in os.listdir(ckpt_dir)
             if f.startswith("br_ppo_") and f.endswith(".pt") and f != "br_ppo_final.pt"]
    files_with_num = [(f, int(f.replace("br_ppo_", "").replace(".pt", ""))) for f in files]
    files_with_num.sort(key=lambda x: x[1])
    if len(files_with_num) <= target_count:
        return files_with_num
    step = max(1, len(files_with_num) // target_count)
    sampled = files_with_num[::step]
    if files_with_num[0] not in sampled:
        sampled.insert(0, files_with_num[0])
    if files_with_num[-1] not in sampled:
        sampled.append(files_with_num[-1])
    return sampled


def load_networks(ckpt_dir, checkpoints, device):
    networks = {}
    for i, (fname, update_num) in enumerate(checkpoints):
        path = os.path.join(ckpt_dir, fname)
        net = AttentionActorCritic().to(device)
        sd, _ = _load_checkpoint(path, "cpu")
        net.load_state_dict(sd, strict=False)
        net.eval()
        networks[update_num] = net
        if (i + 1) % 20 == 0 or i == len(checkpoints) - 1:
            p(f"  Loaded {i+1}/{len(checkpoints)} networks")
    return networks


# ---------------------------------------------------------------------------
# Vectorized metric collection during game batch
# ---------------------------------------------------------------------------
class BatchMetrics:
    """Accumulates metrics for a batch of games using tensors, no Python loops."""

    def __init__(self, B, A, device):
        self.B = B
        self.A = A
        self.device = device

        # Counters per game (B,)
        self.total_deaths = torch.zeros(B, device=device)
        self.zone_deaths = torch.zeros(B, device=device)
        self.bullet_deaths = torch.zeros(B, device=device)
        self.bullets_fired = torch.zeros(B, device=device)
        self.bullets_hit = torch.zeros(B, device=device)
        self.heal_total = torch.zeros(B, device=device)
        self.heal_at_full = torch.zeros(B, device=device)

        # Survival: sum of death frames per game, count of deaths
        self.death_frame_sum = torch.zeros(B, device=device)
        self.death_frame_count = torch.zeros(B, device=device)

        # Sums for averaging later
        self.engagement_dist_sum = torch.zeros(B, device=device)
        self.engagement_dist_count = torch.zeros(B, device=device)
        self.aim_error_sum = torch.zeros(B, device=device)
        self.aim_error_count = torch.zeros(B, device=device)

        # Attention sums (with enemies)
        self.attn_enemy_sum = torch.zeros(B, 4, device=device)  # bullets, deposits, health, agents
        self.attn_enemy_count = torch.zeros(B, device=device)
        # Attention sums (no enemies)
        self.attn_noene_sum = torch.zeros(B, 4, device=device)
        self.attn_noene_count = torch.zeros(B, device=device)

    def record_attention(self, attn_weights, entity_mask, alive, active_env, step):
        """Vectorized attention recording. attn_weights: (B, A, N), entity_mask: (B, A, N)."""
        if step % 5 != 0:
            return

        B, A, N = attn_weights.shape
        # Mask for active alive agents
        valid = alive & active_env.unsqueeze(1)  # (B, A)

        w = attn_weights  # (B, A, N)
        em = entity_mask.float()  # (B, A, N)

        # Sum attention by entity type
        bw = (w[:, :, :K_BULLETS] * em[:, :, :K_BULLETS]).sum(-1)  # (B, A)
        dw = (w[:, :, K_BULLETS:K_BULLETS+D_DEPOSITS] * em[:, :, K_BULLETS:K_BULLETS+D_DEPOSITS]).sum(-1)
        hw = (w[:, :, K_BULLETS+D_DEPOSITS:AGENT_ENTITY_START] * em[:, :, K_BULLETS+D_DEPOSITS:AGENT_ENTITY_START]).sum(-1)
        aw = (w[:, :, AGENT_ENTITY_START:AGENT_ENTITY_START+A_OTHER] * em[:, :, AGENT_ENTITY_START:AGENT_ENTITY_START+A_OTHER]).sum(-1)
        tw = bw + dw + hw + aw  # (B, A)

        # Normalize
        tw_safe = tw.clamp(min=1e-8)
        bw = bw / tw_safe
        dw = dw / tw_safe
        hw = hw / tw_safe
        aw = aw / tw_safe

        # Check if enemies visible
        enemies_vis = em[:, :, AGENT_ENTITY_START:AGENT_ENTITY_START+A_OTHER].any(-1)  # (B, A)
        has_entities = em.any(-1)  # (B, A)
        valid_with_ent = valid & has_entities & (tw > 1e-8)

        # With enemies
        e_mask = valid_with_ent & enemies_vis  # (B, A)
        stacked = torch.stack([bw, dw, hw, aw], dim=-1)  # (B, A, 4)
        self.attn_enemy_sum += (stacked * e_mask.unsqueeze(-1)).sum(1)  # (B, 4)
        self.attn_enemy_count += e_mask.sum(1).float()  # (B,)

        # Without enemies
        n_mask = valid_with_ent & ~enemies_vis
        self.attn_noene_sum += (stacked * n_mask.unsqueeze(-1)).sum(1)
        self.attn_noene_count += n_mask.sum(1).float()

    def record_heals(self, heal_action, alive, health, active_env):
        """heal_action: (B, A) bool, health: (B, A)."""
        valid = alive & active_env.unsqueeze(1) & heal_action  # (B, A)
        self.heal_total += valid.sum(1).float()
        full_hp = health >= (AGENT_MAX_HP - 0.1)
        self.heal_at_full += (valid & full_hp).sum(1).float()

    def record_combat(self, just_fired, just_hit, sim, pre_alive, active_env):
        """Vectorized combat metrics."""
        B, A = just_fired.shape
        valid_fire = just_fired & active_env.unsqueeze(1)
        self.bullets_fired += valid_fire.sum(1).float()
        self.bullets_hit += (just_hit & active_env.unsqueeze(1)).sum(1).float()

        # Engagement distance: for each firer, find nearest alive enemy
        if not valid_fire.any():
            return

        # Positions of all agents
        ax = sim.agent_x  # (B, A)
        ay = sim.agent_y  # (B, A)
        a_dir = sim.agent_dir  # (B, A)

        # Compute pairwise distances between all agents in each env
        dx = ax.unsqueeze(2) - ax.unsqueeze(1)  # (B, A, A)
        dy = ay.unsqueeze(2) - ay.unsqueeze(1)  # (B, A, A)
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)  # (B, A, A)

        # Mask: other agent must be alive, within FOV, and not self
        self_mask = torch.eye(A, device=self.device).bool().unsqueeze(0).expand(B, -1, -1)
        other_alive = pre_alive.unsqueeze(1).expand(B, A, A)  # (B, A, A)
        valid_target = other_alive & ~self_mask & (dist < ENTITY_FOV_RADIUS)

        # Set invalid distances to inf
        dist_masked = dist.clone()
        dist_masked[~valid_target] = float('inf')

        # Nearest enemy per agent
        nearest_dist, nearest_idx = dist_masked.min(dim=2)  # (B, A)
        has_target = nearest_dist < float('inf')
        fire_with_target = valid_fire & has_target  # (B, A)

        # Engagement distance — clamp inf to 0 before multiply to avoid inf*0=nan
        safe_dist = torch.where(fire_with_target, nearest_dist, torch.zeros_like(nearest_dist))
        self.engagement_dist_sum += (safe_dist * fire_with_target.float()).sum(1)
        self.engagement_dist_count += fire_with_target.sum(1).float()

        # Aim error: angle between agent_dir and direction to nearest enemy
        target_x = torch.gather(ax.unsqueeze(1).expand(B, A, A), 2, nearest_idx.unsqueeze(2)).squeeze(2)
        target_y = torch.gather(ay.unsqueeze(1).expand(B, A, A), 2, nearest_idx.unsqueeze(2)).squeeze(2)
        to_target_angle = torch.atan2(target_y - ay, target_x - ax)  # (B, A)
        angle_diff = a_dir - to_target_angle
        aim_error = torch.abs(torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff)))
        aim_error_deg = aim_error * (180.0 / math.pi)
        safe_aim = torch.where(fire_with_target, aim_error_deg, torch.zeros_like(aim_error_deg))

        self.aim_error_sum += (safe_aim * fire_with_target.float()).sum(1)
        self.aim_error_count += fire_with_target.sum(1).float()

    def record_deaths(self, just_died, sim, active_env):
        """Vectorized death recording."""
        B, A = just_died.shape
        valid = just_died & active_env.unsqueeze(1)
        self.total_deaths += valid.sum(1).float()

        # Zone check
        px = sim.agent_x
        py = sim.agent_y
        frame = sim.frame.unsqueeze(1).expand(B, A).float()
        zp = ((frame - ZONE_SHRINK_START) / (ZONE_SHRINK_END - ZONE_SHRINK_START)).clamp(0, 1)
        zr = ZONE_MAX_RADIUS + (ZONE_MIN_RADIUS - ZONE_MAX_RADIUS) * zp
        dist_c = torch.sqrt((px - ARENA_W/2)**2 + (py - ARENA_H/2)**2)
        in_zone = dist_c <= zr

        self.zone_deaths += (valid & ~in_zone).sum(1).float()
        self.bullet_deaths += (valid & in_zone).sum(1).float()

        # Track frame of death for survival time
        frame_at_death = sim.frame.unsqueeze(1).expand(B, A).float()
        self.death_frame_sum += (frame_at_death * valid.float()).sum(1)
        self.death_frame_count += valid.sum(1).float()


# ---------------------------------------------------------------------------
# Run batch
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_batch_and_collect(sim, obs_builder, networks, game_assignments, device):
    B = len(game_assignments)
    A = MAX_AGENTS

    unique_updates = list(set(game_assignments))
    update_to_idx = {u: i for i, u in enumerate(unique_updates)}
    net_list = [networks[u] for u in unique_updates]
    num_nets = len(net_list)

    net_assignment = torch.zeros(B, A, dtype=torch.long, device=device)
    for b, u in enumerate(game_assignments):
        net_assignment[b, :] = update_to_idx[u]

    sim.reset()

    lstm_hx = torch.zeros(B, A, LSTM_HIDDEN, device=device)
    lstm_cx = torch.zeros(B, A, LSTM_HIDDEN, device=device)

    metrics = BatchMetrics(B, A, device)

    # Per-episode resource tracking
    ep_ammo_pickups = torch.zeros(B, A, device=device)
    ep_medkit_pickups = torch.zeros(B, A, device=device)
    prev_ammo = sim.agent_ammo.clone()
    prev_medkits = sim.agent_medkits.clone()

    max_steps = 5000
    step = 0

    while not sim.episode_done.all().item() and step < max_steps:
        step += 1

        actor_obs = obs_builder.actor_obs()
        self_feat, entities, entity_mask = pack_actor_obs(actor_obs)

        BA = B * A
        sf_flat = self_feat.reshape(BA, SELF_DIM)
        ent_flat = entities.reshape(BA, N_ENTITIES, ENTITY_DIM)
        emask_flat = entity_mask.reshape(BA, N_ENTITIES)
        hx_flat = lstm_hx.reshape(BA, LSTM_HIDDEN)
        cx_flat = lstm_cx.reshape(BA, LSTM_HIDDEN)
        net_flat = net_assignment.reshape(BA)

        alive_flat = sim.agent_alive.reshape(BA)
        active_env = ~sim.episode_done
        active_flat = active_env.unsqueeze(1).expand(B, A).reshape(BA) & alive_flat

        all_disc = torch.zeros(BA, NUM_DISCRETE_ACTIONS, dtype=torch.long, device=device)
        all_cont = torch.zeros(BA, device=device)
        new_hx = hx_flat.clone()
        new_cx = cx_flat.clone()
        attn_weights_flat = torch.zeros(BA, N_ENTITIES, device=device)

        for net_idx in range(num_nets):
            mask = (net_flat == net_idx) & active_flat
            if not mask.any():
                continue
            indices = mask.nonzero(as_tuple=True)[0]

            logits, alpha, beta_param, (hx_out, cx_out), attn_w = net_list[net_idx].forward_actor(
                sf_flat[indices], ent_flat[indices], emask_flat[indices],
                hx=hx_flat[indices], cx=cx_flat[indices], return_attention=True)

            logits = _apply_action_masks(logits, sf_flat[indices])
            disc, cont = _sample_actions(logits, alpha, beta_param)

            all_disc[indices] = disc
            all_cont[indices] = cont
            new_hx[indices] = hx_out
            new_cx[indices] = cx_out
            attn_weights_flat[indices] = attn_w

        lstm_hx = new_hx.reshape(B, A, LSTM_HIDDEN)
        lstm_cx = new_cx.reshape(B, A, LSTM_HIDDEN)
        all_disc = all_disc.reshape(B, A, NUM_DISCRETE_ACTIONS)
        all_cont = all_cont.reshape(B, A)
        attn_weights_all = attn_weights_flat.reshape(B, A, N_ENTITIES)

        mx, my, aim, fire, heal = _actions_to_sim(all_disc, all_cont, sim.agent_dir)

        # Record attention
        metrics.record_attention(attn_weights_all, entity_mask, sim.agent_alive, active_env, step)
        # Record heals
        metrics.record_heals(heal, sim.agent_alive, sim.agent_health, active_env)

        # Step sim with action repeat
        for _rep in range(ACTION_REPEAT):
            pre_alive = sim.agent_alive.clone()
            pre_health = sim.agent_health.clone()

            cur_alive = sim.agent_alive.clone()
            _, done = sim.step(
                mx * cur_alive.float(), my * cur_alive.float(),
                aim, fire & cur_alive, heal & cur_alive)

            # Combat metrics
            just_fired = (sim.agent_cooldown == FIRE_COOLDOWN) & pre_alive
            hp_drop = pre_health - sim.agent_health
            just_hit = (hp_drop >= BULLET_DAMAGE * 0.8) & pre_alive
            metrics.record_combat(just_fired, just_hit, sim, pre_alive, active_env)

            # Death metrics
            just_died = pre_alive & ~sim.agent_alive
            metrics.record_deaths(just_died, sim, active_env)

            # Resource tracking
            ammo_gained = (sim.agent_ammo > prev_ammo) & sim.agent_alive & active_env.unsqueeze(1)
            ep_ammo_pickups += ammo_gained.float()
            med_gained = (sim.agent_medkits > prev_medkits) & sim.agent_alive & active_env.unsqueeze(1)
            ep_medkit_pickups += med_gained.float()
            prev_ammo = sim.agent_ammo.clone()
            prev_medkits = sim.agent_medkits.clone()

            if done.all():
                break

        # Reset LSTM for dead
        dead = ~sim.agent_alive
        lstm_hx = lstm_hx * (~dead).unsqueeze(-1).float()
        lstm_cx = lstm_cx * (~dead).unsqueeze(-1).float()

    # Aggregate per-game results to per-checkpoint
    result = {}
    for b_idx in range(B):
        u = game_assignments[b_idx]
        if u not in result:
            result[u] = defaultdict(list)
        r = result[u]

        r["total_deaths"].append(metrics.total_deaths[b_idx].item())
        r["zone_deaths"].append(metrics.zone_deaths[b_idx].item())
        r["bullet_deaths"].append(metrics.bullet_deaths[b_idx].item())
        r["bullets_fired"].append(metrics.bullets_fired[b_idx].item())
        r["bullets_hit"].append(metrics.bullets_hit[b_idx].item())
        r["heal_total"].append(metrics.heal_total[b_idx].item())
        r["heal_at_full"].append(metrics.heal_at_full[b_idx].item())
        r["episode_length"].append(sim.frame[b_idx].item())
        r["kills"].append(sim.agent_kills[b_idx].float().mean().item())
        r["ammo_pickups"].append(ep_ammo_pickups[b_idx].mean().item())
        r["medkit_pickups"].append(ep_medkit_pickups[b_idx].mean().item())

        dfc = metrics.death_frame_count[b_idx].item()
        if dfc > 0:
            r["mean_survival_time"].append(metrics.death_frame_sum[b_idx].item() / dfc)

        edc = metrics.engagement_dist_count[b_idx].item()
        if edc > 0:
            r["mean_engage_dist"].append(metrics.engagement_dist_sum[b_idx].item() / edc)
        aec = metrics.aim_error_count[b_idx].item()
        if aec > 0:
            r["mean_aim_error"].append(metrics.aim_error_sum[b_idx].item() / aec)

        aec2 = metrics.attn_enemy_count[b_idx].item()
        if aec2 > 0:
            for i, key in enumerate(["attn_bullets_e", "attn_deposits_e", "attn_health_e", "attn_agents_e"]):
                r[key].append(metrics.attn_enemy_sum[b_idx, i].item() / aec2)
        anc = metrics.attn_noene_count[b_idx].item()
        if anc > 0:
            for i, key in enumerate(["attn_bullets_n", "attn_deposits_n", "attn_health_n", "attn_agents_n"]):
                r[key].append(metrics.attn_noene_sum[b_idx, i].item() / anc)

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def smooth(y, window=5):
    if len(y) < window:
        return y
    y = np.array(y, dtype=float)
    out = np.empty_like(y)
    for i in range(len(y)):
        lo = max(0, i - window // 2)
        hi = min(len(y), i + window // 2 + 1)
        out[i] = y[lo:hi].mean()
    return out


def plot_all(summaries, updates_sorted, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'figure.facecolor': 'white'})

    x = np.array(updates_sorted)

    def get(key):
        return np.array([summaries[u].get(key, 0) for u in updates_sorted])

    zone_pct = get("zone_death_pct")
    bullet_pct = get("bullet_death_pct")
    acc = get("accuracy")
    eng_dist = get("mean_engage_dist")
    aim_err = get("mean_aim_error")
    ammo = get("mean_ammo")
    medkit = get("mean_medkit")
    ep_len = get("mean_ep_len")
    survival = get("mean_survival_time")
    kills = get("mean_kills")
    heal_waste = get("heal_waste_pct")
    bullets_fired = get("mean_bullets_fired")
    ab_e = get("attn_bullets_e")
    ad_e = get("attn_deposits_e")
    ah_e = get("attn_health_e")
    aa_e = get("attn_agents_e")
    ab_n = get("attn_bullets_n")
    ad_n = get("attn_deposits_n")
    ah_n = get("attn_health_n")
    aa_n = get("attn_agents_n")

    # 1. Death Causes (stacked area)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(x, bullet_pct, zone_pct, labels=["Bullet Deaths", "Zone Deaths"],
                 colors=["#e74c3c", "#3498db"], alpha=0.8)
    ax.set_xlabel("Training Update"); ax.set_ylabel("% of Deaths")
    ax.set_title("Death Causes Over Training"); ax.legend(loc="center right"); ax.set_ylim(0, 100); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "01_death_causes.png"), dpi=150); plt.close()

    # 2. Shot Accuracy
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, acc, "o", color="#e74c3c", markersize=3, alpha=0.4)
    ax.plot(x, smooth(acc, 7), color="#e74c3c", linewidth=2.5, label="Smoothed")
    ax.set_xlabel("Training Update"); ax.set_ylabel("Shot Accuracy (%)")
    ax.set_title("Shot Accuracy Over Training"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "02_shot_accuracy.png"), dpi=150); plt.close()

    # 3. Engagement Distance
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, eng_dist, "o", color="#2ecc71", markersize=3, alpha=0.4)
    ax.plot(x, smooth(eng_dist, 7), color="#2ecc71", linewidth=2.5, label="Mean")
    ax.axhline(y=ENTITY_FOV_RADIUS, color="gray", linestyle=":", alpha=0.5, label=f"FOV ({ENTITY_FOV_RADIUS}px)")
    ax.set_xlabel("Training Update"); ax.set_ylabel("Distance (px)")
    ax.set_title("Engagement Distance Over Training"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "03_engagement_distance.png"), dpi=150); plt.close()

    # 4. Aim Error
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, aim_err, "o", color="#9b59b6", markersize=3, alpha=0.4)
    ax.plot(x, smooth(aim_err, 7), color="#9b59b6", linewidth=2.5, label="Mean")
    ax.axhline(y=90, color="gray", linestyle=":", alpha=0.5, label="Random (90°)")
    ax.set_xlabel("Training Update"); ax.set_ylabel("Aim Error (°)")
    ax.set_title("Aim Error Over Training"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "04_aim_error.png"), dpi=150); plt.close()

    # 5. Resources
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(x, ammo, "o", color="#f39c12", markersize=3, alpha=0.4)
    ax1.plot(x, smooth(ammo, 7), color="#f39c12", linewidth=2.5)
    ax1.set_xlabel("Training Update"); ax1.set_ylabel("Pickups/Ep/Agent"); ax1.set_title("Ammo Collection"); ax1.grid(True, alpha=0.3)
    ax2.plot(x, medkit, "o", color="#e74c3c", markersize=3, alpha=0.4)
    ax2.plot(x, smooth(medkit, 7), color="#e74c3c", linewidth=2.5)
    ax2.set_xlabel("Training Update"); ax2.set_ylabel("Pickups/Ep/Agent"); ax2.set_title("Medkit Collection"); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "05_resource_collection.png"), dpi=150); plt.close()

    # 6. Attention (with enemies)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(x, aa_e*100, ab_e*100, ad_e*100, ah_e*100,
                 labels=["Enemies", "Bullets", "Ammo", "Health"],
                 colors=["#e74c3c", "#f39c12", "#2ecc71", "#e91e63"], alpha=0.8)
    ax.set_xlabel("Training Update"); ax.set_ylabel("Attention Share (%)")
    ax.set_title("Attention Distribution (Enemies Visible)"); ax.legend(loc="center right"); ax.set_ylim(0, 100); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "06_attention_with_enemies.png"), dpi=150); plt.close()

    # 7. Attention (no enemies)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.stackplot(x, ab_n*100, ad_n*100, ah_n*100, aa_n*100,
                 labels=["Bullets", "Ammo", "Health", "Agents (ghost)"],
                 colors=["#f39c12", "#2ecc71", "#e91e63", "#95a5a6"], alpha=0.8)
    ax.set_xlabel("Training Update"); ax.set_ylabel("Attention Share (%)")
    ax.set_title("Attention Distribution (No Enemies)"); ax.legend(loc="center right"); ax.set_ylim(0, 100); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "07_attention_no_enemies.png"), dpi=150); plt.close()

    # 8. Episode Length & Kills
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(x, survival, "o", color="#3498db", markersize=3, alpha=0.4)
    ax1.plot(x, smooth(survival, 7), color="#3498db", linewidth=2.5)
    ax1.axhline(y=MAX_EPISODE_FRAMES, color="red", linestyle=":", alpha=0.5, label=f"Timeout ({MAX_EPISODE_FRAMES})")
    ax1.set_xlabel("Training Update"); ax1.set_ylabel("Frames"); ax1.set_title("Mean Survival Time"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(x, kills, "o", color="#e67e22", markersize=3, alpha=0.4)
    ax2.plot(x, smooth(kills, 7), color="#e67e22", linewidth=2.5)
    ax2.set_xlabel("Training Update"); ax2.set_ylabel("Kills/Ep/Agent"); ax2.set_title("Combat Activity"); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "08_episode_length_kills.png"), dpi=150); plt.close()

    # 9. Bullets Fired Per Episode
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, bullets_fired, "o", color="#e67e22", markersize=3, alpha=0.4)
    ax.plot(x, smooth(bullets_fired, 7), color="#e67e22", linewidth=2.5, label="Smoothed")
    ax.set_xlabel("Training Update"); ax.set_ylabel("Bullets Fired / Game")
    ax.set_title("Shooting Activity Over Training"); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "09_shooting_activity.png"), dpi=150); plt.close()

    # 10. Dashboard
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    panels = [
        (gs[0,0], "Zone Death %", zone_pct, "#3498db"),
        (gs[0,1], "Shot Accuracy %", acc, "#e74c3c"),
        (gs[0,2], "Aim Error (°)", aim_err, "#9b59b6"),
        (gs[1,0], "Engage Dist (px)", eng_dist, "#2ecc71"),
        (gs[1,1], "Ammo/Ep", ammo, "#f39c12"),
        (gs[1,2], "Medkits/Ep", medkit, "#e91e63"),
        (gs[2,0], "Survival Time", survival, "#3498db"),
        (gs[2,1], "Kills/Ep/Agent", kills, "#e67e22"),
        (gs[2,2], "Bullets Fired/Game", bullets_fired, "#e67e22"),
    ]
    for gs_pos, title, data, color in panels:
        ax = fig.add_subplot(gs_pos)
        ax.plot(x, data, "o", color=color, markersize=2, alpha=0.4)
        ax.plot(x, smooth(data, 7), color=color, linewidth=2)
        ax.set_title(title, fontsize=11); ax.grid(True, alpha=0.3); ax.tick_params(labelsize=9)
    fig.suptitle("Agent Behavior Evolution Across Training", fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(os.path.join(output_dir, "10_dashboard.png"), dpi=150); plt.close()

    p(f"  Saved 10 plots to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if "--plot-only" in sys.argv:
        p("Re-plotting from saved metrics...")
        with open(os.path.join(OUTPUT_DIR, "metrics_summary.json")) as f:
            summaries = json.load(f)
        updates_sorted = sorted(summaries.keys(), key=int)
        # Convert string keys to int
        summaries = {int(k): v for k, v in summaries.items()}
        updates_sorted = [int(u) for u in updates_sorted]
        plot_all(summaries, updates_sorted, OUTPUT_DIR)
        p("Done!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p(f"Device: {device}")

    checkpoints = discover_checkpoints(CHECKPOINT_DIR, TARGET_CHECKPOINTS)
    p(f"Found {len(checkpoints)} checkpoints")

    p("Loading networks...")
    networks = load_networks(CHECKPOINT_DIR, checkpoints, device)
    update_nums = sorted(networks.keys())

    total_games = len(update_nums) * GAMES_PER_CHECKPOINT
    game_list = []
    for u in update_nums:
        game_list.extend([u] * GAMES_PER_CHECKPOINT)

    import random
    random.shuffle(game_list)
    while len(game_list) % BATCH_GAMES != 0:
        game_list.append(game_list[-1])

    num_batches = len(game_list) // BATCH_GAMES
    p(f"\nRunning {total_games} games ({num_batches} batches of {BATCH_GAMES})")
    p(f"  {GAMES_PER_CHECKPOINT} games/checkpoint, {len(update_nums)} checkpoints\n")

    sim = BatchedBRSim(num_envs=BATCH_GAMES, max_agents=MAX_AGENTS, device=str(device))
    obs_builder = ObservationBuilder(sim)

    # Accumulate across batches
    all_data = defaultdict(lambda: defaultdict(list))

    total_start = time.time()
    for batch_idx in range(num_batches):
        batch_start = time.time()
        assignments = game_list[batch_idx * BATCH_GAMES : (batch_idx + 1) * BATCH_GAMES]

        batch_results = run_batch_and_collect(sim, obs_builder, networks, assignments, device)

        for u, data in batch_results.items():
            for key, vals in data.items():
                all_data[u][key].extend(vals)

        elapsed = time.time() - total_start
        bt = time.time() - batch_start
        eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
        p(f"  Batch {batch_idx+1}/{num_batches} | {bt:.1f}s | elapsed {elapsed:.0f}s | ETA {eta:.0f}s")

    total_time = time.time() - total_start
    p(f"\nData collection done in {total_time:.1f}s")

    # Compute summaries
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summaries = {}
    for u in update_nums:
        d = all_data[u]
        td = max(1, sum(d["total_deaths"]))
        bf = max(1, sum(d["bullets_fired"]))
        ht = max(1, sum(d["heal_total"]))
        summaries[u] = {
            "zone_death_pct": round(100 * sum(d["zone_deaths"]) / td, 1),
            "bullet_death_pct": round(100 * sum(d["bullet_deaths"]) / td, 1),
            "accuracy": round(100 * sum(d["bullets_hit"]) / bf, 1),
            "mean_engage_dist": round(float(np.mean(d["mean_engage_dist"])) if d["mean_engage_dist"] else 0, 1),
            "mean_aim_error": round(float(np.mean(d["mean_aim_error"])) if d["mean_aim_error"] else 0, 1),
            "mean_ammo": round(float(np.mean(d["ammo_pickups"])), 2),
            "mean_medkit": round(float(np.mean(d["medkit_pickups"])), 2),
            "mean_ep_len": round(float(np.mean(d["episode_length"])), 0),
            "mean_survival_time": round(float(np.mean(d["mean_survival_time"])) if d["mean_survival_time"] else 0, 0),
            "mean_kills": round(float(np.mean(d["kills"])), 2),
            "heal_waste_pct": round(100 * sum(d["heal_at_full"]) / ht, 1),
            "mean_bullets_fired": round(float(np.mean(d["bullets_fired"])) if d["bullets_fired"] else 0, 1),
            "attn_bullets_e": round(float(np.mean(d["attn_bullets_e"])) if d["attn_bullets_e"] else 0, 4),
            "attn_deposits_e": round(float(np.mean(d["attn_deposits_e"])) if d["attn_deposits_e"] else 0, 4),
            "attn_health_e": round(float(np.mean(d["attn_health_e"])) if d["attn_health_e"] else 0, 4),
            "attn_agents_e": round(float(np.mean(d["attn_agents_e"])) if d["attn_agents_e"] else 0, 4),
            "attn_bullets_n": round(float(np.mean(d["attn_bullets_n"])) if d["attn_bullets_n"] else 0, 4),
            "attn_deposits_n": round(float(np.mean(d["attn_deposits_n"])) if d["attn_deposits_n"] else 0, 4),
            "attn_health_n": round(float(np.mean(d["attn_health_n"])) if d["attn_health_n"] else 0, 4),
            "attn_agents_n": round(float(np.mean(d["attn_agents_n"])) if d["attn_agents_n"] else 0, 4),
        }

    with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

    p("\nGenerating plots...")
    plot_all(summaries, update_nums, OUTPUT_DIR)
    p("Done!")


if __name__ == "__main__":
    main()
