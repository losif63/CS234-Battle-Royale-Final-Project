"""
Rule-based baseline agent for the Battle Royale environment.

Strategy:
  1. DODGE: If an enemy bullet is close and heading toward us, dodge it.
  2. GET AMMO: If we have 0 ammo, move toward the nearest ammo pickup.
  3. ATTACK: If we have ammo, line up with the enemy on x or y axis and shoot.

Usage:
    uv run python -m agents.baseline_agent --eval                # evaluate PPO vs baseline (100 eps)
    uv run python -m agents.baseline_agent --eval --episodes 500 # more episodes
    uv run python -m agents.baseline_agent --watch               # watch PPO vs baseline
    uv run python -m agents.baseline_agent --watch-baseline      # watch baseline vs baseline
"""

import argparse
import math
import os
import time

import numpy as np
import torch

from src.env import GameEnv
import src.config as cfg

# Reuse PPO's observation builder and frame stacking constants
from agents.ppo_agent import (
    build_obs,
    PPOAgent,
    ActorCritic,
    SINGLE_OBS_DIM,
    OBS_DIM,
    ACTION_REPEAT,
    MAX_STEPS_PER_EPISODE,
    SAVE_DIR,
)

# Actions
STAY = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
SHOOT_UP = 5
SHOOT_DOWN = 6
SHOOT_LEFT = 7
SHOOT_RIGHT = 8

# Baseline tuning constants
ALIGN_TOLERANCE = 25.0      # pixels: how close on an axis to count as "lined up"
DODGE_RADIUS = 120.0         # pixels: threat radius for incoming bullets
BULLET_APPROACH_DOT = -0.3   # dot product threshold (negative = approaching)


class BaselineAgent:
    """Simple rule-based agent."""

    def act(self, raw_obs: dict, agent_id: int) -> int:
        self_key = f"agent_{agent_id}"
        other_id = 1 - agent_id
        other_key = f"agent_{other_id}"

        sx, sy = raw_obs[self_key]["position"]
        ox, oy = raw_obs[other_key]["position"]
        ammo = raw_obs[self_key]["ammo"]
        bullets = raw_obs["bullets"]
        pickups = raw_obs["ammo_pickups"]

        # --- 1. DODGE incoming enemy bullets ---
        dodge = self._dodge_action(sx, sy, agent_id, bullets)
        if dodge is not None:
            return dodge

        # --- 2. GET AMMO if empty ---
        if ammo == 0 and pickups:
            return self._move_toward_nearest_pickup(sx, sy, pickups)

        # --- 3. ATTACK: line up and shoot ---
        if ammo > 0:
            return self._attack_action(sx, sy, ox, oy)

        # Fallback: move toward enemy
        return self._move_toward(sx, sy, ox, oy)

    def _dodge_action(self, sx, sy, agent_id, bullets) -> int | None:
        """If an enemy bullet is close and heading toward us, dodge perpendicular."""
        closest_threat = None
        closest_dist = float("inf")

        for (bx, by), (vx, vy), owner in bullets:
            if owner == agent_id:
                continue  # own bullet

            dx = bx - sx
            dy = by - sy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > DODGE_RADIUS:
                continue

            # Check if bullet is heading toward us (dot product of velocity and displacement)
            bullet_speed = math.sqrt(vx * vx + vy * vy)
            if bullet_speed < 1e-6:
                continue
            # Normalize displacement and velocity
            ndx, ndy = dx / dist if dist > 0 else 0, dy / dist if dist > 0 else 0
            nvx, nvy = vx / bullet_speed, vy / bullet_speed
            # Bullet velocity should point roughly toward us (displacement from bullet to us is (-dx,-dy))
            dot = nvx * (-dx) + nvy * (-dy)
            # Positive dot = bullet moving toward us
            if dot > 0 and dist < closest_dist:
                closest_dist = dist
                closest_threat = (vx, vy)

        if closest_threat is None:
            return None

        vx, vy = closest_threat
        # Move perpendicular to bullet velocity
        # Two perpendicular directions: (vy, -vx) and (-vy, vx)
        # Pick the one that moves us further from bullet's path
        # Simple heuristic: pick direction that moves away from center of bullet path
        perp1_x, perp1_y = vy, -vx
        perp2_x, perp2_y = -vy, vx

        # Choose the perpendicular direction that doesn't push us into walls
        # Prefer moving toward arena center
        cx, cy = cfg.ARENA_WIDTH / 2, cfg.ARENA_HEIGHT / 2
        score1 = perp1_x * (cx - sx) + perp1_y * (cy - sy)
        score2 = perp2_x * (cx - sx) + perp2_y * (cy - sy)

        if score1 >= score2:
            px, py = perp1_x, perp1_y
        else:
            px, py = perp2_x, perp2_y

        # Convert to action: pick dominant axis
        if abs(px) > abs(py):
            return RIGHT if px > 0 else LEFT
        else:
            return DOWN if py > 0 else UP

    def _move_toward_nearest_pickup(self, sx, sy, pickups) -> int:
        best_dist = float("inf")
        best_px, best_py = sx, sy
        for (px, py) in pickups:
            d = math.sqrt((px - sx) ** 2 + (py - sy) ** 2)
            if d < best_dist:
                best_dist = d
                best_px, best_py = px, py
        return self._move_toward(sx, sy, best_px, best_py)

    def _attack_action(self, sx, sy, ox, oy) -> int:
        """If lined up, shoot. Otherwise move to align."""
        dx = ox - sx
        dy = oy - sy

        # Check if we're aligned on x-axis (same column)
        if abs(dx) <= ALIGN_TOLERANCE:
            return SHOOT_UP if dy < 0 else SHOOT_DOWN

        # Check if we're aligned on y-axis (same row)
        if abs(dy) <= ALIGN_TOLERANCE:
            return SHOOT_RIGHT if dx > 0 else SHOOT_LEFT

        # Not aligned: move to align on the closer axis
        if abs(dx) < abs(dy):
            # Closer to x-alignment: move horizontally to match enemy x
            return RIGHT if dx > 0 else LEFT
        else:
            # Closer to y-alignment: move vertically to match enemy y
            return DOWN if dy > 0 else UP

    def _move_toward(self, sx, sy, tx, ty) -> int:
        dx = tx - sx
        dy = ty - sy
        if abs(dx) > abs(dy):
            return RIGHT if dx > 0 else LEFT
        else:
            return DOWN if dy > 0 else UP


# ---------------------------------------------------------------------------
# Evaluation: PPO vs Baseline
# ---------------------------------------------------------------------------

def evaluate(checkpoint: str, num_episodes: int = 100, verbose: bool = True):
    """Run PPO (agent_0) vs Baseline (agent_1) for num_episodes. Returns stats dict."""
    device = torch.device("cpu")
    ppo = PPOAgent(device=device)
    ppo.load(checkpoint)
    ppo.network.eval()

    baseline = BaselineAgent()
    env = GameEnv()

    ppo_wins = 0
    baseline_wins = 0
    draws = 0
    total_steps = 0

    for ep in range(1, num_episodes + 1):
        env.reset()
        raw_obs = env.get_obs()

        obs0 = build_obs(raw_obs, agent_id=0)

        t = 0
        done = False
        info = {}

        while t < MAX_STEPS_PER_EPISODE and not done:
            # PPO action
            a0 = ppo.act_deterministic(obs0)
            # Baseline action
            a1 = baseline.act(raw_obs, agent_id=1)

            # Smart action repeat
            for frame in range(ACTION_REPEAT):
                eff_a0 = a0 if (a0 < 5 or frame == 0) else 0
                eff_a1 = a1 if (a1 < 5 or frame == 0) else 0

                raw_obs, (r0, r1), done, info = env.step(eff_a0, eff_a1)
                t += 1
                if t >= MAX_STEPS_PER_EPISODE:
                    done = True
                if done:
                    break

            obs0 = build_obs(raw_obs, agent_id=0)

        winner = info.get("winner")
        if winner == 0:
            ppo_wins += 1
        elif winner == 1:
            baseline_wins += 1
        else:
            draws += 1
        total_steps += t

        if verbose and ep % 10 == 0:
            print(
                f"  ep {ep:>4d}/{num_episodes} | "
                f"PPO {ppo_wins} / Baseline {baseline_wins} / Draw {draws} | "
                f"avg_len {total_steps / ep:.0f}"
            )

    stats = {
        "ppo_wins": ppo_wins,
        "baseline_wins": baseline_wins,
        "draws": draws,
        "ppo_win_rate": ppo_wins / num_episodes,
        "baseline_win_rate": baseline_wins / num_episodes,
        "draw_rate": draws / num_episodes,
        "avg_steps": total_steps / num_episodes,
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"Results ({num_episodes} episodes):")
        print(f"  PPO wins:      {ppo_wins:>4d} ({stats['ppo_win_rate']:.1%})")
        print(f"  Baseline wins: {baseline_wins:>4d} ({stats['baseline_win_rate']:.1%})")
        print(f"  Draws:         {draws:>4d} ({stats['draw_rate']:.1%})")
        print(f"  Avg steps:     {stats['avg_steps']:.0f}")
        print(f"{'='*50}")

    env.close()
    return stats


# ---------------------------------------------------------------------------
# Watch: PPO vs Baseline (rendered)
# ---------------------------------------------------------------------------

def watch_vs_baseline(checkpoint: str, num_episodes: int = 5):
    import pygame

    device = torch.device("cpu")
    ppo = PPOAgent(device=device)
    ppo.load(checkpoint)
    ppo.network.eval()

    baseline = BaselineAgent()
    env = GameEnv()

    for ep in range(1, num_episodes + 1):
        env.reset()
        raw_obs = env.get_obs()

        obs0 = build_obs(raw_obs, agent_id=0)

        t = 0
        done = False
        info = {}

        while t < MAX_STEPS_PER_EPISODE and not done:
            a0 = ppo.act_deterministic(obs0)
            a1 = baseline.act(raw_obs, agent_id=1)

            for frame in range(ACTION_REPEAT):
                eff_a0 = a0 if (a0 < 5 or frame == 0) else 0
                eff_a1 = a1 if (a1 < 5 or frame == 0) else 0

                raw_obs, (r0, r1), done, info = env.step(eff_a0, eff_a1)
                t += 1
                env.render(view=True, step=t)

                if t >= MAX_STEPS_PER_EPISODE:
                    done = True
                if done:
                    pygame.time.wait(1000)
                    break

            obs0 = build_obs(raw_obs, agent_id=0)

        winner = info.get("winner")
        if winner == 0:
            result = "PPO wins"
        elif winner == 1:
            result = "Baseline wins"
        else:
            result = "Draw"
        print(f"Episode {ep}: {result} (steps={t})")

    env.close()


# ---------------------------------------------------------------------------
# Watch: Baseline vs Baseline
# ---------------------------------------------------------------------------

def watch_baseline_vs_baseline(num_episodes: int = 5):
    import pygame

    b0 = BaselineAgent()
    b1 = BaselineAgent()
    env = GameEnv()

    for ep in range(1, num_episodes + 1):
        env.reset()
        raw_obs = env.get_obs()
        t = 0
        done = False
        info = {}

        while t < MAX_STEPS_PER_EPISODE and not done:
            a0 = b0.act(raw_obs, agent_id=0)
            a1 = b1.act(raw_obs, agent_id=1)

            for frame in range(ACTION_REPEAT):
                eff_a0 = a0 if (a0 < 5 or frame == 0) else 0
                eff_a1 = a1 if (a1 < 5 or frame == 0) else 0

                raw_obs, (r0, r1), done, info = env.step(eff_a0, eff_a1)
                t += 1
                env.render(view=True, step=t)

                if t >= MAX_STEPS_PER_EPISODE:
                    done = True
                if done:
                    pygame.time.wait(1000)
                    break

        winner = info.get("winner")
        result = f"Player {winner}" if winner is not None else "Draw"
        print(f"Episode {ep}: {result} wins (steps={t})")

    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Rule-based baseline agent")
    parser.add_argument("--eval", action="store_true", help="Evaluate PPO vs baseline")
    parser.add_argument("--watch", action="store_true", help="Watch PPO vs baseline")
    parser.add_argument("--watch-baseline", action="store_true", help="Watch baseline vs baseline")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    ckpt = args.checkpoint or os.path.join(SAVE_DIR, "ppo_final.pt")

    if args.watch_baseline:
        watch_baseline_vs_baseline(args.episodes)
    elif args.watch:
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found: {ckpt}")
            return
        watch_vs_baseline(ckpt, args.episodes)
    elif args.eval:
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found: {ckpt}")
            return
        evaluate(ckpt, args.episodes)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
