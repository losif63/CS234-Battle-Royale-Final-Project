"""
Random agent for the Battle Royale environment.

Usage:
    uv run python -m agents.random_agent --eval                # evaluate PPO vs random (100 eps)
    uv run python -m agents.random_agent --eval --episodes 500 # more episodes
    uv run python -m agents.random_agent --watch               # watch PPO vs random
    uv run python -m agents.random_agent --watch-random        # watch random vs random
"""

import argparse
import os
import random

import numpy as np
import torch

from src.env import GameEnv
from agents.ppo_agent import (
    build_obs,
    PPOAgent,
    OBS_DIM,
    ACTION_REPEAT,
    MAX_STEPS_PER_EPISODE,
    SAVE_DIR,
    NUM_ACTIONS,
)


class RandomAgent:
    """Uniformly random agent."""

    def act(self, raw_obs: dict, agent_id: int) -> int:
        return random.randint(0, NUM_ACTIONS - 1)


# ---------------------------------------------------------------------------
# Evaluation: PPO vs Random
# ---------------------------------------------------------------------------

def evaluate(checkpoint: str, num_episodes: int = 100, verbose: bool = True):
    """Run PPO (agent_0) vs Random (agent_1) for num_episodes. Returns stats dict."""
    device = torch.device("cpu")
    ppo = PPOAgent(device=device)
    ppo.load(checkpoint)
    ppo.network.eval()

    rng_agent = RandomAgent()
    env = GameEnv()

    ppo_wins = 0
    random_wins = 0
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
            a0 = ppo.act_deterministic(obs0)
            a1 = rng_agent.act(raw_obs, agent_id=1)

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
            random_wins += 1
        else:
            draws += 1
        total_steps += t

        if verbose and ep % 10 == 0:
            print(
                f"  ep {ep:>4d}/{num_episodes} | "
                f"PPO {ppo_wins} / Random {random_wins} / Draw {draws} | "
                f"avg_len {total_steps / ep:.0f}"
            )

    stats = {
        "ppo_wins": ppo_wins,
        "random_wins": random_wins,
        "draws": draws,
        "ppo_win_rate": ppo_wins / num_episodes,
        "random_win_rate": random_wins / num_episodes,
        "draw_rate": draws / num_episodes,
        "avg_steps": total_steps / num_episodes,
    }

    if verbose:
        print(f"\n{'='*50}")
        print(f"Results ({num_episodes} episodes):")
        print(f"  PPO wins:    {ppo_wins:>4d} ({stats['ppo_win_rate']:.1%})")
        print(f"  Random wins: {random_wins:>4d} ({stats['random_win_rate']:.1%})")
        print(f"  Draws:       {draws:>4d} ({stats['draw_rate']:.1%})")
        print(f"  Avg steps:   {stats['avg_steps']:.0f}")
        print(f"{'='*50}")

    env.close()
    return stats


# ---------------------------------------------------------------------------
# Watch: PPO vs Random (rendered)
# ---------------------------------------------------------------------------

def watch_vs_random(checkpoint: str, num_episodes: int = 5):
    import pygame

    device = torch.device("cpu")
    ppo = PPOAgent(device=device)
    ppo.load(checkpoint)
    ppo.network.eval()

    rng_agent = RandomAgent()
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
            a1 = rng_agent.act(raw_obs, agent_id=1)

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
            result = "Random wins"
        else:
            result = "Draw"
        print(f"Episode {ep}: {result} (steps={t})")

    env.close()


# ---------------------------------------------------------------------------
# Watch: Random vs Random
# ---------------------------------------------------------------------------

def watch_random_vs_random(num_episodes: int = 5):
    import pygame

    r0 = RandomAgent()
    r1 = RandomAgent()
    env = GameEnv()

    for ep in range(1, num_episodes + 1):
        env.reset()
        raw_obs = env.get_obs()
        t = 0
        done = False
        info = {}

        while t < MAX_STEPS_PER_EPISODE and not done:
            a0 = r0.act(raw_obs, agent_id=0)
            a1 = r1.act(raw_obs, agent_id=1)

            for frame in range(ACTION_REPEAT):
                eff_a0 = a0 if (a0 < 5 or frame == 0) else 0
                eff_a1 = a1 if (a1 < 5 or frame == 0) else 0

                raw_obs, (r0_val, r1_val), done, info = env.step(eff_a0, eff_a1)
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
    parser = argparse.ArgumentParser(description="Random agent for Battle Royale")
    parser.add_argument("--eval", action="store_true", help="Evaluate PPO vs random")
    parser.add_argument("--watch", action="store_true", help="Watch PPO vs random")
    parser.add_argument("--watch-random", action="store_true", help="Watch random vs random")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    ckpt = args.checkpoint or os.path.join(SAVE_DIR, "ppo_final.pt")

    if args.watch_random:
        watch_random_vs_random(args.episodes)
    elif args.watch:
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found: {ckpt}")
            return
        watch_vs_random(ckpt, args.episodes)
    elif args.eval:
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found: {ckpt}")
            return
        evaluate(ckpt, args.episodes)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
