# Author: Jaduk Suh
# Created: November 13th
import random
from src.env import GameEnv
import src.config as cfg
import pygame
import argparse


def main(args):
    pygame.init()

    env = GameEnv()
    env.reset()

    total_reward_0 = 0.0
    total_reward_1 = 0.0
    num_steps = 0
    max_steps = 500

    print(f"Running two random agents for up to {max_steps} steps...")
    print("Actions: 0=STAY, 1-4=move, 5-8=shoot up/down/left/right (only if ammo > 0)")

    while num_steps < max_steps:
        # Each agent picks a random action (0-8). With no ammo, 5-8 effectively do nothing.
        action_0 = random.randint(0, 8)
        action_1 = random.randint(0, 8)

        observation, (reward_0, reward_1), done, info = env.step(action_0, action_1)
        total_reward_0 += reward_0
        total_reward_1 += reward_1
        num_steps += 1

        if args.render:
            env.render(view=True, step=num_steps)

        if num_steps % 50 == 0:
            print(
                f"Step {num_steps}: R0={reward_0:.2f}, R1={reward_1:.2f}, "
                f"Alive={info.get('alive', [])}, "
                f"Ammo pickups={info.get('num_ammo_pickups', 0)}"
            )

        if done:
            print(f"\nGame over at step {num_steps}!")
            print(f"Winner: Player {info.get('winner', -1) + 1}" if info.get('winner') is not None else "Draw")
            break

    print(f"\nFinal results:")
    print(f"  Total steps: {num_steps}")
    print(f"  Total reward P1: {total_reward_0:.2f}")
    print(f"  Total reward P2: {total_reward_1:.2f}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render", "-r", action="store_true", default=False,
        help="Whether to render the game",
    )
    args = parser.parse_args()
    main(args)
