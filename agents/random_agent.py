# Author: Jaduk Suh
# Created: November 13th
import random
from src.env import GameEnv
import src.config as cfg
import pygame
import argparse


def main(args):
    # Initialize pygame first
    pygame.init()
    
    env = GameEnv()
    env.reset()
    
    total_reward = 0.0
    num_steps = 0
    max_steps = 200
    
    print(f"Running random agent for {max_steps} steps...")
    
    while num_steps < max_steps:
        # Random action
        action = random.randint(0, 4)
        
        # Step environment (headless - no rendering)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        num_steps += 1
        
        # Render
        if args.render:
            env.render(view=True, step=num_steps)
        
        # Print progress
        if num_steps % 50 == 0:
            print(f"Step {num_steps}: Reward={reward:.2f}, "
                  f"Total={total_reward:.2f}, "
                  f"Time alive={info.get('time_alive', 0)}, "
                  f"Arrows={info.get('num_arrows', 0)}")
        
        # Check if done
        if done:
            print(f"\nCollision detected at step {num_steps}!")
            print(f"Time alive: {info.get('time_alive', 0)}")
            break
    
    print(f"\nFinal results:")
    print(f"  Total steps: {num_steps}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final time alive: {info.get('time_alive', 0)}")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', '-r', action='store_true', default=False, help='Whether to render the random agent or not')
    args = parser.parse_args()
    main(args)

