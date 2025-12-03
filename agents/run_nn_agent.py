# Author: Jaduk Suh
# Created: November 26th
from src.model import Q
import random
from src.env import GameEnv
import pygame
import argparse
import torch

# Select action based on epsilon-greedy
def select_action(q_net, obs):
    with torch.no_grad():
        q_values = q_net(obs)
        return torch.argmax(q_values).item()

def main(args):
    # Initialize pygame first
    pygame.init()
    
    env = GameEnv()
    env.reset()

    q_net = Q()
    q_net.load_state_dict(torch.load("q_network.pt"))
    q_net.eval()

    total_reward = 0.0
    done = False
    step = 0

    obs = env.get_obs()
    while not done:
        action = select_action(q_net, obs)

        # Go one step
        next_obs, reward, done = env.step(action)
        total_reward += reward
        step += 1
        
        # Render
        if args.render:
            env.render(view=True, step=step)
        obs = next_obs

    print("Finished.")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', '-r', action='store_true', default=False, help='Whether to render the NN agent or not')
    args = parser.parse_args()
    main(args)

