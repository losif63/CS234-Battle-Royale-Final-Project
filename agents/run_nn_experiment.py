# Author: Jaduk Suh
# Created: November 26th
from src.model import Q
import random
from src.env import GameEnv
import pygame
import argparse
import torch
import src.config as cfg
import numpy as np
import json
from tqdm import tqdm

# Select action based on epsilon-greedy
def select_action(q_net, obs):
    with torch.no_grad():
        q_values = q_net(obs)
        return torch.argmax(q_values).item()

def main():
    # Initialize pygame first
    pygame.init()
    
    vision_range = cfg.VISION_RADIUS
    env = GameEnv()
    env.reset()

    q_net = Q()
    q_net.load_state_dict(torch.load(f"q_network_{vision_range}.pt"))
    q_net.eval()

    total_reward = 0.0
    done = False
    num_experiments = 1000
    results = []
    
    for experiment in tqdm(range(num_experiments)):
        env.reset()
        done = False
        num_steps = 0
        obs = env.get_obs()
        while not done:
            action = select_action(q_net, obs)

            # Go one step
            _, reward, done = env.step(action)
            total_reward += reward
            num_steps += 1
        
        results.append(num_steps)

    env.close()

    data = {
        'vision_range': int(vision_range),
        'num_experiments': int(num_experiments),
        'results': results,
        'mean': float(np.mean(results)),
        'std': float(np.std(results)),
        'min': int(np.min(results)),
        'max': int(np.max(results)),
        'median': float(np.median(results)),
    }
    
    # save results
    with open(f'nn_results_{vision_range}.json', 'w') as f:
        json.dump(data, f, indent=4)
    
if __name__ == "__main__":
    main()

