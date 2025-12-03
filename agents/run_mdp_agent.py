# Author: Jaduk Suh
# Created: December 2nd
import random
from src.env import GameEnv
import src.config as cfg
import pygame
import argparse
import numpy as np
import json
import math
from typing import List, Dict, Tuple
from src.objects import Arrow
from tqdm import tqdm

def main(args):
    # Initialize pygame first
    pygame.init()
    
    env = GameEnv()
    env.reset()
    
    vision_range = cfg.VISION_RADIUS
    
    state_space: Dict[Tuple[int, int, int, int], int] = {}
    state_space[(0, 0, 0, 0)] = 0
    idx = 1
    for x in range(-vision_range, vision_range + 1):
        for y in range(-vision_range, vision_range + 1):
            if (x ** 2 + y ** 2) > vision_range ** 2:
                continue
            for speed in range(cfg.ARROW_SPEED_MIN, cfg.ARROW_SPEED_MAX + 1):
                for angle in range(0, 360, 10):
                    state_space[(speed, angle, x, y)] = idx
                    idx += 1
    qstar = np.load('mdp_qstar.npy')
    num_steps = 0
    obs: List[Arrow] = env.get_obs2() 
    print(f"Running MDP agent...")
    while True:
        agent_pos = env.agent.get_position()
        if len(obs) == 0:
            cur_state = (0, 0, 0, 0)
            cur_state_idx = state_space[cur_state]
            q_values = qstar[cur_state_idx, :]
            action = np.argmax(q_values)
        else:
            q_values = []
            for arrow in obs:
                arr_x, arr_y = arrow.get_position()
                arr_x = round(arr_x - agent_pos[0])
                arr_y = round(arr_y - agent_pos[1])
                
                # Check if arrow is within vision range
                dist = math.sqrt(arr_x ** 2 + arr_y ** 2)
                if dist > vision_range:
                    # Arrow out of vision, use terminal state
                    cur_state = (0, 0, 0, 0)
                elif dist < cfg.AGENT_RADIUS + cfg.ARROW_RADIUS:
                    # Collision state - use terminal state
                    cur_state = (0, 0, 0, 0)
                else:
                    # Round angle to match MDP state space (0, 10, 20, ..., 350)
                    rounded_angle = round(arrow.angle / 10) * 10
                    if rounded_angle < 0:
                        rounded_angle += 360
                    rounded_angle = rounded_angle % 360
                    
                    # Clamp position to vision range
                    arr_x = max(-vision_range, min(vision_range, arr_x))
                    arr_y = max(-vision_range, min(vision_range, arr_y))
                    
                    cur_state = (arrow.speed, rounded_angle, arr_x, arr_y)
                
                # Look up state (with error handling)
                if cur_state in state_space:
                    cur_state_idx = state_space[cur_state]
                    q_values.append(qstar[cur_state_idx, :])
                else:
                    # State not found, use terminal state
                    cur_state_idx = state_space[(0, 0, 0, 0)]
                    q_values.append(qstar[cur_state_idx, :])
            
            if len(q_values) > 0:
                q_values = np.stack(q_values, axis=0)  # Shape: (num_arrows, num_actions)
                # q_min = np.min(q_values, axis=0)  # Shape: (num_actions,)
                # action = np.argmax(q_min)  # Scalar action index
                q_sum = np.sum(q_values, axis=0)  # Shape: (num_actions,)
                action = np.argmax(q_sum)  # Scalar action index
            else:
                # No valid arrows, use terminal state
                cur_state_idx = state_space[(0, 0, 0, 0)]
                q_values = qstar[cur_state_idx, :]
                action = np.argmax(q_values)            

        obs, _, done = env.step2(action)
        num_steps += 1
        
        # Render
        if args.render:
            env.render(view=True, step=num_steps)        
        # Check if done
        if done:
            print(f"\nCollision detected at step {num_steps}!")
            break
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', '-r', action='store_true', default=False, help='Whether to render the random agent or not')
    args = parser.parse_args()
    main(args)

