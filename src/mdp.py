# Author: Jaduk Suh
# Created: December 2nd
from typing import Tuple, List, Dict
import src.config as cfg
import math
import numpy as np
from tqdm import tqdm
import json


class MDP():
    def __init__(self, vision_range: int) -> None:
        self.vision_range = vision_range
        self.gamma = 0.9
        self.state_space: Dict[Tuple[int, int, int, int], int] = {}
        self.state_space[(0, 0, 0, 0)] = 0
        idx = 1
        for x in range(-vision_range, vision_range + 1):
            for y in range(-vision_range, vision_range + 1):
                if (x ** 2 + y ** 2) > vision_range ** 2:
                    continue
                for speed in range(cfg.ARROW_SPEED_MIN, cfg.ARROW_SPEED_MAX + 1):
                    for angle in range(0, 360, 10):
                        self.state_space[(speed, angle, x, y)] = idx
                        idx += 1
        
        self.action_space: Dict[str, int] = {   
            'STAY': 0,
            'UP': 1,
            'DOWN': 2,
            'LEFT': 3,
            'RIGHT': 4
        }
        
        self.transition: np.ndarray = np.zeros((len(self.state_space), len(self.action_space)), dtype=np.uint32)
        self.value: np.ndarray = np.zeros((len(self.state_space)))
        self.rewards: np.ndarray = np.zeros((len(self.state_space), len(self.action_space)))

        for speed, angle, x, y in tqdm(self.state_space):
            state_idx = self.state_space[(speed, angle, x, y)]
            for action in self.action_space:
                action_idx = self.action_space[action]
                # Handle terminal states first
                # Collision with arrow
                if math.sqrt(x ** 2 + y ** 2) < cfg.AGENT_RADIUS + cfg.ARROW_RADIUS:
                    self.transition[state_idx, action_idx] = state_idx
                    self.value[state_idx] = 0.0
                    continue
                # Arrow out of vision range
                elif speed == 0 and angle == 0:
                    self.transition[state_idx, action_idx] = state_idx
                    self.value[state_idx] = 0.0
                    continue

                # Normal states
                # Calculate new state
                new_x = (int)(x + speed * math.cos(angle * math.pi / 180))
                new_y = (int)(y + speed * math.sin(angle * math.pi / 180))
                if action == 'UP':
                    new_y -= cfg.AGENT_SPEED
                elif action == 'DOWN':
                    new_y += cfg.AGENT_SPEED
                elif action == 'LEFT':
                    new_x -= cfg.AGENT_SPEED
                elif action == 'RIGHT':
                    new_x += cfg.AGENT_SPEED

                # Next state is out of vision range
                if math.sqrt(new_x ** 2 + new_y ** 2) > self.vision_range:
                    new_x, new_y = 0, 0
                    next_state = (0, 0, new_x, new_y)
                    next_state_idx = self.state_space[next_state]
                    self.transition[state_idx, action_idx] = next_state_idx
                    self.rewards[state_idx, action_idx] = 0.1
                # Next state has collision
                elif math.sqrt(new_x ** 2 + new_y ** 2) < cfg.AGENT_RADIUS + cfg.ARROW_RADIUS: 
                    next_state = (speed, angle, new_x, new_y)
                    next_state_idx = self.state_space[next_state]
                    self.transition[state_idx, action_idx] = next_state_idx
                    self.rewards[state_idx, action_idx] = -10.0
                else:
                    next_state = (speed, angle, new_x, new_y)
                    next_state_idx = self.state_space[next_state]
                    self.transition[state_idx, action_idx] = next_state_idx
                    self.rewards[state_idx, action_idx] = 0.1
                    
    def value_iteration(self) -> None:
        error = float('inf')
        threshold = 1e-4
        epoch = 0
        while error > threshold:
            epoch += 1
            new_value = self.rewards + self.gamma * self.value[self.transition]
            new_value_star = np.max(new_value, axis=-1)

            error = np.max(np.abs(self.value - new_value_star))
            print(f"Epoch {epoch} - Error {error}")
            self.value = new_value_star

    def save_mdp(self) -> None:
        # Convert tuple keys to strings for JSON serialization
        state_space_json = {str(k): v for k, v in self.state_space.items()}
        with open('mdp_states.json', 'w') as f:
            json.dump(state_space_json, f)
        np.save('mdp_values.npy', self.value)
                
if __name__ == '__main__':
    mdp = MDP(150)
    mdp.value_iteration()
    mdp.save_mdp()
                



