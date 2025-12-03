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
        self.gamma = 0.995
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
        
        # Initialize terminal state values
        for speed, angle, x, y in self.state_space:
            state_idx = self.state_space[(speed, angle, x, y)]
            if math.sqrt(x ** 2 + y ** 2) < cfg.AGENT_RADIUS + cfg.ARROW_RADIUS:
                # Collision terminal state - value is reward / (1 - gamma) for absorbing state
                self.value[state_idx] = -50.0 / (1.0 - self.gamma)
            elif speed == 0 and angle == 0:
                # Safe terminal state (no arrow) - value is reward / (1 - gamma)
                self.value[state_idx] = 0.1 / (1.0 - self.gamma)

        for speed, angle, x, y in tqdm(self.state_space):
            state_idx = self.state_space[(speed, angle, x, y)]
            for action in self.action_space:
                action_idx = self.action_space[action]
                # Handle terminal states first
                if speed == 0 and angle == 0:
                    self.transition[state_idx, action_idx] = state_idx
                    continue

                # Collision state -> Transfer to terminal state 
                elif math.sqrt(x ** 2 + y ** 2) < cfg.AGENT_RADIUS + cfg.ARROW_RADIUS:
                    terminal_state = (0, 0, 0, 0)
                    terminal_state_idx = self.state_space[terminal_state]
                    self.transition[state_idx, action_idx] = terminal_state_idx
                    self.rewards[state_idx, action_idx] = -50.0
                    continue
                # Normal states
                # Calculate new state: arrow moves first, then agent moves
                # Arrow position relative to agent: arrow moves in its direction
                new_x = round(x + speed * math.cos(angle * math.pi / 180))
                new_y = round(y - speed * math.sin(angle * math.pi / 180))
                # Agent movement: if agent moves, relative position changes oppositely
                if action == 'UP':
                    new_y += cfg.AGENT_SPEED  # Agent moves up, arrow's relative y increases
                elif action == 'DOWN':
                    new_y -= cfg.AGENT_SPEED  # Agent moves down, arrow's relative y decreases
                elif action == 'LEFT':
                    new_x += cfg.AGENT_SPEED  # Agent moves left, arrow's relative x increases
                elif action == 'RIGHT':
                    new_x -= cfg.AGENT_SPEED  # Agent moves right, arrow's relative x decreases

                # Check if next state is out of vision range
                if math.sqrt(new_x ** 2 + new_y ** 2) > self.vision_range:
                    next_state = (0, 0, 0, 0)
                    next_state_idx = self.state_space[next_state]
                    self.transition[state_idx, action_idx] = next_state_idx
                    self.rewards[state_idx, action_idx] = 0.1
                # Check if next state has collision
                elif math.sqrt(new_x ** 2 + new_y ** 2) < cfg.AGENT_RADIUS + cfg.ARROW_RADIUS: 
                    next_state = (speed, angle, new_x, new_y)
                    next_state_idx = self.state_space[next_state]
                    self.transition[state_idx, action_idx] = next_state_idx
                    self.rewards[state_idx, action_idx] = -50.0
                # Near collision -> Give small penalty
                elif math.sqrt(new_x ** 2 + new_y ** 2) < (cfg.AGENT_RADIUS + cfg.ARROW_RADIUS) * 1.5:
                    next_state = (speed, angle, new_x, new_y)
                    next_state_idx = self.state_space[next_state]
                    self.transition[state_idx, action_idx] = next_state_idx
                    self.rewards[state_idx, action_idx] = -20.0 
                else:
                    # Normal transition - need to find matching state
                    # Check if within vision circle
                    if math.sqrt(new_x ** 2 + new_y ** 2) <= self.vision_range:
                        next_state = (speed, angle, new_x, new_y)
                        if next_state in self.state_space:
                            next_state_idx = self.state_space[next_state]
                            self.transition[state_idx, action_idx] = next_state_idx
                            self.rewards[state_idx, action_idx] = 0.1
                        else:
                            # State not in space, transition to terminal
                            next_state = (0, 0, 0, 0)
                            next_state_idx = self.state_space[next_state]
                            self.transition[state_idx, action_idx] = next_state_idx
                            self.rewards[state_idx, action_idx] = 0.1
                    else:
                        # Out of vision range
                        next_state = (0, 0, 0, 0)
                        next_state_idx = self.state_space[next_state]
                        self.transition[state_idx, action_idx] = next_state_idx
                        self.rewards[state_idx, action_idx] = 0.1
                    
    def value_iteration(self) -> None:
        error = float('inf')
        threshold = 1e-3
        epoch = 0
        
        # Identify terminal states (collision states and safe terminal states)
        terminal_states = set()
        for speed, angle, x, y in self.state_space:
            state_idx = self.state_space[(speed, angle, x, y)]
            if math.sqrt(x ** 2 + y ** 2) < cfg.AGENT_RADIUS + cfg.ARROW_RADIUS:
                terminal_states.add(state_idx)
            elif speed == 0 and angle == 0:
                terminal_states.add(state_idx)
        
        while error > threshold:
            epoch += 1
            new_value = self.rewards + self.gamma * self.value[self.transition]
            new_value_star = np.max(new_value, axis=-1)
            
            # Don't update terminal states - they're absorbing
            for term_idx in terminal_states:
                new_value_star[term_idx] = self.value[term_idx]

            error = np.max(np.abs(self.value - new_value_star))
            print(f"Epoch {epoch} - Error {error}")
            self.value = new_value_star
    
    def calculate_qstar(self) -> None:
        self.qstar = self.rewards + self.gamma * self.value[self.transition]


    def save_mdp(self) -> None:
        # Convert tuple keys to strings for JSON serialization
        np.save('mdp_qstar.npy', self.qstar)
                
if __name__ == '__main__':
    mdp = MDP(cfg.VISION_RADIUS)
    mdp.value_iteration()
    mdp.calculate_qstar()
    mdp.save_mdp()
                



