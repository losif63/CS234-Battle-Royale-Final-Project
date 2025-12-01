# Author: Jaduk Suh
# Created: November 13th
import src.config as cfg
import pygame
from src.objects import Agent, Arrow
from typing import List, Optional, Dict, Tuple
import random
from src.utils import distance, manhattan_distance, spawn_arrow
import math
import torch

SEED = 42

class GameEnv:
    ACTIONS = ["STAY", "UP", "DOWN", "LEFT", "RIGHT"]

    def __init__(self):
        self.rng = random.Random()
        self.rng.seed(SEED)
        self.initialized_render = False
        
        # Game state
        self.agent: Agent = Agent(
            x = cfg.ARENA_WIDTH / 2,
            y = cfg.ARENA_HEIGHT / 2,
            radius= cfg.AGENT_RADIUS,
            speed = cfg.AGENT_SPEED
        )
        self.arrows: List[Arrow] = []
        self.time_alive: int = 0
        self.done: bool = False
    
    # Reset game
    def reset(self):
        # Reset agent
        self.agent.x = cfg.ARENA_WIDTH / 2
        self.agent.y = cfg.ARENA_HEIGHT / 2
        self.arrows = []
        self.time_alive = 0
        self.done = False

    def center_distance_penalty(self, agent_pos):
        cx, cy = cfg.ARENA_WIDTH / 2, cfg.ARENA_HEIGHT / 2
        ax, ay = agent_pos

        # distance from center
        dist = distance(agent_pos, (cx, cy))

        # if inside safe zone → no penalty
        if dist <= cfg.CENTER_RADIUS:
            return 0.0
        
        # if outside → penalty proportional to how far outside
        # excess_dist = dist - cfg.CENTER_RADIUS

        # simple linear penalty:
        return cfg.REWARD_CENTER_OUT

    def center_distance_penalty_multiplier(self, agent_pos):
        cx, cy = cfg.ARENA_WIDTH / 2, cfg.ARENA_HEIGHT / 2
        max_dist = math.sqrt(cx ** 2 + cy ** 2)

        # distance from center
        dist = distance(agent_pos, (cx, cy))

        # simple linear penalty:
        return 1.0 - (dist / max_dist) * 0.7

    # Return observation, reward, game done status, and game info
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        if self.done:
            # If already done, return zero reward
            return ({"observation": self.get_obs(), "info": {}}, 0.0, self.done, {})
        
        # Execute agent action
        action_str = self.ACTIONS[action]
        dx, dy = 0.0, 0.0
        
        if action_str == "UP":
            dy -= cfg.AGENT_SPEED
        elif action_str == "DOWN":
            dy += cfg.AGENT_SPEED
        elif action_str == "LEFT":
            dx -= cfg.AGENT_SPEED
        elif action_str == "RIGHT":
            dx += cfg.AGENT_SPEED
        # "STAY" -> dx, dy remain 0
        
        self.agent.move(dx, dy, cfg.ARENA_WIDTH, cfg.ARENA_HEIGHT)
        
        if random.random() < cfg.ARROW_SPAWN_RATE and len(self.arrows) < cfg.ARROW_MAX_NUMBER:
            x, y, vx, vy = spawn_arrow(
                cfg.ARENA_WIDTH,
                cfg.ARENA_HEIGHT,
                cfg.ARROW_SPEED_MIN,
                cfg.ARROW_SPEED_MAX,
                self.agent.x,
                self.agent.y
            )
            self.arrows.append(Arrow(x, y, vx, vy, cfg.ARROW_RADIUS)
)

        # Update arrows
        for arrow in self.arrows:
            arrow.update()
        
        # Remove out-of-bounds arrows
        self.arrows = [
            arrow for arrow in self.arrows
            if not arrow.is_out_of_bounds(cfg.ARENA_WIDTH, cfg.ARENA_HEIGHT)
        ]

        # Calculate reward 
        reward = cfg.REWARD_PER_STEP
        collision = False
        agent_pos = self.agent.get_position()
        for arrow in self.arrows:
            dist = distance(agent_pos, arrow.get_position())
            man_dist = manhattan_distance(agent_pos, arrow.get_position())
            
            if dist < cfg.AGENT_RADIUS + cfg.ARROW_RADIUS:
                collision = True
                reward = cfg.REWARD_COLLISION
                self.done = True
                break
            
            # Penalize proximity to arrows
            if dist < cfg.VISION_RADIUS:
                reward -= cfg.REWARD_MIN_DIST_ALPHA * (1 / man_dist)
        
        # reward -= self.center_distance_penalty(agent_pos)
        if reward >= 0:
            reward *= self.center_distance_penalty_multiplier(agent_pos)
        else:
            reward /= self.center_distance_penalty_multiplier(agent_pos)
        
        self.time_alive += 1
        
        obs = self.get_obs()
        
        return (obs, reward, self.done)

    # Get observations
    # observation -> Pytorch vector
    # Relative position & velocities of 20 closest arrows
    def get_obs(self) -> Dict:
        obs = torch.zeros((82), dtype=torch.float32)
        agent_pos = self.agent.get_position()
        agent_x, agent_y = agent_pos
        obs[0] = agent_x / cfg.ARENA_WIDTH
        obs[1] = agent_y / cfg.ARENA_HEIGHT

        visible_arrows = []        
        # Filter arrows by vision radius
        for arrow in self.arrows:
            dist = distance(agent_pos, arrow.get_position())
            if dist <= cfg.VISION_RADIUS:
                visible_arrows.append((dist, arrow))        
        visible_arrows.sort()
        for i in range(min(20, len(visible_arrows))):
            dist, arrow = visible_arrows[i]
            arrow_x, arrow_y = arrow.get_position()
            arrow_vx, arrow_vy = arrow.get_velocity()
            obs[4*i+2] = (arrow_x - agent_x) / cfg.VISION_RADIUS
            obs[4*i+3] = (arrow_y - agent_y) / cfg.VISION_RADIUS
            obs[4*i+4] = arrow_vx / cfg.ARROW_SPEED_MAX
            obs[4*i+5] = arrow_vy / cfg.ARROW_SPEED_MAX

        return obs
    
    def render(self, view: bool = True):
        if not view:
            return
        
        if not self.initialized_render:
            pygame.init()
            self.screen = pygame.display.set_mode((cfg.ARENA_WIDTH, cfg.ARENA_HEIGHT))
            pygame.display.set_caption(cfg.WINDOW_TITLE)
            self.clock = pygame.time.Clock()
            self.initialized_render = True
        
        # Handle pygame events (keep window responsive)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True
        
        # Fill background color
        self.screen.fill(cfg.COLOR_BG)
        
        # Draw arena border
        pygame.draw.rect(
            self.screen,
            cfg.COLOR_BORDER,
            (0, 0, cfg.ARENA_WIDTH, cfg.ARENA_HEIGHT),
            width=2
        )
        
        # Draw vision radius (faint circle)
        pygame.draw.circle(
            self.screen,
            cfg.COLOR_VISION,
            (int(self.agent.x), int(self.agent.y)),
            int(cfg.VISION_RADIUS),
            width=1
        )
        
        # Draw arrows
        for arrow in self.arrows:
            x, y = int(arrow.x), int(arrow.y)
            # Draw as a small triangle pointing in direction of motion
            angle = math.atan2(arrow.vy, arrow.vx)
            size = arrow.radius * 1.5
            # Triangle pointing in velocity direction
            # Tip of arrow at front, base at back
            tip_x = x + size * math.cos(angle)
            tip_y = y + size * math.sin(angle)
            # Base points perpendicular to velocity
            perp_angle = angle + math.pi / 2
            base_x1 = x - size * 0.5 * math.cos(angle) + size * 0.3 * math.cos(perp_angle)
            base_y1 = y - size * 0.5 * math.sin(angle) + size * 0.3 * math.sin(perp_angle)
            base_x2 = x - size * 0.5 * math.cos(angle) - size * 0.3 * math.cos(perp_angle)
            base_y2 = y - size * 0.5 * math.sin(angle) - size * 0.3 * math.sin(perp_angle)
            points = [
                (int(tip_x), int(tip_y)),
                (int(base_x1), int(base_y1)),
                (int(base_x2), int(base_y2))
            ]
            pygame.draw.polygon(self.screen, cfg.COLOR_ARROW, points)
        
        # Draw agent
        pygame.draw.circle(
            self.screen,
            cfg.COLOR_AGENT,
            (int(self.agent.x), int(self.agent.y)),
            int(self.agent.radius)
        )
        
        pygame.display.flip()
        self.clock.tick(cfg.FPS)
    
    def close(self):
        if self.initialized_render:
            pygame.quit()
            self.initialized_render = False
            self.screen = None
            self.clock = None