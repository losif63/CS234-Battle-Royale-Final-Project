# Author: Jaduk Suh
# Created: November 13th
import src.config as cfg
import pygame
from src.objects import Agent, Arrow
from typing import List, Optional, Dict, Tuple
import random
from src.utils import distance, spawn_arrow
import math


class GameEnv:
    ACTIONS = ["STAY", "UP", "DOWN", "LEFT", "RIGHT"]

    def __init__(self):
        self.rng = random.Random()
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
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)
        
        # Reset agent
        self.agent.x = cfg.ARENA_WIDTH / 2
        self.agent.y = cfg.ARENA_HEIGHT / 2
        self.arrows = []
        self.time_alive = 0
        self.done = False

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
        
        # TODO: Spawn new arrows
        if random.random() < cfg.ARROW_SPAWN_RATE:
            x, y, vx, vy = spawn_arrow(
                cfg.ARENA_WIDTH,
                cfg.ARENA_HEIGHT,
                cfg.ARROW_SPEED_MIN,
                cfg.ARROW_SPEED_MAX,
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
        
        # TODO: Check collisions and alter reward
        reward = cfg.REWARD_PER_STEP
        collision = False
        for arrow in self.arrows:
            if distance(self.agent.get_position(), arrow.get_position()) < cfg.AGENT_RADIUS + cfg.ARROW_RADIUS:
                collision = True
                reward = cfg.REWARD_COLLISION
                self.done = True
                break

        
        # TODO: Alter reward based on other policy (like nearness to arrow)
        
        self.time_alive += 1
        
        obs = self.get_obs()
        info = {
            "time_alive": self.time_alive,
            "num_arrows": len(self.arrows),
            "collision": collision
        }
        
        return (obs, reward, self.done, info)

    # Get observations -> That is, info about all arrows within agent's vision range 
    def get_obs(self) -> Dict:
        agent_pos = self.agent.get_position()
        visible_arrows = []
        
        # Filter arrows by vision radius
        for arrow in self.arrows:
            dist = distance(agent_pos, arrow.get_position())
            if dist <= cfg.VISION_RADIUS:
                visible_arrows.append((
                    arrow.x,
                    arrow.y,
                    arrow.vx,
                    arrow.vy
                ))
        
        return {
            "agent_pos": agent_pos,
            "arrows": visible_arrows,
        }
    
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