# Author: Jaduk Suh
# Created: November 13th
import src.config as cfg
import pygame
from src.objects import Agent, Bullet, AmmoPickup
from typing import List, Optional, Dict, Tuple
import random
from src.utils import distance, spawn_agents_far_apart, spawn_ammo_pickups
import math

SEED = 42

# Shooting direction to angle in degrees (same convention as Arrow: 0=right, 90=up, 180=left, 270=down)
SHOOT_ANGLES = {
    "SHOOT_UP": 90.0,
    "SHOOT_DOWN": 270.0,
    "SHOOT_LEFT": 180.0,
    "SHOOT_RIGHT": 0.0,
}


class GameEnv:
    ACTIONS = [
        "STAY", "UP", "DOWN", "LEFT", "RIGHT",
        "SHOOT_UP", "SHOOT_DOWN", "SHOOT_LEFT", "SHOOT_RIGHT",
    ]
    NUM_ACTIONS = 9

    def __init__(self):
        self.rng = random.Random()
        self.rng.seed(SEED)
        self.initialized_render = False

        # Game state: two agents
        self.agents: List[Agent] = [
            Agent(
                x=cfg.ARENA_WIDTH / 2 - 80,
                y=cfg.ARENA_HEIGHT / 2,
                radius=cfg.AGENT_RADIUS,
                speed=cfg.AGENT_SPEED,
                ammo=0,
            ),
            Agent(
                x=cfg.ARENA_WIDTH / 2 + 80,
                y=cfg.ARENA_HEIGHT / 2,
                radius=cfg.AGENT_RADIUS,
                speed=cfg.AGENT_SPEED,
                ammo=0,
            ),
        ]
        self.bullets: List[Bullet] = []
        self.ammo_pickups: List[AmmoPickup] = []
        self.time_step: int = 0
        self.done: bool = False
        self.winner: Optional[int] = None  # 0 or 1 if someone was hit, None if game ongoing
        self.alive: List[bool] = [True, True]

    def reset(self):
        positions = spawn_agents_far_apart(
            cfg.ARENA_WIDTH,
            cfg.ARENA_HEIGHT,
            cfg.MIN_AGENT_SPAWN_DISTANCE,
            cfg.AGENT_RADIUS,
            cfg.AGENT_SPAWN_MARGIN,
            self.rng,
        )
        for i, (x, y) in enumerate(positions):
            self.agents[i].x = x
            self.agents[i].y = y
            self.agents[i].ammo = 0
        self.bullets = []
        self.ammo_pickups = spawn_ammo_pickups(
            cfg.ARENA_WIDTH,
            cfg.ARENA_HEIGHT,
            cfg.NUM_AMMO_PICKUPS,
            cfg.AMMO_PICKUP_RADIUS,
            [self.agents[0].get_position(), self.agents[1].get_position()],
            cfg.AGENT_RADIUS,
            self.rng,
        )
        self.time_step = 0
        self.done = False
        self.winner = None
        self.alive = [True, True]

    def _apply_action(self, agent_id: int, action: int):
        agent = self.agents[agent_id]
        if action < 0 or action >= self.NUM_ACTIONS:
            action = 0
        action_str = self.ACTIONS[action]

        # Move
        dx, dy = 0.0, 0.0
        if action_str == "UP":
            dy -= cfg.AGENT_SPEED
        elif action_str == "DOWN":
            dy += cfg.AGENT_SPEED
        elif action_str == "LEFT":
            dx -= cfg.AGENT_SPEED
        elif action_str == "RIGHT":
            dx += cfg.AGENT_SPEED

        if action_str in ("STAY", "UP", "DOWN", "LEFT", "RIGHT"):
            agent.move(dx, dy, cfg.ARENA_WIDTH, cfg.ARENA_HEIGHT)
            return

        # Shoot (only if has ammo)
        if action_str in SHOOT_ANGLES and agent.ammo > 0:
            angle = SHOOT_ANGLES[action_str]
            # Spawn bullet at agent center
            self.bullets.append(
                Bullet(
                    x=agent.x,
                    y=agent.y,
                    speed=cfg.BULLET_SPEED,
                    angle=angle,
                    radius=cfg.BULLET_RADIUS,
                    owner_id=agent_id,
                )
            )
            agent.ammo -= 1

    def step(
        self, action_0: int, action_1: int
    ) -> Tuple[Dict, Tuple[float, float], bool, Dict]:
        if self.done:
            obs = self.get_obs()
            return (obs, (0.0, 0.0), self.done, self.get_info())

        # Apply both agents' actions (move or shoot)
        if self.alive[0]:
            self._apply_action(0, action_0)
        if self.alive[1]:
            self._apply_action(1, action_1)

        # Ammo pickups: if agent overlaps pickup, gain ammo and remove pickup
        for agent_id in (0, 1):
            if not self.alive[agent_id]:
                continue
            pos = self.agents[agent_id].get_position()
            to_remove = []
            for p in self.ammo_pickups:
                if distance(pos, p.get_position()) < cfg.AGENT_RADIUS + cfg.AMMO_PICKUP_RADIUS:
                    self.agents[agent_id].ammo += cfg.AMMO_PER_PICKUP
                    to_remove.append(p)
            for p in to_remove:
                self.ammo_pickups.remove(p)

        # Update bullets
        for bullet in self.bullets:
            bullet.update()

        # Remove out-of-bounds bullets
        self.bullets = [
            b
            for b in self.bullets
            if not b.is_out_of_bounds(cfg.ARENA_WIDTH, cfg.ARENA_HEIGHT)
        ]

        # Bullet–agent collision: bullet hits the *other* agent
        reward_0 = cfg.REWARD_PER_STEP
        reward_1 = cfg.REWARD_PER_STEP
        for bullet in self.bullets:
            if self.done:
                break
            bpos = bullet.get_position()
            for agent_id in (0, 1):
                if not self.alive[agent_id]:
                    continue
                if bullet.owner_id == agent_id:
                    continue  # can't hit yourself
                apos = self.agents[agent_id].get_position()
                if distance(bpos, apos) < cfg.AGENT_RADIUS + cfg.BULLET_RADIUS:
                    self.alive[agent_id] = False
                    self.done = True
                    self.winner = bullet.owner_id
                    if agent_id == 0:
                        reward_0 = cfg.REWARD_COLLISION
                        reward_1 = cfg.REWARD_HIT_ENEMY
                    else:
                        reward_1 = cfg.REWARD_COLLISION
                        reward_0 = cfg.REWARD_HIT_ENEMY
                    break

        self.time_step += 1
        obs = self.get_obs()
        info = self.get_info()
        return (obs, (reward_0, reward_1), self.done, info)

    def get_obs(self) -> Dict:
        """Observation dict with keys 'agent_0', 'agent_1' (each has position, ammo, etc.)."""
        return {
            "agent_0": {
                "position": self.agents[0].get_position(),
                "ammo": self.agents[0].ammo,
                "alive": self.alive[0],
            },
            "agent_1": {
                "position": self.agents[1].get_position(),
                "ammo": self.agents[1].ammo,
                "alive": self.alive[1],
            },
            "bullets": [
                (b.get_position(), b.get_velocity(), b.owner_id) for b in self.bullets
            ],
            "ammo_pickups": [p.get_position() for p in self.ammo_pickups],
        }

    def get_info(self) -> Dict:
        return {
            "time_step": self.time_step,
            "winner": self.winner,
            "alive": list(self.alive),
            "num_bullets": len(self.bullets),
            "num_ammo_pickups": len(self.ammo_pickups),
        }

    def render(self, view: bool = True, step: int = 0):
        if not view:
            return

        if not self.initialized_render:
            pygame.init()
            self.screen = pygame.display.set_mode((cfg.ARENA_WIDTH, cfg.ARENA_HEIGHT))
            pygame.display.set_caption(cfg.WINDOW_TITLE)
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.initialized_render = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.done = True

        self.screen.fill(cfg.COLOR_BG)

        pygame.draw.rect(
            self.screen,
            cfg.COLOR_BORDER,
            (0, 0, cfg.ARENA_WIDTH, cfg.ARENA_HEIGHT),
            width=2,
        )

        # Ammo pickups
        for p in self.ammo_pickups:
            pygame.draw.circle(
                self.screen,
                cfg.COLOR_AMMO_PICKUP,
                (int(p.x), int(p.y)),
                int(p.radius),
            )

        # Bullets
        for bullet in self.bullets:
            x, y = int(bullet.x), int(bullet.y)
            vx, vy = bullet.get_velocity()
            angle = math.atan2(vy, vx)
            size = bullet.radius * 1.5
            tip_x = x + size * math.cos(angle)
            tip_y = y + size * math.sin(angle)
            perp = angle + math.pi / 2
            base_x1 = x - size * 0.5 * math.cos(angle) + size * 0.3 * math.cos(perp)
            base_y1 = y - size * 0.5 * math.sin(angle) + size * 0.3 * math.sin(perp)
            base_x2 = x - size * 0.5 * math.cos(angle) - size * 0.3 * math.cos(perp)
            base_y2 = y - size * 0.5 * math.sin(angle) - size * 0.3 * math.sin(perp)
            points = [
                (int(tip_x), int(tip_y)),
                (int(base_x1), int(base_y1)),
                (int(base_x2), int(base_y2)),
            ]
            pygame.draw.polygon(self.screen, cfg.COLOR_BULLET, points)

        # Agents
        colors = [cfg.COLOR_AGENT, cfg.COLOR_AGENT_2]
        for i, agent in enumerate(self.agents):
            if not self.alive[i]:
                continue
            pygame.draw.circle(
                self.screen,
                colors[i],
                (int(agent.x), int(agent.y)),
                int(agent.radius),
            )
            ammo_text = self.font.render(f"P{i+1}: {agent.ammo}", True, (255, 255, 255))
            self.screen.blit(ammo_text, (10, 10 + i * 22))

        frame_text = self.font.render(f"Frame: {step}", True, (255, 255, 255))
        self.screen.blit(frame_text, (10, 60))

        if self.done and self.winner is not None:
            win_text = self.font.render(
                f"Player {self.winner + 1} wins!", True, (255, 255, 0)
            )
            self.screen.blit(
                win_text,
                (cfg.ARENA_WIDTH // 2 - 80, cfg.ARENA_HEIGHT // 2 - 18),
            )

        pygame.display.flip()
        self.clock.tick(cfg.FPS)

    def close(self):
        if self.initialized_render:
            pygame.quit()
            self.initialized_render = False
            self.screen = None
            self.clock = None
