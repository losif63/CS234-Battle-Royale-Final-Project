# Author: Jaduk Suh
# Created: November 13th
import src.config as cfg
import pygame
import numpy as np
from typing import Optional, Dict, Tuple
import random
from src.utils import spawn_agents_far_apart, spawn_ammo_pickups
import math

SEED = 42

# Precomputed velocity vectors for each shoot action (unit vectors * BULLET_SPEED)
# SHOOT_UP=5, SHOOT_DOWN=6, SHOOT_LEFT=7, SHOOT_RIGHT=8
SHOOT_VELOCITIES = {
    5: np.array([0.0, -cfg.BULLET_SPEED], dtype=np.float32),   # UP
    6: np.array([0.0, cfg.BULLET_SPEED], dtype=np.float32),    # DOWN
    7: np.array([-cfg.BULLET_SPEED, 0.0], dtype=np.float32),   # LEFT
    8: np.array([cfg.BULLET_SPEED, 0.0], dtype=np.float32),    # RIGHT
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

        # Agent state
        self.agent_positions = np.array([
            [cfg.ARENA_WIDTH / 2 - 80, cfg.ARENA_HEIGHT / 2],
            [cfg.ARENA_WIDTH / 2 + 80, cfg.ARENA_HEIGHT / 2],
        ], dtype=np.float32)
        self.agent_ammo = np.zeros(2, dtype=np.int32)
        self.agent_alive = np.array([True, True])

        # Bullet buffer (fixed-size, tracked by active mask)
        self.bullet_positions = np.zeros((cfg.MAX_BULLETS, 2), dtype=np.float32)
        self.bullet_velocities = np.zeros((cfg.MAX_BULLETS, 2), dtype=np.float32)
        self.bullet_owners = np.zeros(cfg.MAX_BULLETS, dtype=np.int32)
        self.bullet_active = np.zeros(cfg.MAX_BULLETS, dtype=bool)

        # Ammo pickups
        self.pickup_positions = np.zeros((cfg.NUM_AMMO_PICKUPS, 2), dtype=np.float32)
        self.pickup_active = np.zeros(cfg.NUM_AMMO_PICKUPS, dtype=bool)

        # Scalar state
        self.time_step: int = 0
        self.done: bool = False
        self.winner: Optional[int] = None

    def reset(self):
        # Spawn agents
        positions = spawn_agents_far_apart(
            cfg.ARENA_WIDTH,
            cfg.ARENA_HEIGHT,
            cfg.MIN_AGENT_SPAWN_DISTANCE,
            cfg.AGENT_RADIUS,
            cfg.AGENT_SPAWN_MARGIN,
            self.rng,
        )
        for i, (x, y) in enumerate(positions):
            self.agent_positions[i, 0] = x
            self.agent_positions[i, 1] = y
        self.agent_ammo[:] = 0
        self.agent_alive[:] = True

        # Clear bullets
        self.bullet_active[:] = False

        # Spawn ammo pickups
        pickups = spawn_ammo_pickups(
            cfg.ARENA_WIDTH,
            cfg.ARENA_HEIGHT,
            cfg.NUM_AMMO_PICKUPS,
            cfg.AMMO_PICKUP_RADIUS,
            [tuple(self.agent_positions[0]), tuple(self.agent_positions[1])],
            cfg.AGENT_RADIUS,
            self.rng,
        )
        for i, p in enumerate(pickups):
            self.pickup_positions[i, 0] = p.x
            self.pickup_positions[i, 1] = p.y
        self.pickup_active[:len(pickups)] = True
        self.pickup_active[len(pickups):] = False

        self.time_step = 0
        self.done = False
        self.winner = None

    def _apply_action(self, agent_id: int, action: int):
        if action < 0 or action >= self.NUM_ACTIONS:
            action = 0

        # Movement
        if action <= 4:
            dx, dy = 0.0, 0.0
            if action == 1:    # UP
                dy = -cfg.AGENT_SPEED
            elif action == 2:  # DOWN
                dy = cfg.AGENT_SPEED
            elif action == 3:  # LEFT
                dx = -cfg.AGENT_SPEED
            elif action == 4:  # RIGHT
                dx = cfg.AGENT_SPEED

            self.agent_positions[agent_id, 0] += dx
            self.agent_positions[agent_id, 1] += dy

            # Clamp to arena
            self.agent_positions[agent_id, 0] = np.clip(
                self.agent_positions[agent_id, 0],
                cfg.AGENT_RADIUS, cfg.ARENA_WIDTH - cfg.AGENT_RADIUS,
            )
            self.agent_positions[agent_id, 1] = np.clip(
                self.agent_positions[agent_id, 1],
                cfg.AGENT_RADIUS, cfg.ARENA_HEIGHT - cfg.AGENT_RADIUS,
            )
            return

        # Shoot (actions 5-8), only if has ammo
        if action in SHOOT_VELOCITIES and self.agent_ammo[agent_id] > 0:
            # Find first inactive bullet slot
            inactive = ~self.bullet_active
            if not inactive.any():
                return  # buffer full, skip shot

            slot = np.argmax(inactive)
            self.bullet_positions[slot] = self.agent_positions[agent_id]
            self.bullet_velocities[slot] = SHOOT_VELOCITIES[action]
            self.bullet_owners[slot] = agent_id
            self.bullet_active[slot] = True
            self.agent_ammo[agent_id] -= 1

    def step(
        self, action_0: int, action_1: int
    ) -> Tuple[Dict, Tuple[float, float], bool, Dict]:
        if self.done:
            obs = self.get_obs()
            return (obs, (0.0, 0.0), self.done, self.get_info())

        # Apply both agents' actions
        if self.agent_alive[0]:
            self._apply_action(0, action_0)
        if self.agent_alive[1]:
            self._apply_action(1, action_1)

        # Ammo pickups: check distance from each agent to each active pickup
        for agent_id in range(2):
            if not self.agent_alive[agent_id]:
                continue
            active_mask = self.pickup_active.copy()
            if not active_mask.any():
                continue

            # Distance from this agent to all pickups
            diff = self.pickup_positions - self.agent_positions[agent_id]  # (NUM_PICKUPS, 2)
            dists = np.sqrt(np.sum(diff ** 2, axis=1))  # (NUM_PICKUPS,)
            collected = active_mask & (dists < cfg.AGENT_RADIUS + cfg.AMMO_PICKUP_RADIUS)
            num_collected = collected.sum()
            if num_collected > 0:
                self.agent_ammo[agent_id] += int(num_collected) * cfg.AMMO_PER_PICKUP
                self.pickup_active[collected] = False

        # Update bullet positions
        active = self.bullet_active
        if active.any():
            self.bullet_positions[active] += self.bullet_velocities[active]

        # Remove out-of-bounds bullets
        if active.any():
            margin = cfg.BULLET_RADIUS * 2
            oob = (
                (self.bullet_positions[:, 0] < -margin) |
                (self.bullet_positions[:, 0] > cfg.ARENA_WIDTH + margin) |
                (self.bullet_positions[:, 1] < -margin) |
                (self.bullet_positions[:, 1] > cfg.ARENA_HEIGHT + margin)
            )
            self.bullet_active[oob & active] = False

        # Bullet-agent collision
        reward_0 = cfg.REWARD_PER_STEP
        reward_1 = cfg.REWARD_PER_STEP
        active = self.bullet_active
        if active.any():
            hit_radius = cfg.AGENT_RADIUS + cfg.BULLET_RADIUS
            for agent_id in range(2):
                if not self.agent_alive[agent_id]:
                    continue
                # Distance from all active bullets to this agent
                diff = self.bullet_positions[active] - self.agent_positions[agent_id]
                dists = np.sqrt(np.sum(diff ** 2, axis=1))
                # Bullets that hit: close enough AND not owned by this agent
                owners = self.bullet_owners[active]
                hits = (dists < hit_radius) & (owners != agent_id)
                if hits.any():
                    self.agent_alive[agent_id] = False
                    self.done = True
                    shooter = int(owners[np.argmax(hits)])
                    self.winner = shooter
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
        """Observation dict with numpy arrays."""
        active_bullets = self.bullet_active
        bullet_pos = self.bullet_positions[active_bullets]       # (n_active, 2)
        bullet_vel = self.bullet_velocities[active_bullets]      # (n_active, 2)
        bullet_own = self.bullet_owners[active_bullets]          # (n_active,)

        active_pickups = self.pickup_active
        pickup_pos = self.pickup_positions[active_pickups]       # (n_active, 2)

        return {
            "agent_0": {
                "position": self.agent_positions[0].copy(),
                "ammo": int(self.agent_ammo[0]),
                "alive": bool(self.agent_alive[0]),
            },
            "agent_1": {
                "position": self.agent_positions[1].copy(),
                "ammo": int(self.agent_ammo[1]),
                "alive": bool(self.agent_alive[1]),
            },
            "bullets": [
                (bullet_pos[i].copy(), bullet_vel[i].copy(), int(bullet_own[i]))
                for i in range(len(bullet_pos))
            ],
            "ammo_pickups": [pickup_pos[i].copy() for i in range(len(pickup_pos))],
        }

    def get_info(self) -> Dict:
        return {
            "time_step": self.time_step,
            "winner": self.winner,
            "alive": self.agent_alive.tolist(),
            "num_bullets": int(self.bullet_active.sum()),
            "num_ammo_pickups": int(self.pickup_active.sum()),
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
        for i in range(cfg.NUM_AMMO_PICKUPS):
            if not self.pickup_active[i]:
                continue
            px, py = int(self.pickup_positions[i, 0]), int(self.pickup_positions[i, 1])
            pygame.draw.circle(
                self.screen,
                cfg.COLOR_AMMO_PICKUP,
                (px, py),
                int(cfg.AMMO_PICKUP_RADIUS),
            )

        # Bullets
        for i in range(cfg.MAX_BULLETS):
            if not self.bullet_active[i]:
                continue
            x, y = int(self.bullet_positions[i, 0]), int(self.bullet_positions[i, 1])
            vx, vy = self.bullet_velocities[i]
            angle = math.atan2(vy, vx)
            size = cfg.BULLET_RADIUS * 1.5
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
        for i in range(2):
            if not self.agent_alive[i]:
                continue
            ax, ay = int(self.agent_positions[i, 0]), int(self.agent_positions[i, 1])
            pygame.draw.circle(
                self.screen,
                colors[i],
                (ax, ay),
                int(cfg.AGENT_RADIUS),
            )
            ammo_text = self.font.render(f"P{i+1}: {self.agent_ammo[i]}", True, (255, 255, 255))
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
