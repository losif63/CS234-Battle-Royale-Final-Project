# Author: Jaduk Suh
# Created: November 13th
from dataclasses import dataclass
from typing import Tuple
import math


@dataclass
class Agent:
    x: float
    y: float
    radius: float
    speed: float
    ammo: int = 0
    vx: float = 0.0
    vy: float = 0.0

    def set_velocity(self, vx: float, vy: float):
        """Set the agent's velocity (called by actions)."""
        self.vx = vx
        self.vy = vy

    def update(self, arena_width: int, arena_height: int):
        """Move the agent by its current velocity, clamping to arena bounds."""
        self.x += self.vx
        self.y += self.vy
        self.x = max(self.radius, min(arena_width - self.radius, self.x))
        self.y = max(self.radius, min(arena_height - self.radius, self.y))

    def move(self, dx: float, dy: float, arena_width: int, arena_height: int):
        self.x += dx
        self.y += dy
        self.x = max(self.radius, min(arena_width - self.radius, self.x))
        self.y = max(self.radius, min(arena_height - self.radius, self.y))

    def get_position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def get_velocity(self) -> Tuple[float, float]:
        return (self.vx, self.vy)


@dataclass
class Bullet:
    """Bullet shot by an agent. Reuses same motion logic as arrows (angle in degrees)."""
    x: float
    y: float
    speed: float
    angle: float  # degrees: 0 = right, 90 = up, 180 = left, 270 = down
    radius: float
    owner_id: int = 0  # which agent shot this (0 or 1)

    def update(self):
        self.x += self.speed * math.cos(self.angle * math.pi / 180.0)
        self.y -= self.speed * math.sin(self.angle * math.pi / 180.0)

    def is_out_of_bounds(self, arena_width: int, arena_height: int) -> bool:
        margin = self.radius * 2
        return (
            self.x < -margin
            or self.x > arena_width + margin
            or self.y < -margin
            or self.y > arena_height + margin
        )

    def get_position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def get_velocity(self) -> Tuple[float, float]:
        return (
            self.speed * math.cos(self.angle * math.pi / 180.0),
            -self.speed * math.sin(self.angle * math.pi / 180.0),
        )


# Legacy alias for compatibility
Arrow = Bullet


@dataclass
class AmmoPickup:
    x: float
    y: float
    radius: float

    def get_position(self) -> Tuple[float, float]:
        return (self.x, self.y)

