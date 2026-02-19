# Author: Jaduk Suh
# Created: November 13th
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math


@dataclass
class Agent:
    x: float
    y: float
    radius: float
    speed: float
    ammo: int = 0

    def move(
        self,
        dx: float,
        dy: float,
        arena_width: int,
        arena_height: int,
        walls: Optional[List["Wall"]] = None,
    ):
        old_x, old_y = self.x, self.y

        self.x += dx
        self.y += dy

        # Ensure agent doesn't go off arena
        self.x = max(self.radius, min(arena_width - self.radius, self.x))
        self.y = max(self.radius, min(arena_height - self.radius, self.y))

        # If colliding with any wall, revert to original position
        if walls is not None:
            for wall in walls:
                if wall.collides_with_circle(self.x, self.y, self.radius):
                    self.x, self.y = old_x, old_y
                    return

    def get_position(self) -> Tuple[float, float]:
        return (self.x, self.y)


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


@dataclass
class Wall:
    x: float
    y: float
    width_units: int
    height_units: int
    unit: int = 20

    def get_xbounds(self) -> Tuple[float, float]:
        return (self.x, self.x + self.width_units * self.unit)

    def get_ybounds(self) -> Tuple[float, float]:
        return (self.y, self.y + self.height_units * self.unit)

    def collides_with_circle(self, cx: float, cy: float, radius: float) -> bool:
        """Return True if the circle (center cx, cy with given radius) overlaps this wall rect."""
        left, right = self.get_xbounds()
        top, bottom = self.get_ybounds()
        closest_x = max(left, min(right, cx))
        closest_y = max(top, min(bottom, cy))
        dist_sq = (cx - closest_x) ** 2 + (cy - closest_y) ** 2
        return dist_sq < radius**2