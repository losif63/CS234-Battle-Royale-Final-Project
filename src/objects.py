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
    
    def move(self, dx: float, dy: float, arena_width: int, arena_height: int):
        self.x += dx
        self.y += dy
        
        # Ensure agent don't go off arena
        self.x = max(self.radius, min(arena_width - self.radius, self.x))
        self.y = max(self.radius, min(arena_height - self.radius, self.y))
    
    def get_position(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Arrow:
    x: float
    y: float
    speed: int
    angle: float
    radius: float
    
    def update(self):
        self.x += self.speed * math.cos(self.angle * math.pi / 180.0)
        self.y -= self.speed * math.sin(self.angle * math.pi / 180.0)
    
    # Check if arrow is out of arena
    # If arrow goes out of arena, it can later be removed from the game
    def is_out_of_bounds(self, arena_width: int, arena_height: int) -> bool:
        margin = self.radius * 2
        return (self.x < -margin or self.x > arena_width + margin or
                self.y < -margin or self.y > arena_height + margin)
    
    def get_position(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def get_velocity(self) -> Tuple[float, float]:
        return (self.speed * math.cos(self.angle * math.pi / 180.0), -self.speed * math.sin(self.angle * math.pi / 180.0))

