# Author: Jaduk Suh
# Created: November 13th
import math
import random
from typing import Tuple


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def manhattan_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.fabs(p1[0] - p2[0]) + math.fabs(p1[1] - p2[1])

# Spawn arrow on edge of arena
# TODO: Make arrow point toward agent with certain probability
def spawn_arrow(
    arena_width: int,
    arena_height: int,
    speed_min: int,
    speed_max: int,
    agent_x: float,
    agent_y: float,
    toward_agent_prob=0.2
) -> Tuple[float, float, int, float]:
    # Choose a random edge (0=top, 1=right, 2=bottom, 3=left)
    edge = random.randint(0, 3)
    agent_prob = random.random() 

    speed = random.randint(speed_min, speed_max)
    if edge == 0:  # Top
        x = random.uniform(0, arena_width)
        y = 0
        if agent_prob < toward_agent_prob:
            angle = math.atan2(y - agent_y, agent_x - x) * 180.0 / math.pi  # Convert radians to degrees
        else:
            angle = random.random() * 120.0 + 210.0 # Arrow should point downward
    elif edge == 1:  # Right
        x = arena_width
        y = random.uniform(0, arena_height)
        if agent_prob < toward_agent_prob:
            angle = math.atan2(y - agent_y, agent_x - x) * 180.0 / math.pi  # Convert radians to degrees
        else:
            angle = random.random() * 120.0 + 120.0 # Arrow shuld point left
    elif edge == 2:  # Bottom
        x = random.uniform(0, arena_width)
        y = arena_height
        if agent_prob < toward_agent_prob:
            angle = math.atan2(y - agent_y, agent_x - x) * 180.0 / math.pi  # Convert radians to degrees
        else:
            angle = random.random() * 120.0 + 30.0 # Arrow should point up
    else:  # Left
        x = 0
        y = random.uniform(0, arena_height)
        if agent_prob < toward_agent_prob:
            angle = math.atan2(y - agent_y, agent_x - x) * 180.0 / math.pi  # Convert radians to degrees
        else:
            angle = random.random() * 120.0 + 300.0 # Arrow should point right
    # Normalize angle to [0, 360)
    if angle < 0:
        angle += 360.0
    if angle >= 360.0:
        angle -= 360.0
            
    return (x, y, speed, angle)

def detect_object_collision(
    p1, r1, p2, r2,
) -> bool:
    dist = distance(p1, p2)
    return dist < (r1 + r2)