# Author: Jaduk Suh
# Created: November 13th
import math
import random
from typing import Tuple


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Spawn arrow on edge of arena
# TODO: Make arrow point toward agent with certain probability
def spawn_arrow(
    arena_width: int,
    arena_height: int,
    speed_min: float,
    speed_max: float,
    agent_x: float,
    agent_y: float,
    toward_agent_prob=0.2
) -> Tuple[float, float, float, float]:
    # Choose a random edge (0=top, 1=right, 2=bottom, 3=left)
    edge = random.randint(0, 3)

    agent_prob = random.random() 

    if edge == 0:  # Top
        x = random.uniform(0, arena_width)
        y = 0
        if agent_prob < toward_agent_prob:
            vx = agent_x - x
            vy = agent_y - y
        else:
            vx = random.uniform(-2.0, 2.0)
            vy = random.uniform(1.0, 3.0) # Arrow should point downward
    elif edge == 1:  # Right
        x = arena_width
        y = random.uniform(0, arena_height)
        if agent_prob < toward_agent_prob:
            vx = agent_x - x
            vy = agent_y - y
        else:
            vx = random.uniform(-3.0, -1.0) # Arrow shuld point left
            vy = random.uniform(-2.0, 2.0)
    elif edge == 2:  # Bottom
        x = random.uniform(0, arena_width)
        y = arena_height
        if agent_prob < toward_agent_prob:
            vx = agent_x - x
            vy = agent_y - y
        else:
            vx = random.uniform(-2.0, 2.0)
            vy = random.uniform(-3.0, -1.0) # ARrow should point up
    else:  # Left
        x = 0
        y = random.uniform(0, arena_height)
        if agent_prob < toward_agent_prob:
            vx = agent_x - x
            vy = agent_y - y
        else:
            vx = random.uniform(1.0, 3.0) # Arrow should point right
            vy = random.uniform(-2.0, 2.0) 
    
    # Normalize and scale to random speed
    v_norm = math.sqrt(vx ** 2 + vy ** 2)
    speed = random.uniform(speed_min, speed_max)
    gain = speed / v_norm
    
    return (x, y, vx * gain, vy * gain)

def detect_object_collision(
    p1, r1, p2, r2,
) -> bool:
    dist = distance(p1, p2)
    return dist < (r1 + r2)