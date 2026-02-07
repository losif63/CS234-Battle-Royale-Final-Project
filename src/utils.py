# Author: Jaduk Suh
# Created: November 13th
import math
import random
from typing import Tuple, List

from src.objects import AmmoPickup


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def manhattan_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.fabs(p1[0] - p2[0]) + math.fabs(p1[1] - p2[1])


def spawn_agents_far_apart(
    arena_width: int,
    arena_height: int,
    min_distance: float,
    agent_radius: float,
    margin: float,
    rng: random.Random,
) -> List[Tuple[float, float]]:
    """Return [(x0, y0), (x1, y1)] so that agents are at least min_distance apart."""
    low_x = margin + agent_radius
    high_x = arena_width - margin - agent_radius
    low_y = margin + agent_radius
    high_y = arena_height - margin - agent_radius
    max_attempts = 200
    for _ in range(max_attempts):
        x0 = rng.uniform(low_x, high_x)
        y0 = rng.uniform(low_y, high_y)
        x1 = rng.uniform(low_x, high_x)
        y1 = rng.uniform(low_y, high_y)
        if distance((x0, y0), (x1, y1)) >= min_distance:
            return [(x0, y0), (x1, y1)]
    # Fallback: place on opposite sides
    x0 = low_x + (high_x - low_x) * 0.25
    y0 = (low_y + high_y) / 2
    x1 = low_x + (high_x - low_x) * 0.75
    y1 = (low_y + high_y) / 2
    return [(x0, y0), (x1, y1)]


def spawn_ammo_pickups(
    arena_width: int,
    arena_height: int,
    num_pickups: int,
    pickup_radius: float,
    agent_positions: List[Tuple[float, float]],
    agent_radius: float,
    rng: random.Random,
) -> List[AmmoPickup]:
    """Spawn ammo pickups at random positions, not overlapping agents."""
    margin = pickup_radius + agent_radius + 10
    low_x = margin
    high_x = arena_width - margin
    low_y = margin
    high_y = arena_height - margin
    pickups: List[AmmoPickup] = []
    min_dist_from_agent = agent_radius + pickup_radius + 15
    max_attempts = 50 * num_pickups
    attempts = 0
    while len(pickups) < num_pickups and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(low_x, high_x)
        y = rng.uniform(low_y, high_y)
        pos = (x, y)
        if any(distance(pos, apos) < min_dist_from_agent for apos in agent_positions):
            continue
        if any(distance(pos, p.get_position()) < pickup_radius * 2 for p in pickups):
            continue
        pickups.append(AmmoPickup(x=x, y=y, radius=pickup_radius))
    return pickups


def detect_object_collision(p1, r1, p2, r2) -> bool:
    dist = distance(p1, p2)
    return dist < (r1 + r2)