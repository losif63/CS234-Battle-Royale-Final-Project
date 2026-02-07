# Author: Jaduk Suh
# Created: November 13th

# Arena shape
ARENA_WIDTH = 800
ARENA_HEIGHT = 600

# Rendering
FPS = 60
WINDOW_TITLE: str = "Battle Royale"

# Colors (RGB tuples)
COLOR_BG = (20, 20, 30)
COLOR_BORDER = (100, 100, 120)
COLOR_AGENT = (100, 200, 255)
COLOR_AGENT_2 = (255, 180, 100)
COLOR_BULLET = (255, 100, 100)
COLOR_AMMO_PICKUP = (100, 255, 100)
COLOR_VISION = (100, 200, 255)

# Agent properties
AGENT_RADIUS = 15.0
AGENT_SPEED = 5.0

# Spawn: agents must be at least this far apart
MIN_AGENT_SPAWN_DISTANCE = 250
AGENT_SPAWN_MARGIN = 50  # min distance from arena walls

# Bullet properties (reused from arrow logic)
BULLET_SPEED = 12.0
BULLET_RADIUS = 6.0

# Ammo pickups
AMMO_PICKUP_RADIUS = 20.0
AMMO_PER_PICKUP = 10
NUM_AMMO_PICKUPS = 6

# Vision / Fog of War (for future use)
VISION_RADIUS = 150

# Reward (for training / evaluation)
REWARD_PER_STEP = 0.1
REWARD_COLLISION = -10.0
REWARD_HIT_ENEMY = 10.0
REWARD_MIN_DIST_ALPHA = 2.5
REWARD_CENTER_ALPHA = 0.1
WALL_PENALTY_ALPHA = 5 * REWARD_MIN_DIST_ALPHA

# Threshold
WALL_THRESHOLD = 20
COS_THRESHOLD = 0.7
CENTER_RADIUS = 120
REWARD_CENTER_OUT = 0.02
