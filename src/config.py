# Author: Jaduk Suh
# Created: November 13th

# Arena shape
ARENA_WIDTH = 800
ARENA_HEIGHT = 600

# Rendering
FPS = 60
WINDOW_TITLE: str = "Ninja Dodge Game"

# Colors (RGB tuples)
COLOR_BG = (20, 20, 30)
COLOR_BORDER = (100, 100, 120)
COLOR_AGENT = (100, 200, 255)
COLOR_ARROW = (255, 100, 100)
COLOR_VISION = (100, 200, 255)

# Agent properties
AGENT_RADIUS = 15.0
AGENT_SPEED = 5.0

# Arrow properties
ARROW_SPAWN_RATE = 0.1  # Probability per step
ARROW_SPEED_MIN = 2
ARROW_SPEED_MAX = 8
ARROW_RADIUS = 8.0
ARROW_MAX_NUMBER = 50 # Keep 50 arrows at a time

# Vision / Fog of War
VISION_RADIUS = 150

# Reward
REWARD_PER_STEP = 0.1
REWARD_COLLISION = -10.0
REWARD_MIN_DIST_ALPHA = 2.5  # Optional: negative reward for proximity
REWARD_CENTER_ALPHA = 0.1
WALL_PENALTY_ALPHA = 5 * REWARD_MIN_DIST_ALPHA


# Threshold
WALL_THRESHOLD = 20
COS_THRESHOLD = 0.7 # 0.9~1.0 heading toward agent | 0.5 ~ 0.7 partially heading toward agent | 
CENTER_RADIUS = 120
REWARD_CENTER_OUT = 0.02
