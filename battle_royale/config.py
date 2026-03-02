import colorsys
import numpy as np

# Arena (scaled proportionally: ~2.5x area of 1600x1200 for 10 agents)
ARENA_W = 4000
ARENA_H = 3000
_DIAG = (ARENA_W**2 + ARENA_H**2) ** 0.5  # ~5000

# Agent
NUM_AGENTS = 10
AGENT_RADIUS = 15
AGENT_SPEED = 5.0
AGENT_FRICTION = 0.6
AGENT_MAX_HP = 100

# Episode
MAX_EPISODE_FRAMES = 4800  # 80s at 60fps (longer for big map)
WIN_PAUSE_FRAMES = 120     # 2s pause on win
WIN_FLASH_FRAMES = 24      # flash winner color for first 24 frames (~0.4s)

# Controller deadzones
MOVE_DEADZONE = 0.15
AIM_DEADZONE = 0.25

# Rendering
AIM_LINE_LENGTH = 40
FPS = 60

# Colors
COLOR_BG = (30, 30, 30)
COLOR_WALL = (120, 120, 120)
COLOR_AGENT = (60, 130, 220)
COLOR_AIM = (255, 220, 60)
COLOR_HP_BAR = (80, 200, 80)
COLOR_HP_BG = (60, 60, 60)
COLOR_HUD_TEXT = (220, 220, 220)
COLOR_AMMO_DEPOSIT = (50, 220, 130)

# Ammo
NUM_AMMO_DEPOSITS = 12
AMMO_PER_PICKUP = 10
AMMO_START = 5
AMMO_MAX = 50
AMMO_PICKUP_RADIUS = 25.0
AMMO_RESPAWN_FRAMES = 900    # 15s at 60fps
FIRE_MOVE_PENALTY = 0.4      # velocity multiplier while firing (cooldown active)

# Health pickups
NUM_HEALTH_PICKUPS = 15
MEDKIT_HEAL_AMOUNT = 0.5   # fraction of AGENT_MAX_HP
MEDKIT_MAX = 3              # max carried
HEAL_CHANNEL_FRAMES = 60    # 1s at 60fps
HEALTH_PICKUP_RADIUS = 25.0
COLOR_HEALTH_PICKUP = (220, 50, 80)

# Zone (shrinking safe area) — proportional to arena diagonal
ZONE_MAX_RADIUS = 0.50 * _DIAG
ZONE_MIN_RADIUS = 0.06 * _DIAG
ZONE_SHRINK_START = 600       # frame when zone starts shrinking
ZONE_SHRINK_END = 3600        # frame when zone reaches minimum
ZONE_DAMAGE_PER_FRAME = 0.5   # HP/frame outside zone (~200 frames to kill)

# Field of view (fixed gameplay constant)
ENTITY_FOV_RADIUS = 550.0

# Walls: (x, y, w, h) axis-aligned rects
WALL_THICKNESS = 20
BORDER_WALLS = np.array([
    [0, 0, ARENA_W, WALL_THICKNESS],                          # top
    [0, ARENA_H - WALL_THICKNESS, ARENA_W, WALL_THICKNESS],   # bottom
    [0, 0, WALL_THICKNESS, ARENA_H],                          # left
    [ARENA_W - WALL_THICKNESS, 0, WALL_THICKNESS, ARENA_H],   # right
], dtype=np.float64)
NUM_INTERIOR_WALLS = 50
NUM_WALLS = len(BORDER_WALLS) + NUM_INTERIOR_WALLS

# Bullets
BULLET_SPEED = 20.0
BULLET_RADIUS = 6.0
BULLET_DAMAGE = 25.0
MAX_BULLETS = 80
FIRE_COOLDOWN = 6
COLOR_BULLET = (255, 100, 100)

# Per-agent color palette (golden-angle spaced hues for max separation)
def _make_agent_colors(n):
    colors = []
    for i in range(n):
        hue = (i * 137.508) % 360 / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors

AGENT_COLORS = _make_agent_colors(NUM_AGENTS)
