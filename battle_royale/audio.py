"""Distance-based sound effects for the battle royale game."""

import math
import os

# Force SDL to route audio through PulseAudio (PipeWire compat layer)
# so it reaches the correct output device.
os.environ.setdefault("SDL_AUDIODRIVER", "pulseaudio")

import pygame

from .config import _DIAG

# Max audible distance (beyond this, volume = 0)
MAX_SOUND_DIST = 0.45 * _DIAG
NUM_CHANNELS = 16


class AudioManager:
    def __init__(self):
        # Don't re-init mixer — pygame.init() already did it.
        # Just ensure we have enough channels for overlapping sounds.
        pygame.mixer.set_num_channels(NUM_CHANNELS)

        snd_dir = os.path.join(os.path.dirname(__file__), "sounds")
        self.shot_sound = pygame.mixer.Sound(os.path.join(snd_dir, "shot.wav"))
        self.hit_sound = pygame.mixer.Sound(os.path.join(snd_dir, "hit.wav"))

    def _volume_for_distance(self, sx, sy, lx, ly):
        """Linear falloff: 1.0 at distance 0, 0.0 at MAX_SOUND_DIST."""
        d = math.sqrt((sx - lx) ** 2 + (sy - ly) ** 2)
        return max(0.0, 1.0 - d / MAX_SOUND_DIST)

    def play_shot(self, source_x, source_y, listener_x, listener_y):
        vol = self._volume_for_distance(source_x, source_y, listener_x, listener_y)
        if vol > 0.01:
            ch = pygame.mixer.find_channel()
            if ch:
                ch.set_volume(vol)
                ch.play(self.shot_sound)

    def play_hit(self, target_x, target_y, listener_x, listener_y):
        vol = self._volume_for_distance(target_x, target_y, listener_x, listener_y)
        if vol > 0.01:
            ch = pygame.mixer.find_channel()
            if ch:
                ch.set_volume(vol)
                ch.play(self.hit_sound)
