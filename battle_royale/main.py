"""Entry point: input handler + game loop."""

import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("SDL_AUDIODRIVER", "pulseaudio")

import math
import torch
import pygame
from .config import MOVE_DEADZONE, AIM_DEADZONE, NUM_AGENTS, FIRE_COOLDOWN
from .sim import BatchedBRSim
from .renderer import Renderer
from .audio import AudioManager


class InputHandler:
    def __init__(self):
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

    def read(self) -> tuple[float, float, float, bool, bool]:
        """Returns (move_x, move_y, aim_angle, fire, heal). aim_angle=NaN if no aim input."""
        if self.joystick is not None:
            return self._read_controller()
        return self._read_kb_mouse()

    def _read_controller(self) -> tuple[float, float, float, bool, bool]:
        js = self.joystick
        # Left stick: movement
        mx = js.get_axis(0)
        my = js.get_axis(1)
        if math.sqrt(mx**2 + my**2) < MOVE_DEADZONE:
            mx, my = 0.0, 0.0

        # Right stick: aim (axes 3,4 — axis 2 is left trigger on most controllers)
        ax = js.get_axis(3) if js.get_numaxes() > 3 else 0.0
        ay = js.get_axis(4) if js.get_numaxes() > 4 else 0.0
        aim_mag = math.sqrt(ax**2 + ay**2)
        if aim_mag < AIM_DEADZONE:
            aim_angle = float("nan")
        else:
            aim_angle = math.atan2(ay, ax)

        # Right trigger: fire (axis 5, rests at -1.0, fully pressed = 1.0)
        rt = js.get_axis(5) if js.get_numaxes() > 5 else -1.0
        fire = rt > 0.5

        # Left bumper (button 4): heal
        heal = js.get_button(4) if js.get_numbuttons() > 4 else False

        return mx, my, aim_angle, fire, heal

    def _read_kb_mouse(self) -> tuple[float, float, float, bool, bool]:
        keys = pygame.key.get_pressed()
        mx, my = 0.0, 0.0
        if keys[pygame.K_w]:
            my -= 1.0
        if keys[pygame.K_s]:
            my += 1.0
        if keys[pygame.K_a]:
            mx -= 1.0
        if keys[pygame.K_d]:
            mx += 1.0

        # Mouse aim computed in main() from actual agent position
        aim_angle = float("nan")

        # Left mouse button or space: fire
        fire = pygame.mouse.get_pressed()[0] or keys[pygame.K_SPACE]

        # Q: heal
        heal = keys[pygame.K_q]

        return mx, my, aim_angle, fire, heal


def _bot_actions(sim, player_agent=0):
    """Generate actions for bot agents (all agents except player_agent).
    Returns (move_x, move_y, aim_angle, fire) tensors of shape (B, A).
    """
    B, A = sim.B, sim.A
    dev = sim.device

    # Random movement for bots
    move_x = torch.randn(B, A, device=dev) * 0.8
    move_y = torch.randn(B, A, device=dev) * 0.8

    # Aim toward nearest alive enemy
    aim_angle = torch.full((B, A), float("nan"), device=dev)

    for a in range(A):
        if a == player_agent:
            continue
        bot_alive = sim.agent_alive[:, a]  # (B,)
        if not bot_alive.any():
            continue

        # Compute distances to all other agents
        dx = sim.agent_x - sim.agent_x[:, a:a+1]  # (B, A)
        dy = sim.agent_y - sim.agent_y[:, a:a+1]  # (B, A)
        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)  # (B, A)

        # Mask out self and dead agents
        mask = sim.agent_alive.clone()
        mask[:, a] = False
        dist = torch.where(mask, dist, torch.tensor(float("inf"), device=dev))

        # Find nearest enemy
        nearest = dist.argmin(dim=1)  # (B,)
        target_dx = sim.agent_x[torch.arange(B, device=dev), nearest] - sim.agent_x[:, a]
        target_dy = sim.agent_y[torch.arange(B, device=dev), nearest] - sim.agent_y[:, a]
        angle = torch.atan2(target_dy, target_dx)

        aim_angle[:, a] = torch.where(bot_alive, angle, aim_angle[:, a])

    # Fire with 30% probability
    fire = (torch.rand(B, A, device=dev) < 0.3) & sim.agent_alive

    # Zero out player slot (will be filled by input handler)
    move_x[:, player_agent] = 0.0
    move_y[:, player_agent] = 0.0
    fire[:, player_agent] = False

    return move_x, move_y, aim_angle, fire


def main():
    pygame.init()
    sim = BatchedBRSim(num_envs=1, max_agents=NUM_AGENTS)
    renderer = Renderer()
    inp = InputHandler()
    audio = AudioManager()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    sim.reset()

        # Get bot actions
        move_x, move_y, aim_angle, fire = _bot_actions(sim, player_agent=0)
        heal = torch.zeros(sim.B, sim.A, dtype=torch.bool, device=sim.device)

        # Get player input
        mx, my, p_aim, firing, healing = inp.read()

        # Compute mouse aim from actual agent position
        if inp.joystick is None:
            mouse_x, mouse_y = renderer.window_to_game(*pygame.mouse.get_pos())
            ax = sim.agent_x[0, 0].item()
            ay = sim.agent_y[0, 0].item()
            dx = mouse_x - ax
            dy = mouse_y - ay
            if dx**2 + dy**2 > 1.0:
                p_aim = math.atan2(dy, dx)

        # Fill player slot
        move_x[0, 0] = mx
        move_y[0, 0] = my
        aim_angle[0, 0] = p_aim
        fire[0, 0] = firing
        heal[0, 0] = healing

        # Snapshot state before step for event detection
        prev_health = sim.agent_health[0].clone()

        rewards, done = sim.step(move_x, move_y, aim_angle, fire, heal)

        # Player (agent 0) is the listener
        lx = sim.agent_x[0, 0].item()
        ly = sim.agent_y[0, 0].item()

        # Detect shots: cooldown == FIRE_COOLDOWN means agent fired this frame
        for a in range(sim.A):
            if sim.agent_cooldown[0, a].item() == FIRE_COOLDOWN:
                sx = sim.agent_x[0, a].item()
                sy = sim.agent_y[0, a].item()
                audio.play_shot(sx, sy, lx, ly)

        # Detect hits: health decreased this frame
        for a in range(sim.A):
            if sim.agent_health[0, a].item() < prev_health[a].item():
                hx = sim.agent_x[0, a].item()
                hy = sim.agent_y[0, a].item()
                audio.play_hit(hx, hy, lx, ly)

        state = sim.get_state(0)
        pause_done = renderer.render(state)

        # Auto-reset after win pause
        if pause_done:
            sim.reset()

    pygame.quit()


if __name__ == "__main__":
    main()
