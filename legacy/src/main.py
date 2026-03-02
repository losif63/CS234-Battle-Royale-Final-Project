# Author: Jaduk Suh
# Created: November 13th
import pygame
from argparse import ArgumentParser
from src.env import GameEnv

# Action indices: 0=STAY, 1=UP, 2=DOWN, 3=LEFT, 4=RIGHT, 5=SHOOT_UP, 6=SHOOT_DOWN, 7=SHOOT_LEFT, 8=SHOOT_RIGHT

# Player 1: WASD move, RFGH shoot (R=up, F=down, G=left, H=right)
MOVE_MAP_P1 = {pygame.K_w: 1, pygame.K_s: 2, pygame.K_a: 3, pygame.K_d: 4}
SHOOT_MAP_P1 = {pygame.K_r: 5, pygame.K_f: 6, pygame.K_g: 7, pygame.K_h: 8}
ACTION_MAP_P1 = {**MOVE_MAP_P1, **SHOOT_MAP_P1}

# Player 2: Arrow keys move, I/J/K/L shoot (I=up, K=down, J=left, L=right)
MOVE_MAP_P2 = {
    pygame.K_UP: 1,
    pygame.K_DOWN: 2,
    pygame.K_LEFT: 3,
    pygame.K_RIGHT: 4,
}
SHOOT_MAP_P2 = {pygame.K_i: 5, pygame.K_k: 6, pygame.K_j: 7, pygame.K_l: 8}
ACTION_MAP_P2 = {**MOVE_MAP_P2, **SHOOT_MAP_P2}


def main():
    pygame.init()

    env = GameEnv()
    env.reset()

    # Shoot actions (5-8) are one-shot: only on KEYDOWN, not while held
    shoot_this_frame_0 = None  # set by KEYDOWN, consumed once, then cleared
    shoot_this_frame_1 = None

    print("Battle Royale - 2 players")
    print("Player 1: WASD move, R/F/G/H shoot (up/down/left/right)")
    print("Player 2: Arrows move, I/K/J/L shoot (up/down/left/right)")
    print("Pick up green ammo for 10 bullets. ESC or close window to quit.")

    running = True
    step = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key in SHOOT_MAP_P1:
                    shoot_this_frame_0 = SHOOT_MAP_P1[event.key]
                if event.key in SHOOT_MAP_P2:
                    shoot_this_frame_1 = SHOOT_MAP_P2[event.key]

        keys = pygame.key.get_pressed()
        # Movement: continuous from get_pressed(). Shoot: only if we got KEYDOWN this frame.
        if shoot_this_frame_0 is not None:
            action_0 = shoot_this_frame_0
            shoot_this_frame_0 = None
        else:
            action_0 = 0
            for k, a in MOVE_MAP_P1.items():
                if keys[k]:
                    action_0 = a
                    break
        if shoot_this_frame_1 is not None:
            action_1 = shoot_this_frame_1
            shoot_this_frame_1 = None
        else:
            action_1 = 0
            for k, a in MOVE_MAP_P2.items():
                if keys[k]:
                    action_1 = a
                    break

        observation, (reward_0, reward_1), done, info = env.step(action_0, action_1)
        env.render(view=True, step=step)
        step += 1

        if done:
            pygame.time.wait(2000)
            running = False
            break

    env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    main()
