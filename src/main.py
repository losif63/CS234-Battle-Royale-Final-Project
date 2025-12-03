# Author: Jaduk Suh
# Created: November 13th
import pygame
from argparse import ArgumentParser
from src.env import GameEnv


def main():
    # Initialize pygame first
    pygame.init()
    
    env = GameEnv()
    env.reset()
    
    # Action mapping: pygame keys to action indices
    action_map = {
        pygame.K_UP: 1,      # UP
        pygame.K_DOWN: 2,    # DOWN
        pygame.K_LEFT: 3,    # LEFT
        pygame.K_RIGHT: 4,   # RIGHT
    }
    
    running = True
    current_action = 0  # STAY by default
    
    print("Ninja Dodge - Use arrow keys to move, ESC or close window to quit")
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in action_map:
                    current_action = action_map[event.key]
            elif event.type == pygame.KEYUP:
                # Stop when key is released
                if event.key in action_map:
                    current_action = 0
        
        # Get pressed keys (for continuous movement)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            current_action = 1
        elif keys[pygame.K_DOWN]:
            current_action = 2
        elif keys[pygame.K_LEFT]:
            current_action = 3
        elif keys[pygame.K_RIGHT]:
            current_action = 4
        else:
            current_action = 0
        
        # Step environment
        observation, reward, done = env.step(current_action)
        
        # Render
        env.render(view=True)
        
        # Check if done (collision)
        if done:
            pygame.time.wait(2000)
            
            # Uncomment below code to Restart after game over
            # env.reset()
            # print("Resetting...")
            
            # Terminate
            running = False
            break
    
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    main()

