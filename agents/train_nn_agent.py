# Author: Jaduk Suh
# Created: November 26th
from src.model import Q
import random
from src.env import GameEnv
import pygame
import argparse
import torch

# Select action based on epsilon-greedy
def select_action(q_net, obs, epsilon):
    if random.random() < epsilon:
        return random.randint(0, 4)
    
    with torch.no_grad():
        q_values = q_net(obs)
        return torch.argmax(q_values).item()

def main(args):
    # Initialize pygame first
    pygame.init()
    
    env = GameEnv()
    env.reset()

    q_net = Q()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    num_episodes = 1000
    max_steps_per_episode = 3600
    
    print(f"Q-Learning for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0.0
        done = False
        step = 0

        obs = env.get_obs()
        while not done and step < max_steps_per_episode:
            # Select action with epsilon-greedy method
            action = select_action(q_net, obs, epsilon)

            # Go one step
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Render
            if args.render:
                env.render(view=True)
            

            
            # TD Learning
            q_values = q_net(obs)
            q_sa = q_values[action]
            
            with torch.no_grad():
                if done:
                    target = reward
                else:
                    target = reward + gamma * torch.max(q_net(next_obs)).item()
            loss = (target - q_sa) ** 2
            loss.backward()
            optimizer.step()
            obs = next_obs

        # Decay epsilon for epsilon-greedy
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
        print(f"Episode {episode+1}/{num_episodes} | "
              f"Steps: {step} | "
              f"Total reward: {total_reward:.2f} | "
              f"Epsilon: {epsilon:.3f}")
    
    print("\nTraining finished.")
    env.close()

    torch.save(q_net.state_dict(), "q_network.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', '-r', action='store_true', default=False, help='Whether to render the training process or not')
    args = parser.parse_args()
    main(args)

