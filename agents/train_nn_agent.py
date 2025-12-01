# Author: Jaduk Suh
# Created: November 26th
from src.model import Q
import random
from src.env import GameEnv
import pygame
import argparse
import torch
from collections import deque

SEED = 42

class ReplayBuffer():
    def __init__(self):
        self.obs = deque()
        self.actions = deque()
        self.rewards = deque()
        self.next_obs = deque()
        self.max_capacity = 20000
        return

    def __len__(self):
        return len(self.obs)

    def push(self, o, a, r, next_o):
        if len(self.obs) >= self.max_capacity:
            self.obs.popleft()
            self.actions.popleft()
            self.rewards.popleft()
            self.next_obs.popleft()
        
        self.obs.append(o)
        self.actions.append(a)
        self.rewards.append(r)
        self.next_obs.append(next_o)
        return
    
    def sample(self, batch_size):
        idxs = random.sample(range(len(self.obs)), batch_size)
        obs_batch = torch.stack([self.obs[i] for i in idxs])
        actions_batch = torch.tensor([self.actions[i] for i in idxs], dtype=torch.long)
        rewards_batch = torch.tensor([self.rewards[i] for i in idxs], dtype=torch.float32)
        next_obs_batch = torch.stack([self.next_obs[i] for i in idxs])
        return obs_batch, actions_batch, rewards_batch, next_obs_batch
        

# Select action based on epsilon-greedy
def select_action(q_net, obs, epsilon):
    q_values = q_net(obs)
    if random.random() < epsilon:
        return q_values, random.randint(0, 4)
    
    with torch.no_grad():
        return q_values, torch.argmax(q_values).item()

def main(args):
    # Initialize pygame first
    pygame.init()
    
    env = GameEnv()
    env.reset()

    q_net = Q()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=5e-4, weight_decay=1e-5)
    buffer = ReplayBuffer()
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9993

    num_episodes = 3000
    max_steps_per_episode = 3600

    warmup_steps = 1000
    batch_size = 16
    
    print(f"Q-Learning for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0.0
        done = False
        step = 0

        obs = env.get_obs()
        while not done and step < max_steps_per_episode:
            # Select action with epsilon-greedy method
            q_values, action = select_action(q_net, obs, epsilon)

            # Go one step
            next_obs, reward, done = env.step(action)
            total_reward += reward
            step += 1
            
            # Render
            if args.render:
                env.render(view=True)

            buffer.push(obs, action, reward, next_obs)
            obs = next_obs

            if len(buffer) >= warmup_steps:
                optimizer.zero_grad()

                # TD Learning
                # Experience Replay
                obs_b, act_b, rew_b, next_obs_b = buffer.sample(batch_size)
                q_values = q_net(obs_b)
                q_sa = q_values.gather(1, act_b.unsqueeze(1)).squeeze(1)

                 # Target: r + gamma * max_a' Q(s', a')
                with torch.no_grad():
                    q_next = q_net(next_obs_b)          # [B, num_actions]
                    max_q_next = q_next.max(dim=1).values
                    targets = rew_b + gamma * max_q_next
                
                loss = torch.mean((targets - q_sa) ** 2)
                loss.backward()
                optimizer.step()
            
            # q_sa = q_values[action]
            # with torch.no_grad():
            #     if done:
            #         target = reward
            #     else:
            #         target = reward + gamma * torch.max(q_net(next_obs))
            # loss = (target - q_sa) ** 2
            # loss.backward()
            # optimizer.step()
            

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
    random.seed(SEED)
    torch.manual_seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', '-r', action='store_true', default=False, help='Whether to render the training process or not')
    args = parser.parse_args()
    main(args)

