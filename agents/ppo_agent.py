"""
PPO agent for the Battle Royale environment.
Self-play: one actor-critic network controls both agents (observations are perspective-flipped).

Usage:
    uv run python -m agents.ppo_agent            # train (default 1000 updates)
    uv run python -m agents.ppo_agent --render    # watch a trained agent play
    uv run python -m agents.ppo_agent --resume agents/checkpoints/ppo_update_500.pt
    uv run python -m agents.ppo_agent --updates 2000
"""

import argparse
import copy
import math
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.env import GameEnv
import src.config as cfg

# ---------------------------------------------------------------------------
# Observation constants (must match DQN agent)
# ---------------------------------------------------------------------------
SINGLE_OBS_DIM = 38
OBS_DIM = SINGLE_OBS_DIM
CRITIC_OBS_DIM = OBS_DIM * 2  # omniscient critic sees both agents' obs
NUM_ACTIONS = 9
K_PICKUPS = 3          # was 6 — rarely need more than 3 nearest
K_BULLETS = 3          # was 10 — rarely >3 enemy bullets on screen
PICKUP_FEATURES = 3
BULLET_FEATURES = 5
MAX_AMMO_NORM = 30.0
ENEMY_POS_NORM = 250.0

# ---------------------------------------------------------------------------
# PPO Hyperparameters
# ---------------------------------------------------------------------------
ACTION_REPEAT = 4
MAX_STEPS_PER_EPISODE = 1000

NUM_ENVS = 128
STEPS_PER_ROLLOUT = 128
MINI_BATCH_SIZE = 256
PPO_EPOCHS = 4
GAMMA = 0.999
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEFF = 0.5
ENT_COEFF = 0.05
LR = 3e-4
MAX_GRAD_NORM = 0.5
DEFAULT_NUM_UPDATES = 10000

# League / ELO self-play
POOL_MAX_SIZE = 50          # max past policies to keep
POOL_SNAPSHOT_EVERY = 10    # add current policy to pool every N updates
ELO_K = 32                  # ELO K-factor
ELO_INIT = 1000             # starting ELO for new snapshots
OPPONENT_LATEST_PROB = 0.5  # probability of playing the most recent snapshot
STATIONARY_OPP_PROB = 0.15   # 15% stationary opponents (easy targets to learn attacking)

SAVE_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
SAVE_EVERY = 100  # updates
LOG_EVERY = 10    # updates


# ---------------------------------------------------------------------------
# Observation builder (copied from DQN agent)
# ---------------------------------------------------------------------------

def build_obs(raw_obs: dict, agent_id: int) -> np.ndarray:
    """Build a compact 38-d observation vector from the env's dict observation.

    Perspective is from `agent_id` (0 or 1). Only enemy bullets are included.
    Layout:
        [0:2]   self position (x/W, y/H)
        [2:6]   wall distances (left, right, top, bottom) normalized to [0,1]
        [6:8]   self velocity (vx/speed, vy/speed)
        [8:9]   self ammo (normalized)
        [9:11]  relative enemy position (dx/norm, dy/norm)
        [11:13] enemy velocity (vx/speed, vy/speed)
        [13:14] enemy ammo (normalized)
        [14:23] 3 nearest ammo pickups, each 3 floats (dx, dy, active)
        [23:38] 3 nearest enemy bullets, each 5 floats (dx, dy, vx, vy, active)
    """
    self_key = f"agent_{agent_id}"
    other_id = 1 - agent_id
    other_key = f"agent_{other_id}"

    sx, sy = raw_obs[self_key]["position"]
    ox, oy = raw_obs[other_key]["position"]
    svx, svy = raw_obs[self_key]["velocity"]
    ovx, ovy = raw_obs[other_key]["velocity"]

    obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)

    # Self position normalized
    obs[0] = sx / cfg.ARENA_WIDTH
    obs[1] = sy / cfg.ARENA_HEIGHT

    # Wall distances normalized to [0,1] (0 = touching wall, 1 = far side)
    obs[2] = sx / cfg.ARENA_WIDTH                       # dist to left wall
    obs[3] = (cfg.ARENA_WIDTH - sx) / cfg.ARENA_WIDTH    # dist to right wall
    obs[4] = sy / cfg.ARENA_HEIGHT                       # dist to top wall
    obs[5] = (cfg.ARENA_HEIGHT - sy) / cfg.ARENA_HEIGHT  # dist to bottom wall

    # Self velocity normalized by agent speed
    obs[6] = svx / cfg.AGENT_SPEED
    obs[7] = svy / cfg.AGENT_SPEED

    # Self ammo normalized
    obs[8] = min(raw_obs[self_key]["ammo"] / MAX_AMMO_NORM, 1.0)

    # Relative enemy position — normalized by smaller constant so alignment is detectable
    obs[9] = (ox - sx) / ENEMY_POS_NORM
    obs[10] = (oy - sy) / ENEMY_POS_NORM

    # Enemy velocity normalized by agent speed
    obs[11] = ovx / cfg.AGENT_SPEED
    obs[12] = ovy / cfg.AGENT_SPEED

    # Enemy ammo — critical for strategy (rush if unarmed, be cautious if armed)
    obs[13] = min(raw_obs[other_key]["ammo"] / MAX_AMMO_NORM, 1.0)

    # Ammo pickups: sort by distance to self, take K nearest
    pickups = raw_obs["ammo_pickups"]
    if pickups:
        pdists = []
        for (px, py) in pickups:
            dx = px - sx
            dy = py - sy
            dist = math.sqrt(dx * dx + dy * dy)
            pdists.append((dist, dx, dy))
        pdists.sort(key=lambda t: t[0])

        for i, (_, dx, dy) in enumerate(pdists[:K_PICKUPS]):
            base = 14 + i * PICKUP_FEATURES
            obs[base + 0] = dx / cfg.ARENA_WIDTH
            obs[base + 1] = dy / cfg.ARENA_HEIGHT
            obs[base + 2] = 1.0  # active flag

    # Enemy bullets only: sort by distance to self, take K nearest
    bullet_offset = 14 + K_PICKUPS * PICKUP_FEATURES  # = 23
    bullets = raw_obs["bullets"]
    enemy_bullets = [(pos, vel) for pos, vel, owner in bullets if owner != agent_id]
    if enemy_bullets:
        dists = []
        for (bx, by), (vx, vy) in enemy_bullets:
            dx = bx - sx
            dy = by - sy
            dist = math.sqrt(dx * dx + dy * dy)
            dists.append((dist, dx, dy, vx, vy))
        dists.sort(key=lambda t: t[0])

        for i, (_, dx, dy, vx, vy) in enumerate(dists[:K_BULLETS]):
            base = bullet_offset + i * BULLET_FEATURES
            obs[base + 0] = dx / cfg.ARENA_WIDTH
            obs[base + 1] = dy / cfg.ARENA_HEIGHT
            obs[base + 2] = vx / cfg.BULLET_SPEED
            obs[base + 3] = vy / cfg.BULLET_SPEED
            obs[base + 4] = 1.0  # active flag

    return obs


# ---------------------------------------------------------------------------
# Batched Environment Wrapper
# ---------------------------------------------------------------------------

class BatchedEnv:
    """Runs num_envs GameEnv instances in lockstep for self-play PPO with frame stacking."""

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.envs = [GameEnv(seed=1000 + i) for i in range(num_envs)]
        self.step_counts = np.zeros(num_envs, dtype=np.int32)
        self.obs0 = np.zeros((num_envs, OBS_DIM), dtype=np.float32)
        self.obs1 = np.zeros((num_envs, OBS_DIM), dtype=np.float32)
        # Store latest raw obs for each env (needed for pool opponent)
        self.raw_obs_list: list[dict] = [{}] * num_envs

    def reset(self) -> None:
        """Reset all envs and initialize observations."""
        for i, env in enumerate(self.envs):
            env.reset()
            raw_obs = env.get_obs()
            self.raw_obs_list[i] = raw_obs
            self.obs0[i] = build_obs(raw_obs, agent_id=0)
            self.obs1[i] = build_obs(raw_obs, agent_id=1)
        self.step_counts[:] = 0

    def get_obs_both(self) -> tuple[np.ndarray, np.ndarray]:
        """Get observations for both agents across all envs."""
        return self.obs0, self.obs1

    def step(
        self, actions_0: np.ndarray, actions_1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all envs with smart action repeat.

        Move actions (0-4) repeat ACTION_REPEAT times.
        Shoot actions (5-8) execute once, then STAY for remaining frames.

        Returns:
            obs0, obs1: (num_envs, OBS_DIM) stacked next observations
            rewards0, rewards1: (num_envs,) accumulated rewards
            dones: (num_envs,) whether episode ended
            infos: list of info dicts
        """
        rewards0 = np.zeros(self.num_envs, dtype=np.float32)
        rewards1 = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        infos = [{}] * self.num_envs

        for i, env in enumerate(self.envs):
            a0 = int(actions_0[i])
            a1 = int(actions_1[i])
            done = False

            for frame in range(ACTION_REPEAT):
                # Shoot actions only fire on the first frame, then STAY
                eff_a0 = a0 if (a0 < 5 or frame == 0) else 0
                eff_a1 = a1 if (a1 < 5 or frame == 0) else 0

                raw_obs, (r0, r1), done, info = env.step(eff_a0, eff_a1)
                rewards0[i] += r0
                rewards1[i] += r1
                self.step_counts[i] += 1

                if self.step_counts[i] >= MAX_STEPS_PER_EPISODE:
                    done = True
                    # Draw penalty: worse than dying — incentivize fighting
                    if info.get("winner") is None:
                        rewards0[i] += cfg.REWARD_COLLISION * 1.5  # -15
                        rewards1[i] += cfg.REWARD_COLLISION * 1.5  # -15

                if done:
                    break

            dones[i] = done
            infos[i] = info

            # Update observations
            f0 = build_obs(raw_obs, agent_id=0)
            f1 = build_obs(raw_obs, agent_id=1)
            self.obs0[i] = f0
            self.obs1[i] = f1
            self.raw_obs_list[i] = raw_obs

        # Auto-reset done envs
        for i in range(self.num_envs):
            if dones[i]:
                self.envs[i].reset()
                raw_obs = self.envs[i].get_obs()
                self.raw_obs_list[i] = raw_obs
                self.obs0[i] = build_obs(raw_obs, agent_id=0)
                self.obs1[i] = build_obs(raw_obs, agent_id=1)
                self.step_counts[i] = 0

        return self.obs0.copy(), self.obs1.copy(), rewards0, rewards1, dones, infos

    def close(self):
        for env in self.envs:
            env.close()


# ---------------------------------------------------------------------------
# Actor-Critic Network
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, critic_obs_dim: int = CRITIC_OBS_DIM,
                 num_actions: int = NUM_ACTIONS):
        super().__init__()
        # Policy trunk: sees single-agent obs only
        self.policy_trunk = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, num_actions)

        # Value trunk: omniscient — sees both agents' obs
        self.value_trunk = nn.Sequential(
            nn.Linear(critic_obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(128, 1)

    def policy(self, x: torch.Tensor) -> torch.Tensor:
        """Returns policy logits only."""
        return self.policy_head(self.policy_trunk(x))

    def value(self, critic_obs: torch.Tensor) -> torch.Tensor:
        """Returns value estimate from omniscient observation."""
        return self.value_head(self.value_trunk(critic_obs)).squeeze(-1)

    def get_action_and_value(
        self, x: torch.Tensor, critic_obs: torch.Tensor,
        action: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action (or evaluate given action) and return action, log_prob, entropy, value.

        x: policy observation (single-agent perspective)
        critic_obs: omniscient observation (both agents concatenated)
        action_mask: (batch, num_actions) bool tensor.  True = allowed, False = masked out.
        """
        logits = self.policy(x)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.value(critic_obs)
        return action, log_prob, entropy, value


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

def _ammo_mask(obs_t: torch.Tensor) -> torch.Tensor:
    """Build action mask: disable shoot actions (5-8) when ammo == 0.

    obs_t: (batch, obs_dim) — obs[:, 2] is normalized ammo.
    Returns: (batch, NUM_ACTIONS) bool, True = allowed.
    """
    batch = obs_t.shape[0]
    mask = torch.ones(batch, NUM_ACTIONS, dtype=torch.bool, device=obs_t.device)
    no_ammo = obs_t[:, 8] <= 0.0
    mask[no_ammo, 5:9] = False
    return mask


class PPOAgent:
    def __init__(self, device: torch.device | None = None):
        self.device = device or torch.device("cpu")
        self.network = ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)
        self.update_count = 0

    def act(self, obs: np.ndarray, critic_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select actions for a batch of observations.

        Returns: actions, log_probs, values (all as numpy arrays).
        """
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).to(self.device)
            critic_t = torch.from_numpy(critic_obs).to(self.device)
            mask = _ammo_mask(obs_t)
            action, log_prob, _, value = self.network.get_action_and_value(obs_t, critic_t, action_mask=mask)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def act_deterministic(self, obs: np.ndarray) -> int:
        """Select the greedy action for a single observation (policy only, no value needed)."""
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            mask = _ammo_mask(obs_t)
            logits = self.network.policy(obs_t)
            logits = logits.masked_fill(~mask, -1e8)
            return int(logits.argmax(dim=1).item())

    def act_stochastic(self, obs: np.ndarray, temperature: float = 0.5) -> int:
        """Sample an action from the policy with temperature control.

        Lower temperature = more deterministic, higher = more random.
        """
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            mask = _ammo_mask(obs_t)
            logits = self.network.policy(obs_t)
            logits = logits.masked_fill(~mask, -1e8)
            return int(torch.distributions.Categorical(logits=logits / temperature).sample().item())

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute GAE advantages and returns.

        Args:
            rewards: (T, B) rewards
            values: (T, B) value estimates
            dones: (T, B) episode done flags
            last_values: (B,) bootstrap values for the last step

        Returns:
            advantages: (T, B)
            returns: (T, B)
        """
        T, B = rewards.shape
        advantages = np.zeros((T, B), dtype=np.float32)
        last_gae = np.zeros(B, dtype=np.float32)

        for t in reversed(range(T)):
            if t == T - 1:
                next_values = last_values
            else:
                next_values = values[t + 1]

            not_done = 1.0 - dones[t].astype(np.float32)
            delta = rewards[t] + GAMMA * next_values * not_done - values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * not_done * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict:
        """Run PPO_EPOCHS of mini-batch updates.

        All inputs should be flattened: (N,) or (N, obs_dim).
        """
        N = obs.shape[0]
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for _ in range(PPO_EPOCHS):
            indices = torch.randperm(N, device=self.device)
            for start in range(0, N, MINI_BATCH_SIZE):
                end = min(start + MINI_BATCH_SIZE, N)
                mb_idx = indices[start:end]

                mb_obs = obs[mb_idx]
                mb_critic_obs = critic_obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages[mb_idx]
                mb_returns = returns[mb_idx]

                mb_mask = _ammo_mask(mb_obs)
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    mb_obs, mb_critic_obs, mb_actions, action_mask=mb_mask
                )

                # Clipped surrogate objective
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(new_values, mb_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + VF_COEFF * value_loss + ENT_COEFF * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += (-entropy_loss.item())
                num_updates += 1

        self.update_count += 1

        return {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    # -- save / load --------------------------------------------------------

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "update_count": self.update_count,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(ckpt["network"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.update_count = ckpt["update_count"]


# ---------------------------------------------------------------------------
# Policy Pool (league self-play with ELO)
# ---------------------------------------------------------------------------

class PolicyPool:
    """Maintains a pool of past policy snapshots with ELO ratings."""

    def __init__(self, device: torch.device):
        self.device = device
        self.snapshots: list[dict] = []  # list of {"weights": state_dict, "elo": float}
        self.current_elo = ELO_INIT

    def add_snapshot(self, network: ActorCritic):
        """Snapshot the current network into the pool."""
        weights = copy.deepcopy(network.state_dict())
        self.snapshots.append({"weights": weights, "elo": self.current_elo})
        # Trim to max size (keep most recent)
        if len(self.snapshots) > POOL_MAX_SIZE:
            self.snapshots.pop(0)

    def sample_opponent(self) -> tuple[ActorCritic, int]:
        """Sample an opponent from the pool. Returns (network, pool_index)."""
        opponent = ActorCritic().to(self.device)
        opponent.eval()

        if not self.snapshots:
            # No snapshots yet — opponent is random (untrained network)
            return opponent, -1

        # 50% latest, 50% uniform random from pool
        if random.random() < OPPONENT_LATEST_PROB:
            idx = len(self.snapshots) - 1
        else:
            idx = random.randrange(len(self.snapshots))

        opponent.load_state_dict(self.snapshots[idx]["weights"])
        return opponent, idx

    def update_elo(self, pool_idx: int, current_won: bool):
        """Update ELO for both current policy and the opponent."""
        if pool_idx < 0 or pool_idx >= len(self.snapshots):
            return

        opp_elo = self.snapshots[pool_idx]["elo"]
        expected_current = 1.0 / (1.0 + 10.0 ** ((opp_elo - self.current_elo) / 400.0))
        score = 1.0 if current_won else 0.0

        self.current_elo += ELO_K * (score - expected_current)
        self.snapshots[pool_idx]["elo"] += ELO_K * ((1.0 - score) - (1.0 - expected_current))

    def get_stats(self) -> str:
        if not self.snapshots:
            return "pool empty"
        elos = [s["elo"] for s in self.snapshots]
        return f"pool={len(self.snapshots)} cur_elo={self.current_elo:.0f} pool_elo=[{min(elos):.0f},{max(elos):.0f}]"


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(num_updates: int = DEFAULT_NUM_UPDATES, resume: str | None = None):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Num envs: {NUM_ENVS}, Steps per rollout: {STEPS_PER_ROLLOUT}")
    print(f"Transitions per update: {NUM_ENVS * STEPS_PER_ROLLOUT} (agent_0 only)")
    print(f"Total updates: {num_updates}")
    print(f"League: pool_max={POOL_MAX_SIZE}, snapshot_every={POOL_SNAPSHOT_EVERY}")
    print(f"Opponent mix: stationary {STATIONARY_OPP_PROB:.0%}, self-play pool {1-STATIONARY_OPP_PROB:.0%}")
    print()

    agent = PPOAgent(device=device)
    pool = PolicyPool(device=device)
    batched_env = BatchedEnv(NUM_ENVS)

    start_update = 0
    if resume:
        agent.load(resume)
        start_update = agent.update_count
        print(f"Resumed from {resume} (update_count={start_update})")

    # Seed the pool with the initial (random) policy
    pool.add_snapshot(agent.network)

    # Get initial observations for both agents
    batched_env.reset()
    obs0, obs1 = batched_env.get_obs_both()

    # Episode tracking
    ep_rewards = deque(maxlen=100)
    ep_lengths = deque(maxlen=100)
    ep_kills = deque(maxlen=100)   # 1 if episode ended in a kill, 0 otherwise
    ep_wins = deque(maxlen=100)    # 1 if agent_0 (current policy) won

    # Per-env accumulators for tracking episode stats
    running_rewards = np.zeros(NUM_ENVS, dtype=np.float32)
    running_lengths = np.zeros(NUM_ENVS, dtype=np.int32)

    total_env_steps = 0
    wall_start = time.time()

    for update in range(start_update, start_update + num_updates):
        update_start = time.time()

        # -- Learning rate annealing (linear decay to 0) --
        frac = 1.0 - (update - start_update) / max(1, num_updates)
        lr_now = LR * frac
        for param_group in agent.optimizer.param_groups:
            param_group["lr"] = lr_now
        update_env_steps = 0

        # -- Sample opponent --
        use_stationary = random.random() < STATIONARY_OPP_PROB
        opponent_net = None
        opp_idx = -1
        if not use_stationary:
            opponent_net, opp_idx = pool.sample_opponent()

        # -- Rollout storage (agent_0 / current policy only) --
        all_obs = np.zeros((STEPS_PER_ROLLOUT, NUM_ENVS, OBS_DIM), dtype=np.float32)
        all_critic_obs = np.zeros((STEPS_PER_ROLLOUT, NUM_ENVS, CRITIC_OBS_DIM), dtype=np.float32)
        all_actions = np.zeros((STEPS_PER_ROLLOUT, NUM_ENVS), dtype=np.int64)
        all_log_probs = np.zeros((STEPS_PER_ROLLOUT, NUM_ENVS), dtype=np.float32)
        all_values = np.zeros((STEPS_PER_ROLLOUT, NUM_ENVS), dtype=np.float32)
        all_rewards = np.zeros((STEPS_PER_ROLLOUT, NUM_ENVS), dtype=np.float32)
        all_dones = np.zeros((STEPS_PER_ROLLOUT, NUM_ENVS), dtype=np.float32)

        # -- Collect rollout --
        for step in range(STEPS_PER_ROLLOUT):
            all_obs[step] = obs0
            critic_obs = np.concatenate([obs0, obs1], axis=1)  # (NUM_ENVS, CRITIC_OBS_DIM)
            all_critic_obs[step] = critic_obs

            # Agent 0 (current policy — learning)
            actions_0, log_probs, values = agent.act(obs0, critic_obs)
            all_actions[step] = actions_0
            all_log_probs[step] = log_probs
            all_values[step] = values

            # Agent 1 (opponent — stationary or pool snapshot)
            if use_stationary:
                actions_1 = np.zeros(NUM_ENVS, dtype=np.int64)  # action 0 = STAY
            else:
                with torch.no_grad():
                    obs1_t = torch.from_numpy(obs1).to(device)
                    opp_mask = _ammo_mask(obs1_t)
                    opp_logits = opponent_net.policy(obs1_t)
                    opp_logits = opp_logits.masked_fill(~opp_mask, -1e8)
                    actions_1 = torch.distributions.Categorical(logits=opp_logits).sample().cpu().numpy()

            # Step all envs
            next_obs0, next_obs1, rewards0, rewards1, dones, infos = batched_env.step(
                actions_0, actions_1
            )

            update_env_steps += NUM_ENVS * ACTION_REPEAT

            all_rewards[step] = rewards0
            all_dones[step] = dones.astype(np.float32)

            # Track episode stats
            running_rewards += rewards0
            running_lengths += 1

            for i in range(NUM_ENVS):
                if dones[i]:
                    ep_rewards.append(running_rewards[i])
                    ep_lengths.append(running_lengths[i])
                    winner = infos[i].get("winner")
                    ep_kills.append(1 if winner is not None else 0)
                    current_won = (winner == 0)
                    ep_wins.append(1 if current_won else 0)
                    if winner is not None:
                        pool.update_elo(opp_idx, current_won)
                    running_rewards[i] = 0.0
                    running_lengths[i] = 0

            obs0 = next_obs0
            obs1 = next_obs1

        total_env_steps += update_env_steps

        # -- Snapshot current policy into pool --
        if (update + 1) % POOL_SNAPSHOT_EVERY == 0:
            pool.add_snapshot(agent.network)

        # -- Bootstrap value for last step --
        with torch.no_grad():
            critic_obs_last = np.concatenate([obs0, obs1], axis=1)
            critic_t = torch.from_numpy(critic_obs_last).to(device)
            last_values = agent.network.value(critic_t).cpu().numpy()

        # -- Compute GAE --
        advantages, returns = agent.compute_gae(
            all_rewards, all_values, all_dones, last_values
        )

        # -- Flatten rollout data: (T, B) -> (T*B,) --
        T = STEPS_PER_ROLLOUT
        B = NUM_ENVS
        flat_obs = torch.from_numpy(all_obs.reshape(T * B, OBS_DIM)).to(device)
        flat_critic_obs = torch.from_numpy(all_critic_obs.reshape(T * B, CRITIC_OBS_DIM)).to(device)
        flat_actions = torch.from_numpy(all_actions.reshape(T * B)).to(device)
        flat_log_probs = torch.from_numpy(all_log_probs.reshape(T * B)).to(device)
        flat_advantages = torch.from_numpy(advantages.reshape(T * B)).to(device)
        flat_returns = torch.from_numpy(returns.reshape(T * B)).to(device)

        # -- PPO update --
        loss_info = agent.update(
            flat_obs, flat_critic_obs, flat_actions, flat_log_probs, flat_advantages, flat_returns
        )

        # -- Logging --
        update_elapsed = time.time() - update_start
        fps = update_env_steps / update_elapsed if update_elapsed > 0 else 0

        if (update + 1) % LOG_EVERY == 0 or update == start_update:
            avg_reward = np.mean(ep_rewards) if ep_rewards else 0.0
            avg_length = np.mean(ep_lengths) if ep_lengths else 0.0
            kill_rate = np.mean(ep_kills) if ep_kills else 0.0
            win_rate = np.mean(ep_wins) if ep_wins else 0.0

            wall_elapsed = time.time() - wall_start
            overall_fps = total_env_steps / wall_elapsed if wall_elapsed > 0 else 0

            opp_label = "ST" if use_stationary else "SP"
            print(
                f"Update {update + 1:>5d}/{start_update + num_updates} [{opp_label}] | "
                f"FPS {fps:>7.0f} (avg {overall_fps:>7.0f}) | "
                f"avg_r {avg_reward:>7.2f} | "
                f"kill {kill_rate:.2f} win {win_rate:.2f} | "
                f"ep_len {avg_length:>5.1f} | "
                f"{pool.get_stats()} | "
                f"ent {loss_info['entropy']:.3f} lr {lr_now:.1e}"
            )

        # -- Save checkpoint --
        if (update + 1) % SAVE_EVERY == 0:
            path = os.path.join(SAVE_DIR, f"ppo_update_{update + 1}.pt")
            agent.save(path)
            print(f"  -> saved {path}")

    # Final save
    final_path = os.path.join(SAVE_DIR, "ppo_final.pt")
    agent.save(final_path)
    print(f"\nTraining done. Final model saved to {final_path}")

    batched_env.close()


# ---------------------------------------------------------------------------
# Watch a trained agent play
# ---------------------------------------------------------------------------

def _find_latest_checkpoint() -> str | None:
    """Find the checkpoint with the highest update number in SAVE_DIR."""
    import glob
    pattern = os.path.join(SAVE_DIR, "ppo_update_*.pt")
    files = glob.glob(pattern)
    if not files:
        # Fall back to ppo_final.pt
        final = os.path.join(SAVE_DIR, "ppo_final.pt")
        return final if os.path.exists(final) else None
    # Extract update numbers and pick the max
    def _num(f):
        base = os.path.basename(f)
        return int(base.replace("ppo_update_", "").replace(".pt", ""))
    return max(files, key=_num)


def watch(checkpoint: str | None = None, max_frames: int = MAX_STEPS_PER_EPISODE):
    """Watch two AIs fight. Auto-loads latest checkpoint between episodes."""
    import pygame  # noqa: F811

    device = torch.device("cpu")
    agent = PPOAgent(device=device)

    loaded_ckpt = checkpoint or _find_latest_checkpoint()
    if not loaded_ckpt or not os.path.exists(loaded_ckpt):
        print(f"No checkpoint found. Train first.")
        return
    agent.load(loaded_ckpt)
    agent.network.eval()
    print(f"Loaded {loaded_ckpt}")

    env = GameEnv(seed=random.randint(0, 2**31))
    ep = 0
    wins = {0: 0, 1: 0, "draw": 0}

    while True:
        # Hot-reload: check for newer checkpoint between episodes
        latest = _find_latest_checkpoint()
        if latest and latest != loaded_ckpt:
            try:
                agent.load(latest)
                agent.network.eval()
                loaded_ckpt = latest
                print(f"Hot-reloaded {loaded_ckpt}")
            except Exception:
                pass  # checkpoint might be mid-write, skip

        ep += 1
        env.rng.seed(random.randint(0, 2**31))
        env.reset()
        raw_obs = env.get_obs()
        obs0 = build_obs(raw_obs, agent_id=0)
        obs1 = build_obs(raw_obs, agent_id=1)

        ep_reward = 0.0
        t = 0
        done = False
        info = {}

        # Get initial value estimates
        win_prob_blue = 0.5

        while t < max_frames and not done:
            a0 = agent.act_stochastic(obs0)
            a1 = agent.act_stochastic(obs1)

            # Compute win probability from omniscient critic
            with torch.no_grad():
                critic_obs0 = np.concatenate([obs0, obs1])
                critic_obs1 = np.concatenate([obs1, obs0])
                critic_batch = torch.from_numpy(
                    np.stack([critic_obs0, critic_obs1])
                ).to(device)
                values = agent.network.value(critic_batch).cpu().numpy()
                # Softmax over the two values to get win probability
                v0, v1 = float(values[0]), float(values[1])
                max_v = max(v0, v1)
                exp0 = math.exp(v0 - max_v)
                exp1 = math.exp(v1 - max_v)
                win_prob_blue = exp0 / (exp0 + exp1)

            for frame in range(ACTION_REPEAT):
                eff_a0 = a0 if (a0 < 5 or frame == 0) else 0
                eff_a1 = a1 if (a1 < 5 or frame == 0) else 0

                raw_obs, (r0, r1), done, info = env.step(eff_a0, eff_a1)
                ep_reward += r0
                t += 1
                env.render(view=True, step=t, flip=False)

                # Draw win probability bar overlay
                if env.screen is not None:
                    bar_w, bar_h = 200, 16
                    bar_x = cfg.ARENA_WIDTH - bar_w - 10
                    bar_y = 10
                    blue_w = int(bar_w * win_prob_blue)
                    # Background
                    pygame.draw.rect(env.screen, (40, 40, 50), (bar_x, bar_y, bar_w, bar_h))
                    # Blue portion
                    if blue_w > 0:
                        pygame.draw.rect(env.screen, cfg.COLOR_AGENT, (bar_x, bar_y, blue_w, bar_h))
                    # Orange portion
                    if blue_w < bar_w:
                        pygame.draw.rect(env.screen, cfg.COLOR_AGENT_2, (bar_x + blue_w, bar_y, bar_w - blue_w, bar_h))
                    # Border
                    pygame.draw.rect(env.screen, (180, 180, 180), (bar_x, bar_y, bar_w, bar_h), 1)
                    # Percentage labels
                    small_font = pygame.font.Font(None, 22)
                    blue_pct = small_font.render(f"{win_prob_blue:.0%}", True, (255, 255, 255))
                    orange_pct = small_font.render(f"{1 - win_prob_blue:.0%}", True, (255, 255, 255))
                    env.screen.blit(blue_pct, (bar_x + 4, bar_y + 1))
                    env.screen.blit(orange_pct, (bar_x + bar_w - orange_pct.get_width() - 4, bar_y + 1))
                    pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        env.close()
                        return

                if t >= max_frames:
                    done = True

                if done:
                    pygame.time.wait(1000)
                    break

            obs0 = build_obs(raw_obs, agent_id=0)
            obs1 = build_obs(raw_obs, agent_id=1)

        winner = info.get("winner")
        if winner is not None:
            wins[winner] += 1
        else:
            wins["draw"] += 1

        total = wins[0] + wins[1] + wins["draw"]
        winner_str = {0: "Blue", 1: "Orange"}.get(winner, "Draw")
        print(f"Episode {ep}: {winner_str} wins, steps={t} | Blue {wins[0]/total:.1%} Orange {wins[1]/total:.1%} draw {wins['draw']/total:.1%}")



# ---------------------------------------------------------------------------
# Play against the trained agent (human = Player 1, PPO = Player 2)
# ---------------------------------------------------------------------------

def play(checkpoint: str):
    import pygame  # noqa: F811

    # Player 1 controls: WASD move, arrow keys shoot
    MOVE_MAP = {pygame.K_w: 1, pygame.K_s: 2, pygame.K_a: 3, pygame.K_d: 4}
    SHOOT_MAP = {pygame.K_UP: 5, pygame.K_DOWN: 6, pygame.K_LEFT: 7, pygame.K_RIGHT: 8}

    device = torch.device("cpu")
    agent = PPOAgent(device=device)
    agent.load(checkpoint)
    agent.network.eval()

    env = GameEnv()

    print("You are Player 1 (blue). PPO is Player 2 (orange).")
    print("  Move: WASD")
    print("  Shoot: Arrow keys")
    print("  ESC to quit. Pick up green ammo for 10 bullets.")

    # Force pygame init + first render before event loop
    env.reset()
    env.render(view=True, step=0)

    running = True
    while running:
        env.reset()
        raw_obs = env.get_obs()

        obs1 = build_obs(raw_obs, agent_id=1)

        shoot_queued = None
        t = 0
        done = False

        while t < MAX_STEPS_PER_EPISODE and not done and running:
            # --- Poll events + read held keys ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key in SHOOT_MAP:
                        shoot_queued = SHOOT_MAP[event.key]

            if not running:
                break

            # --- PPO action (decided once per cycle) ---
            ai_action = agent.act_deterministic(obs1)

            # --- Step with smart repeat ---
            for frame in range(ACTION_REPEAT):
                # Poll events every frame so we never miss a keypress
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        if event.key in SHOOT_MAP:
                            shoot_queued = SHOOT_MAP[event.key]

                if not running:
                    break

                # Human action: shoot if queued, otherwise move from held keys
                if shoot_queued is not None:
                    eff_human = shoot_queued
                    shoot_queued = None
                else:
                    keys = pygame.key.get_pressed()
                    eff_human = 0
                    for k, a in MOVE_MAP.items():
                        if keys[k]:
                            eff_human = a
                            break

                eff_ai = ai_action if (ai_action < 5 or frame == 0) else 0

                raw_obs, (r0, r1), done, info = env.step(eff_human, eff_ai)
                t += 1
                env.render(view=True, step=t)

                if t >= MAX_STEPS_PER_EPISODE:
                    done = True
                if done:
                    break

            obs1 = build_obs(raw_obs, agent_id=1)

        if done:
            winner = info.get("winner")
            if winner == 0:
                print(f"You lose! (step {t})")
            elif winner == 1:
                print(f"You win! (step {t})")
            else:
                print(f"Draw! (step {t})")
            pygame.time.wait(2000)

    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PPO agent for Battle Royale")
    parser.add_argument("--watch", action="store_true", help="Watch two AIs fight")
    parser.add_argument("--play", action="store_true", help="Play against the trained agent")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint for --render/--play mode",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--updates", type=int, default=DEFAULT_NUM_UPDATES,
        help="Number of PPO updates (default: 1000)",
    )
    parser.add_argument(
        "--frames", type=int, default=MAX_STEPS_PER_EPISODE,
        help="Max steps per episode in watch mode (default: 500)",
    )
    args = parser.parse_args()

    ckpt = args.checkpoint

    if args.watch or args.play:
        if args.play:
            if ckpt is None:
                ckpt = _find_latest_checkpoint()
            if not ckpt or not os.path.exists(ckpt):
                print("No checkpoint found. Train first.")
                return
            play(ckpt)
        else:
            watch(checkpoint=ckpt, max_frames=args.frames)
    else:
        train(num_updates=args.updates, resume=args.resume)


if __name__ == "__main__":
    main()
