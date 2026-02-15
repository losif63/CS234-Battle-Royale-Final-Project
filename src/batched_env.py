import numpy as np
import src.config as cfg

# Precomputed velocity vectors: action_index -> (vx, vy)
_SHOOT_VELS = {
    5: np.array([0.0, -cfg.BULLET_SPEED], dtype=np.float32),   # UP
    6: np.array([0.0, cfg.BULLET_SPEED], dtype=np.float32),    # DOWN
    7: np.array([-cfg.BULLET_SPEED, 0.0], dtype=np.float32),   # LEFT
    8: np.array([cfg.BULLET_SPEED, 0.0], dtype=np.float32),    # RIGHT
}


class BatchedGameEnv:
    """Vectorized battle royale: N games stepped simultaneously with numpy."""

    NUM_ACTIONS = 9

    def __init__(self, num_envs: int, seed: int = 42):
        self.num_envs = num_envs
        self.rng = np.random.RandomState(seed)

        # Agent state — (N, 2_agents, ...)
        self.agent_positions = np.zeros((num_envs, 2, 2), dtype=np.float32)
        self.agent_ammo = np.zeros((num_envs, 2), dtype=np.int32)
        self.agent_alive = np.ones((num_envs, 2), dtype=bool)

        # Bullet buffer — (N, MAX_BULLETS, ...)
        M = cfg.MAX_BULLETS
        self.bullet_positions = np.zeros((num_envs, M, 2), dtype=np.float32)
        self.bullet_velocities = np.zeros((num_envs, M, 2), dtype=np.float32)
        self.bullet_owners = np.zeros((num_envs, M), dtype=np.int32)
        self.bullet_active = np.zeros((num_envs, M), dtype=bool)

        # Pickup state — (N, NUM_PICKUPS, ...)
        P = cfg.NUM_AMMO_PICKUPS
        self.pickup_positions = np.zeros((num_envs, P, 2), dtype=np.float32)
        self.pickup_active = np.zeros((num_envs, P), dtype=bool)

        # Per-env scalars
        self.done = np.zeros(num_envs, dtype=bool)
        self.rewards = np.zeros((num_envs, 2), dtype=np.float32)
        self.winners = np.full(num_envs, -1, dtype=np.int32)
        self.time_steps = np.zeros(num_envs, dtype=np.int32)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, mask=None):
        """Reset all envs, or only those where mask is True."""
        if mask is None:
            mask = np.ones(self.num_envs, dtype=bool)
        n = int(mask.sum())
        if n == 0:
            return

        # --- Spawn agents far apart (vectorized rejection sampling) ---
        margin = cfg.AGENT_SPAWN_MARGIN + cfg.AGENT_RADIUS
        lo = np.array([margin, margin], dtype=np.float32)
        hi = np.array([cfg.ARENA_WIDTH - margin, cfg.ARENA_HEIGHT - margin], dtype=np.float32)

        pos = np.zeros((n, 2, 2), dtype=np.float32)
        remaining = np.ones(n, dtype=bool)

        for _ in range(200):
            k = int(remaining.sum())
            if k == 0:
                break
            cand = self.rng.uniform(lo, hi, size=(k, 2, 2)).astype(np.float32)
            diff = cand[:, 0] - cand[:, 1]
            dists = np.sqrt((diff ** 2).sum(axis=1))
            good = dists >= cfg.MIN_AGENT_SPAWN_DISTANCE

            idx = np.where(remaining)[0]
            pos[idx[good]] = cand[good]
            remaining[idx[good]] = False

        # Fallback for stragglers
        if remaining.any():
            pos[remaining, 0] = [lo[0] + (hi[0] - lo[0]) * 0.25, (lo[1] + hi[1]) / 2]
            pos[remaining, 1] = [lo[0] + (hi[0] - lo[0]) * 0.75, (lo[1] + hi[1]) / 2]

        self.agent_positions[mask] = pos
        self.agent_ammo[mask] = 0
        self.agent_alive[mask] = True

        # --- Clear bullets ---
        self.bullet_active[mask] = False

        # --- Spawn pickups (uniform random, good enough for training) ---
        p_margin = cfg.AMMO_PICKUP_RADIUS + cfg.AGENT_RADIUS + 10
        p_lo = np.array([p_margin, p_margin], dtype=np.float32)
        p_hi = np.array([cfg.ARENA_WIDTH - p_margin, cfg.ARENA_HEIGHT - p_margin], dtype=np.float32)
        self.pickup_positions[mask] = self.rng.uniform(
            p_lo, p_hi, size=(n, cfg.NUM_AMMO_PICKUPS, 2)
        ).astype(np.float32)
        self.pickup_active[mask] = True

        self.done[mask] = False
        self.winners[mask] = -1
        self.time_steps[mask] = 0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions: np.ndarray):
        """
        Step all envs. Auto-resets done envs at the start.

        Args:
            actions: int array of shape (num_envs, 2), values 0-8.

        Returns:
            rewards: float32 (num_envs, 2)
            dones:   bool    (num_envs,)
        """
        # Auto-reset finished envs
        if self.done.any():
            self.reset(mask=self.done)

        self.rewards[:] = cfg.REWARD_PER_STEP

        # --- Apply actions per agent ---
        for agent_id in range(2):
            a = actions[:, agent_id]                    # (N,)
            alive = self.agent_alive[:, agent_id]       # (N,)

            # Movement (actions 1-4)
            for act, axis, sign in [(1, 1, -1), (2, 1, 1), (3, 0, -1), (4, 0, 1)]:
                m = alive & (a == act)
                self.agent_positions[m, agent_id, axis] += sign * cfg.AGENT_SPEED

            # Clamp to arena
            np.clip(self.agent_positions[:, agent_id, 0],
                    cfg.AGENT_RADIUS, cfg.ARENA_WIDTH - cfg.AGENT_RADIUS,
                    out=self.agent_positions[:, agent_id, 0])
            np.clip(self.agent_positions[:, agent_id, 1],
                    cfg.AGENT_RADIUS, cfg.ARENA_HEIGHT - cfg.AGENT_RADIUS,
                    out=self.agent_positions[:, agent_id, 1])

            # Shooting (actions 5-8)
            for act_idx, vel in _SHOOT_VELS.items():
                shoot = alive & (a == act_idx) & (self.agent_ammo[:, agent_id] > 0)
                if not shoot.any():
                    continue

                shoot_idx = np.where(shoot)[0]
                inactive = ~self.bullet_active[shoot_idx]       # (k, MAX_BULLETS)
                has_free = inactive.any(axis=1)                 # (k,)
                if not has_free.any():
                    continue

                valid = shoot_idx[has_free]
                slots = np.argmax(~self.bullet_active[valid], axis=1)

                self.bullet_positions[valid, slots] = self.agent_positions[valid, agent_id]
                self.bullet_velocities[valid, slots] = vel
                self.bullet_owners[valid, slots] = agent_id
                self.bullet_active[valid, slots] = True
                self.agent_ammo[valid, agent_id] -= 1

        # --- Ammo pickup collisions ---
        for agent_id in range(2):
            alive = self.agent_alive[:, agent_id]
            if not alive.any():
                continue

            # (N, 1, 2) - (N, P, 2) -> (N, P, 2)
            diff = self.pickup_positions - self.agent_positions[:, agent_id : agent_id + 1, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))            # (N, P)

            collected = (
                alive[:, None]
                & self.pickup_active
                & (dists < cfg.AGENT_RADIUS + cfg.AMMO_PICKUP_RADIUS)
            )
            n_collected = collected.sum(axis=1)                 # (N,)
            self.agent_ammo[:, agent_id] += n_collected * cfg.AMMO_PER_PICKUP
            self.pickup_active &= ~collected

        # --- Update bullets ---
        active = self.bullet_active
        if active.any():
            self.bullet_positions[active] += self.bullet_velocities[active]

        # --- Remove out-of-bounds bullets ---
        margin = cfg.BULLET_RADIUS * 2
        oob = (
            (self.bullet_positions[:, :, 0] < -margin)
            | (self.bullet_positions[:, :, 0] > cfg.ARENA_WIDTH + margin)
            | (self.bullet_positions[:, :, 1] < -margin)
            | (self.bullet_positions[:, :, 1] > cfg.ARENA_HEIGHT + margin)
        )
        self.bullet_active &= ~oob

        # --- Bullet–agent collision ---
        hit_radius = cfg.AGENT_RADIUS + cfg.BULLET_RADIUS
        newly_done = np.zeros(self.num_envs, dtype=bool)

        for agent_id in range(2):
            other_id = 1 - agent_id
            eligible = self.agent_alive[:, agent_id] & ~newly_done

            # (N, 1, 2) broadcast with (N, M, 2)
            diff = self.bullet_positions - self.agent_positions[:, agent_id : agent_id + 1, :]
            dists = np.sqrt((diff ** 2).sum(axis=2))            # (N, M)

            hits = self.bullet_active & (dists < hit_radius) & (self.bullet_owners != agent_id)
            env_hit = eligible & hits.any(axis=1)

            if env_hit.any():
                self.agent_alive[env_hit, agent_id] = False
                self.done[env_hit] = True
                self.winners[env_hit] = other_id
                self.rewards[env_hit, agent_id] = cfg.REWARD_COLLISION
                self.rewards[env_hit, other_id] = cfg.REWARD_HIT_ENEMY
                newly_done[env_hit] = True

        self.time_steps += 1
        return self.rewards.copy(), self.done.copy()
