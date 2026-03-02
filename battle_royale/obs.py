"""Observation builder: reads BatchedBRSim state and produces actor/critic observations."""

import math
import torch
from .sim import BatchedBRSim
from .config import (
    ARENA_W, ARENA_H, AGENT_SPEED, AGENT_MAX_HP, AGENT_RADIUS,
    FIRE_COOLDOWN, MAX_EPISODE_FRAMES, BULLET_SPEED,
    AMMO_MAX, NUM_AMMO_DEPOSITS, ENTITY_FOV_RADIUS,
    MEDKIT_MAX, HEAL_CHANNEL_FRAMES, NUM_HEALTH_PICKUPS,
    ZONE_MAX_RADIUS, ZONE_MIN_RADIUS, ZONE_SHRINK_START, ZONE_SHRINK_END,
)


class ObservationBuilder:
    def __init__(self, sim: BatchedBRSim, num_lidar_rays: int = 12, lidar_range: float = ENTITY_FOV_RADIUS,
                 max_visible_bullets: int = 10):
        self.sim = sim
        self.num_lidar_rays = num_lidar_rays
        self.lidar_range = lidar_range
        self.max_visible_bullets = max_visible_bullets
        self.fov_radius = ENTITY_FOV_RADIUS

        R = num_lidar_rays
        A = sim.A

        # Precompute ray angle offsets: 2*pi*r/R for r in 0..R-1
        self.ray_offsets = torch.linspace(
            0, 2 * math.pi * (R - 1) / R, R,
            dtype=torch.float32, device=sim.device,
        )

        # Precompute other-agent gather indices (A, A-1)
        indices = []
        for i in range(A):
            indices.append([j for j in range(A) if j != i])
        self.other_idx = torch.tensor(indices, dtype=torch.long, device=sim.device)

        # Wall edges accessed from sim directly (per-env, may change on reset)

        self._diag = math.sqrt(ARENA_W ** 2 + ARENA_H ** 2)

    # ------------------------------------------------------------------
    # Actor observations
    # ------------------------------------------------------------------

    def actor_obs(self) -> dict:
        sim = self.sim
        B, A = sim.B, sim.A

        # Zone: signed distance to edge (positive=inside, negative=outside)
        zone_progress = ((sim.frame.float() - ZONE_SHRINK_START) /
                         (ZONE_SHRINK_END - ZONE_SHRINK_START)).clamp(0, 1)  # (B,)
        zone_radius = ZONE_MAX_RADIUS + (ZONE_MIN_RADIUS - ZONE_MAX_RADIUS) * zone_progress
        dist_to_center = ((sim.agent_x - ARENA_W / 2) ** 2 +
                          (sim.agent_y - ARENA_H / 2) ** 2).sqrt()  # (B, A)
        zone_obs = (zone_radius.unsqueeze(1) - dist_to_center) / self._diag  # (B, A)

        # Own state (B, A, 10)
        own = torch.stack([
            sim.agent_x / ARENA_W,
            sim.agent_y / ARENA_H,
            sim.agent_vx / AGENT_SPEED,
            sim.agent_vy / AGENT_SPEED,
            sim.agent_health / AGENT_MAX_HP,
            sim.agent_cooldown.float() / FIRE_COOLDOWN,
            sim.agent_ammo / AMMO_MAX,
            sim.agent_medkits.float() / MEDKIT_MAX,
            sim.agent_heal_progress.float() / HEAL_CHANNEL_FRAMES,
            zone_obs,
        ], dim=-1)

        # Lidar (B, A, R)
        lidar = self._compute_lidar()

        # Visible agents (B, A, A-1, 7) and mask (B, A, A-1)
        agents, agent_mask = self._compute_agent_features()

        # Nearest bullets (B, A, K, 4) and mask (B, A, K)
        bullets, bullet_mask = self._compute_bullet_features()

        # Deposit features (B, A, D, 2) and mask (B, A, D)
        deposits, deposit_mask = self._compute_deposit_features()

        # Health pickup features (B, A, H, 2) and mask (B, A, H)
        health_pickups, health_pickup_mask = self._compute_health_pickup_features()

        # Global (B, A, 2)
        num_alive = sim.agent_alive.float().sum(dim=1)  # (B,)
        global_obs = torch.stack([
            num_alive / A,
            sim.frame.float() / MAX_EPISODE_FRAMES,
        ], dim=-1)  # (B, 2)
        global_obs = global_obs.unsqueeze(1).expand(B, A, 2)

        return {
            "own": own,
            "lidar": lidar,
            "agents": agents,
            "agent_mask": agent_mask,
            "bullets": bullets,
            "bullet_mask": bullet_mask,
            "deposits": deposits,
            "deposit_mask": deposit_mask,
            "health_pickups": health_pickups,
            "health_pickup_mask": health_pickup_mask,
            "global": global_obs,
        }

    # ------------------------------------------------------------------
    # Lidar
    # ------------------------------------------------------------------

    def _compute_lidar(self) -> torch.Tensor:
        sim = self.sim
        R = self.num_lidar_rays

        # Absolute ray angles (B, A, R)
        angles = sim.agent_dir.unsqueeze(-1) + self.ray_offsets
        dx = torch.cos(angles)
        dy = torch.sin(angles)

        # Origins (B, A, 1)
        ox = sim.agent_x.unsqueeze(-1)
        oy = sim.agent_y.unsqueeze(-1)

        wall_dist = self._lidar_walls(ox, oy, dx, dy)    # (B, A, R)

        return wall_dist  # (B, A, R) — walls only

    def _lidar_walls(self, ox, oy, dx, dy) -> torch.Tensor:
        """Ray-AABB slab method, broadcast over (B,A,R,W)."""
        eps = 1e-8
        rng = self.lidar_range

        # Expand to (B, A, R, 1) and (1, 1, 1, W) for broadcast
        ox4 = ox.unsqueeze(-1)  # (B, A, 1, 1)
        oy4 = oy.unsqueeze(-1)
        # Safe inverse directions — epsilon avoids div-by-zero,
        # pushes parallel-ray t values to ±huge (clamped away later)
        safe_dx = torch.where(dx.abs() < eps, torch.full_like(dx, eps), dx)
        safe_dy = torch.where(dy.abs() < eps, torch.full_like(dy, eps), dy)
        inv_dx = (1.0 / safe_dx).unsqueeze(-1)  # (B, A, R, 1)
        inv_dy = (1.0 / safe_dy).unsqueeze(-1)

        # Wall slabs (B, W) → (B, 1, 1, W) for broadcast with (B, A, R, 1)
        wx1 = self.sim._wall_x1[:, None, None, :]
        wx2 = self.sim._wall_x2[:, None, None, :]
        wy1 = self.sim._wall_y1[:, None, None, :]
        wy2 = self.sim._wall_y2[:, None, None, :]
        tx1 = (wx1 - ox4) * inv_dx  # (B, A, R, W)
        tx2 = (wx2 - ox4) * inv_dx
        ty1 = (wy1 - oy4) * inv_dy
        ty2 = (wy2 - oy4) * inv_dy

        t_enter = torch.maximum(torch.minimum(tx1, tx2),
                                torch.minimum(ty1, ty2))
        t_exit = torch.minimum(torch.maximum(tx1, tx2),
                               torch.maximum(ty1, ty2))

        valid = (t_enter < t_exit) & (t_exit > 0)
        hit_t = torch.where(valid, torch.clamp(t_enter, min=0),
                            torch.full_like(t_enter, rng))

        wall_dist, _ = hit_t.min(dim=-1)  # (B, A, R)
        return torch.clamp(wall_dist, max=rng) / rng

    # ------------------------------------------------------------------
    # Bullet features (nearest K) with FOV mask
    # ------------------------------------------------------------------

    def _compute_bullet_features(self):
        """Nearest-K bullet features (B, A, K, 4) and FOV mask (B, A, K)."""
        sim = self.sim
        B, A = sim.B, sim.A
        K = self.max_visible_bullets
        M = sim.M  # max bullets
        fov_sq = self.fov_radius ** 2

        # Agent positions (B, A, 1)
        ax = sim.agent_x.unsqueeze(-1)  # (B, A, 1)
        ay = sim.agent_y.unsqueeze(-1)

        # Bullet positions (B, 1, M)
        bx = sim.bullet_x.unsqueeze(1)
        by = sim.bullet_y.unsqueeze(1)
        bvx = sim.bullet_vx.unsqueeze(1)
        bvy = sim.bullet_vy.unsqueeze(1)
        active = sim.bullet_active.unsqueeze(1).float()  # (B, 1, M)

        # Relative position (B, A, M)
        rel_x = (bx - ax) / self._diag
        rel_y = (by - ay) / self._diag

        # Distance for sorting (B, A, M)
        dist_sq = (bx - ax) ** 2 + (by - ay) ** 2

        # FOV + active mask
        in_fov = (dist_sq < fov_sq) & (active > 0.5)  # (B, A, M)

        # Inactive or out-of-FOV bullets get huge distance
        dist_sq_masked = dist_sq + (~in_fov).float() * 1e8

        # Nearest K indices (B, A, K)
        _, topk_idx = dist_sq_masked.topk(K, dim=-1, largest=False)

        # Gather features (B, A, K)
        rel_x = rel_x.gather(-1, topk_idx)
        rel_y = rel_y.gather(-1, topk_idx)
        vx = bvx.expand(B, A, M).gather(-1, topk_idx) / BULLET_SPEED
        vy = bvy.expand(B, A, M).gather(-1, topk_idx) / BULLET_SPEED
        mask = in_fov.float().expand(B, A, M).gather(-1, topk_idx)  # (B, A, K)

        features = torch.stack([rel_x, rel_y, vx, vy], dim=-1)  # (B, A, K, 4)
        features = features * mask.unsqueeze(-1)  # zero out inactive/out-of-FOV slots

        return features, mask > 0.5

    # ------------------------------------------------------------------
    # Agent features with FOV mask
    # ------------------------------------------------------------------

    def _compute_agent_features(self):
        """Per-other-agent features (B,A,A-1,8) and FOV mask (B,A,A-1)."""
        sim = self.sim
        B, A = sim.B, sim.A
        fov_sq = self.fov_radius ** 2

        # Gather other agent data (B, A, A-1)
        other_x = sim.agent_x[:, self.other_idx]
        other_y = sim.agent_y[:, self.other_idx]
        other_vx = sim.agent_vx[:, self.other_idx]
        other_vy = sim.agent_vy[:, self.other_idx]
        other_health = sim.agent_health[:, self.other_idx]
        other_alive = sim.agent_alive[:, self.other_idx]

        # Own state (B, A, 1)
        my_x = sim.agent_x.unsqueeze(-1)
        my_y = sim.agent_y.unsqueeze(-1)
        my_vx = sim.agent_vx.unsqueeze(-1)
        my_vy = sim.agent_vy.unsqueeze(-1)
        my_dir = sim.agent_dir.unsqueeze(-1)

        # Relative position
        dx = other_x - my_x
        dy = other_y - my_y
        rel_x = dx / ARENA_W
        rel_y = dy / ARENA_H

        # Relative velocity
        rel_vx = (other_vx - my_vx) / AGENT_SPEED
        rel_vy = (other_vy - my_vy) / AGENT_SPEED

        # Health
        health = other_health / AGENT_MAX_HP

        # Euclidean distance
        dist_sq = dx ** 2 + dy ** 2
        dist = torch.sqrt(dist_sq + 1e-8)
        norm_dist = dist / self._diag

        # Relative angle from my aim direction, wrapped to [-pi, pi], normalized to [-1, 1]
        angle_to_target = torch.atan2(dy, dx)
        rel_angle = angle_to_target - my_dir
        rel_angle = torch.atan2(torch.sin(rel_angle), torch.cos(rel_angle))
        rel_angle = rel_angle / math.pi

        # Enemy aim direction relative to vector toward me (0 = aiming at me)
        other_dir = sim.agent_dir[:, self.other_idx]  # (B, A, A-1)
        angle_to_me = torch.atan2(-dy, -dx)  # from enemy toward me
        enemy_aim_rel = other_dir - angle_to_me
        enemy_aim_rel = torch.atan2(torch.sin(enemy_aim_rel), torch.cos(enemy_aim_rel))
        enemy_aim_rel = enemy_aim_rel / math.pi  # [-1, 1], 0 = aiming at me

        features = torch.stack([
            rel_x, rel_y, rel_vx, rel_vy, health, norm_dist, rel_angle, enemy_aim_rel,
        ], dim=-1)  # (B, A, A-1, 8)

        # Mask: alive AND within FOV
        in_fov = dist_sq < fov_sq
        mask = other_alive & in_fov  # (B, A, A-1)
        features = features * mask.unsqueeze(-1).float()

        return features, mask

    # ------------------------------------------------------------------
    # Deposit features with FOV mask
    # ------------------------------------------------------------------

    def _compute_deposit_features(self):
        """Per-deposit features (B, A, D, 2): rel_x, rel_y + FOV mask (B, A, D)."""
        sim = self.sim
        B, A = sim.B, sim.A
        D = sim.D
        fov_sq = self.fov_radius ** 2

        # Agent positions (B, A, 1)
        ax = sim.agent_x.unsqueeze(-1)
        ay = sim.agent_y.unsqueeze(-1)

        # Deposit positions (B, 1, D)
        dx_pos = sim.deposit_x.unsqueeze(1)
        dy_pos = sim.deposit_y.unsqueeze(1)

        # Relative position (B, A, D)
        rel_x = (dx_pos - ax) / ARENA_W
        rel_y = (dy_pos - ay) / ARENA_H

        # Distance for FOV check
        dist_sq = (dx_pos - ax) ** 2 + (dy_pos - ay) ** 2
        # Must be alive AND within FOV
        mask = (dist_sq < fov_sq) & sim.deposit_alive.unsqueeze(1)  # (B, A, D)

        features = torch.stack([rel_x, rel_y], dim=-1)  # (B, A, D, 2)
        features = features * mask.unsqueeze(-1).float()

        return features, mask

    # ------------------------------------------------------------------
    # Health pickup features with FOV mask
    # ------------------------------------------------------------------

    def _compute_health_pickup_features(self):
        """Per-health-pickup features (B, A, H, 2): rel_x, rel_y + FOV mask (B, A, H)."""
        sim = self.sim
        B, A = sim.B, sim.A
        H = sim.H
        fov_sq = self.fov_radius ** 2

        # Agent positions (B, A, 1)
        ax = sim.agent_x.unsqueeze(-1)
        ay = sim.agent_y.unsqueeze(-1)

        # Health pickup positions (B, 1, H)
        hx = sim.health_pickup_x.unsqueeze(1)
        hy = sim.health_pickup_y.unsqueeze(1)

        # Relative position (B, A, H)
        rel_x = (hx - ax) / ARENA_W
        rel_y = (hy - ay) / ARENA_H

        # Distance for FOV check
        dist_sq = (hx - ax) ** 2 + (hy - ay) ** 2
        # Must be alive AND within FOV
        mask = (dist_sq < fov_sq) & sim.health_pickup_alive.unsqueeze(1)  # (B, A, H)

        features = torch.stack([rel_x, rel_y], dim=-1)  # (B, A, H, 2)
        features = features * mask.unsqueeze(-1).float()

        return features, mask
