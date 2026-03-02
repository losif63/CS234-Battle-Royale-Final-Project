"""Fully-batched 2D battle royale simulator. All state is (B, A) torch tensors."""

import torch
import numpy as np
from .config import (
    ARENA_W, ARENA_H, AGENT_RADIUS, AGENT_SPEED, AGENT_FRICTION,
    AGENT_MAX_HP, BORDER_WALLS, NUM_INTERIOR_WALLS,
    BULLET_SPEED, BULLET_RADIUS, MAX_BULLETS,
    FIRE_COOLDOWN, BULLET_DAMAGE, MAX_EPISODE_FRAMES,
    NUM_AMMO_DEPOSITS, AMMO_PER_PICKUP, AMMO_START, AMMO_MAX,
    AMMO_PICKUP_RADIUS, AMMO_RESPAWN_FRAMES, WALL_THICKNESS,
    NUM_HEALTH_PICKUPS, MEDKIT_HEAL_AMOUNT, MEDKIT_MAX,
    HEAL_CHANNEL_FRAMES, HEALTH_PICKUP_RADIUS,
    ZONE_MAX_RADIUS, ZONE_MIN_RADIUS, ZONE_SHRINK_START, ZONE_SHRINK_END,
    ZONE_DAMAGE_PER_FRAME, FIRE_MOVE_PENALTY,
)


class BatchedBRSim:
    def __init__(self, num_envs: int = 1, max_agents: int = 1, device: str = "auto"):
        self.B = num_envs
        self.A = max_agents
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Per-env walls (B, W, 4) — border walls shared, interior randomized
        self.N_BORDER = len(BORDER_WALLS)
        self.N_INTERIOR = NUM_INTERIOR_WALLS
        self.W = self.N_BORDER + self.N_INTERIOR
        border = torch.tensor(BORDER_WALLS, dtype=torch.float32, device=self.device)
        self.walls = torch.zeros(self.B, self.W, 4, device=self.device)
        self.walls[:, :self.N_BORDER] = border.unsqueeze(0)

        # Edge tensors (B, W) — recomputed after wall changes
        self._wall_x1 = self.walls[:, :, 0]
        self._wall_y1 = self.walls[:, :, 1]
        self._wall_x2 = self.walls[:, :, 0] + self.walls[:, :, 2]
        self._wall_y2 = self.walls[:, :, 1] + self.walls[:, :, 3]

        # Agent state arrays (B, A)
        self.agent_x = torch.zeros(self.B, self.A, device=self.device)
        self.agent_y = torch.zeros(self.B, self.A, device=self.device)
        self.agent_vx = torch.zeros(self.B, self.A, device=self.device)
        self.agent_vy = torch.zeros(self.B, self.A, device=self.device)
        self.agent_dir = torch.zeros(self.B, self.A, device=self.device)
        self.agent_health = torch.full((self.B, self.A), float(AGENT_MAX_HP), device=self.device)
        self.agent_alive = torch.ones(self.B, self.A, dtype=torch.bool, device=self.device)
        self.agent_cooldown = torch.zeros(self.B, self.A, dtype=torch.long, device=self.device)
        self.frame = torch.zeros(self.B, dtype=torch.long, device=self.device)

        # Episode tracking
        self.agent_place = torch.zeros(self.B, self.A, dtype=torch.long, device=self.device)
        self.death_rank_counter = torch.full((self.B,), self.A, dtype=torch.long, device=self.device)
        self.episode_done = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        self.episode_rewards = torch.zeros(self.B, self.A, device=self.device)

        # Bullet state arrays (B, M)
        self.M = MAX_BULLETS
        self.bullet_x = torch.zeros(self.B, self.M, device=self.device)
        self.bullet_y = torch.zeros(self.B, self.M, device=self.device)
        self.bullet_vx = torch.zeros(self.B, self.M, device=self.device)
        self.bullet_vy = torch.zeros(self.B, self.M, device=self.device)
        self.bullet_active = torch.zeros(self.B, self.M, dtype=torch.bool, device=self.device)
        self.bullet_owner = torch.full((self.B, self.M), -1, dtype=torch.long, device=self.device)
        self.bullet_spawn_x = torch.zeros(self.B, self.M, device=self.device)
        self.bullet_spawn_y = torch.zeros(self.B, self.M, device=self.device)

        # Ammo
        self.D = NUM_AMMO_DEPOSITS
        self.agent_ammo = torch.full((self.B, self.A), AMMO_START, dtype=torch.float32, device=self.device)
        self.deposit_x = torch.zeros(self.B, self.D, device=self.device)
        self.deposit_y = torch.zeros(self.B, self.D, device=self.device)
        self.deposit_ammo = torch.full((self.B, self.D), float(AMMO_PER_PICKUP), device=self.device)
        self.deposit_alive = torch.ones(self.B, self.D, dtype=torch.bool, device=self.device)
        self.deposit_respawn_timer = torch.zeros(self.B, self.D, dtype=torch.long, device=self.device)

        # Health pickups
        self.H = NUM_HEALTH_PICKUPS
        self.agent_medkits = torch.zeros(self.B, self.A, dtype=torch.long, device=self.device)
        self.agent_heal_progress = torch.zeros(self.B, self.A, dtype=torch.long, device=self.device)
        self.health_pickup_x = torch.zeros(self.B, self.H, device=self.device)
        self.health_pickup_y = torch.zeros(self.B, self.H, device=self.device)
        self.health_pickup_count = torch.ones(self.B, self.H, dtype=torch.long, device=self.device)
        self.health_pickup_alive = torch.ones(self.B, self.H, dtype=torch.bool, device=self.device)

        self.reset()

    def reset(self, mask=None):
        """Reset envs. If mask is None, reset all; otherwise mask is (B,) bool."""
        if mask is None:
            mask = torch.ones(self.B, dtype=torch.bool, device=self.device)

        buf = AGENT_RADIUS + 30
        n_reset = mask.sum().item()
        if n_reset == 0:
            return
        # Randomize interior walls first (deposits/pickups need valid walls)
        self._randomize_interior_walls(mask)

        # Spawn agents evenly spaced around arena perimeter with jitter
        # Distribute along perimeter: top → right → bottom → left
        perimeter = 2 * (ARENA_W + ARENA_H)
        # Random offset per env so spawn positions aren't deterministic
        offset = torch.rand(n_reset, 1, device=self.device) * perimeter
        spacing = torch.arange(self.A, device=self.device).unsqueeze(0) * (perimeter / self.A)
        pos_along = (offset + spacing) % perimeter  # (n_reset, A)

        # Convert perimeter position to (x, y)
        spawn_x = torch.zeros(n_reset, self.A, device=self.device)
        spawn_y = torch.zeros(n_reset, self.A, device=self.device)
        # Top edge: [0, W)
        top = pos_along < ARENA_W
        spawn_x = torch.where(top, pos_along, spawn_x)
        spawn_y = torch.where(top, torch.full_like(spawn_y, buf), spawn_y)
        # Right edge: [W, W+H)
        right = (pos_along >= ARENA_W) & (pos_along < ARENA_W + ARENA_H)
        spawn_x = torch.where(right, torch.full_like(spawn_x, ARENA_W - buf), spawn_x)
        spawn_y = torch.where(right, pos_along - ARENA_W, spawn_y)
        # Bottom edge: [W+H, 2W+H)
        bottom = (pos_along >= ARENA_W + ARENA_H) & (pos_along < 2 * ARENA_W + ARENA_H)
        spawn_x = torch.where(bottom, ARENA_W - (pos_along - ARENA_W - ARENA_H), spawn_x)
        spawn_y = torch.where(bottom, torch.full_like(spawn_y, ARENA_H - buf), spawn_y)
        # Left edge: [2W+H, 2W+2H)
        left = pos_along >= 2 * ARENA_W + ARENA_H
        spawn_x = torch.where(left, torch.full_like(spawn_x, buf), spawn_x)
        spawn_y = torch.where(left, ARENA_H - (pos_along - 2 * ARENA_W - ARENA_H), spawn_y)

        jitter = 30.0
        self.agent_x[mask] = spawn_x + (torch.rand(n_reset, self.A, device=self.device) - 0.5) * jitter
        self.agent_y[mask] = spawn_y + (torch.rand(n_reset, self.A, device=self.device) - 0.5) * jitter
        self.agent_vx[mask] = 0.0
        self.agent_vy[mask] = 0.0
        # Face toward arena center
        cx_dir = ARENA_W / 2 - self.agent_x[mask]
        cy_dir = ARENA_H / 2 - self.agent_y[mask]
        self.agent_dir[mask] = torch.atan2(cy_dir, cx_dir)
        self.agent_health[mask] = AGENT_MAX_HP
        self.agent_alive[mask] = True
        self.agent_cooldown[mask] = 0
        self.bullet_active[mask] = False
        self.frame[mask] = 0

        # Ammo reset
        self.agent_ammo[mask] = AMMO_START
        self.deposit_ammo[mask] = AMMO_PER_PICKUP
        self.deposit_alive[mask] = True
        self.deposit_respawn_timer[mask] = 0
        self._randomize_deposits(mask)

        # Health pickup reset
        self.agent_medkits[mask] = 0
        self.agent_heal_progress[mask] = 0
        self.health_pickup_count[mask] = 1
        self.health_pickup_alive[mask] = True
        self._randomize_health_pickups(mask)

        # Episode tracking reset
        self.agent_place[mask] = 0
        self.death_rank_counter[mask] = self.A
        self.episode_done[mask] = False
        self.episode_rewards[mask] = 0.0

        self._resolve_wall_collisions(mask)

    def step(self, move_x, move_y, aim_angle, fire=None, heal=None):
        """Advance one tick. Returns (rewards (B, A), done (B,))."""
        if fire is None:
            fire = torch.zeros(self.B, self.A, dtype=torch.bool, device=self.device)
        if heal is None:
            heal = torch.zeros(self.B, self.A, dtype=torch.bool, device=self.device)

        # Don't process finished episodes
        active_env = ~self.episode_done  # (B,)

        # Normalize move magnitude to <= 1
        mag = torch.sqrt(move_x**2 + move_y**2)
        mag_clamp = torch.clamp(mag, min=1.0)
        move_x = move_x / mag_clamp
        move_y = move_y / mag_clamp

        # Set velocity from input, apply friction when no input
        has_input = mag > 1e-6
        self.agent_vx = torch.where(has_input, move_x * AGENT_SPEED, self.agent_vx * AGENT_FRICTION)
        self.agent_vy = torch.where(has_input, move_y * AGENT_SPEED, self.agent_vy * AGENT_FRICTION)

        # Slow movement while firing (cooldown active from recent shot)
        is_firing = self.agent_cooldown > 0
        fire_slow = torch.where(is_firing, FIRE_MOVE_PENALTY, 1.0)
        self.agent_vx = self.agent_vx * fire_slow
        self.agent_vy = self.agent_vy * fire_slow

        # --- Healing (before position integration so freeze takes effect) ---
        self._process_healing(heal)
        is_healing = self.agent_heal_progress > 0

        # Update aim direction (NaN = keep previous, blocked while healing)
        valid_aim = ~torch.isnan(aim_angle) & ~is_healing
        self.agent_dir = torch.where(valid_aim, aim_angle, self.agent_dir)

        # Integrate position
        self.agent_x += self.agent_vx
        self.agent_y += self.agent_vy

        # Resolve agent-wall collisions
        self._resolve_wall_collisions()

        # --- Ammo pickups ---
        self._process_ammo_pickups()

        # --- Ammo respawn (inside zone) ---
        self._respawn_deposits()

        # --- Health pickups ---
        self._process_health_pickups()

        # --- Bullets ---
        self.agent_cooldown = torch.clamp(self.agent_cooldown - 1, min=0)

        self._spawn_bullets(fire & ~is_healing)

        # Move active bullets
        self.bullet_x += self.bullet_vx * self.bullet_active
        self.bullet_y += self.bullet_vy * self.bullet_active

        self._bullet_wall_collisions()

        # Capture alive state before collisions for death tracking
        was_alive = self.agent_alive.clone()

        damage_dealt = self._bullet_agent_collisions()
        self._bullet_bounds_check()

        # Zone damage
        zone_progress = ((self.frame.float() - ZONE_SHRINK_START) /
                         (ZONE_SHRINK_END - ZONE_SHRINK_START)).clamp(0, 1)
        zone_radius = ZONE_MAX_RADIUS + (ZONE_MIN_RADIUS - ZONE_MAX_RADIUS) * zone_progress
        dist_to_center = ((self.agent_x - ARENA_W / 2) ** 2 +
                          (self.agent_y - ARENA_H / 2) ** 2).sqrt()
        outside_zone = (dist_to_center > zone_radius.unsqueeze(1)) & self.agent_alive
        self.agent_health -= ZONE_DAMAGE_PER_FRAME * outside_zone.float()
        self.agent_alive &= self.agent_health > 0

        self.frame += 1

        # --- Episode logic ---
        rewards = torch.zeros(self.B, self.A, device=self.device)

        # Reward for dealing damage (normalized by max HP)
        rewards += damage_dealt / AGENT_MAX_HP * 0.5

        # Detect newly dead agents (only in active episodes) — fully branchless
        just_died = was_alive & ~self.agent_alive & active_env[:, None]

        # Assign placement to newly dead agents
        self.agent_place = torch.where(
            just_died,
            self.death_rank_counter[:, None].expand_as(self.agent_place),
            self.agent_place,
        )
        place_reward = 1.0 - (self.agent_place - 1).float() / max(self.A - 1, 1)
        rewards = torch.where(just_died, place_reward, rewards)

        deaths_per_env = just_died.sum(dim=1)
        self.death_rank_counter -= deaths_per_env

        # Death loot: drop ammo/medkits at death location (branchless per-agent loop)
        for victim_a in range(self.A):
            died = just_died[:, victim_a]
            # Drop ammo as deposit — find first free slot, write unconditionally
            has_ammo = died & (self.agent_ammo[:, victim_a] > 0)
            free = ~self.deposit_alive  # (B, D)
            can_spawn = free.any(dim=1) & has_ammo
            slot = free.to(torch.int).argmax(dim=1)  # first free slot (or 0 if none)
            # Scatter unconditionally; use can_spawn mask to gate values
            brange = torch.arange(self.B, device=self.device)
            self.deposit_x[brange, slot] = torch.where(can_spawn, self.agent_x[:, victim_a], self.deposit_x[brange, slot])
            self.deposit_y[brange, slot] = torch.where(can_spawn, self.agent_y[:, victim_a], self.deposit_y[brange, slot])
            self.deposit_ammo[brange, slot] = torch.where(can_spawn, self.agent_ammo[:, victim_a], self.deposit_ammo[brange, slot])
            self.deposit_alive[brange, slot] = self.deposit_alive[brange, slot] | can_spawn

            # Drop medkits as health pickup
            has_meds = died & (self.agent_medkits[:, victim_a] > 0)
            free_hp = ~self.health_pickup_alive  # (B, H)
            can_spawn_hp = free_hp.any(dim=1) & has_meds
            slot_hp = free_hp.to(torch.int).argmax(dim=1)
            self.health_pickup_x[brange, slot_hp] = torch.where(can_spawn_hp, self.agent_x[:, victim_a], self.health_pickup_x[brange, slot_hp])
            self.health_pickup_y[brange, slot_hp] = torch.where(can_spawn_hp, self.agent_y[:, victim_a], self.health_pickup_y[brange, slot_hp])
            self.health_pickup_count[brange, slot_hp] = torch.where(can_spawn_hp, self.agent_medkits[:, victim_a], self.health_pickup_count[brange, slot_hp])
            self.health_pickup_alive[brange, slot_hp] = self.health_pickup_alive[brange, slot_hp] | can_spawn_hp

        # Check episode end conditions
        num_alive = self.agent_alive.sum(dim=1)
        newly_done = (num_alive <= 1) & ~self.episode_done

        # Assign place 1 to survivors in newly done envs
        survivor_mask = self.agent_alive & newly_done[:, None]
        self.agent_place = torch.where(
            survivor_mask,
            torch.ones_like(self.agent_place),
            self.agent_place,
        )
        rewards = torch.where(survivor_mask, torch.ones_like(rewards), rewards)

        self.episode_done |= newly_done

        # Timeout
        timed_out = (self.frame >= MAX_EPISODE_FRAMES) & ~self.episode_done
        timeout_survivors = self.agent_alive & timed_out[:, None]
        self.agent_place = torch.where(
            timeout_survivors,
            torch.ones_like(self.agent_place),
            self.agent_place,
        )
        self.episode_done |= timed_out

        self.episode_rewards += rewards

        return rewards, self.episode_done.clone()

    def _recompute_wall_edges(self):
        """Recompute cached wall edge tensors from self.walls (B, W, 4)."""
        self._wall_x1 = self.walls[:, :, 0]
        self._wall_y1 = self.walls[:, :, 1]
        self._wall_x2 = self.walls[:, :, 0] + self.walls[:, :, 2]
        self._wall_y2 = self.walls[:, :, 1] + self.walls[:, :, 3]

    def _randomize_interior_walls(self, mask):
        """Generate random interior walls for masked envs."""
        n_reset = mask.sum().item()
        if n_reset == 0:
            return
        N = self.N_INTERIOR
        margin = WALL_THICKNESS + 30

        # Random wall type: 0=horizontal bar, 1=vertical bar, 2=block
        wall_type = torch.randint(0, 3, (n_reset, N), device=self.device)
        is_h = wall_type == 0
        is_v = wall_type == 1

        rand_a = torch.rand(n_reset, N, device=self.device)
        rand_b = torch.rand(n_reset, N, device=self.device)

        # Scale wall dimensions proportionally to arena size
        _s = min(ARENA_W, ARENA_H) / 1200.0
        # Horizontal: w=100-200 scaled, h=WALL_THICKNESS
        # Vertical:   w=WALL_THICKNESS, h=100-200 scaled
        # Block:      w=h=50-100 scaled
        w = torch.where(is_h, rand_a * (100 * _s) + (100 * _s),
            torch.where(is_v, torch.full_like(rand_a, WALL_THICKNESS),
                         rand_a * (50 * _s) + (50 * _s)))
        h = torch.where(is_h, torch.full_like(rand_b, WALL_THICKNESS),
            torch.where(is_v, rand_b * (100 * _s) + (100 * _s),
                         w))  # block: square

        x = torch.rand(n_reset, N, device=self.device) * (ARENA_W - 2 * margin - w) + margin
        y = torch.rand(n_reset, N, device=self.device) * (ARENA_H - 2 * margin - h) + margin

        interior = torch.stack([x, y, w, h], dim=-1)  # (n_reset, N, 4)
        self.walls[mask, self.N_BORDER:] = interior
        self._recompute_wall_edges()

    def _point_in_any_wall(self, px, py, env_mask, margin=AMMO_PICKUP_RADIUS):
        """Check if points overlap any wall AABB (with margin).
        px/py: (n_reset, D), env_mask: (B,) bool."""
        # (n_reset, D, 1) vs (n_reset, 1, W)
        wx1 = self._wall_x1[env_mask].unsqueeze(1)  # (n_reset, 1, W)
        wy1 = self._wall_y1[env_mask].unsqueeze(1)
        wx2 = self._wall_x2[env_mask].unsqueeze(1)
        wy2 = self._wall_y2[env_mask].unsqueeze(1)
        x = px.unsqueeze(-1)  # (n_reset, D, 1)
        y = py.unsqueeze(-1)
        inside = (
            (x >= (wx1 - margin)) &
            (x <= (wx2 + margin)) &
            (y >= (wy1 - margin)) &
            (y <= (wy2 + margin))
        )  # (n_reset, D, W)
        return inside.any(dim=-1)  # (n_reset, D)

    def _randomize_deposits(self, mask):
        """Place deposits at random positions avoiding walls. Rejection sampling."""
        n_reset = mask.sum().item()
        if n_reset == 0:
            return
        buf = WALL_THICKNESS + 10
        # Generate candidates and reject those inside walls (up to 20 attempts)
        x = torch.rand(n_reset, self.D, device=self.device) * (ARENA_W - 2 * buf) + buf
        y = torch.rand(n_reset, self.D, device=self.device) * (ARENA_H - 2 * buf) + buf
        for _ in range(20):
            bad = self._point_in_any_wall(x, y, mask)
            if not bad.any():
                break
            new_x = torch.rand(n_reset, self.D, device=self.device) * (ARENA_W - 2 * buf) + buf
            new_y = torch.rand(n_reset, self.D, device=self.device) * (ARENA_H - 2 * buf) + buf
            x = torch.where(bad, new_x, x)
            y = torch.where(bad, new_y, y)
        self.deposit_x[mask] = x
        self.deposit_y[mask] = y

    def _process_ammo_pickups(self):
        """Check agent-deposit distance, add ammo, mark collected deposits as dead."""
        # (B, A, 1) vs (B, 1, D)
        dx = self.agent_x.unsqueeze(-1) - self.deposit_x.unsqueeze(1)
        dy = self.agent_y.unsqueeze(-1) - self.deposit_y.unsqueeze(1)
        dist_sq = dx ** 2 + dy ** 2  # (B, A, D)

        pickup = (
            (dist_sq < AMMO_PICKUP_RADIUS ** 2) &
            self.agent_alive.unsqueeze(-1) &
            self.deposit_alive.unsqueeze(1)
        )  # (B, A, D)

        # Any agent picks up this deposit -> (B, D)
        deposit_collected = pickup.any(dim=1)

        # Add ammo to agents (variable amount per deposit)
        ammo_gained = (pickup.float() * self.deposit_ammo.unsqueeze(1)).sum(dim=-1)  # (B, A)
        self.agent_ammo = torch.clamp(self.agent_ammo + ammo_gained, max=AMMO_MAX)

        # Kill collected deposits, start respawn timer
        newly_collected = deposit_collected & self.deposit_alive
        self.deposit_respawn_timer = torch.where(
            newly_collected, torch.full_like(self.deposit_respawn_timer, AMMO_RESPAWN_FRAMES),
            self.deposit_respawn_timer)
        self.deposit_alive = self.deposit_alive & ~deposit_collected

    def _respawn_deposits(self):
        """Tick respawn timers; respawn dead deposits inside current zone. Branchless."""
        dead = ~self.deposit_alive & (self.deposit_respawn_timer > 0)
        self.deposit_respawn_timer -= dead.long()
        ready = ~self.deposit_alive & (self.deposit_respawn_timer <= 0) & dead

        # Compute current zone radius (always — cheap)
        zone_progress = ((self.frame.float() - ZONE_SHRINK_START) /
                         (ZONE_SHRINK_END - ZONE_SHRINK_START)).clamp(0, 1)  # (B,)
        zone_radius = ZONE_MAX_RADIUS + (ZONE_MIN_RADIUS - ZONE_MAX_RADIUS) * zone_progress
        zr = zone_radius * 0.8  # spawn within 80% of zone to avoid edge

        # Random position inside zone circle for ALL slots (masked by ready)
        angle = torch.rand(self.B, self.D, device=self.device) * 2 * 3.14159
        r = torch.sqrt(torch.rand(self.B, self.D, device=self.device)) * zr.unsqueeze(1)
        new_x = (ARENA_W / 2 + torch.cos(angle) * r).clamp(AMMO_PICKUP_RADIUS, ARENA_W - AMMO_PICKUP_RADIUS)
        new_y = (ARENA_H / 2 + torch.sin(angle) * r).clamp(AMMO_PICKUP_RADIUS, ARENA_H - AMMO_PICKUP_RADIUS)

        self.deposit_x = torch.where(ready, new_x, self.deposit_x)
        self.deposit_y = torch.where(ready, new_y, self.deposit_y)
        self.deposit_ammo = torch.where(ready, torch.full_like(self.deposit_ammo, AMMO_PER_PICKUP), self.deposit_ammo)
        self.deposit_alive = self.deposit_alive | ready

    def _randomize_health_pickups(self, mask):
        """Place health pickups at random positions avoiding walls. Rejection sampling."""
        n_reset = mask.sum().item()
        if n_reset == 0:
            return
        buf = WALL_THICKNESS + 10
        x = torch.rand(n_reset, self.H, device=self.device) * (ARENA_W - 2 * buf) + buf
        y = torch.rand(n_reset, self.H, device=self.device) * (ARENA_H - 2 * buf) + buf
        for _ in range(20):
            bad = self._point_in_any_wall(x, y, mask, margin=HEALTH_PICKUP_RADIUS)
            if not bad.any():
                break
            new_x = torch.rand(n_reset, self.H, device=self.device) * (ARENA_W - 2 * buf) + buf
            new_y = torch.rand(n_reset, self.H, device=self.device) * (ARENA_H - 2 * buf) + buf
            x = torch.where(bad, new_x, x)
            y = torch.where(bad, new_y, y)
        self.health_pickup_x[mask] = x
        self.health_pickup_y[mask] = y

    def _process_health_pickups(self):
        """Check agent-health pickup distance, add medkits, mark collected as dead."""
        # (B, A, 1) vs (B, 1, H)
        dx = self.agent_x.unsqueeze(-1) - self.health_pickup_x.unsqueeze(1)
        dy = self.agent_y.unsqueeze(-1) - self.health_pickup_y.unsqueeze(1)
        dist_sq = dx ** 2 + dy ** 2  # (B, A, H)

        pickup = (
            (dist_sq < HEALTH_PICKUP_RADIUS ** 2) &
            self.agent_alive.unsqueeze(-1) &
            self.health_pickup_alive.unsqueeze(1)
        )  # (B, A, H)

        # Any agent picks up this pickup -> (B, H)
        hp_collected = pickup.any(dim=1)

        # Add medkits to agents (variable count per pickup)
        medkits_gained = (pickup.long() * self.health_pickup_count.unsqueeze(1)).sum(dim=-1)  # (B, A)
        self.agent_medkits = torch.clamp(self.agent_medkits + medkits_gained, max=MEDKIT_MAX)

        # Kill collected health pickups (no respawn)
        self.health_pickup_alive = self.health_pickup_alive & ~hp_collected

    def _process_healing(self, heal_action):
        """Process heal channeling. Freezes movement while channeling."""
        wants_heal = heal_action & (self.agent_medkits > 0) & self.agent_alive

        # If wants_heal: freeze movement, increment progress
        self.agent_vx = torch.where(wants_heal, torch.zeros_like(self.agent_vx), self.agent_vx)
        self.agent_vy = torch.where(wants_heal, torch.zeros_like(self.agent_vy), self.agent_vy)
        self.agent_heal_progress = torch.where(
            wants_heal,
            self.agent_heal_progress + 1,
            torch.zeros_like(self.agent_heal_progress),  # cancel if not healing
        )

        # Check if channel complete (branchless — no .any() sync)
        channel_done = self.agent_heal_progress >= HEAL_CHANNEL_FRAMES
        heal_amount = MEDKIT_HEAL_AMOUNT * AGENT_MAX_HP
        self.agent_health = torch.where(
            channel_done,
            torch.clamp(self.agent_health + heal_amount, max=float(AGENT_MAX_HP)),
            self.agent_health,
        )
        self.agent_medkits = torch.where(channel_done, self.agent_medkits - 1, self.agent_medkits)
        self.agent_heal_progress = torch.where(
            channel_done,
            torch.zeros_like(self.agent_heal_progress),
            self.agent_heal_progress,
        )

    def _spawn_bullets(self, fire):
        """Spawn bullets. Loop over A (small, 1-4), vectorized over B. Branchless."""
        can_fire = fire & (self.agent_cooldown == 0) & self.agent_alive & (self.agent_ammo > 0)  # (B, A)
        brange = torch.arange(self.B, device=self.device)

        for a in range(self.A):
            firing = can_fire[:, a]  # (B,)

            # Find first free bullet slot per env
            has_free = (~self.bullet_active).any(dim=1)  # (B,)
            spawn_mask = firing & has_free  # (B,)

            # argmax on int gives first True index (0 if none free — masked away)
            first_free = (~self.bullet_active).to(torch.int).argmax(dim=1)  # (B,)

            d = self.agent_dir[:, a]
            sx = self.agent_x[:, a] + torch.cos(d) * AGENT_RADIUS
            sy = self.agent_y[:, a] + torch.sin(d) * AGENT_RADIUS
            vx = torch.cos(d) * BULLET_SPEED
            vy = torch.sin(d) * BULLET_SPEED

            # Write unconditionally to first_free slot, masked by spawn_mask
            s = first_free
            self.bullet_x[brange, s] = torch.where(spawn_mask, sx, self.bullet_x[brange, s])
            self.bullet_y[brange, s] = torch.where(spawn_mask, sy, self.bullet_y[brange, s])
            self.bullet_spawn_x[brange, s] = torch.where(spawn_mask, sx, self.bullet_spawn_x[brange, s])
            self.bullet_spawn_y[brange, s] = torch.where(spawn_mask, sy, self.bullet_spawn_y[brange, s])
            self.bullet_vx[brange, s] = torch.where(spawn_mask, vx, self.bullet_vx[brange, s])
            self.bullet_vy[brange, s] = torch.where(spawn_mask, vy, self.bullet_vy[brange, s])
            self.bullet_active[brange, s] = self.bullet_active[brange, s] | spawn_mask
            self.bullet_owner[brange, s] = torch.where(spawn_mask, torch.full_like(first_free, a), self.bullet_owner[brange, s])
            self.agent_cooldown[:, a] = torch.where(spawn_mask, torch.full_like(self.agent_cooldown[:, a], FIRE_COOLDOWN), self.agent_cooldown[:, a])
            self.agent_ammo[:, a] = torch.where(spawn_mask.float() > 0, self.agent_ammo[:, a] - 1, self.agent_ammo[:, a])

    def _bullet_wall_collisions(self):
        """Kill bullets that overlap walls. Broadcast (B, M, 1) vs (B, 1, W)."""
        r = BULLET_RADIUS
        wx1 = self._wall_x1[:, None, :]  # (B, 1, W)
        wy1 = self._wall_y1[:, None, :]
        wx2 = self._wall_x2[:, None, :]
        wy2 = self._wall_y2[:, None, :]

        bx = self.bullet_x[:, :, None]  # (B, M, 1)
        by = self.bullet_y[:, :, None]

        closest_x = torch.clamp(bx, min=wx1, max=wx2)
        closest_y = torch.clamp(by, min=wy1, max=wy2)
        dx = bx - closest_x
        dy = by - closest_y
        dist_sq = dx**2 + dy**2

        hit_any_wall = (dist_sq < r**2).any(dim=2)  # (B, M)
        self.bullet_active &= ~hit_any_wall

    def _bullet_agent_collisions(self):
        """Bullet-agent circle-circle collision. Broadcast (B, M, 1) vs (B, 1, A).
        Returns damage_dealt (B, A): total damage this agent's bullets dealt."""
        r_sum = BULLET_RADIUS + AGENT_RADIUS

        # (B, M, 1) vs (B, 1, A)
        dx = self.bullet_x[:, :, None] - self.agent_x[:, None, :]
        dy = self.bullet_y[:, :, None] - self.agent_y[:, None, :]
        dist_sq = dx**2 + dy**2  # (B, M, A)

        # Can't hit own owner
        owner_match = self.bullet_owner[:, :, None] == torch.arange(self.A, device=self.device)[None, None, :]
        hit = (dist_sq < r_sum**2) & self.bullet_active[:, :, None] & self.agent_alive[:, None, :] & ~owner_match

        # Damage agents: sum hits per agent across bullets
        hit_count = hit.sum(dim=1).float()  # (B, A) - how many bullets hit
        self.agent_health -= hit_count * BULLET_DAMAGE
        self.agent_alive &= self.agent_health > 0

        # Credit damage to bullet owners: for each hitting bullet, credit owner
        # hit: (B, M, A) -> any agent hit: (B, M)
        bullet_hit_any = hit.any(dim=2)  # (B, M)
        damage_dealt = torch.zeros(self.B, self.A, device=self.device)
        owners = self.bullet_owner.clamp(0)
        damage_dealt.scatter_add_(1, owners, bullet_hit_any.float() * BULLET_DAMAGE)

        # Deactivate bullets that hit any agent
        self.bullet_active &= ~bullet_hit_any

        return damage_dealt

    def _bullet_bounds_check(self):
        """Kill bullets outside arena."""
        margin = BULLET_RADIUS * 2
        out = (
            (self.bullet_x < -margin) |
            (self.bullet_x > ARENA_W + margin) |
            (self.bullet_y < -margin) |
            (self.bullet_y > ARENA_H + margin)
        )
        self.bullet_active &= ~out

    def _resolve_wall_collisions(self, mask=None, iterations: int = 3):
        """Circle-AABB collision, fully vectorized over (B, A, W).
        If mask is provided, only resolve for those envs."""
        for _ in range(iterations):
            self._resolve_wall_collisions_once(mask)

    def _resolve_wall_collisions_once(self, mask=None):
        """Single pass of circle-AABB collision resolution."""
        r = AGENT_RADIUS

        if mask is not None:
            wx1 = self._wall_x1[mask, None, :]  # (n, 1, W)
            wy1 = self._wall_y1[mask, None, :]
            wx2 = self._wall_x2[mask, None, :]
            wy2 = self._wall_y2[mask, None, :]
            cx = self.agent_x[mask, :, None]  # (n, A, 1)
            cy = self.agent_y[mask, :, None]
        else:
            wx1 = self._wall_x1[:, None, :]  # (B, 1, W)
            wy1 = self._wall_y1[:, None, :]
            wx2 = self._wall_x2[:, None, :]
            wy2 = self._wall_y2[:, None, :]
            cx = self.agent_x[:, :, None]  # (B, A, 1)
            cy = self.agent_y[:, :, None]

        closest_x = torch.clamp(cx, min=wx1, max=wx2)
        closest_y = torch.clamp(cy, min=wy1, max=wy2)

        dx = cx - closest_x
        dy = cy - closest_y
        dist_sq = dx**2 + dy**2

        # Case 1: center OUTSIDE the AABB
        dist = torch.sqrt(torch.clamp(dist_sq, min=1e-12))
        outside = dist_sq > 1e-8
        penetration_outside = r - dist
        overlap_outside = (penetration_outside > 0) & outside

        nx_out = torch.where(outside, dx / dist, torch.zeros_like(dx))
        ny_out = torch.where(outside, dy / dist, torch.zeros_like(dy))

        push_x_out = torch.where(overlap_outside, nx_out * penetration_outside, torch.zeros_like(dx))
        push_y_out = torch.where(overlap_outside, ny_out * penetration_outside, torch.zeros_like(dy))

        # Case 2: center INSIDE the AABB
        inside = ~outside

        d_left = cx - wx1
        d_right = wx2 - cx
        d_top = cy - wy1
        d_bottom = wy2 - cy

        min_d = torch.minimum(torch.minimum(d_left, d_right), torch.minimum(d_top, d_bottom))

        push_x_in = torch.zeros_like(cx)
        push_y_in = torch.zeros_like(cy)

        is_left = inside & (min_d == d_left)
        is_right = inside & (min_d == d_right) & ~is_left
        is_top = inside & (min_d == d_top) & ~is_left & ~is_right
        is_bottom = inside & ~is_left & ~is_right & ~is_top

        push_x_in = torch.where(is_left, -(d_left + r), push_x_in)
        push_x_in = torch.where(is_right, d_right + r, push_x_in)
        push_y_in = torch.where(is_top, -(d_top + r), push_y_in)
        push_y_in = torch.where(is_bottom, d_bottom + r, push_y_in)

        push_x = push_x_out + push_x_in
        push_y = push_y_out + push_y_in

        total_push_x = push_x.sum(dim=2)
        total_push_y = push_y.sum(dim=2)

        if mask is not None:
            self.agent_x[mask] += total_push_x
            self.agent_y[mask] += total_push_y

            vx = self.agent_vx[mask]
            vy = self.agent_vy[mask]
            self.agent_vx[mask] = torch.where(
                (total_push_x > 0) & (vx < 0), torch.zeros_like(vx),
                torch.where((total_push_x < 0) & (vx > 0), torch.zeros_like(vx), vx))
            self.agent_vy[mask] = torch.where(
                (total_push_y > 0) & (vy < 0), torch.zeros_like(vy),
                torch.where((total_push_y < 0) & (vy > 0), torch.zeros_like(vy), vy))
        else:
            self.agent_x += total_push_x
            self.agent_y += total_push_y

            self.agent_vx = torch.where(
                (total_push_x > 0) & (self.agent_vx < 0), torch.zeros_like(self.agent_vx),
                torch.where(
                    (total_push_x < 0) & (self.agent_vx > 0), torch.zeros_like(self.agent_vx),
                    self.agent_vx))
            self.agent_vy = torch.where(
                (total_push_y > 0) & (self.agent_vy < 0), torch.zeros_like(self.agent_vy),
                torch.where(
                    (total_push_y < 0) & (self.agent_vy > 0), torch.zeros_like(self.agent_vy),
                    self.agent_vy))

    def get_state(self, env_idx: int = 0) -> dict:
        """Extract one env's state as numpy arrays for the renderer."""
        active = self.bullet_active[env_idx]

        # Find winner ID for display
        num_alive = self.agent_alive[env_idx].sum().item()
        winner_id = -1
        if self.episode_done[env_idx].item() and num_alive >= 1:
            alive_indices = torch.where(self.agent_alive[env_idx])[0]
            if len(alive_indices) == 1:
                winner_id = alive_indices[0].item()

        frame = int(self.frame[env_idx].item())
        zp = max(0.0, min(1.0, (frame - ZONE_SHRINK_START) / (ZONE_SHRINK_END - ZONE_SHRINK_START)))
        zone_radius = ZONE_MAX_RADIUS + (ZONE_MIN_RADIUS - ZONE_MAX_RADIUS) * zp

        return {
            "agent_x": self.agent_x[env_idx].cpu().numpy(),
            "agent_y": self.agent_y[env_idx].cpu().numpy(),
            "agent_dir": self.agent_dir[env_idx].cpu().numpy(),
            "agent_health": self.agent_health[env_idx].cpu().numpy(),
            "agent_alive": self.agent_alive[env_idx].cpu().numpy(),
            "walls": self.walls[env_idx].cpu().numpy(),
            "frame": int(self.frame[env_idx].item()),
            "num_agents": self.A,
            "bullet_x": self.bullet_x[env_idx][active].cpu().numpy(),
            "bullet_y": self.bullet_y[env_idx][active].cpu().numpy(),
            "bullet_vx": self.bullet_vx[env_idx][active].cpu().numpy(),
            "bullet_vy": self.bullet_vy[env_idx][active].cpu().numpy(),
            "bullet_spawn_x": self.bullet_spawn_x[env_idx][active].cpu().numpy(),
            "bullet_spawn_y": self.bullet_spawn_y[env_idx][active].cpu().numpy(),
            "episode_done": bool(self.episode_done[env_idx].item()),
            "winner_id": winner_id,
            "deposit_x": self.deposit_x[env_idx].cpu().numpy(),
            "deposit_y": self.deposit_y[env_idx].cpu().numpy(),
            "deposit_alive": self.deposit_alive[env_idx].cpu().numpy(),
            "agent_ammo": self.agent_ammo[env_idx].cpu().numpy(),
            "health_pickup_x": self.health_pickup_x[env_idx].cpu().numpy(),
            "health_pickup_y": self.health_pickup_y[env_idx].cpu().numpy(),
            "health_pickup_alive": self.health_pickup_alive[env_idx].cpu().numpy(),
            "agent_medkits": self.agent_medkits[env_idx].cpu().numpy(),
            "agent_heal_progress": self.agent_heal_progress[env_idx].cpu().numpy(),
            "zone_radius": zone_radius,
        }
