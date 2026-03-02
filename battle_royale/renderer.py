"""Pygame renderer for one env from the batched sim."""

import math
import pygame
from .config import (
    ARENA_W, ARENA_H, AGENT_RADIUS, AIM_LINE_LENGTH, FPS,
    BULLET_RADIUS, COLOR_BG, COLOR_WALL, COLOR_AIM,
    COLOR_HP_BAR, COLOR_HP_BG, COLOR_HUD_TEXT, COLOR_BULLET, AGENT_MAX_HP,
    AGENT_COLORS, WIN_PAUSE_FRAMES, WIN_FLASH_FRAMES, COLOR_AMMO_DEPOSIT,
    ENTITY_FOV_RADIUS, COLOR_HEALTH_PICKUP, HEAL_CHANNEL_FRAMES,
)



class Renderer:
    def __init__(self, instant_restart=False, camera_follow=None):
        pygame.init()
        self.camera_follow = camera_follow

        if camera_follow is not None:
            self.vp_size = int(ENTITY_FOV_RADIUS * 2)
            self.screen = pygame.display.set_mode(
                (self.vp_size, self.vp_size), pygame.RESIZABLE)
            pygame.display.set_caption("Battle Royale (First Person)")
            self.viewport = pygame.Surface((self.vp_size, self.vp_size))
            self._build_fov_mask()
            self._vp_wx = 0.0
            self._vp_wy = 0.0
        else:
            self.screen = pygame.display.set_mode(
                (ARENA_W, ARENA_H), pygame.RESIZABLE)
            pygame.display.set_caption("Battle Royale")

        self.game_surf = pygame.Surface((ARENA_W, ARENA_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)
        self.win_font = pygame.font.SysFont("monospace", 48, bold=True)
        self.sub_font = pygame.font.SysFont("monospace", 20)
        self.small_font = pygame.font.SysFont("monospace", 12)
        self.instant_restart = instant_restart
        self.win_pause_counter = 0
        self.trail_surf = pygame.Surface((ARENA_W, ARENA_H), pygame.SRCALPHA)

    def _build_fov_mask(self):
        """Pre-build circular FOV mask (opaque black outside, transparent inside)."""
        sz = self.vp_size
        r = int(ENTITY_FOV_RADIUS)
        self.fov_mask = pygame.Surface((sz, sz), pygame.SRCALPHA)
        self.fov_mask.fill((0, 0, 0, 255))
        # Punch transparent hole via alpha subtraction
        hole = pygame.Surface((sz, sz), pygame.SRCALPHA)
        pygame.draw.circle(hole, (0, 0, 0, 255), (sz // 2, sz // 2), r)
        self.fov_mask.blit(hole, (0, 0), special_flags=pygame.BLEND_RGBA_SUB)

    def _extract_viewport(self, state):
        """Extract viewport centered on followed agent, apply FOV mask."""
        idx = self.camera_follow
        sz = self.vp_size

        if state["agent_alive"][idx]:
            px = float(state["agent_x"][idx])
            py = float(state["agent_y"][idx])
            self._last_follow_pos = (px, py)
        else:
            px, py = getattr(self, '_last_follow_pos',
                             (ARENA_W / 2, ARENA_H / 2))

        vx = px - sz / 2
        vy = py - sz / 2
        self._vp_wx = vx
        self._vp_wy = vy

        vp = self.viewport
        vp.fill(COLOR_BG)

        # Clip source rect to game_surf bounds
        src_l = max(0, int(vx))
        src_t = max(0, int(vy))
        src_r = min(ARENA_W, int(vx + sz))
        src_b = min(ARENA_H, int(vy + sz))
        dst_x = src_l - int(vx)
        dst_y = src_t - int(vy)
        if src_r > src_l and src_b > src_t:
            vp.blit(self.game_surf, (dst_x, dst_y),
                    (src_l, src_t, src_r - src_l, src_b - src_t))

        # Apply circular FOV mask
        vp.blit(self.fov_mask, (0, 0))
        return vp

    def window_to_game(self, wx, wy):
        """Convert window coordinates to game coordinates."""
        sw, sh = self.screen.get_size()
        if self.camera_follow is not None:
            sz = self.vp_size
            return (wx * sz / sw + self._vp_wx,
                    wy * sz / sh + self._vp_wy)
        return wx * ARENA_W / sw, wy * ARENA_H / sh

    def render(self, state: dict) -> bool:
        """Render one frame. Returns True if win pause is finished."""
        surf = self.game_surf
        surf.fill(COLOR_BG)

        # Draw walls
        for i in range(state["walls"].shape[0]):
            x, y, w, h = state["walls"][i]
            pygame.draw.rect(surf, COLOR_WALL, (int(x), int(y), int(w), int(h)))

        # Draw zone boundary
        if "zone_radius" in state:
            zr = int(state["zone_radius"])
            if zr < max(ARENA_W, ARENA_H):
                pygame.draw.circle(surf, (255, 80, 80), (ARENA_W // 2, ARENA_H // 2), zr, 2)

        # Draw ammo deposits as diamonds (only alive ones)
        if "deposit_x" in state:
            dep_x = state["deposit_x"]
            dep_y = state["deposit_y"]
            dep_alive = state.get("deposit_alive")
            for i in range(len(dep_x)):
                if dep_alive is not None and not dep_alive[i]:
                    continue
                dx, dy = int(dep_x[i]), int(dep_y[i])
                sz = 10
                points = [(dx, dy - sz), (dx + sz, dy), (dx, dy + sz), (dx - sz, dy)]
                pygame.draw.polygon(surf, COLOR_AMMO_DEPOSIT, points)
                label = self.small_font.render("A", True, (255, 255, 255))
                lw, lh = label.get_size()
                surf.blit(label, (dx - lw // 2, dy - lh // 2))

        # Draw health pickups as plus/cross signs
        if "health_pickup_x" in state:
            hp_x = state["health_pickup_x"]
            hp_y = state["health_pickup_y"]
            hp_alive = state.get("health_pickup_alive")
            for i in range(len(hp_x)):
                if hp_alive is not None and not hp_alive[i]:
                    continue
                hx, hy = int(hp_x[i]), int(hp_y[i])
                sz = 10
                # Draw plus sign
                pygame.draw.line(surf, COLOR_HEALTH_PICKUP, (hx - sz, hy), (hx + sz, hy), 4)
                pygame.draw.line(surf, COLOR_HEALTH_PICKUP, (hx, hy - sz), (hx, hy + sz), 4)
                label = self.small_font.render("H", True, (255, 255, 255))
                lw, lh = label.get_size()
                surf.blit(label, (hx - lw // 2, hy + sz + 2))

        # Draw agents
        for i in range(state["num_agents"]):
            if not state["agent_alive"][i]:
                continue
            ax = state["agent_x"][i]
            ay = state["agent_y"][i]
            d = state["agent_dir"][i]
            hp = state["agent_health"][i]

            cx, cy = int(ax), int(ay)

            color = AGENT_COLORS[i % len(AGENT_COLORS)]

            # FOV circle (hidden in FP mode — gives away positions through fog)
            if self.camera_follow is None:
                fov_color = (*color[:3], 30) if len(color) >= 3 else (100, 100, 100, 30)
                fov_surf = pygame.Surface((int(ENTITY_FOV_RADIUS * 2), int(ENTITY_FOV_RADIUS * 2)), pygame.SRCALPHA)
                pygame.draw.circle(fov_surf, fov_color,
                                   (int(ENTITY_FOV_RADIUS), int(ENTITY_FOV_RADIUS)), int(ENTITY_FOV_RADIUS))
                surf.blit(fov_surf, (cx - int(ENTITY_FOV_RADIUS), cy - int(ENTITY_FOV_RADIUS)))

            # Body — per-agent color from palette
            pygame.draw.circle(surf, color, (cx, cy), AGENT_RADIUS)

            # Agent ID label
            label = self.font.render(str(i), True, (255, 255, 255))
            lw, lh = label.get_size()
            surf.blit(label, (cx - lw // 2, cy - lh // 2))

            # Aim line
            end_x = ax + math.cos(d) * AIM_LINE_LENGTH
            end_y = ay + math.sin(d) * AIM_LINE_LENGTH
            pygame.draw.line(surf, COLOR_AIM, (cx, cy), (int(end_x), int(end_y)), 2)

            # HP bar above agent
            bar_w = 30
            bar_h = 4
            bar_x = cx - bar_w // 2
            bar_y = cy - AGENT_RADIUS - 10
            hp_frac = hp / AGENT_MAX_HP
            pygame.draw.rect(surf, COLOR_HP_BG, (bar_x, bar_y, bar_w, bar_h))
            pygame.draw.rect(surf, COLOR_HP_BAR, (bar_x, bar_y, int(bar_w * hp_frac), bar_h))

            # Ammo + medkit count above HP bar
            if "agent_ammo" in state:
                ammo = int(state["agent_ammo"][i])
                medkits = int(state["agent_medkits"][i]) if "agent_medkits" in state else 0
                hud_label = self.small_font.render(f"{ammo} {medkits}H", True, COLOR_AMMO_DEPOSIT)
                aw, ah = hud_label.get_size()
                surf.blit(hud_label, (cx - aw // 2, bar_y - ah - 1))

            # Heal progress bar (below agent) when channeling
            if "agent_heal_progress" in state:
                heal_prog = int(state["agent_heal_progress"][i])
                if heal_prog > 0:
                    prog_w = 30
                    prog_h = 3
                    prog_x = cx - prog_w // 2
                    prog_y = cy + AGENT_RADIUS + 4
                    prog_frac = heal_prog / HEAL_CHANNEL_FRAMES
                    pygame.draw.rect(surf, (60, 60, 60), (prog_x, prog_y, prog_w, prog_h))
                    pygame.draw.rect(surf, COLOR_HEALTH_PICKUP, (prog_x, prog_y, int(prog_w * prog_frac), prog_h))

        # Draw bullets with fading trail from spawn point
        bx = state["bullet_x"]
        by = state["bullet_y"]
        bvx = state["bullet_vx"]
        bvy = state["bullet_vy"]
        bsx = state.get("bullet_spawn_x", bx)
        bsy = state.get("bullet_spawn_y", by)
        N_SEGS = 8
        self.trail_surf.fill((0, 0, 0, 0))
        for i in range(len(bx)):
            x, y = float(bx[i]), float(by[i])
            sx, sy = float(bsx[i]), float(bsy[i])

            # Fading trail: N segments from spawn (faded) to bullet (bright)
            for s in range(N_SEGS):
                t0 = s / N_SEGS
                t1 = (s + 1) / N_SEGS
                x0 = sx + (x - sx) * t0
                y0 = sy + (y - sy) * t0
                x1 = sx + (x - sx) * t1
                y1 = sy + (y - sy) * t1
                alpha = int(160 * t1)
                pygame.draw.line(self.trail_surf, (180, 180, 180, alpha),
                                 (int(x0), int(y0)), (int(x1), int(y1)), 1)

            # Bullet head (solid triangle)
            vx, vy = float(bvx[i]), float(bvy[i])
            angle = math.atan2(vy, vx)
            size = BULLET_RADIUS * 1.5
            tip = (x + size * math.cos(angle), y + size * math.sin(angle))
            perp = angle + math.pi / 2
            back_x = x - size * 0.5 * math.cos(angle)
            back_y = y - size * 0.5 * math.sin(angle)
            base1 = (back_x + size * 0.3 * math.cos(perp), back_y + size * 0.3 * math.sin(perp))
            base2 = (back_x - size * 0.3 * math.cos(perp), back_y - size * 0.3 * math.sin(perp))
            pygame.draw.polygon(surf, COLOR_BULLET, [tip, base1, base2])

        surf.blit(self.trail_surf, (0, 0))

        # --- Viewport extraction (first-person) or full arena ---
        if self.camera_follow is not None:
            target = self._extract_viewport(state)
            tw, th = self.vp_size, self.vp_size
        else:
            target = surf
            tw, th = ARENA_W, ARENA_H

        # HUD
        alive_count = sum(1 for i in range(state["num_agents"]) if state["agent_alive"][i])
        hud_text = self.font.render(
            f"Frame: {state['frame']}  Alive: {alive_count}/{state['num_agents']}",
            True, COLOR_HUD_TEXT,
        )
        target.blit(hud_text, (tw - 320, 25))

        # Win overlay
        pause_done = False
        if state.get("episode_done", False):
            if self.instant_restart:
                pause_done = True
            else:
                if self.win_pause_counter == 0:
                    self.win_pause_counter = WIN_PAUSE_FRAMES

                winner_id = state.get("winner_id", -1)
                frames_elapsed = WIN_PAUSE_FRAMES - self.win_pause_counter

                # Flash phase: solid winner color pulsing on/off
                if winner_id >= 0 and frames_elapsed < WIN_FLASH_FRAMES:
                    color = AGENT_COLORS[winner_id % len(AGENT_COLORS)]
                    # 8-frame cycle: solid for 5, off for 3
                    if frames_elapsed % 8 < 5:
                        target.fill(color)
                        win_text = self.win_font.render(
                            f"Agent {winner_id} Wins!", True, (255, 255, 255))
                        wtw, wth = win_text.get_size()
                        target.blit(win_text,
                            (tw // 2 - wtw // 2, th // 2 - wth // 2))
                else:
                    # Normal overlay after flash
                    overlay = pygame.Surface((tw, th), pygame.SRCALPHA)
                    overlay.fill((0, 0, 0, 140))
                    target.blit(overlay, (0, 0))

                    if winner_id >= 0:
                        win_text = self.win_font.render(
                            f"Agent {winner_id} Wins!", True, (255, 255, 100))
                    else:
                        win_text = self.win_font.render("Draw!", True, (255, 255, 100))
                    wtw, wth = win_text.get_size()
                    target.blit(win_text,
                        (tw // 2 - wtw // 2, th // 2 - wth // 2 - 20))

                    sub_text = self.sub_font.render(
                        "Next round starting...", True, (200, 200, 200))
                    stw, sth = sub_text.get_size()
                    target.blit(sub_text,
                        (tw // 2 - stw // 2, th // 2 + wth // 2))

                self.win_pause_counter -= 1
                if self.win_pause_counter <= 0:
                    self.win_pause_counter = 0
                    pause_done = True
        else:
            self.win_pause_counter = 0

        # Scale to window and display
        pygame.transform.scale(target, self.screen.get_size(), self.screen)
        pygame.display.flip()
        self.clock.tick(FPS)
        return pause_done
