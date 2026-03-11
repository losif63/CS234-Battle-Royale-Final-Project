"""Watch and play modes for the battle royale PPO agent."""

import math
import os

import torch

from .sim import BatchedBRSim
from .obs import ObservationBuilder
from .network import (
    AttentionActorCritic, pack_actor_obs,
    _actions_to_sim, _apply_action_masks, _sample_actions, _greedy_actions,
    MAX_AGENTS, LSTM_HIDDEN, N_ENTITIES, ENTITY_DIM, SELF_DIM,
    NUM_DISCRETE_ACTIONS,
)
from .config import FIRE_COOLDOWN


def watch(checkpoint: str | None = None, run_name: str | None = None):
    import pygame
    from .renderer import Renderer
    from .train import ACTION_REPEAT, RUNS_DIR, _find_latest_checkpoint, _load_checkpoint

    device = torch.device("cpu")

    search_dir = os.path.join(RUNS_DIR, run_name, "checkpoints") if run_name else None
    network = AttentionActorCritic()
    ckpt_path = checkpoint or _find_latest_checkpoint(search_dir)
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("No checkpoint found. Train first.")
        return

    sd, ckpt = _load_checkpoint(ckpt_path, device)
    network.load_state_dict(sd, strict=False)
    network.eval()
    print(f"Loaded {ckpt_path} (update {ckpt.get('update_count', '?')})")

    pygame.init()
    sim = BatchedBRSim(num_envs=1, max_agents=MAX_AGENTS, device="cpu")
    obs_builder = ObservationBuilder(sim)
    renderer = Renderer(instant_restart=True)

    running = True
    ep = 0
    while running:
        ep += 1
        sim.reset()
        # Reset LSTM state each episode
        lstm_hx = torch.zeros(MAX_AGENTS, LSTM_HIDDEN)
        lstm_cx = torch.zeros(MAX_AGENTS, LSTM_HIDDEN)

        # Hot-reload checkpoint between episodes
        latest = _find_latest_checkpoint(search_dir)
        if latest and latest != ckpt_path:
            try:
                sd, _ = _load_checkpoint(latest, device)
                network.load_state_dict(sd, strict=False)
                network.eval()
                ckpt_path = latest
                print(f"Hot-reloaded {ckpt_path}")
            except Exception:
                pass

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        sim.reset()
                        lstm_hx.zero_()
                        lstm_cx.zero_()

            if not running:
                break

            # All agents use the same policy, sampled with temperature
            with torch.no_grad():
                actor_obs = obs_builder.actor_obs()
                self_feat, entities, entity_mask = pack_actor_obs(actor_obs)
                sf = self_feat.reshape(MAX_AGENTS, -1)
                ent = entities.reshape(MAX_AGENTS, N_ENTITIES, ENTITY_DIM)
                emask = entity_mask.reshape(MAX_AGENTS, N_ENTITIES)
                logits, alpha, beta_param, (lstm_hx, lstm_cx) = network.forward_actor(
                    sf, ent, emask, hx=lstm_hx, cx=lstm_cx)
                logits = _apply_action_masks(logits, sf)
                temp = 0.5
                logits = tuple(l / temp for l in logits)
                # Temperature for Beta: alpha_t = 1 + (alpha-1)/temp
                alpha_t = 1.0 + (alpha - 1.0) / temp
                beta_t = 1.0 + (beta_param - 1.0) / temp
                disc, cont = _sample_actions(logits, alpha_t, beta_t)
                mx, my, aim, fire, heal = _actions_to_sim(disc, cont, sim.agent_dir[0])
                move_x = mx.unsqueeze(0)
                move_y = my.unsqueeze(0)
                aim_angle = aim.unsqueeze(0)
                fire = fire.unsqueeze(0)
                heal = heal.unsqueeze(0)

            # Repeat action for ACTION_REPEAT frames
            pause_done = False
            for _rep in range(ACTION_REPEAT):
                rewards, done = sim.step(move_x, move_y, aim_angle, fire, heal)
                state = sim.get_state(0)
                pause_done = renderer.render(state)
                if pause_done:
                    break

            if pause_done:
                break  # next episode

    pygame.quit()


def play(checkpoint: str | None = None, run_name: str | None = None,
         first_person: bool = False):
    """Play as Agent 0 (human) against AI agents."""
    import sys
    if sys.platform.startswith("linux"):
        os.environ.setdefault("SDL_AUDIODRIVER", "pulseaudio")
    import pygame
    from .renderer import Renderer
    from .main import InputHandler
    from .audio import AudioManager
    from .train import ACTION_REPEAT, RUNS_DIR, _find_latest_checkpoint, _load_checkpoint

    device = torch.device("cpu")
    AI_AGENTS = MAX_AGENTS - 1  # number of AI opponents
    search_dir = os.path.join(RUNS_DIR, run_name, "checkpoints") if run_name else None

    network = AttentionActorCritic()
    ckpt_path = checkpoint or _find_latest_checkpoint(search_dir)
    if not ckpt_path or not os.path.exists(ckpt_path):
        print("No checkpoint found. Train first.")
        return

    sd, ckpt = _load_checkpoint(ckpt_path, device)
    network.load_state_dict(sd, strict=False)
    network.eval()
    print(f"Loaded {ckpt_path} (update {ckpt.get('update_count', '?')})")

    pygame.init()
    sim = BatchedBRSim(num_envs=1, max_agents=MAX_AGENTS, device="cpu")
    obs_builder = ObservationBuilder(sim)
    renderer = Renderer(camera_follow=0 if first_person else None)
    inp = InputHandler()
    audio = AudioManager()

    if inp.joystick:
        print(f"Controller: {inp.joystick.get_name()}")
        print("Controls: L-stick = move, R-stick = aim, R-trigger = fire, L-bumper = heal")
    else:
        print("Controls: WASD = move, Mouse = aim, Click/Space = fire, Q = heal")

    running = True
    ep = 0
    while running:
        ep += 1
        sim.reset()
        lstm_hx = torch.zeros(AI_AGENTS, LSTM_HIDDEN)
        lstm_cx = torch.zeros(AI_AGENTS, LSTM_HIDDEN)

        # Hot-reload checkpoint between episodes
        latest = _find_latest_checkpoint(search_dir)
        if latest and latest != ckpt_path:
            try:
                sd, _ = _load_checkpoint(latest, device)
                network.load_state_dict(sd, strict=False)
                network.eval()
                ckpt_path = latest
                print(f"Hot-reloaded {ckpt_path}")
            except Exception:
                pass

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        sim.reset()
                        lstm_hx.zero_()
                        lstm_cx.zero_()

            if not running:
                break

            # --- Human input (Agent 0) ---
            human_mx, human_my, human_aim, human_fire, human_heal = inp.read()

            if inp.joystick is None and math.isnan(human_aim):
                mouse_x, mouse_y = renderer.window_to_game(*pygame.mouse.get_pos())
                agent0_x = float(sim.agent_x[0, 0].item())
                agent0_y = float(sim.agent_y[0, 0].item())
                dx = mouse_x - agent0_x
                dy = mouse_y - agent0_y
                if dx ** 2 + dy ** 2 > 1.0:
                    human_aim = math.atan2(dy, dx)

            # --- AI input (Agents 1..A-1) ---
            with torch.no_grad():
                actor_obs = obs_builder.actor_obs()
                self_feat, entities, entity_mask = pack_actor_obs(actor_obs)
                sf_ai = self_feat[0, 1:].reshape(AI_AGENTS, -1)
                ent_ai = entities[0, 1:].reshape(AI_AGENTS, N_ENTITIES, ENTITY_DIM)
                emask_ai = entity_mask[0, 1:].reshape(AI_AGENTS, N_ENTITIES)
                logits_ai, alpha_ai, beta_ai, (lstm_hx, lstm_cx) = network.forward_actor(
                    sf_ai, ent_ai, emask_ai, hx=lstm_hx, cx=lstm_cx)
                logits_ai = _apply_action_masks(logits_ai, sf_ai)
                ai_disc, ai_cont = _greedy_actions(logits_ai, alpha_ai, beta_ai)
                ai_mx, ai_my, ai_aim, ai_fire, ai_heal = _actions_to_sim(
                    ai_disc, ai_cont, sim.agent_dir[0, 1:])

            # Build action tensors (1, A)
            move_x = torch.cat([torch.tensor([human_mx]), ai_mx]).unsqueeze(0)
            move_y = torch.cat([torch.tensor([human_my]), ai_my]).unsqueeze(0)
            aim_angle = torch.cat([torch.tensor([human_aim]), ai_aim]).unsqueeze(0)
            fire = torch.cat([torch.tensor([human_fire]), ai_fire]).unsqueeze(0).bool()
            heal = torch.cat([torch.tensor([human_heal]), ai_heal]).unsqueeze(0).bool()

            prev_health = sim.agent_health[0].clone()

            rewards, done = sim.step(move_x, move_y, aim_angle, fire, heal)

            # Distance-based sound effects (listener = agent 0)
            lx = sim.agent_x[0, 0].item()
            ly = sim.agent_y[0, 0].item()
            for a in range(sim.A):
                if sim.agent_cooldown[0, a].item() == FIRE_COOLDOWN:
                    audio.play_shot(sim.agent_x[0, a].item(), sim.agent_y[0, a].item(), lx, ly)
                if sim.agent_health[0, a].item() < prev_health[a].item():
                    audio.play_hit(sim.agent_x[0, a].item(), sim.agent_y[0, a].item(), lx, ly)

            state = sim.get_state(0)
            pause_done = renderer.render(state)

            # FP mode: auto-restart when player dies
            if first_person and not sim.agent_alive[0, 0]:
                break

            if pause_done:
                break

    pygame.quit()


def debug_lidar():
    """Debug visualization: one agent you control with WASD, lidar rays drawn."""
    import pygame
    from .renderer import Renderer

    pygame.init()
    sim = BatchedBRSim(num_envs=1, max_agents=MAX_AGENTS, device="cpu")
    obs_builder = ObservationBuilder(sim)
    renderer = Renderer()
    small_font = pygame.font.SysFont("monospace", 11)

    R = obs_builder.num_lidar_rays
    lidar_range = obs_builder.lidar_range

    print(f"Debug lidar: {R} rays, range {lidar_range}")
    print("Controls: WASD = move, Mouse = aim direction, R = reset")

    running = True
    sim.reset()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    sim.reset()

        if not running:
            break

        # Human controls agent 0
        keys = pygame.key.get_pressed()
        human_mx, human_my = 0.0, 0.0
        if keys[pygame.K_a]: human_mx -= 1.0
        if keys[pygame.K_d]: human_mx += 1.0
        if keys[pygame.K_w]: human_my -= 1.0
        if keys[pygame.K_s]: human_my += 1.0

        mouse_x, mouse_y = renderer.window_to_game(*pygame.mouse.get_pos())
        ax = float(sim.agent_x[0, 0])
        ay = float(sim.agent_y[0, 0])
        human_aim = math.atan2(mouse_y - ay, mouse_x - ax)

        # Other agents stand still
        move_x = torch.zeros(1, MAX_AGENTS)
        move_y = torch.zeros(1, MAX_AGENTS)
        aim_angle = torch.zeros(1, MAX_AGENTS)
        fire = torch.zeros(1, MAX_AGENTS, dtype=torch.bool)
        move_x[0, 0] = human_mx
        move_y[0, 0] = human_my
        aim_angle[0, 0] = human_aim

        sim.step(move_x, move_y, aim_angle, fire)

        # Get lidar BEFORE rendering
        with torch.no_grad():
            actor_obs = obs_builder.actor_obs()
            lidar = actor_obs["lidar"][0, 0]  # (R,) — normalized wall distances

        # Render base scene
        state = sim.get_state(0)
        renderer.render(state)

        # Draw lidar rays on top
        ax = float(sim.agent_x[0, 0])
        ay = float(sim.agent_y[0, 0])
        agent_dir = float(sim.agent_dir[0, 0])

        for r in range(R):
            angle = agent_dir + float(obs_builder.ray_offsets[r])
            norm_dist = float(lidar[r])
            pixel_dist = norm_dist * lidar_range

            end_x = ax + math.cos(angle) * pixel_dist
            end_y = ay + math.sin(angle) * pixel_dist

            # Color: green (far) -> red (close)
            t = 1.0 - norm_dist
            cr = int(min(255, 255 * t * 2))
            cg = int(min(255, 255 * (1 - t) * 2))
            color = (cr, cg, 0)

            pygame.draw.line(renderer.screen, color,
                             (int(ax), int(ay)), (int(end_x), int(end_y)), 1)
            # Small dot at hit point
            pygame.draw.circle(renderer.screen, (255, 255, 255),
                               (int(end_x), int(end_y)), 3)
            # Value label
            label = small_font.render(f"{norm_dist:.2f}", True, (255, 255, 255))
            renderer.screen.blit(label, (int(end_x) + 4, int(end_y) - 6))

        pygame.display.flip()
        renderer.clock.tick(60)

    pygame.quit()
