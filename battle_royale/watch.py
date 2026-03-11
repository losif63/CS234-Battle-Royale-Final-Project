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
    NUM_DISCRETE_ACTIONS, MAX_VISIBLE_BULLETS,
)
from .config import (
    FIRE_COOLDOWN, NUM_AMMO_DEPOSITS, NUM_HEALTH_PICKUPS, AGENT_COLORS,
)


def _build_attention_lines(agent_idx, attn_weights, entity_mask, sim, obs_builder, color, max_w):
    """Build a list of attention line dicts for the renderer.

    Args:
        agent_idx: which agent we're visualizing (index in [0, A))
        attn_weights: (N,) tensor — attention weight per entity slot
        entity_mask: (N,) bool tensor — which slots are valid
        sim: BatchedBRSim (env 0 is used)
        obs_builder: ObservationBuilder
        color: (r, g, b) — agent's color for the lines
        max_w: global max attention weight for normalization

    Returns:
        list of dicts: {"from": (x,y), "to": (x,y), "color": (r,g,b), "alpha": int}
    """
    lines = []
    K = MAX_VISIBLE_BULLETS
    D = NUM_AMMO_DEPOSITS
    H = NUM_HEALTH_PICKUPS
    A_other = MAX_AGENTS - 1

    # Source position
    src_x = float(sim.agent_x[0, agent_idx])
    src_y = float(sim.agent_y[0, agent_idx])

    def _add_line(slot_idx, tx, ty):
        w = attn_weights[slot_idx].item()
        alpha = int(128 * w / max_w) if max_w > 1e-8 else 0
        lines.append({"from": (src_x, src_y), "to": (tx, ty),
                       "color": color, "alpha": alpha})

    offset = 0

    # Bullets (slots 0..K-1)
    bullet_positions = _get_bullet_world_positions(agent_idx, sim, obs_builder)
    for i in range(K):
        if entity_mask[offset + i]:
            _add_line(offset + i, *bullet_positions[i])
    offset += K

    # Ammo deposits (slots K..K+D-1)
    for i in range(D):
        if entity_mask[offset + i]:
            _add_line(offset + i, float(sim.deposit_x[0, i]), float(sim.deposit_y[0, i]))
    offset += D

    # Health pickups (slots K+D..K+D+H-1)
    for i in range(H):
        if entity_mask[offset + i]:
            _add_line(offset + i, float(sim.health_pickup_x[0, i]), float(sim.health_pickup_y[0, i]))
    offset += H

    # Other agents (slots K+D+H..)
    other_indices = obs_builder.other_idx[agent_idx]
    for i in range(A_other):
        if entity_mask[offset + i]:
            oid = int(other_indices[i])
            _add_line(offset + i, float(sim.agent_x[0, oid]), float(sim.agent_y[0, oid]))

    return lines


def _get_bullet_world_positions(agent_idx, sim, obs_builder):
    """Re-derive the nearest-K bullet world positions for a given agent.

    Returns list of (x, y) tuples, length K. Invalid slots get (0, 0).
    """
    K = obs_builder.max_visible_bullets
    fov_sq = obs_builder.fov_radius ** 2

    ax = float(sim.agent_x[0, agent_idx])
    ay = float(sim.agent_y[0, agent_idx])

    # Gather all active bullets
    active = sim.bullet_active[0]  # (M,)
    bx = sim.bullet_x[0]  # (M,)
    by = sim.bullet_y[0]  # (M,)

    dist_sq = (bx - ax) ** 2 + (by - ay) ** 2
    in_fov = (dist_sq < fov_sq) & active
    dist_sq_masked = dist_sq.clone()
    dist_sq_masked[~in_fov] = 1e8

    _, topk_idx = dist_sq_masked.topk(K, largest=False)

    positions = []
    for i in range(K):
        bidx = int(topk_idx[i])
        if in_fov[bidx]:
            positions.append((float(bx[bidx]), float(by[bidx])))
        else:
            positions.append((0.0, 0.0))
    return positions


def watch(checkpoint: str | None = None, run_name: str | None = None,
          opponent: str | None = None, show_attention: bool = False,
          first_person: bool = False):
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
    print(f"Agent 0: {ckpt_path} (update {ckpt.get('update_count', '?')})")

    # Separate opponent network if --opponent is set; otherwise all share one policy
    AI_AGENTS = MAX_AGENTS - 1
    opp_network = None
    if opponent:
        opp_network = AttentionActorCritic()
        opp_sd, opp_ckpt = _load_checkpoint(opponent, device)
        opp_network.load_state_dict(opp_sd, strict=False)
        opp_network.eval()
        print(f"Opponents: {opponent} (update {opp_ckpt.get('update_count', '?')})")

    pygame.init()
    sim = BatchedBRSim(num_envs=1, max_agents=MAX_AGENTS, device="cpu")
    obs_builder = ObservationBuilder(sim)
    renderer = Renderer(instant_restart=not first_person, show_attention=show_attention,
                        camera_follow=0 if first_person else None,
                        fov_mask=False)

    running = True
    ep = 0
    while running:
        ep += 1
        sim.reset()
        # Reset LSTM state each episode
        if opp_network:
            lstm_hx_0 = torch.zeros(1, LSTM_HIDDEN)
            lstm_cx_0 = torch.zeros(1, LSTM_HIDDEN)
            lstm_hx_opp = torch.zeros(AI_AGENTS, LSTM_HIDDEN)
            lstm_cx_opp = torch.zeros(AI_AGENTS, LSTM_HIDDEN)
        else:
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
                        if opp_network:
                            lstm_hx_0.zero_(); lstm_cx_0.zero_()
                            lstm_hx_opp.zero_(); lstm_cx_opp.zero_()
                        else:
                            lstm_hx.zero_(); lstm_cx.zero_()

            if not running:
                break

            with torch.no_grad():
                actor_obs = obs_builder.actor_obs()
                self_feat, entities, entity_mask = pack_actor_obs(actor_obs)
                temp = 0.5

                if opp_network:
                    # Agent 0: learner network
                    sf0 = self_feat[0, 0:1]
                    ent0 = entities[0, 0:1]
                    emask0 = entity_mask[0, 0:1]
                    if show_attention:
                        log0, a0, b0, (lstm_hx_0, lstm_cx_0), attn_w_0 = network.forward_actor(
                            sf0, ent0, emask0, hx=lstm_hx_0, cx=lstm_cx_0,
                            return_attention=True)
                    else:
                        log0, a0, b0, (lstm_hx_0, lstm_cx_0) = network.forward_actor(
                            sf0, ent0, emask0, hx=lstm_hx_0, cx=lstm_cx_0)
                    log0 = _apply_action_masks(log0, sf0)
                    log0 = tuple(l / temp for l in log0)
                    a0_t = 1.0 + (a0 - 1.0) / temp
                    b0_t = 1.0 + (b0 - 1.0) / temp
                    d0, c0 = _sample_actions(log0, a0_t, b0_t)
                    mx0, my0, aim0, fire0, heal0 = _actions_to_sim(d0, c0, sim.agent_dir[0, 0:1])

                    # Agents 1..A-1: opponent network
                    sf_opp = self_feat[0, 1:]
                    ent_opp = entities[0, 1:]
                    emask_opp = entity_mask[0, 1:]
                    if show_attention:
                        log_opp, a_opp, b_opp, (lstm_hx_opp, lstm_cx_opp), attn_w_opp = opp_network.forward_actor(
                            sf_opp, ent_opp, emask_opp, hx=lstm_hx_opp, cx=lstm_cx_opp,
                            return_attention=True)
                    else:
                        log_opp, a_opp, b_opp, (lstm_hx_opp, lstm_cx_opp) = opp_network.forward_actor(
                            sf_opp, ent_opp, emask_opp, hx=lstm_hx_opp, cx=lstm_cx_opp)
                    log_opp = _apply_action_masks(log_opp, sf_opp)
                    log_opp = tuple(l / temp for l in log_opp)
                    a_opp_t = 1.0 + (a_opp - 1.0) / temp
                    b_opp_t = 1.0 + (b_opp - 1.0) / temp
                    d_opp, c_opp = _sample_actions(log_opp, a_opp_t, b_opp_t)
                    mx_opp, my_opp, aim_opp, fire_opp, heal_opp = _actions_to_sim(
                        d_opp, c_opp, sim.agent_dir[0, 1:])

                    move_x = torch.cat([mx0, mx_opp]).unsqueeze(0)
                    move_y = torch.cat([my0, my_opp]).unsqueeze(0)
                    aim_angle = torch.cat([aim0, aim_opp]).unsqueeze(0)
                    fire = torch.cat([fire0, fire_opp]).unsqueeze(0)
                    heal = torch.cat([heal0, heal_opp]).unsqueeze(0)
                else:
                    # All agents share the same policy
                    sf = self_feat.reshape(MAX_AGENTS, -1)
                    ent = entities.reshape(MAX_AGENTS, N_ENTITIES, ENTITY_DIM)
                    emask = entity_mask.reshape(MAX_AGENTS, N_ENTITIES)
                    if show_attention:
                        logits, alpha, beta_param, (lstm_hx, lstm_cx), attn_w = network.forward_actor(
                            sf, ent, emask, hx=lstm_hx, cx=lstm_cx,
                            return_attention=True)
                    else:
                        logits, alpha, beta_param, (lstm_hx, lstm_cx) = network.forward_actor(
                            sf, ent, emask, hx=lstm_hx, cx=lstm_cx)
                    logits = _apply_action_masks(logits, sf)
                    logits = tuple(l / temp for l in logits)
                    alpha_t = 1.0 + (alpha - 1.0) / temp
                    beta_t = 1.0 + (beta_param - 1.0) / temp
                    disc, cont = _sample_actions(logits, alpha_t, beta_t)
                    mx, my, aim, fire, heal = _actions_to_sim(disc, cont, sim.agent_dir[0])
                    move_x = mx.unsqueeze(0)
                    move_y = my.unsqueeze(0)
                    aim_angle = aim.unsqueeze(0)
                    fire = fire.unsqueeze(0)
                    heal = heal.unsqueeze(0)

            # Build attention lines for ALL alive agents
            if show_attention:
                all_lines = []
                if opp_network:
                    # Compute global max across all agents
                    weights_list = []
                    masks_list = []
                    if sim.agent_alive[0, 0]:
                        weights_list.append(attn_w_0[0])
                        masks_list.append(emask0[0])
                    for a in range(1, MAX_AGENTS):
                        if sim.agent_alive[0, a]:
                            weights_list.append(attn_w_opp[a - 1])
                            masks_list.append(emask_opp[a - 1])
                    global_max = max((w[m].max().item() for w, m in zip(weights_list, masks_list)
                                     if m.any()), default=0.0)

                    if sim.agent_alive[0, 0]:
                        color = AGENT_COLORS[0 % len(AGENT_COLORS)]
                        all_lines.extend(_build_attention_lines(
                            0, attn_w_0[0], emask0[0], sim, obs_builder, color, global_max))
                    for a in range(1, MAX_AGENTS):
                        if sim.agent_alive[0, a]:
                            color = AGENT_COLORS[a % len(AGENT_COLORS)]
                            all_lines.extend(_build_attention_lines(
                                a, attn_w_opp[a - 1], emask_opp[a - 1], sim, obs_builder, color, global_max))
                else:
                    # Compute global max across all agents
                    global_max = max((attn_w[a][emask[a]].max().item()
                                     for a in range(MAX_AGENTS)
                                     if sim.agent_alive[0, a] and emask[a].any()), default=0.0)
                    for a in range(MAX_AGENTS):
                        if sim.agent_alive[0, a]:
                            color = AGENT_COLORS[a % len(AGENT_COLORS)]
                            all_lines.extend(_build_attention_lines(
                                a, attn_w[a], emask[a], sim, obs_builder, color, global_max))
                renderer.attention_lines = all_lines if all_lines else None

            # Compute win probabilities from critic
            with torch.no_grad():
                sf_all = self_feat[0]  # (A, SELF_DIM)
                ent_all = entities[0]  # (A, N, ENTITY_DIM)
                emask_all = entity_mask[0]  # (A, N)
                if opp_network:
                    hx_all = torch.cat([lstm_hx_0, lstm_hx_opp], dim=0)
                else:
                    hx_all = lstm_hx
                values = network.forward_critic(sf_all, ent_all, emask_all, hx=hx_all)
                alive_mask = sim.agent_alive[0].bool()
                # Softmax over alive agents, dead get 0
                v = values.clone()
                v[~alive_mask] = float('-inf')
                win_probs = torch.softmax(v, dim=0)
                win_probs[~alive_mask] = 0.0
                win_probs_list = win_probs.tolist()

            # Repeat action for ACTION_REPEAT frames
            pause_done = False
            for _rep in range(ACTION_REPEAT):
                rewards, done = sim.step(move_x, move_y, aim_angle, fire, heal)
                state = sim.get_state(0)
                state["win_probs"] = win_probs_list
                pause_done = renderer.render(state)
                if pause_done:
                    break

            # FP mode: auto-restart when agent 0 dies
            if first_person and not sim.agent_alive[0, 0]:
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

            # Compute win probabilities from critic
            with torch.no_grad():
                sf_all = self_feat[0]  # (A, SELF_DIM)
                ent_all = entities[0]  # (A, N, ENTITY_DIM)
                emask_all = entity_mask[0]  # (A, N)
                hx_crit = torch.cat([torch.zeros(1, LSTM_HIDDEN), lstm_hx], dim=0)
                values = network.forward_critic(sf_all, ent_all, emask_all, hx=hx_crit)
                alive_mask = sim.agent_alive[0].bool()
                v = values.clone()
                v[~alive_mask] = float('-inf')
                win_probs = torch.softmax(v, dim=0)
                win_probs[~alive_mask] = 0.0

            state = sim.get_state(0)
            state["win_probs"] = win_probs.tolist()
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
