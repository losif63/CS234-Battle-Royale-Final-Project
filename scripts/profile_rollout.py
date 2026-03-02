"""Profile one rollout iteration to find bottlenecks."""

import time
import torch

from battle_royale.sim import BatchedBRSim
from battle_royale.obs import ObservationBuilder
from battle_royale.train import (
    AttentionActorCritic, pack_actor_obs, _actions_to_sim, _sample_actions,
    _apply_action_masks,
    NUM_ENVS, MAX_AGENTS, STEPS_PER_ROLLOUT, ACTION_REPEAT,
    SELF_DIM, N_ENTITIES, ENTITY_DIM, LSTM_HIDDEN, NUM_DISCRETE_ACTIONS,
)

device = torch.device("cuda")
B, A, T = NUM_ENVS, MAX_AGENTS, STEPS_PER_ROLLOUT

print(f"B={B}, A={A}, T={T}, ACTION_REPEAT={ACTION_REPEAT}")

# Setup
sim = BatchedBRSim(num_envs=B, max_agents=A, device="cuda")
obs_builder = ObservationBuilder(sim)
network = AttentionActorCritic().to(device)
network = torch.compile(network)
opp_network = AttentionActorCritic().to(device)
opp_network = torch.compile(opp_network)
opp_network.eval()

learner_hx = torch.zeros(B, LSTM_HIDDEN, device=device)
learner_cx = torch.zeros(B, LSTM_HIDDEN, device=device)
opp_hx = torch.zeros(B * (A - 1), LSTM_HIDDEN, device=device)
opp_cx = torch.zeros(B * (A - 1), LSTM_HIDDEN, device=device)

# Warmup (let torch.compile trace)
print("Warming up torch.compile...")
for _ in range(3):
    obs = obs_builder.actor_obs()
    sf, ent, emask = pack_actor_obs(obs)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        network.get_action_and_value(sf[:, 0], ent[:, 0], emask[:, 0], hx=learner_hx, cx=learner_cx)
        logits, alpha, beta, (o_hx_w, o_cx_w) = opp_network.forward_actor(
            sf[:, 1:].reshape(B * (A - 1), SELF_DIM),
            ent[:, 1:].reshape(B * (A - 1), N_ENTITIES, ENTITY_DIM),
            emask[:, 1:].reshape(B * (A - 1), N_ENTITIES),
            hx=opp_hx, cx=opp_cx)
        logits = _apply_action_masks(logits, sf[:, 1:].reshape(B * (A - 1), SELF_DIM))
        o_disc, o_cont = _sample_actions(logits, alpha, beta)
    # Build dummy actions for sim step
    all_disc = torch.zeros(B, A, NUM_DISCRETE_ACTIONS, dtype=torch.long, device=device)
    all_cont = torch.zeros(B, A, device=device)
    mx, my, aim, fire, heal = _actions_to_sim(all_disc, all_cont, sim.agent_dir)
    sim.step(mx, my, aim, fire, heal)

torch.cuda.synchronize()
print("Warmup done.\n")

# Profile one full rollout step (T steps)
t_obs_total = 0.0
t_pack_total = 0.0
t_learner_total = 0.0
t_opp_total = 0.0
t_action_cvt_total = 0.0
t_sim_total = 0.0
t_bookkeep_total = 0.0

def sync():
    torch.cuda.synchronize()
    return time.perf_counter()

overall_start = sync()

for t in range(T):
    # 1. Observation building
    t0 = sync()
    actor_obs = obs_builder.actor_obs()
    t1 = sync()

    # 2. Pack obs
    self_feat, entities, entity_mask = pack_actor_obs(actor_obs)
    sf_l = self_feat[:, 0]
    ent_l = entities[:, 0]
    emask_l = entity_mask[:, 0]
    sf_o = self_feat[:, 1:].reshape(B * (A - 1), SELF_DIM)
    ent_o = entities[:, 1:].reshape(B * (A - 1), N_ENTITIES, ENTITY_DIM)
    emask_o = entity_mask[:, 1:].reshape(B * (A - 1), N_ENTITIES)
    t2 = sync()

    # 3. Learner network
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        l_disc, l_cont, l_lp, _, _, l_val, l_hx, l_cx = network.get_action_and_value(
            sf_l, ent_l, emask_l, hx=learner_hx, cx=learner_cx)
    t3 = sync()

    # 4. Opponent network
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        o_logits, o_alpha, o_beta, (o_hx_out, o_cx_out) = opp_network.forward_actor(
            sf_o, ent_o, emask_o, hx=opp_hx, cx=opp_cx)
        o_logits = _apply_action_masks(o_logits, sf_o)
        o_disc, o_cont = _sample_actions(o_logits, o_alpha, o_beta)
    t4 = sync()

    # 5. Action conversion
    all_disc = torch.empty(B, A, NUM_DISCRETE_ACTIONS, dtype=torch.long, device=device)
    all_disc[:, 0] = l_disc
    all_disc[:, 1:] = o_disc.reshape(B, A - 1, NUM_DISCRETE_ACTIONS)
    all_cont = torch.empty(B, A, device=device)
    all_cont[:, 0] = l_cont
    all_cont[:, 1:] = o_cont.reshape(B, A - 1)
    mx, my, aim, fire, heal = _actions_to_sim(all_disc, all_cont, sim.agent_dir)
    t5 = sync()

    # 6. Sim steps (ACTION_REPEAT)
    for _rep in range(ACTION_REPEAT):
        cur_alive = sim.agent_alive.clone()
        move_x = mx * cur_alive
        move_y = my * cur_alive
        fire_bool = fire & cur_alive
        heal_bool = heal & cur_alive
        rewards, episode_done = sim.step(move_x, move_y, aim, fire_bool, heal_bool)
        sim.reset(mask=episode_done)
    t6 = sync()

    # 7. Bookkeeping (LSTM update, buffer store)
    learner_hx = l_hx.detach()
    learner_cx = l_cx.detach()
    opp_hx = o_hx_out.detach()
    opp_cx = o_cx_out.detach()
    t7 = sync()

    t_obs_total += t1 - t0
    t_pack_total += t2 - t1
    t_learner_total += t3 - t2
    t_opp_total += t4 - t3
    t_action_cvt_total += t5 - t4
    t_sim_total += t6 - t5
    t_bookkeep_total += t7 - t6

overall_end = sync()
overall = overall_end - overall_start

print(f"=== Rollout profile ({T} steps, {ACTION_REPEAT} action_repeat) ===")
print(f"{'Component':<20} {'Total (s)':>10} {'Per step (ms)':>14} {'% of total':>10}")
print("-" * 58)
for name, val in [
    ("obs_build", t_obs_total),
    ("pack_obs", t_pack_total),
    ("learner_net", t_learner_total),
    ("opponent_net", t_opp_total),
    ("action_convert", t_action_cvt_total),
    ("sim_step", t_sim_total),
    ("bookkeeping", t_bookkeep_total),
]:
    print(f"{name:<20} {val:>10.3f} {val/T*1000:>13.1f} {val/overall*100:>9.1f}%")
print("-" * 58)
print(f"{'TOTAL':<20} {overall:>10.3f} {overall/T*1000:>13.1f} {'100.0':>9}%")

# Also profile individual obs components
print(f"\n=== Obs builder breakdown (1 call) ===")
torch.cuda.synchronize()

t0 = sync()
_ = obs_builder._compute_lidar()
t1 = sync()
_ = obs_builder._compute_agent_features()
t2 = sync()
_ = obs_builder._compute_bullet_features()
t3 = sync()
_ = obs_builder._compute_deposit_features()
t4 = sync()
_ = obs_builder._compute_health_pickup_features()
t5 = sync()

print(f"  lidar:          {(t1-t0)*1000:.1f} ms")
print(f"  agent_features: {(t2-t1)*1000:.1f} ms")
print(f"  bullet_features:{(t3-t2)*1000:.1f} ms")
print(f"  deposit_feat:   {(t4-t3)*1000:.1f} ms")
print(f"  health_feat:    {(t5-t4)*1000:.1f} ms")
print(f"  total obs:      {(t5-t0)*1000:.1f} ms")

# Profile sim.step breakdown
print(f"\n=== sim.step breakdown (1 call) ===")
mx_t = torch.zeros(B, A, device=device)
my_t = torch.zeros(B, A, device=device)
aim_t = torch.zeros(B, A, device=device)
fire_t = torch.zeros(B, A, dtype=torch.bool, device=device)
heal_t = torch.zeros(B, A, dtype=torch.bool, device=device)

# We can't easily break down sim.step without modifying it,
# so just time a few calls
times = []
for _ in range(10):
    t0 = sync()
    sim.step(mx_t, my_t, aim_t, fire_t, heal_t)
    t1 = sync()
    times.append(t1 - t0)
    sim.reset(mask=torch.zeros(B, dtype=torch.bool, device=device))

avg = sum(times) / len(times)
print(f"  avg sim.step: {avg*1000:.1f} ms  (over 10 calls)")
print(f"  per rollout ({ACTION_REPEAT} repeats x {T} steps = {ACTION_REPEAT*T} calls): {avg*ACTION_REPEAT*T:.3f} s")
