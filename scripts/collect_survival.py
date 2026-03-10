"""Quick targeted collection of survival time metric only, then replot."""
import os, sys, json, time, torch, numpy as np, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from battle_royale.sim import BatchedBRSim
from battle_royale.obs import ObservationBuilder
from battle_royale.network import (
    AttentionActorCritic, pack_actor_obs, _actions_to_sim, _apply_action_masks, _sample_actions,
    MAX_AGENTS, LSTM_HIDDEN, N_ENTITIES, ENTITY_DIM, SELF_DIM, NUM_DISCRETE_ACTIONS,
)
from battle_royale.config import FIRE_COOLDOWN, MAX_EPISODE_FRAMES
from battle_royale.train import _load_checkpoint, ACTION_REPEAT

device = torch.device('cuda')
B = 200
A = MAX_AGENTS
GAMES_PER = 20
CKPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "battle_royale", "runs", "apex_rp", "checkpoints")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "behavioral_plots")

# Load existing metrics
with open(os.path.join(OUTPUT_DIR, "metrics_summary.json")) as f:
    summaries = json.load(f)

# Discover checkpoints matching existing summaries
files = [f for f in os.listdir(CKPT_DIR)
         if f.startswith("br_ppo_") and f.endswith(".pt") and f != "br_ppo_final.pt"]
files_with_num = [(f, int(f.replace("br_ppo_", "").replace(".pt", ""))) for f in files]
files_with_num.sort(key=lambda x: x[1])
step = max(1, len(files_with_num) // 80)
sampled = files_with_num[::step]
if files_with_num[0] not in sampled:
    sampled.insert(0, files_with_num[0])
if files_with_num[-1] not in sampled:
    sampled.append(files_with_num[-1])

update_nums = sorted([n for _, n in sampled if str(n) in summaries])
print(f"{len(update_nums)} checkpoints", flush=True)

# Load networks
networks = {}
for i, u in enumerate(update_nums):
    net = AttentionActorCritic().to(device)
    sd, _ = _load_checkpoint(os.path.join(CKPT_DIR, f"br_ppo_{u}.pt"), "cpu")
    net.load_state_dict(sd, strict=False)
    net.eval()
    networks[u] = net
    if (i + 1) % 30 == 0:
        print(f"  Loaded {i+1}/{len(update_nums)}", flush=True)
print(f"  Loaded all {len(update_nums)} networks", flush=True)

# Build game list
game_list = []
for u in update_nums:
    game_list.extend([u] * GAMES_PER)
random.shuffle(game_list)
while len(game_list) % B != 0:
    game_list.append(game_list[-1])

num_batches = len(game_list) // B
print(f"Running {len(update_nums)*GAMES_PER} games ({num_batches} batches)", flush=True)

sim = BatchedBRSim(num_envs=B, max_agents=A, device="cuda")
obs_builder = ObservationBuilder(sim)

all_survival = defaultdict(list)
t0 = time.time()

for bi in range(num_batches):
    assignments = game_list[bi * B:(bi + 1) * B]
    unique_updates = list(set(assignments))
    update_to_idx = {u: i for i, u in enumerate(unique_updates)}
    net_list = [networks[u] for u in unique_updates]
    num_nets = len(net_list)

    net_assignment = torch.zeros(B, A, dtype=torch.long, device=device)
    for b, u in enumerate(assignments):
        net_assignment[b, :] = update_to_idx[u]

    sim.reset()
    hx = torch.zeros(B, A, LSTM_HIDDEN, device=device)
    cx = torch.zeros(B, A, LSTM_HIDDEN, device=device)

    death_frame_sum = torch.zeros(B, device=device)
    death_frame_count = torch.zeros(B, device=device)

    for stp in range(5000):
        if sim.episode_done.all():
            break
        with torch.no_grad():
            ao = obs_builder.actor_obs()
            sf, ent, emask = pack_actor_obs(ao)
            BA = B * A
            sf_f = sf.reshape(BA, SELF_DIM)
            ent_f = ent.reshape(BA, N_ENTITIES, ENTITY_DIM)
            em_f = emask.reshape(BA, N_ENTITIES)
            hx_f = hx.reshape(BA, LSTM_HIDDEN)
            cx_f = cx.reshape(BA, LSTM_HIDDEN)
            nf = net_assignment.reshape(BA)
            alive_f = sim.agent_alive.reshape(BA)
            active = ~sim.episode_done
            active_f = active.unsqueeze(1).expand(B, A).reshape(BA) & alive_f

            ad = torch.zeros(BA, NUM_DISCRETE_ACTIONS, dtype=torch.long, device=device)
            ac = torch.zeros(BA, device=device)
            nh = hx_f.clone()
            nc = cx_f.clone()

            for ni in range(num_nets):
                m = (nf == ni) & active_f
                if not m.any():
                    continue
                idx = m.nonzero(as_tuple=True)[0]
                lo, al, be, (ho, co) = net_list[ni].forward_actor(
                    sf_f[idx], ent_f[idx], em_f[idx], hx=hx_f[idx], cx=cx_f[idx])
                lo = _apply_action_masks(lo, sf_f[idx])
                d, c = _sample_actions(lo, al, be)
                ad[idx] = d
                ac[idx] = c
                nh[idx] = ho
                nc[idx] = co

            hx = nh.reshape(B, A, LSTM_HIDDEN)
            cx = nc.reshape(B, A, LSTM_HIDDEN)
            mx, my, aim, fire, heal = _actions_to_sim(
                ad.reshape(B, A, NUM_DISCRETE_ACTIONS), ac.reshape(B, A), sim.agent_dir)

        for _ in range(ACTION_REPEAT):
            pa = sim.agent_alive.clone()
            _, done = sim.step(
                mx * sim.agent_alive.float(), my * sim.agent_alive.float(),
                aim, fire & sim.agent_alive, heal & sim.agent_alive)
            died = pa & ~sim.agent_alive
            valid = died & active.unsqueeze(1)
            ff = sim.frame.unsqueeze(1).expand(B, A).float()
            death_frame_sum += (ff * valid.float()).sum(1)
            death_frame_count += valid.sum(1).float()
            if done.all():
                break

        dead = ~sim.agent_alive
        hx = hx * (~dead).unsqueeze(-1).float()
        cx = cx * (~dead).unsqueeze(-1).float()

    for b in range(B):
        u = assignments[b]
        dc = death_frame_count[b].item()
        if dc > 0:
            all_survival[u].append(death_frame_sum[b].item() / dc)

    el = time.time() - t0
    eta = el / (bi + 1) * (num_batches - bi - 1)
    print(f"  Batch {bi+1}/{num_batches} | elapsed {el:.0f}s | ETA {eta:.0f}s", flush=True)

# Merge into summaries
for u in update_nums:
    k = str(u)
    if k in summaries and all_survival[u]:
        summaries[k]["mean_survival_time"] = round(float(np.mean(all_survival[u])), 0)
    else:
        summaries[k]["mean_survival_time"] = 0

with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w") as f:
    json.dump(summaries, f, indent=2)
print("Saved updated metrics", flush=True)
