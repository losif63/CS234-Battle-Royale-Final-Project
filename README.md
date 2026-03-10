# Battle Royale RL

A 10-player free-for-all battle royale game with PPO-trained agents. The entire game simulation runs as batched PyTorch tensor operations on GPU, enabling 30,000 parallel environments during training. Agents learn to aim, shoot, dodge, gather resources, heal, and navigate a shrinking safe zone — all from self-play with no human demonstrations.

## Game Mechanics

### Arena

- **Size:** 4000 x 3000 pixels
- **Walls:** 4 border walls + 50 randomized interior walls per environment (20px thick)
- **Field of view:** Each agent can only see entities within a 550px radius

### Agents

- **Count:** 10 per match
- **Health:** 100 HP (4 bullet hits to kill)
- **Speed:** 5.0 px/frame with 0.6 friction
- **Radius:** 15 px
- **Starting ammo:** 5

### Bullets

- **Damage:** 25 HP per hit
- **Speed:** 20 px/frame
- **Radius:** 6 px
- **Fire cooldown:** 6 frames (0.1s)
- **Max active bullets:** 80 per environment
- **Movement penalty:** 0.4x speed while fire cooldown is active

### Ammo Deposits

- **Count:** 12 per map
- **Pickup radius:** 25 px
- **Ammo per pickup:** 10 bullets
- **Max carried ammo:** 50
- **Respawn time:** 15 seconds (900 frames)

### Health Pickups

- **Count:** 15 per map
- **Pickup radius:** 25 px
- **Heal amount:** 50% of max HP (50 HP)
- **Max carried medkits:** 3
- **Heal channel time:** 1 second (60 frames) — healing is interruptible

### Shrinking Zone

- **Max radius:** 50% of arena diagonal (~2500 px)
- **Min radius:** 6% of arena diagonal (~300 px)
- **Shrink window:** Frames 600–3600 (10s–60s at 60 FPS)
- **Damage outside zone:** 0.5 HP/frame (~200 frames to kill from full HP)

### Episodes

- **Max length:** 4800 frames (80 seconds at 60 FPS)
- **End condition:** One agent remaining or max frames reached

## Architecture

### Simulator (`battle_royale/sim.py`)

Fully batched PyTorch simulator — all game state is stored as `(B, A)` tensors where `B` is the number of parallel environments and `A` is agents per environment. Physics, collision detection, bullet tracing, pickup logic, zone damage, and death tracking all operate on the full batch simultaneously with no Python loops over environments.

### Observation Space (`battle_royale/obs.py`)

The observation is a structured dictionary packed into three tensors for the network:

**Self features** (24-dim):
| Feature | Dim | Description |
|---------|-----|-------------|
| Position | 2 | `(x/W, y/H)` normalized |
| Velocity | 2 | `(vx, vy)` / max speed |
| Health | 1 | HP / max HP |
| Cooldown | 1 | fire cooldown / max cooldown |
| Ammo | 1 | ammo / max ammo |
| Medkits | 1 | medkits / max medkits |
| Heal progress | 1 | channel progress / channel time |
| Zone distance | 1 | signed distance to zone edge / diagonal |
| Lidar | 12 | wall distances across 12 evenly-spaced rays (normalized by range) |
| Global | 2 | fraction of agents alive, frame / max frames |

**Entities** (46 entities, 12-dim each):
Each entity has a 4-dim type one-hot `[bullet, ammo, health, agent]` + 8 feature dims (zero-padded for shorter types):

| Entity type | Count | Features |
|-------------|-------|----------|
| Bullets (nearest) | 10 | relative dx, dy, vx, vy (normalized) |
| Ammo deposits | 12 | relative dx, dy (normalized) |
| Health pickups | 15 | relative dx, dy (normalized) |
| Other agents | 9 | dx, dy, vx, vy, health, cooldown, ammo, direction |

**Entity mask**: Boolean tensor marking which entity slots are occupied (for attention masking).

### Neural Network (`battle_royale/network.py`)

**AttentionActorCritic** — shared encoder with separate actor/critic heads:

```
Self features (24) ──> Linear(24, 64) + ReLU ──> query
                                                     \
Entities (46 x 12) ──> Linear(12, 64) + ReLU ──> key/value ──> Cross-Attention (4 heads, 64-dim)
                                                                         |
                                                            Concat [self_embed, attn_out] (128)
                                                                         |
                                                              MLP(128 → 128 → 128)
                                                                         |
                                                              LSTMCell(128, 128) ──────────┐
                                                                         |                  |
                                                            Skip connection (reactive + hx) |
                                                                         |                  |
                                                              Actor MLP(128 → 128)          |
                                                                    /    |    \              |
                                                  move_x(3) move_y(3) fire(2) heal(2)       |
                                                          + Beta rotation head              |
                                                                                            |
                                                                    Critic: MLP(128+128 → 128 → 1)
                                                                    (encoder output + LSTM hidden)
```

- **Cross-attention** lets the agent selectively attend to relevant entities (nearby enemies, bullets, pickups)
- **LSTMCell** provides temporal memory across frames within an episode
- **Skip connection** from pre-LSTM features preserves reactive information
- **Orthogonal initialization** with gain=sqrt(2) for hidden layers, gain=0.01 for action heads

### Action Space

**Discrete actions** (4 categorical heads):
| Head | Options | Description |
|------|---------|-------------|
| move_x | 3 | left / none / right |
| move_y | 3 | up / none / down |
| fire | 2 | no / yes |
| heal | 2 | no / yes |

**Continuous action** (1 Beta distribution):
| Head | Range | Description |
|------|-------|-------------|
| rotation | Beta(α, β) → [0,1] → [-π, π] | Aim direction delta from current facing |

**Action masking:**
- Fire is masked when ammo = 0
- Heal is masked when health = max or medkits = 0

### Reward Structure

| Event | Reward |
|-------|--------|
| Damage dealt | `+damage / max_HP * 0.5` per step (encourages combat) |
| Death placement | `1.0 - (place - 1) / (A - 1)`: 1st = +1.0, last = 0.0 |
| Last survivor | +1.0 bonus |

## Training (`battle_royale/train.py`)

### PPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Parallel environments | 30,000 |
| Agents per env | 10 |
| Rollout length | 32 steps |
| Action repeat | 3 frames per decision |
| Mini-batch size | 131,072 transitions |
| PPO epochs | 3 |
| Discount (γ) | 0.999 |
| GAE (λ) | 0.95 |
| Clip epsilon | 0.2 |
| Learning rate | 3e-4 (linear annealing) |
| Value loss coefficient | 0.5 |
| Entropy coefficient | 0.003 (base) / 0.03 (heal head) |
| Max gradient norm | 0.5 |

### Self-Play with Policy Pool

Training uses **asymmetric self-play**: agent 0 is the learner, agents 1-9 are opponents sampled from a policy pool.

- Pool stores up to 20 past snapshots of the learner
- New snapshot added every 50 updates
- 20% of games are played against a fixed **anchor** (the initial random policy, ELO 1000)
- **ELO tracking**: learner and pool members have ELO ratings updated after each rollout based on learner win rate

### Performance Optimizations

- `torch.compile` applied to simulator, observation builder, and network
- Mixed precision training (`float16` autocast + GradScaler)
- GPU-side metric accumulation (no CPU sync during rollouts)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for CUDA memory

### Checkpoints

Saved every 100 updates to `battle_royale/runs/<run_name>/checkpoints/br_ppo_<N>.pt`. Each checkpoint contains:
- Network weights
- Optimizer state
- Policy pool snapshots and ELO ratings
- Update count

TensorBoard logs are written to `battle_royale/runs/<run_name>/`.

## Usage

### Install dependencies

```bash
uv sync
```

### Train

```bash
# Start new training run
uv run python -m battle_royale.train --run my_run --updates 5000

# Resume from checkpoint
uv run python -m battle_royale.train --run my_run --resume battle_royale/runs/my_run/checkpoints/br_ppo_1000.pt

# Resume with fresh optimizer (keep weights only)
uv run python -m battle_royale.train --run my_run --resume <path> --resume-weights-only
```

### Watch trained agents

```bash
# Auto-loads latest checkpoint
uv run python -m battle_royale.train --watch

# From a specific run
uv run python -m battle_royale.train --watch --run my_run

# From a specific checkpoint
uv run python -m battle_royale.train --watch --checkpoint path/to/checkpoint.pt
```

Hot-reloads the latest checkpoint between episodes, so you can watch while training.

### Play against the AI

```bash
# Top-down view
uv run python -m battle_royale.train --play

# First-person camera (FOV fog of war, auto-restart on death)
uv run python -m battle_royale.train --play --fp
```

**Keyboard + Mouse:**
| Action | Control |
|--------|---------|
| Move | WASD |
| Aim | Mouse position |
| Fire | Left click or Space |
| Heal | Q |

**Controller:**
| Action | Control |
|--------|---------|
| Move | Left stick |
| Aim | Right stick |
| Fire | Right trigger |
| Heal | Left bumper |

**Other keys:** R = restart episode, Escape = quit.

### Debug lidar

```bash
uv run python -m battle_royale.train --debug-lidar
```

Visualizes the 12 lidar rays from agent 0's perspective with WASD movement and mouse aim. Rays are colored green (far) to red (close).

### Monitor training

```bash
uv run tensorboard --logdir battle_royale/runs/
```

### Analysis scripts

```bash
# Comprehensive gameplay analysis (fire accuracy, heal usage, movement, deaths, etc.)
uv run python scripts/analyze_gameplay.py

# Detailed heal action analysis with logit diagnostics
uv run python scripts/analyze_heal_usage.py

# Parse TensorBoard logs to CSV/plots
uv run python scripts/analyze_tb.py

# Profile rollout performance (FPS benchmarking)
uv run python scripts/profile_rollout.py
```

