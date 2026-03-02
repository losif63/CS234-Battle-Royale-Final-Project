# Battle Royale Agents

1v1. 800×600 arena. Agents spawn with 0 ammo, collect pickups (10 bullets each, 6 on map). 9 actions: stay, move ×4, shoot ×4. Bullets are axis-aligned. One hit kills.

Rewards:
**+10** kill
**−10** die
**−15** draw (timeout at 1000 steps)
no step penalty.

---

## Commands

```bash
# Evaluate (headless, 100 episodes by default)
uv run python -m agents.baseline_agent --eval
uv run python -m agents.random_agent   --eval
uv run python -m agents.baseline_agent --eval --episodes 500
uv run python -m agents.baseline_agent --eval --checkpoint agents/checkpoints/ppo_update_500.pt

# Watch PPO fight
uv run python -m agents.ppo_agent --watch               # PPO vs PPO
uv run python -m agents.baseline_agent --watch          # PPO vs baseline
uv run python -m agents.random_agent --watch            # PPO vs random

# Watch reference matchups
uv run python -m agents.baseline_agent --watch-baseline # baseline vs baseline
uv run python -m agents.random_agent --watch-random     # random vs random

# Play against the trained agent (WASD move, arrow keys shoot)
uv run python -m agents.ppo_agent --play

# Train
uv run python -m agents.ppo_agent
uv run python -m agents.ppo_agent --updates 5000
uv run python -m agents.ppo_agent --resume agents/checkpoints/ppo_update_1000.pt
```

---

## Development notes

> tried DQN first. training was glacially slow — DQN needs a replay buffer, which means stepping one env at a time per agent. couldn't use the batched env. shelved it.

> wrote a rule-based baseline: dodge incoming bullets → fetch ammo if empty → align on x or y axis and shoot. simple priority queue, works surprisingly well.

> first PPO obs (29-dim): self (x, y, ammo), enemy (x, y, ammo), 3 nearest pickup positions, 3 nearest enemy bullets. self-play with pool of past snapshots. agent learned to dodge bullets but couldn't beat baseline after 10k updates.

> realized the problem: giving the network absolute enemy coordinates (ox, oy) means it has to learn subtraction to figure out direction. switched to relative displacement (dx, dy) = (enemy − self), normalized by 250px so a 25px alignment gap ≈ 0.1 in obs space. same fix for pickup positions. immediate improvement.

> added baseline into training as a curriculum: start mostly against baseline, gradually shift to self-play. agent got good at beating baseline but then regressed when self-play took over — it learned to exploit its past self in ways that don't generalize back to baseline.

> best win rate against baseline with curriculum: ~50%.

> noticed a tell: stand still and the agent ignores you completely. it had no incentive to engage, so it just hid. added an existential step penalty to force shorter episodes — rounds got faster, but agents also got more reckless. not clearly better.

> added wall distances to obs (4 floats: distance to each wall, normalized). agent immediately stopped cornering itself. one of the cleanest single-feature wins.

> made the critic omniscient: concatenate both agents' 38-dim obs as the value input (76-dim critic, 38-dim policy). better value estimates without leaking info into the policy.

> biggest single improvement: discount factor. γ=0.99 → effective horizon ~100 steps. agent was myopic, wouldn't plan around ammo. bumped to γ=0.998 (~500 steps), then γ=0.999 (~1000 steps, matching the episode length). agent started treating ammo as a long-term resource worth positioning for.

> stripped baseline out of training entirely. pure self-play against a rolling pool of up to 50 past snapshots, with ELO tracking. 15% of rounds use a stationary opponent to keep the agent from becoming passive. linear LR annealing to 0 across training; entropy coeff 0.05 to prevent premature policy collapse.

> result: **~80% win rate against baseline without ever training against it.**
