"""Deterministic comparison: GameEnv vs BatchedGameEnv(num_envs=1).

Sets both envs to identical initial state, runs fixed action sequences
that exercise all game mechanics, and asserts states match after each step.
"""

import numpy as np
import src.config as cfg
from src.env import GameEnv
from src.batched_env import BatchedGameEnv
from src.objects import Bullet


def sync_state_single_to_batched(single: GameEnv, batched: BatchedGameEnv):
    """Copy state from single env into batched env slot 0."""
    for i in range(2):
        batched.agent_positions[0, i, 0] = single.agents[i].x
        batched.agent_positions[0, i, 1] = single.agents[i].y
        batched.agent_ammo[0, i] = single.agents[i].ammo
    batched.agent_alive[0] = single.alive

    batched.bullet_active[0] = False
    for j, b in enumerate(single.bullets):
        batched.bullet_positions[0, j, 0] = b.x
        batched.bullet_positions[0, j, 1] = b.y
        vx, vy = b.get_velocity()
        batched.bullet_velocities[0, j, 0] = vx
        batched.bullet_velocities[0, j, 1] = vy
        batched.bullet_owners[0, j] = b.owner_id
        batched.bullet_active[0, j] = True

    batched.pickup_active[0] = False
    for j, p in enumerate(single.ammo_pickups):
        batched.pickup_positions[0, j, 0] = p.x
        batched.pickup_positions[0, j, 1] = p.y
        batched.pickup_active[0, j] = True

    batched.done[0] = single.done
    batched.winners[0] = single.winner if single.winner is not None else -1
    batched.time_steps[0] = single.time_step


def assert_states_match(single: GameEnv, batched: BatchedGameEnv, step: int):
    """Assert that single env state matches batched env slot 0."""
    for i in range(2):
        np.testing.assert_allclose(
            [single.agents[i].x, single.agents[i].y],
            batched.agent_positions[0, i],
            atol=1e-4,
            err_msg=f"Step {step}: agent {i} position mismatch",
        )
        assert single.agents[i].ammo == batched.agent_ammo[0, i], \
            f"Step {step}: agent {i} ammo mismatch: {single.agents[i].ammo} vs {batched.agent_ammo[0, i]}"

    assert list(single.alive) == batched.agent_alive[0].tolist(), \
        f"Step {step}: alive mismatch: {single.alive} vs {batched.agent_alive[0].tolist()}"

    # Bullets: compare active bullets (order should match since we sync initially)
    n_single = len(single.bullets)
    n_batched = int(batched.bullet_active[0].sum())
    assert n_single == n_batched, \
        f"Step {step}: bullet count mismatch: {n_single} vs {n_batched}"

    batched_active_idx = np.where(batched.bullet_active[0])[0]
    for j, b in enumerate(single.bullets):
        bi = batched_active_idx[j]
        np.testing.assert_allclose(
            [b.x, b.y],
            batched.bullet_positions[0, bi],
            atol=1e-4,
            err_msg=f"Step {step}: bullet {j} position mismatch",
        )
        bvx, bvy = b.get_velocity()
        np.testing.assert_allclose(
            [bvx, bvy],
            batched.bullet_velocities[0, bi],
            atol=1e-4,
            err_msg=f"Step {step}: bullet {j} velocity mismatch",
        )
        assert b.owner_id == batched.bullet_owners[0, bi], \
            f"Step {step}: bullet {j} owner mismatch"

    # Pickups
    n_pickups_single = len(single.ammo_pickups)
    n_pickups_batched = int(batched.pickup_active[0].sum())
    assert n_pickups_single == n_pickups_batched, \
        f"Step {step}: pickup count mismatch: {n_pickups_single} vs {n_pickups_batched}"

    assert single.done == batched.done[0], \
        f"Step {step}: done mismatch: {single.done} vs {batched.done[0]}"


def setup_envs(agent0_pos, agent1_pos, pickups, agent0_ammo=0, agent1_ammo=0):
    """Create both envs with identical manually-set state."""
    single = GameEnv()
    single.agents[0].x, single.agents[0].y = agent0_pos
    single.agents[1].x, single.agents[1].y = agent1_pos
    single.agents[0].ammo = agent0_ammo
    single.agents[1].ammo = agent1_ammo
    single.bullets = []
    single.ammo_pickups = []
    for px, py in pickups:
        single.ammo_pickups.append(
            __import__("src.objects", fromlist=["AmmoPickup"]).AmmoPickup(
                x=px, y=py, radius=cfg.AMMO_PICKUP_RADIUS
            )
        )
    single.alive = [True, True]
    single.done = False
    single.winner = None
    single.time_step = 0

    batched = BatchedGameEnv(num_envs=1)
    sync_state_single_to_batched(single, batched)

    return single, batched


def step_both(single, batched, a0, a1):
    """Step both envs with the same actions, return rewards."""
    _, (r0_s, r1_s), done_s, _ = single.step(a0, a1)
    rewards_b, dones_b = batched.step(np.array([[a0, a1]]))
    return (r0_s, r1_s), (float(rewards_b[0, 0]), float(rewards_b[0, 1]))


def test_movement():
    """Agents move in all directions, clamp at walls."""
    single, batched = setup_envs((400, 300), (200, 300), [])

    actions = [
        (1, 2),  # agent0 UP, agent1 DOWN
        (2, 1),  # agent0 DOWN, agent1 UP
        (3, 4),  # agent0 LEFT, agent1 RIGHT
        (4, 3),  # agent0 RIGHT, agent1 LEFT
        (0, 0),  # both STAY
    ]
    for step_i, (a0, a1) in enumerate(actions):
        step_both(single, batched, a0, a1)
        assert_states_match(single, batched, step_i)

    # Push agent to wall
    single.agents[0].x = cfg.AGENT_RADIUS + 2
    sync_state_single_to_batched(single, batched)
    step_both(single, batched, 3, 0)  # agent0 LEFT into wall
    assert_states_match(single, batched, 99)


def test_ammo_pickup():
    """Agent walks onto a pickup, gains ammo, pickup disappears."""
    pickup_pos = (405, 300)  # just to the right of agent0
    single, batched = setup_envs((400, 300), (200, 300), [pickup_pos])

    # Move agent0 RIGHT onto the pickup
    step_both(single, batched, 4, 0)
    assert_states_match(single, batched, 0)
    assert single.agents[0].ammo == cfg.AMMO_PER_PICKUP
    assert len(single.ammo_pickups) == 0


def test_shooting():
    """Agent shoots, bullet spawns, travels, goes OOB."""
    single, batched = setup_envs((400, 300), (200, 300), [], agent0_ammo=3)

    # Agent0 shoots RIGHT
    step_both(single, batched, 8, 0)
    assert_states_match(single, batched, 0)
    assert len(single.bullets) == 1
    assert single.agents[0].ammo == 2

    # Let bullet travel
    for i in range(1, 5):
        step_both(single, batched, 0, 0)
        assert_states_match(single, batched, i)

    # Run until bullet goes OOB
    for i in range(5, 100):
        step_both(single, batched, 0, 0)
        assert_states_match(single, batched, i)
        if len(single.bullets) == 0:
            break
    assert len(single.bullets) == 0


def test_bullet_hit():
    """Bullet hits opposing agent, game ends with correct rewards."""
    # Place agents close, agent0 shoots RIGHT toward agent1
    single, batched = setup_envs((100, 300), (140, 300), [], agent0_ammo=5)

    # Shoot
    r_single, r_batched = step_both(single, batched, 8, 0)
    assert_states_match(single, batched, 0)

    # Step until hit
    for i in range(1, 20):
        r_single, r_batched = step_both(single, batched, 0, 0)
        assert_states_match(single, batched, i)
        if single.done:
            break

    assert single.done
    assert single.winner == 0
    np.testing.assert_allclose(r_single[0], r_batched[0], atol=1e-4)
    np.testing.assert_allclose(r_single[1], r_batched[1], atol=1e-4)


def test_no_ammo_shoot():
    """Shooting with no ammo does nothing."""
    single, batched = setup_envs((400, 300), (200, 300), [], agent0_ammo=0)

    step_both(single, batched, 8, 0)  # try to shoot with 0 ammo
    assert_states_match(single, batched, 0)
    assert len(single.bullets) == 0


def test_rewards_match():
    """Rewards match between envs across multiple steps."""
    single, batched = setup_envs((400, 300), (200, 300), [(405, 300)])

    actions = [(4, 0), (0, 0), (8, 0), (0, 0), (0, 0)]
    for i, (a0, a1) in enumerate(actions):
        r_single, r_batched = step_both(single, batched, a0, a1)
        np.testing.assert_allclose(r_single[0], r_batched[0], atol=1e-4,
                                   err_msg=f"Step {i}: reward_0 mismatch")
        np.testing.assert_allclose(r_single[1], r_batched[1], atol=1e-4,
                                   err_msg=f"Step {i}: reward_1 mismatch")
        assert_states_match(single, batched, i)


if __name__ == "__main__":
    tests = [
        test_movement,
        test_ammo_pickup,
        test_shooting,
        test_bullet_hit,
        test_no_ammo_shoot,
        test_rewards_match,
    ]
    for t in tests:
        t()
        print(f"  PASS: {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
