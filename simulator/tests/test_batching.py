"""Batched execution must agree exactly with stepping battles one at a time.

`vmap` over `step` is the whole point of the JAX port, and it is only useful if
it is indistinguishable from the sequential engine. These tests pin that, and
pin the action-masking contract that a batched policy relies on.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from psjax import consts as C
from psjax.data import load_data
from psjax.engine import step
from psjax.env import BattleEnv
from psjax.teams import legal_action_mask, new_battle

DATA = load_data()
N = 6


def test_vmapped_step_matches_sequential_step():
    keys = jax.random.split(jax.random.PRNGKey(7), N)
    batched = jax.vmap(lambda k: new_battle(k, DATA))(keys)
    singles = [new_battle(k, DATA) for k in keys]

    jstep = jax.jit(step)
    vstep = jax.jit(jax.vmap(step, in_axes=(0, 0, None)))

    for turn in range(10):
        actions = jnp.stack([jnp.array([(turn + i) % 4, (turn + 2 * i) % 4], jnp.int32)
                             for i in range(N)])
        batched = vstep(batched, actions, DATA)
        singles = [jstep(s, actions[i], DATA) for i, s in enumerate(singles)]

    for field in batched._fields:
        if field == "key":
            continue
        got = getattr(batched, field)
        want = jnp.stack([getattr(s, field) for s in singles])
        assert bool(jnp.all(got == want)), f"batched {field} diverged from sequential"


def test_env_step_batch_shapes_and_rewards():
    env = BattleEnv(DATA)
    states = env.reset_batch(jax.random.split(jax.random.PRNGKey(0), N))
    actions = jnp.zeros((N, 2), jnp.int32)
    states, obs, rewards, done = env.step_batch(states, actions)
    assert states.hp.shape == (N, 2, C.TEAM_SIZE)
    assert obs.shape == (N, 2, env.obs_dim)
    assert rewards.shape == (N, 2)
    assert done.shape == (N,)
    assert bool(jnp.all(jnp.isfinite(obs)))


def test_batched_rollouts_finish_and_pay_out_zero_sum():
    from psjax.env import rollout_batch
    env = BattleEnv(DATA)
    states = env.reset_batch(jax.random.split(jax.random.PRNGKey(3), N))
    final, rewards = rollout_batch(env, states, jax.random.split(jax.random.PRNGKey(4), N),
                                   max_steps=400)
    assert bool(jnp.all(final.phase == C.PHASE_END)), \
        f"battles unfinished after 400 steps: {final.phase}"
    assert bool(jnp.all(jnp.sum(rewards, axis=1) == 0)), "rewards were not zero-sum"
    assert bool(jnp.all((final.active >= 0) & (final.active < C.TEAM_SIZE)))


def test_batched_action_masks_are_always_satisfiable():
    env = BattleEnv(DATA)
    states = env.reset_batch(jax.random.split(jax.random.PRNGKey(11), N))
    key = jax.random.PRNGKey(12)
    vmask = jax.jit(jax.vmap(legal_action_mask, in_axes=(None, 0)))
    for _ in range(60):
        masks = vmask(DATA, states)
        assert bool(jnp.all(jnp.any(masks, axis=2))), "a player had no legal action"
        key, k = jax.random.split(key)
        actions = jax.vmap(env.sample_actions)(states, jax.random.split(k, N))
        states, _, _, _ = env.step_batch(states, actions)
        assert bool(jnp.all((states.active >= 0) & (states.active < C.TEAM_SIZE)))
