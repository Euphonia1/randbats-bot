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


def test_a_batch_of_complete_battles_holds_together():
    """Play a batch of battles to the end and check the results are coherent.

    The other tests here pin single steps; this one exercises whole games, which
    is where switching, fainting, replacement prompts and the win condition all
    have to agree with each other over many turns at once.
    """
    from psjax.env import rollout_batch
    battles, cap = 128, 500
    env = BattleEnv(DATA)
    states = env.reset_batch(jax.random.split(jax.random.PRNGKey(20), battles))
    final, rewards = rollout_batch(env, states,
                                   jax.random.split(jax.random.PRNGKey(21), battles), cap)

    done = final.phase == C.PHASE_END
    alive = jnp.sum(final.hp > 0, axis=2)                 # [battles, 2]
    winner = final.winner
    decisive = (winner == 0) | (winner == 1)
    idx = jnp.arange(battles)

    # Random play occasionally stalls past the cap; that is not a failure, but
    # the vast majority must finish or something is wrong with the win condition.
    assert int(jnp.sum(done)) >= int(0.95 * battles), \
        f"only {int(jnp.sum(done))}/{battles} battles finished within {cap} steps"

    assert bool(jnp.all(jnp.where(done, winner >= 0, True))), \
        "a battle ended without a winner"
    assert bool(jnp.all(jnp.sum(rewards, axis=1) == 0)), "rewards were not zero-sum"
    assert bool(jnp.all(jnp.max(jnp.abs(rewards), axis=1) <= 1)), \
        "a battle paid out more than once"
    assert bool(jnp.all(jnp.where(decisive, alive[idx, 1 - jnp.clip(winner, 0, 1)] == 0,
                                  True))), "declared a winner with the loser still alive"
    assert bool(jnp.all(jnp.where(decisive, alive[idx, jnp.clip(winner, 0, 1)] > 0,
                                  True))), "declared a winner that was wiped out"
    assert bool(jnp.all(jnp.where(winner == 2, jnp.sum(alive, axis=1) == 0, True))), \
        "a tie was declared with someone still standing"

    # State invariants must survive a whole game, not just one step.
    assert bool(jnp.all((final.hp >= 0) & (final.hp <= final.maxhp)))
    assert bool(jnp.all(final.pp >= 0))
    assert bool(jnp.all(jnp.abs(final.boosts) <= 6))
    assert bool(jnp.all((final.active >= 0) & (final.active < C.TEAM_SIZE)))


def test_neither_player_has_a_structural_edge():
    """With both sides playing randomly, wins should split near evenly.

    A real bias here would point at something asymmetric in turn order, switch
    handling or the win check. The tolerance is loose enough not to flake: a fair
    coin over ~250 decisive battles sits inside 40% / 60% far beyond chance.
    """
    from psjax.env import rollout_batch
    battles = 256
    env = BattleEnv(DATA)
    states = env.reset_batch(jax.random.split(jax.random.PRNGKey(30), battles))
    final, _ = rollout_batch(env, states,
                             jax.random.split(jax.random.PRNGKey(31), battles), 500)
    p0 = int(jnp.sum(final.winner == 0))
    p1 = int(jnp.sum(final.winner == 1))
    share = p0 / max(p0 + p1, 1)
    assert 0.40 <= share <= 0.60, \
        f"player 0 won {p0}/{p0 + p1} decisive battles ({share:.0%}) -- suspiciously lopsided"


def _uturn_battle(seed):
    """Weavile (fast, U-turn) + a Blissey on the bench, against a slow attacker."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from scenario import build
    fast = {"species": "Weavile", "ability": "Pressure", "moves": ["uturn", "iceshard"]}
    return build({"p1": fast,
                  "p1team": [fast, {"species": "Blissey", "ability": "Natural Cure",
                                    "moves": ["splash"]}],
                  "p2": {"species": "Snorlax", "ability": "Thick Fat",
                         "moves": ["seismictoss"]}},
                 jax.random.PRNGKey(seed))


def test_battles_in_different_phases_step_together():
    """A batch desynchronises, and each battle must still follow its own path.

    A self-switch opens a replacement prompt in *some* battles and not others, so
    within one batched call the same action integer means a replacement slot in
    one battle and a move slot in the next. `lax.cond` on a batched predicate
    evaluates both readings and selects per element, so this works -- but it is
    the property a batched policy depends on most, so it is pinned here.
    """
    n = 6
    singles = [_uturn_battle(100 + i) for i in range(n)]
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), *singles)
    jstep, vstep = jax.jit(step), jax.jit(jax.vmap(step, in_axes=(0, 0, None)))

    # Even battles U-turn (opening a prompt); odd ones just attack.
    first = jnp.stack([jnp.array([0 if i % 2 == 0 else 1, 0], jnp.int32)
                       for i in range(n)])
    batch = vstep(batch, first, DATA)
    singles = [jstep(s, first[i], DATA) for i, s in enumerate(singles)]

    for i in range(n):
        if i % 2 == 0:
            assert int(batch.phase[i]) == C.PHASE_SWITCH, f"battle {i} did not pause"
            assert int(batch.pending_side[i]) == 1, "the opponent's move was not held"
        else:
            assert int(batch.phase[i]) == C.PHASE_MOVE, f"battle {i} paused unexpectedly"

    # The mask has to reflect each battle's own phase, or a policy cannot act.
    masks = jax.vmap(legal_action_mask, in_axes=(None, 0))(DATA, batch)
    for i in range(n):
        offered = [int(a) for a in jnp.nonzero(masks[i, 0])[0]]
        if i % 2 == 0:
            assert all(a >= C.ACTION_SWITCH_BASE for a in offered), \
                f"battle {i} was offered moves while owing a replacement: {offered}"
        else:
            assert any(a < C.ACTION_SWITCH_BASE for a in offered), \
                f"battle {i} was offered no move: {offered}"

    # One call, two meanings: a replacement slot for some, a move slot for others.
    second = jnp.stack([
        jnp.array([C.ACTION_SWITCH_BASE + 1 if i % 2 == 0 else 1, 0], jnp.int32)
        for i in range(n)])
    batch = vstep(batch, second, DATA)
    singles = [jstep(s, second[i], DATA) for i, s in enumerate(singles)]

    for field in batch._fields:
        if field == "key":
            continue
        got = getattr(batch, field)
        want = jnp.stack([getattr(s, field) for s in singles])
        assert bool(jnp.all(got == want)), \
            f"mixed-phase batch diverged from stepping alone, on {field}"

    # Turn counters legitimately desynchronise: answering a prompt is a decision
    # point, not a turn, so the battles that paused are one turn behind.
    turns = [int(t) for t in batch.turn]
    assert turns[0] < turns[1], f"expected the paused battles to lag: {turns}"


def test_a_finished_battle_in_a_batch_does_not_disturb_the_others():
    env = BattleEnv(DATA)
    states = env.reset_batch(jax.random.split(jax.random.PRNGKey(40), 4))
    # Force battle 0 to be over; the rest carry on.
    states = states._replace(
        phase=states.phase.at[0].set(jnp.int8(C.PHASE_END)),
        winner=states.winner.at[0].set(jnp.int8(0)))
    before = states.hp[0]
    actions = jnp.zeros((4, 2), jnp.int32)
    states, _, rewards, done = env.step_batch(states, actions)
    assert bool(done[0]) and bool(jnp.all(states.hp[0] == before)), \
        "a finished battle kept mutating inside the batch"
    assert bool(jnp.all(rewards[0] == 0)), "a finished battle paid out again"
    assert not bool(jnp.any(done[1:])), "other battles were dragged to an end"
