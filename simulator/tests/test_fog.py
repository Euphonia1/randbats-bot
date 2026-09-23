"""Partial observability must hide the opponent's secrets and nothing else.

Two properties matter and pull against each other: the censored observation has
to withhold what a real opponent could not know, and the simulation underneath
has to be bit-identical to the unwrapped environment. A wrapper that quietly
perturbs the battle would be worse than no wrapper at all.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from psjax import consts as C
from psjax.data import load_data
from psjax.env import BattleEnv
from psjax.fog import FogOfWarEnv, FogState

DATA = load_data()
ENV = FogOfWarEnv(BattleEnv(DATA))


def _rollout(env, fs, key, steps):
    """Play `steps` decision points with uniform random legal actions."""
    out = [fs]
    for i in range(steps):
        k = jax.random.fold_in(key, i)
        actions = env.sample_actions(fs, k)
        fs, _, _, _ = env.step(fs, actions)
        out.append(fs)
    return out


# --- what the opponent may see ------------------------------------------------

def test_no_moves_are_revealed_at_the_start():
    fs = ENV.reset(jax.random.PRNGKey(0))
    assert not bool(jnp.any(fs.revealed_moves)), \
        "nobody has used a move before the first turn"


def test_leads_are_on_the_field_from_the_start():
    fs = ENV.reset(jax.random.PRNGKey(1))
    for p in range(C.NUM_PLAYERS):
        assert bool(fs.revealed[p, fs.battle.active[p]]), \
            "the lead is visible before either player acts"
        assert int(jnp.sum(fs.revealed[p])) == 1, \
            "only the lead has been seen"


def test_censoring_blanks_the_opponents_unused_moves():
    fs = ENV.reset(jax.random.PRNGKey(2))
    hidden = ENV._censor(fs, 1)
    assert bool(jnp.all(hidden.moves[1] == -1)), \
        "an unused move must read as an empty slot"
    assert bool(jnp.all(hidden.pp[1] == fs.battle.maxpp[1])), \
        "an unused move must read as full PP"


def test_censoring_leaves_the_viewer_untouched():
    """Everything about player 0 survives censoring player 1."""
    fs = ENV.reset(jax.random.PRNGKey(3))
    hidden = ENV._censor(fs, 1)
    for name in ("moves", "pp", "hp", "status", "species", "maxhp"):
        a = getattr(fs.battle, name)[0]
        b = getattr(hidden, name)[0]
        assert bool(jnp.array_equal(a, b)), f"{name} changed for the viewer"


def test_censoring_changes_only_moves_pp_and_hp():
    fs = ENV.reset(jax.random.PRNGKey(4))
    hidden = ENV._censor(fs, 1)
    allowed = {"moves", "pp", "hp", "maxhp"}
    for name in fs.battle._fields:
        if name in allowed:
            continue
        a, b = getattr(fs.battle, name), getattr(hidden, name)
        assert bool(jnp.array_equal(a, b)), f"{name} must not be rewritten"


def test_using_a_move_reveals_it():
    fs = ENV.reset(jax.random.PRNGKey(5))
    slot0 = int(fs.battle.active[0])
    # Action 0 is the first move slot; both sides use it.
    fs2, _, _, _ = ENV.step(fs, jnp.array([0, 0], jnp.int32))
    assert bool(fs2.revealed_moves[0, slot0, 0]), \
        "a move that spent PP has been disclosed"
    assert int(jnp.sum(fs2.revealed_moves[0, slot0])) == 1, \
        "only the move that was used is disclosed"


def test_revelation_never_reverses():
    fs = ENV.reset(jax.random.PRNGKey(6))
    states = _rollout(ENV, fs, jax.random.PRNGKey(7), 40)
    for a, b in zip(states, states[1:]):
        assert bool(jnp.all(b.revealed_moves >= a.revealed_moves)), \
            "a disclosed move cannot become secret again"
        assert bool(jnp.all(b.revealed >= a.revealed)), \
            "a Pokemon that has been seen cannot become unseen"


def test_switching_reveals_the_incoming_pokemon():
    fs = ENV.reset(jax.random.PRNGKey(8))
    mask = ENV.legal_actions(fs)[0]
    switches = [a for a in range(C.ACTION_SWITCH_BASE, C.NUM_ACTIONS)
                if bool(mask[a])]
    assert switches, "the lead should have a bench to switch to"
    target = switches[0] - C.ACTION_SWITCH_BASE
    fs2, _, _, _ = ENV.step(fs, jnp.array([switches[0], 0], jnp.int32))
    assert bool(fs2.revealed[0, target]), \
        "a Pokemon that switches in has been seen"


def test_opponent_hp_reads_as_a_percentage():
    """The opponent's bar is a percentage, and hides the maximum behind it."""
    fs = ENV.reset(jax.random.PRNGKey(9))
    states = _rollout(ENV, fs, jax.random.PRNGKey(10), 12)
    seen_damage = False
    for s in states:
        hidden = ENV._censor(s, 1)
        true_hp = s.battle.hp[1]
        true_max = jnp.maximum(s.battle.maxhp[1], 1)
        expect = jnp.ceil(true_hp.astype(jnp.float32) * 100.0 / true_max)

        assert bool(jnp.all(hidden.maxhp[1] == 100)), \
            "the real maximum must not survive censoring"
        assert bool(jnp.all(hidden.hp[1] == expect.astype(hidden.hp.dtype))), \
            "censored HP must be the percentage Showdown would display"
        assert bool(jnp.all(hidden.hp[1] <= 100))
        # Fainted reads as fainted, so the alive bit stays truthful.
        assert bool(jnp.all((hidden.hp[1] > 0) == (true_hp > 0)))
        seen_damage = seen_damage or bool(jnp.any(true_hp < s.battle.maxhp[1]))
    assert seen_damage, "the rollout never damaged anything, so this proved little"


def test_exact_opponent_hp_is_not_recoverable():
    """Two totals that share a percentage must censor to the same reading."""
    fs = ENV.reset(jax.random.PRNGKey(16))
    st = fs.battle
    # 150/300 and 90/180 are both exactly half.
    st = st._replace(
        hp=st.hp.at[1, 0].set(150).at[1, 1].set(90),
        maxhp=st.maxhp.at[1, 0].set(300).at[1, 1].set(180))
    hidden = ENV._censor(fs._replace(battle=st), 1)
    assert int(hidden.hp[1, 0]) == int(hidden.hp[1, 1]) == 50
    assert int(hidden.maxhp[1, 0]) == int(hidden.maxhp[1, 1]) == 100


# --- the simulation underneath ------------------------------------------------

def test_dynamics_are_identical_to_the_unwrapped_env():
    """Censoring must not leak back into the battle."""
    plain = BattleEnv(DATA)
    key = jax.random.PRNGKey(11)
    fs = ENV.reset(key)
    st = plain.reset(key)
    for i in range(30):
        actions = plain.sample_actions(st, jax.random.fold_in(key, i))
        fs, _, r_fog, d_fog = ENV.step(fs, actions)
        st, _, r_plain, d_plain = plain.step(st, actions)
        assert bool(jnp.array_equal(r_fog, r_plain)), f"rewards diverged at {i}"
        assert bool(jnp.array_equal(d_fog, d_plain)), f"done diverged at {i}"
    for name in fs.battle._fields:
        if name == "key":
            continue
        assert bool(jnp.array_equal(
            getattr(fs.battle, name), getattr(st, name))), \
            f"battle state diverged in {name}"


def test_observation_keeps_the_unwrapped_layout():
    fs = ENV.reset(jax.random.PRNGKey(12))
    obs = ENV.observe(fs)
    assert obs.shape == (C.NUM_PLAYERS, ENV.obs_dim)
    assert ENV.obs_dim == BattleEnv(DATA).obs_dim, \
        "a policy trained under fog must load against the plain env"
    assert bool(jnp.all(jnp.isfinite(obs)))


def test_fog_actually_withholds_something():
    fs = ENV.reset(jax.random.PRNGKey(13))
    assert bool(jnp.any(ENV.observe(fs) != ENV.env.observe(fs.battle))), \
        "the wrapper would be pointless if it changed nothing"


# --- batching -----------------------------------------------------------------

def test_batched_fog_matches_single_battles():
    n = 4
    keys = jax.random.split(jax.random.PRNGKey(14), n)
    batched = ENV.reset_batch(keys)
    singles = [ENV.reset(k) for k in keys]

    actions = jnp.stack([jnp.array([0, 0], jnp.int32)] * n)
    batched, obs_b, r_b, d_b = ENV.step_batch(batched, actions)
    for i, s in enumerate(singles):
        s, obs_s, r_s, d_s = ENV.step(s, actions[i])
        assert bool(jnp.allclose(obs_b[i], obs_s)), f"obs differ for battle {i}"
        assert bool(jnp.array_equal(r_b[i], r_s)), f"rewards differ for battle {i}"
        assert bool(jnp.array_equal(
            batched.revealed_moves[i], s.revealed_moves)), \
            f"disclosure differs for battle {i}"


def test_fog_state_is_a_pytree():
    fs = ENV.reset(jax.random.PRNGKey(15))
    leaves = jax.tree.leaves(fs)
    assert leaves, "FogState must flatten for vmap and scan"
    assert isinstance(jax.tree.map(lambda x: x, fs), FogState)
