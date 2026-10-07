"""A batched, jit-friendly environment wrapper.

`BattleEnv` exposes a small `reset` / `step` API over `engine.step`, plus an
observation encoder. Every method is a pure function of its arguments, so the
whole thing composes with `jax.vmap` for thousands of parallel battles and with
`jax.lax.scan` for rollouts that never leave the device.

    env = BattleEnv()
    states = env.reset(jax.random.split(key, 4096))          # 4096 battles
    states, obs, rewards, done = env.step(states, actions)
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from . import consts as C
from .data import load_data
from .engine import step as engine_step
from .mechanics import effective_speed
from .state import BattleState
from .teams import legal_action_mask, new_battle


class BattleEnv:
    """Gen 9 Random Battle singles as a two-player environment."""

    def __init__(self, data=None):
        self.data = data if data is not None else load_data()

    # --- lifecycle ----------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, key) -> BattleState:
        return new_battle(key, self.data)

    def reset_batch(self, keys) -> BattleState:
        """`keys` is `[N, 2]`; returns a `BattleState` with a leading `N` axis."""
        return jax.vmap(self.reset)(keys)

    # Batching runs through `jax.vmap`. Compilation is a one-off ~15s at any
    # batch size, and throughput is roughly fifty times that of stepping battles
    # one at a time. Keeping it that way depends on two structural properties of
    # the engine -- see the note in `psjax/__init__.py` before changing them.

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, state: BattleState, actions):
        """One decision point. Returns `(state, obs, rewards, done)`."""
        return self._step(state, actions)

    def _step(self, state: BattleState, actions, data=None):
        """The body of `step`, without the jit.

        Kept separate so `step_sharded` can put it inside a `pmap`: wrapping an
        already-jitted callable in `pmap` re-traces on every call. `data` is an
        explicit argument for the same reason -- closed over, `pmap` re-broadcasts
        all 88 tables to every device on every call.

        `rewards` is `[2]`, +1 for the winner and -1 for the loser at the end of
        the battle and 0 otherwise; a tie pays 0 to both.
        """
        data = self.data if data is None else data
        was_over = state.phase == C.PHASE_END
        new_state = engine_step(state, actions, data)
        # A finished battle absorbs further steps rather than corrupting itself.
        new_state = jax.tree.map(
            lambda old, new: jnp.where(was_over, old, new), state, new_state)

        done = new_state.phase == C.PHASE_END
        just_ended = done & jnp.logical_not(was_over)
        winner = new_state.winner
        rewards = jnp.where(
            just_ended,
            jnp.where(winner == 2, jnp.zeros(2),
                      jnp.where(jnp.arange(2) == winner, 1.0, -1.0)),
            jnp.zeros(2))
        return new_state, self._observe(new_state, data), rewards, done

    @functools.partial(jax.jit, static_argnums=0)
    def step_batch(self, states, actions):
        """Step a batch of battles in parallel. Leading axis is the batch."""
        return jax.vmap(self.step)(states, actions)

    # --- action masking -----------------------------------------------------

    @functools.partial(jax.jit, static_argnums=0)
    def legal_actions(self, state) -> jnp.ndarray:
        """`[2, NUM_ACTIONS]` bool mask."""
        return legal_action_mask(self.data, state)

    def sample_actions(self, state, key) -> jnp.ndarray:
        """Uniform random legal actions, for smoke tests and random baselines."""
        mask = self.legal_actions(state)
        keys = jax.random.split(key, C.NUM_PLAYERS)
        return jax.vmap(lambda m, k: jax.random.categorical(
            k, jnp.where(m, 0.0, -1e9)))(mask, keys)

    # --- observation --------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=0)
    def observe(self, state: BattleState, data=None) -> jnp.ndarray:
        """`[2, OBS_DIM]` float array: each row is that player's own view."""
        return self._observe(state, data)

    def _observe(self, state: BattleState, data=None) -> jnp.ndarray:
        """A `[2, OBS_DIM]` float array: each row is that player's own view.

        Deliberately plain: HP fractions, statuses, boosts, types, field state and
        the active Pokemon's moves. It carries full information about both sides
        (no fog of war), which suits self-play; a partially observed variant would
        mask the opponent's bench.
        """
        data = self.data if data is None else data

        def one_side(me):
            you = 1 - me
            rows = []
            for side in (me, you):
                i = state.active[side].astype(jnp.int32)
                hp_frac = state.hp[side] / jnp.maximum(state.maxhp[side], 1)
                rows.append(hp_frac.astype(jnp.float32))                    # 6
                rows.append((state.hp[side] > 0).astype(jnp.float32))       # 6
                rows.append(jax.nn.one_hot(state.status[side], C.NUM_STATUS)
                            .reshape(-1))                                   # 6*7
                # How long each sleeper has slept (never how long it has left).
                asleep = state.status[side] == C.SLP
                rows.append(jnp.where(asleep, state.sleep_attempts[side] / 3.0, 0.0)
                            .astype(jnp.float32))                           # 6
                rows.append((asleep & state.rest_sleep[side]).astype(jnp.float32))  # 6
                rows.append(state.boosts[side].astype(jnp.float32) / 6.0)   # 7
                rows.append(jax.nn.one_hot(state.types[side, i] + 1,
                                           C.NUM_TYPES + 1).reshape(-1))    # 2*20
                rows.append((state.volatiles[side] > 0).astype(jnp.float32))
                rows.append(state.side_conditions[side].astype(jnp.float32) / 5.0)
                rows.append(jnp.array([effective_speed(state, side) / 500.0],
                                      jnp.float32))
                rows.append(state.terastallized[side].astype(jnp.float32))  # 6
                # The active Pokemon's moves: type, category, power, PP left.
                moves = jnp.maximum(state.moves[side, i], 0)
                valid = (state.moves[side, i] >= 0).astype(jnp.float32)
                rows.append(jax.nn.one_hot(data["move_type"][moves],
                                           C.NUM_TYPES).reshape(-1))
                rows.append(jax.nn.one_hot(data["move_category"][moves], 3)
                            .reshape(-1))
                rows.append(data["move_base_power"][moves].astype(jnp.float32)
                            / 150.0)
                rows.append(state.pp[side, i].astype(jnp.float32) /
                            jnp.maximum(state.maxpp[side, i], 1).astype(jnp.float32))
                rows.append(valid)
            rows.append(jax.nn.one_hot(state.weather, C.NUM_WEATHER))
            rows.append(jax.nn.one_hot(state.terrain, C.NUM_TERRAIN))
            rows.append(jnp.array([state.trick_room > 0, state.gravity > 0],
                                  jnp.float32))
            rows.append(jnp.array([state.turn / 100.0], jnp.float32))
            return jnp.concatenate([jnp.atleast_1d(r).astype(jnp.float32) for r in rows])

        return jnp.stack([one_side(0), one_side(1)])

    @functools.cached_property
    def obs_dim(self) -> int:
        dummy = self.reset(jax.random.PRNGKey(0))
        return int(self.observe(dummy).shape[-1])


def rollout(env: BattleEnv, state: BattleState, key, max_steps: int = 300):
    """Play one battle out with uniform random legal actions.

    A `lax.scan`, so the whole rollout stays on device. Use `rollout_batch` to
    play many battles at once.
    """
    def body(carry, k):
        st = carry
        k_act, _ = jax.random.split(k)
        actions = env.sample_actions(st, k_act)
        st, _, rewards, done = env.step(st, actions)
        return st, rewards

    keys = jax.random.split(key, max_steps)
    final, rewards = jax.lax.scan(body, state, keys)
    return final, jnp.sum(rewards, axis=0)


@functools.partial(jax.jit, static_argnums=(0, 3),
                   static_argnames=("max_steps",))
def rollout_batch(env: BattleEnv, states: BattleState, keys, max_steps: int = 300):
    """Play out a batch of battles in parallel.

    `states` and `keys` carry a leading `N` axis. Every battle runs the full
    `max_steps` scan; finished ones absorb further steps (see `BattleEnv.step`),
    so the result is the same as stopping each at its own end.

    The `jit` is what makes this reusable rather than merely correct. A bare
    `vmap` of the scan is an eager XLA call: it compiles the entire unrolled
    rollout on *every* invocation, which for this engine is about fifteen
    seconds a call and swamps the microseconds of actual work. Jitted, the
    compile is cached against `(env, shapes, max_steps)` and paid once.
    """
    return jax.vmap(lambda s, k: rollout(env, s, k, max_steps))(states, keys)
