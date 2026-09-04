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

    # NOTE on batching. `jax.vmap` over `reset`, `observe` and the individual
    # mechanics helpers is fine, but `jax.vmap(step)` currently takes many
    # minutes to compile even at a batch of 8: the blow-up is inside
    # `moves.execute_move`, and it is a compile-time problem, not a runtime one.
    # (`residuals`, `switch_to` and the 31-branch effect switch each vmap in
    # under a second, so the large `lax.switch`es are not the cause.) Until that
    # is tracked down, `step_batch` maps sequentially with `lax.map`, which
    # compiles the unbatched body once and still runs entirely on device.

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, state: BattleState, actions):
        """One decision point. Returns `(state, obs, rewards, done)`.

        `rewards` is `[2]`, +1 for the winner and -1 for the loser at the end of
        the battle and 0 otherwise; a tie pays 0 to both.
        """
        was_over = state.phase == C.PHASE_END
        new_state = engine_step(state, actions, self.data)
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
        return new_state, self.observe(new_state), rewards, done

    def step_batch(self, states, actions):
        """Step a batch of battles. Sequential on device -- see the note above."""
        return jax.lax.map(lambda xs: self.step(xs[0], xs[1]), (states, actions))

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
    def observe(self, state: BattleState) -> jnp.ndarray:
        """A `[2, OBS_DIM]` float array: each row is that player's own view.

        Deliberately plain: HP fractions, statuses, boosts, types, field state and
        the active Pokemon's moves. It carries full information about both sides
        (no fog of war), which suits self-play; a partially observed variant would
        mask the opponent's bench.
        """
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
                rows.append(jax.nn.one_hot(self.data["move_type"][moves],
                                           C.NUM_TYPES).reshape(-1))
                rows.append(jax.nn.one_hot(self.data["move_category"][moves], 3)
                            .reshape(-1))
                rows.append(self.data["move_base_power"][moves].astype(jnp.float32)
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

    A `lax.scan`, so the whole rollout stays on device. For many battles at once
    use `rollout_batch`, which maps this sequentially rather than vmapping it.
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


def rollout_batch(env: BattleEnv, states: BattleState, keys, max_steps: int = 300):
    """Play out a batch of battles. `states` and `keys` carry a leading `N` axis."""
    return jax.lax.map(
        lambda xs: rollout(env, xs[0], xs[1], max_steps), (states, keys))
