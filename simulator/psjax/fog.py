"""Partial observability: showing each player only what Showdown would show them.

`BattleEnv.observe` is deliberately full-information, which suits self-play but
trains a policy that cannot be deployed against a real opponent. This wrapper
closes the gap without touching the engine: it keeps its own record of what each
side has revealed, and rewrites the *opponent's* half of the state before handing
it to the ordinary observation encoder.

    env = FogOfWarEnv()
    fs = env.reset(key)                       # a FogState, not a BattleState
    fs, obs, rewards, done = env.step(fs, actions)

Everything vmaps exactly as the unwrapped environment does, and `fs.battle` is
the real `BattleState` throughout -- the masking is applied to a throwaway copy
for the encoder's benefit and never feeds back into the simulation.

What actually needed hiding
---------------------------
Less than it looks. The encoder exposes, for each side: per-slot HP fractions,
alive bits, statuses and Tera flags, plus the *active* Pokemon's types, boosts,
volatiles, speed and moves. Of that, only two items are genuinely secret:

- **The active Pokemon's moveset.** The encoder writes all four moves' type,
  category, power and PP from the moment it appears. A real opponent learns
  them one at a time, as they are used. This is the leak worth closing.
- **Exact HP.** Showdown reports the opponent's HP as a whole percent, which
  also conceals the maximum -- a real player cannot tell 50% of 300 from 50%
  of 180.

The per-slot bench figures look like a leak and are not. A Pokemon that has
never been active in singles has never been hit and has never been statused, so
its HP is full and its status is clear by construction -- the encoder is
reporting values the opponent could already infer. Hiding them would be
theatre. Likewise the alive bits: Showdown shows the fainted count. Bench
species and movesets never enter the observation at all.

One approximation remains, noted rather than fixed: the opponent's effective
speed is exposed exactly, where a real player infers it from turn order.
"""
from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import consts as C
from .env import BattleEnv
from .state import BattleState, set_at


class FogState(NamedTuple):
    """A battle plus the bookkeeping of what each side has given away.

    `revealed_moves[p, slot, m]` is True once player `p` has used move `m` of
    that slot in front of the opponent -- so the flags describe what `p` has
    *disclosed*, and it is the other player who benefits from reading them.
    """
    battle: BattleState
    revealed: jnp.ndarray         # [2, 6] bool, slot has been on the field
    revealed_moves: jnp.ndarray   # [2, 6, 4] bool, move has been used


def _on_field(state: BattleState) -> jnp.ndarray:
    """`[2, 6]` bool: the slot each player currently has out."""
    slots = jnp.arange(C.TEAM_SIZE)
    return slots[None, :] == state.active[:, None].astype(jnp.int32)


class FogOfWarEnv:
    """`BattleEnv` with the opponent's secrets withheld."""

    def __init__(self, env: BattleEnv | None = None):
        self.env = env if env is not None else BattleEnv()
        self.data = self.env.data

    @property
    def obs_dim(self) -> int:
        """Unchanged: masking rewrites values, never the layout."""
        return self.env.obs_dim

    # --- lifecycle ----------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=0)
    def reset(self, key) -> FogState:
        state = self.env.reset(key)
        return FogState(
            battle=state,
            # Both leads are on the field before either player acts.
            revealed=_on_field(state),
            revealed_moves=jnp.zeros(
                (C.NUM_PLAYERS, C.TEAM_SIZE, 4), dtype=bool))

    def reset_batch(self, keys) -> FogState:
        return jax.vmap(self.reset)(keys)

    @functools.partial(jax.jit, static_argnums=0)
    def step(self, fs: FogState, actions):
        """One decision point. Returns `(fog_state, obs, rewards, done)`."""
        pp_before = fs.battle.pp
        state, _, rewards, done = self.env.step(fs.battle, actions)

        # A move is disclosed by spending its PP. This reads the disclosure off
        # public state rather than instrumenting the engine, which keeps the
        # wrapper a wrapper. Pressure and Spite take more than one PP; either
        # way the count falls, so the flag still trips on the turn it should.
        fs = FogState(
            battle=state,
            revealed=fs.revealed | _on_field(state),
            revealed_moves=fs.revealed_moves | (pp_before > state.pp))
        return fs, self.observe(fs), rewards, done

    def step_batch(self, fs: FogState, actions):
        return jax.vmap(self.step)(fs, actions)

    # --- masking ------------------------------------------------------------

    def _censor(self, fs: FogState, you: int) -> BattleState:
        """`fs.battle` with player `you`'s undisclosed information removed.

        `you` is a Python int, so the slice assignments below are static.
        """
        st = fs.battle
        seen = fs.revealed_moves[you]                        # [6, 4]

        # An unused move reads as an empty slot, which the encoder already
        # handles: it one-hots `-1` into its "no move" bucket and clears the
        # validity bit. Its PP shows full, which is what an opponent assumes.
        moves = jnp.where(seen, st.moves[you], -1)
        pp = jnp.where(seen, st.pp[you], st.maxpp[you])

        # Showdown gives the opponent a health bar in whole percent, not a
        # number. The encoder reads nothing from hp and maxhp but their ratio,
        # so reporting the percentage over a denominator of 100 hands it exactly
        # that precision -- rather than inventing an HP value, which cannot be
        # done without either leaking the true total or lying about it.
        maxhp = jnp.maximum(st.maxhp[you], 1)
        pct = jnp.ceil(st.hp[you].astype(jnp.float32) * 100.0 / maxhp)

        # Illusion: until it is hit, the active Pokemon looks like the party
        # member it is impersonating, so the opponent sees that one's typing.
        active = st.active[you].astype(jnp.int32)
        disguise = st.illusion[you].astype(jnp.int32)
        shown = st.types[you, jnp.where(disguise >= 0, disguise, active)]

        return st._replace(
            moves=set_at(st.moves, you, moves),
            pp=set_at(st.pp, you, pp),
            hp=set_at(st.hp, you, pct),
            maxhp=set_at(st.maxhp, you, 100),
            types=set_at(st.types, (you, active), shown))

    @functools.partial(jax.jit, static_argnums=0)
    def observe(self, fs: FogState) -> jnp.ndarray:
        """`[2, OBS_DIM]`: row `i` is what player `i` is entitled to see.

        Each row is encoded from a state censored against *that* player, so the
        encoder runs twice. It is cheap next to `step`, and it keeps the layout
        identical to the unwrapped environment -- a policy trained on one will
        load against the other.
        """
        def view(me: int) -> jnp.ndarray:
            return self.env.observe(self._censor(fs, 1 - me))[me]

        return jnp.stack([view(0), view(1)])

    # --- pass-throughs ------------------------------------------------------

    @functools.partial(jax.jit, static_argnums=0)
    def legal_actions(self, fs: FogState) -> jnp.ndarray:
        """Unchanged: legality is a fact about your own side, not the opponent's."""
        return self.env.legal_actions(fs.battle)

    def sample_actions(self, fs: FogState, key) -> jnp.ndarray:
        return self.env.sample_actions(fs.battle, key)
