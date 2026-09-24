"""The battle state: a flat pytree of fixed-shape arrays.

Shape conventions, with `P = 2` players and `T = 6` team slots:

    [P, T]      per-Pokemon        (hp, status, item, ...)
    [P, T, 4]   per-move-slot      (moves, pp)
    [P]         per-side / active  (boosts live here: they belong to the slot,
                                    not the Pokemon, and reset on switch)
    scalar      field state        (weather, terrain, turn, ...)

Every field has a static shape and dtype, so a `BattleState` can be batched with
`vmap` (add a leading dimension) and carried through `lax.scan`.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import consts as C

P, T, M = C.NUM_PLAYERS, C.TEAM_SIZE, C.MOVES_PER_POKEMON


class BattleState(NamedTuple):
    # --- team ---------------------------------------------------------------
    species: jnp.ndarray        # [P,T] int16
    level: jnp.ndarray          # [P,T] int8
    hp: jnp.ndarray             # [P,T] int16, 0 == fainted
    maxhp: jnp.ndarray          # [P,T] int16
    stats: jnp.ndarray          # [P,T,6] int16, final stats; index 0 unused
    types: jnp.ndarray          # [P,T,2] int8, current types (Tera/Burn Up mutate)
    moves: jnp.ndarray          # [P,T,4] int16
    pp: jnp.ndarray             # [P,T,4] int8
    maxpp: jnp.ndarray          # [P,T,4] int8
    item: jnp.ndarray           # [P,T] int16, 0 == none/consumed
    ability: jnp.ndarray        # [P,T] int16, current (Trace/Mummy mutate)
    status: jnp.ndarray         # [P,T] int8
    status_turns: jnp.ndarray   # [P,T] int8, sleep turns left / toxic counter
    tera_type: jnp.ndarray      # [P,T] int8
    terastallized: jnp.ndarray  # [P,T] bool

    # --- active slot --------------------------------------------------------
    active: jnp.ndarray             # [P] int8, team index of the active Pokemon
    boosts: jnp.ndarray             # [P,7] int8, -6..+6
    volatiles: jnp.ndarray          # [P,NUM_VOLATILES] int8, turns left; 0 absent
    sub_hp: jnp.ndarray             # [P] int16
    disabled_slot: jnp.ndarray      # [P] int8, -1 none
    encore_slot: jnp.ndarray        # [P] int8, -1 none
    locked_slot: jnp.ndarray        # [P] int8, -1 none (Outrage, charge moves)
    choice_slot: jnp.ndarray        # [P] int8, -1 none
    last_move: jnp.ndarray          # [P] int16, -1 none
    boosted_stat: jnp.ndarray       # [P] int8, Protosynthesis/Quark Drive stat
    times_hit: jnp.ndarray          # [P] int8, Rage Fist
    protect_streak: jnp.ndarray     # [P] int8, consecutive protects
    damage_taken: jnp.ndarray       # [P] int16, damage this turn (Counter/Avalanche)
    damage_category: jnp.ndarray    # [P] int8, category of that damage; -1 none
    moved_this_turn: jnp.ndarray    # [P] bool
    switched_this_turn: jnp.ndarray # [P] bool
    fainted_count: jnp.ndarray      # [P] int8, Last Respects / Supreme Overlord
    moves_since_switch: jnp.ndarray # [P] int8, gates Fake Out / First Impression
    last_move_failed: jnp.ndarray   # [P] bool, Stomping Tantrum
    stats_lowered: jnp.ndarray      # [P] bool, Lash Out (this turn)

    # --- field --------------------------------------------------------------
    weather: jnp.ndarray            # scalar int8
    weather_turns: jnp.ndarray      # scalar int8
    terrain: jnp.ndarray            # scalar int8
    terrain_turns: jnp.ndarray      # scalar int8
    trick_room: jnp.ndarray         # scalar int8, turns left
    gravity: jnp.ndarray            # scalar int8, turns left
    side_conditions: jnp.ndarray    # [P,NUM_SIDE_CONDITIONS] int8

    # --- bookkeeping --------------------------------------------------------
    turn: jnp.ndarray               # scalar int32
    phase: jnp.ndarray              # scalar int8, PHASE_*
    force_switch: jnp.ndarray       # [P] bool, this player owes a replacement
    phazed: jnp.ndarray             # [P] bool, forced out by the opponent this turn
    #: A self-switch (U-turn) suspends the turn while its user picks a
    #: replacement, so the opponent's already-locked action has to be held over
    #: and run once the replacement is in. -1 when nothing is pending.
    pending_side: jnp.ndarray       # scalar int8
    pending_action: jnp.ndarray     # scalar int8
    winner: jnp.ndarray             # scalar int8, -1 ongoing, 0/1 winner, 2 tie
    key: jnp.ndarray                # PRNG key

    # --- convenience --------------------------------------------------------
    @property
    def fainted(self) -> jnp.ndarray:
        """[P,T] bool."""
        return self.hp <= 0

    def active_index(self):
        """Index arrays for gathering the active Pokemon: `state.hp[pi, ai]`."""
        return jnp.arange(P), self.active.astype(jnp.int32)


# --- writing without a scatter -----------------------------------------------
# `x.at[i].set(v)` is a scatter, and on GPU every scatter is a kernel of its own:
# nothing fuses through it, and XLA usually copies the operand first so the write
# can happen in place. The engine did a few hundred of them per step, and they
# and their copies were half the GPU time and most of the kernel launches. A
# select against an index mask is elementwise, so it fuses with its neighbours,
# and it touches a handful of bytes per battle either way.

def set_at(field, index, value, when=True):
    """`field.at[index].set(value)`, as an elementwise select rather than a scatter.

    `index` addresses leading axes: one index or a tuple, each entry an int, a
    traced scalar, or `slice(None)` for a whole axis. `value` broadcasts against
    the axes that are not indexed. Nothing is written where `when` is false, nor
    for an index outside `[0, size)` -- unlike `.at[]`, a negative index does not
    wrap, so the engine's `-1` for "none" writes nothing.
    """
    index = index if isinstance(index, tuple) else (index,)
    mask = jnp.asarray(when, bool)
    for axis, i in enumerate(index):
        if isinstance(i, slice):
            assert i == slice(None), "only whole-axis slices are supported"
            continue
        shape = [1] * field.ndim
        shape[axis] = field.shape[axis]
        mask = mask & (jnp.arange(field.shape[axis]).reshape(shape) == i)
    return jnp.where(mask, jnp.asarray(value, field.dtype), field)


def add_at(field, index, amount, when=True):
    """`field.at[index].add(amount)`, as `set_at` does `.set`."""
    return set_at(field, index, field + jnp.asarray(amount, field.dtype), when)


def select_state(pred, if_true, if_false):
    """`if_true` where `pred`, else `if_false`, field by field.

    This is what `lax.cond` becomes under `vmap` anyway -- both branches run and
    the results are selected -- but written out. A batched `lax.cond` broadcasts
    every operand of its branches to the batch first, and those operands include
    the data tables the branches read: the type chart went out as a
    `[batch, 19, 19]` array on every step.
    """
    return jax.tree.map(lambda a, b: jnp.where(pred, a, b), if_true, if_false)


def gather_active(field: jnp.ndarray, active: jnp.ndarray) -> jnp.ndarray:
    """Take the active slot out of a `[P,T,...]` field, giving `[P,...]`."""
    return jnp.take_along_axis(
        field, active.astype(jnp.int32).reshape(P, 1, *([1] * (field.ndim - 2))), axis=1
    ).squeeze(1)


def set_active(field: jnp.ndarray, active: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    """Write `value` ([P,...]) back into the active slot of a `[P,T,...]` field."""
    idx = active.astype(jnp.int32).reshape(P, 1, *([1] * (field.ndim - 2)))
    return jnp.put_along_axis(field, idx, jnp.expand_dims(value, 1), axis=1,
                              inplace=False)


def empty_state(key: jnp.ndarray) -> BattleState:
    """An all-zero state with correct shapes; `teams.py` fills in the Pokemon."""
    i8 = lambda *s: jnp.zeros(s, jnp.int8)
    i16 = lambda *s: jnp.zeros(s, jnp.int16)
    return BattleState(
        species=i16(P, T), level=jnp.full((P, T), 100, jnp.int8),
        hp=i16(P, T), maxhp=jnp.ones((P, T), jnp.int16),
        stats=jnp.ones((P, T, 6), jnp.int16),
        types=jnp.full((P, T, 2), C.TYPE_NONE, jnp.int8),
        moves=jnp.full((P, T, M), -1, jnp.int16), pp=i8(P, T, M), maxpp=i8(P, T, M),
        item=i16(P, T), ability=i16(P, T), status=i8(P, T), status_turns=i8(P, T),
        tera_type=jnp.full((P, T), C.TYPE_NONE, jnp.int8),
        terastallized=jnp.zeros((P, T), bool),
        active=i8(P), boosts=i8(P, C.NUM_BOOSTS),
        volatiles=i8(P, C.NUM_VOLATILES), sub_hp=i16(P),
        disabled_slot=jnp.full((P,), -1, jnp.int8),
        encore_slot=jnp.full((P,), -1, jnp.int8),
        locked_slot=jnp.full((P,), -1, jnp.int8),
        choice_slot=jnp.full((P,), -1, jnp.int8),
        last_move=jnp.full((P,), -1, jnp.int16),
        boosted_stat=jnp.full((P,), -1, jnp.int8),
        times_hit=i8(P), protect_streak=i8(P), damage_taken=i16(P),
        damage_category=jnp.full((P,), -1, jnp.int8),
        moved_this_turn=jnp.zeros((P,), bool), switched_this_turn=jnp.zeros((P,), bool),
        fainted_count=i8(P), moves_since_switch=i8(P),
        last_move_failed=jnp.zeros((P,), bool), stats_lowered=jnp.zeros((P,), bool),
        weather=jnp.int8(0), weather_turns=jnp.int8(0),
        terrain=jnp.int8(0), terrain_turns=jnp.int8(0),
        trick_room=jnp.int8(0), gravity=jnp.int8(0),
        side_conditions=i8(P, C.NUM_SIDE_CONDITIONS),
        turn=jnp.int32(0), phase=jnp.int8(C.PHASE_MOVE),
        force_switch=jnp.zeros((P,), bool), phazed=jnp.zeros((P,), bool),
        pending_side=jnp.int8(-1), pending_action=jnp.int8(0),
        winner=jnp.int8(-1), key=key,
    )


def reset_slot_state(state: BattleState, player: jnp.ndarray) -> BattleState:
    """Clear everything tied to the active slot rather than the Pokemon.

    Called on every switch-out: boosts, volatiles and the choice lock belong to
    the slot and do not follow a Pokemon to the bench. Baton Pass re-applies the
    parts it carries afterwards.
    """
    p = player
    z = lambda a, v=0: set_at(a, p, v)
    return state._replace(
        boosts=z(state.boosts), volatiles=z(state.volatiles),
        sub_hp=z(state.sub_hp), disabled_slot=z(state.disabled_slot, -1),
        encore_slot=z(state.encore_slot, -1), locked_slot=z(state.locked_slot, -1),
        choice_slot=z(state.choice_slot, -1), last_move=z(state.last_move, -1),
        boosted_stat=z(state.boosted_stat, -1), times_hit=z(state.times_hit),
        protect_streak=z(state.protect_streak),
        moves_since_switch=z(state.moves_since_switch),
        last_move_failed=z(state.last_move_failed, False),
    )
