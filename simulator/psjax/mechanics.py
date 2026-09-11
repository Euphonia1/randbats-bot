"""Shared in-battle derived quantities: speed, grounding, boosts, status, HP.

These sit between the raw `BattleState` and the turn logic. Everything takes and
returns a `BattleState`, and every index may be a traced value, so the whole
module composes under `jit` and `vmap`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import consts as C
from .damage import current_types
from .hooks import A, I
from .stats import boost_multiply, chain_modify, idiv

M1 = 4096


def m(x: float) -> int:
    return int(x * 4096)


# --- lookups on the active Pokemon -------------------------------------------

def act(state, side):
    """The team index of `side`'s active Pokemon."""
    return state.active[side].astype(jnp.int32)


def active_types(state, side):
    i = act(state, side)
    return current_types(slot_get(state.types, side, i),
                         slot_get(state.terastallized, side, i),
                         slot_get(state.tera_type, side, i))


def is_grounded(state, side) -> jnp.ndarray:
    """Ground moves, hazards and terrain all key off this."""
    i = act(state, side)
    types = active_types(state, side)
    ungrounded = (jnp.any(types == C.FLYING) |
                  (slot_get(state.ability, side, i) == A.LEVITATE) |
                  (slot_get(state.item, side, i) == I.AIRBALLOON) |
                  (state.volatiles[side, C.V_MAGNETRISE] > 0) |
                  (state.volatiles[side, C.V_ROOST] > 0))
    forced = (state.gravity > 0) | (state.volatiles[side, C.V_INGRAIN] > 0)
    return forced | jnp.logical_not(ungrounded)


def weather_active(state) -> jnp.ndarray:
    """Air Lock or Cloud Nine on either active Pokemon suppresses the weather."""
    abilities = state.ability[jnp.arange(C.NUM_PLAYERS), state.active.astype(jnp.int32)]
    suppressed = jnp.any((abilities == A.AIRLOCK) | (abilities == A.CLOUDNINE))
    return jnp.logical_not(suppressed) & (state.weather != C.WEATHER_NONE)


def effective_weather(state):
    return jnp.where(weather_active(state), state.weather, jnp.int8(C.WEATHER_NONE))


def effective_speed(state, side) -> jnp.ndarray:
    """Speed after boosts, abilities, items, Tailwind and paralysis."""
    i = act(state, side)
    spe = boost_multiply(slot_get(state.stats, side, i)[C.SPE], state.boosts[side, C.B_SPE])
    ab, it = slot_get(state.ability, side, i), slot_get(state.item, side, i)
    weather = effective_weather(state)
    status = slot_get(state.status, side, i)

    mod = jnp.int32(M1)
    apply = lambda mod, cond, f: jnp.where(cond, chain_modify(mod, f), mod)
    sun = (weather == C.SUN) | (weather == C.HARSH_SUN)
    rain = (weather == C.RAIN) | (weather == C.HEAVY_RAIN)
    mod = apply(mod, (ab == A.CHLOROPHYLL) & sun, m(2.0))
    mod = apply(mod, (ab == A.SWIFTSWIM) & rain, m(2.0))
    mod = apply(mod, (ab == A.SANDRUSH) & (weather == C.SAND), m(2.0))
    mod = apply(mod, (ab == A.SLUSHRUSH) & (weather == C.SNOW), m(2.0))
    mod = apply(mod, (ab == A.QUICKFEET) & (status != C.STATUS_NONE), m(1.5))
    mod = apply(mod, (ab == A.SLOWSTART) & (state.volatiles[side, C.V_SLOWSTART] > 0), m(0.5))
    mod = apply(mod, (ab == A.SURGESURFER) & (state.terrain == C.ELECTRIC_TERRAIN), m(2.0))
    # Unburden doubles Speed once the holder has lost its item.
    mod = apply(mod, (ab == A.UNBURDEN) &
                (state.volatiles[side, C.V_UNBURDEN] > 0) & (it == 0), m(2.0))
    # Protosynthesis / Quark Drive give Speed x1.5 rather than the x1.3 the
    # other stats get.
    mod = apply(mod, state.boosted_stat[side] == C.SPE, m(1.5))
    mod = apply(mod, it == I.CHOICESCARF, m(1.5))
    mod = apply(mod, state.side_conditions[side, C.SC_TAILWIND] > 0, m(2.0))
    spe = chain_modify(spe, mod)

    # Paralysis halves Speed from Gen 7 on, and Quick Feet ignores it.
    spe = jnp.where((status == C.PAR) & (ab != A.QUICKFEET), spe // 2, spe)
    return jnp.maximum(spe, 1)


# --- indexing the active slot without a gather -------------------------------
# `state.hp[side, slot]` looks innocuous, but `slot` is battle data and therefore
# traced, so it lowers to a gather over the team axis -- and under `vmap` a
# batched gather. There are dozens of these per move. Selecting over the six
# slots instead keeps everything elementwise.

def slot_get(field, side, slot):
    """`field[side, slot]` for a `[P, T]` or `[P, T, K]` field, without a gather."""
    row = field[side]
    mask = jnp.arange(C.TEAM_SIZE) == slot
    if row.ndim > 1:
        mask = mask.reshape(-1, *([1] * (row.ndim - 1)))
    return jnp.sum(jnp.where(mask, row, 0), axis=0)


def slot_set(field, side, slot, value, when=True):
    """`field.at[side, slot].set(value)` without a scatter."""
    row = field[side]
    mask = (jnp.arange(C.TEAM_SIZE) == slot) & when
    if row.ndim > 1:
        mask = mask.reshape(-1, *([1] * (row.ndim - 1)))
    return field.at[side].set(jnp.where(mask, jnp.asarray(value, field.dtype), row))


# --- HP ----------------------------------------------------------------------

def damage_pokemon(state, side, slot, amount):
    """Subtract HP, clamped at 0. Fainting is resolved separately."""
    amount = jnp.maximum(amount.astype(jnp.int32), 0)
    hp = slot_get(state.hp, side, slot).astype(jnp.int32)
    new = jnp.maximum(hp - amount, 0).astype(jnp.int16)
    return state._replace(hp=slot_set(state.hp, side, slot, new)), (hp - new)


def heal_pokemon(state, side, slot, amount):
    """Add HP, clamped at max. Healing a fainted Pokemon is a no-op."""
    amount = jnp.maximum(amount.astype(jnp.int32), 0)
    hp = slot_get(state.hp, side, slot).astype(jnp.int32)
    maxhp = slot_get(state.maxhp, side, slot).astype(jnp.int32)
    new = jnp.where(hp <= 0, hp, jnp.minimum(hp + amount, maxhp)).astype(jnp.int16)
    return state._replace(hp=slot_set(state.hp, side, slot, new)), (new - hp)


def fraction_of_max(state, side, slot, num, den):
    """`floor(maxhp * num / den)`, at least 1 -- Showdown's standard chip amount."""
    maxhp = slot_get(state.maxhp, side, slot).astype(jnp.int32)
    return jnp.maximum(idiv(maxhp * num, den), 1)


# --- boosts ------------------------------------------------------------------

def boost_immune(state, side, lowering):
    """Abilities that block stat drops from an opponent."""
    i = act(state, side)
    ab = slot_get(state.ability, side, i)
    blanket = ((ab == A.CLEARBODY) | (ab == A.WHITESMOKE) | (ab == A.FULLMETALBODY))
    item = slot_get(state.item, side, i) == I.CLEARAMULET
    return lowering & (blanket | item |
                       (state.side_conditions[side, C.SC_MIST] > 0))


def apply_boosts(state, side, delta, from_opponent=False):
    """Apply a `[7]` boost delta, clamped to -6..+6.

    Contrary inverts it; Simple doubles it; Clear Body and friends block drops
    that come from the opponent. Returns the state and the delta actually applied
    (Defiant and Competitive key off that).
    """
    i = act(state, side)
    ab = slot_get(state.ability, side, i)
    delta = delta.astype(jnp.int32)
    delta = jnp.where(ab == A.CONTRARY, -delta, delta)
    delta = jnp.where(ab == A.SIMPLE, delta * 2, delta)

    lowering = delta < 0
    blocked = boost_immune(state, side, lowering) & from_opponent
    # Targeted stat-drop protections.
    blocked = blocked | (lowering & from_opponent & (
        ((ab == A.HYPERCUTTER) & (jnp.arange(7) == C.B_ATK)) |
        ((ab == A.BIGPECKS) & (jnp.arange(7) == C.B_DEF)) |
        ((ab == A.KEENEYE) & (jnp.arange(7) == C.B_ACC))))
    delta = jnp.where(blocked, 0, delta)

    old = state.boosts[side].astype(jnp.int32)
    new = jnp.clip(old + delta, -6, 6)
    applied = new - old
    return state._replace(boosts=state.boosts.at[side].set(new.astype(jnp.int8))), applied


# --- status ------------------------------------------------------------------

def status_immune(data, state, side, status, key, source_ability=None):
    """True if `side`'s active cannot be given `status` right now.

    `source_ability` is the ability of whoever is inflicting it, so Corrosion can
    poison Steel and Poison types.
    """
    i = act(state, side)
    ab = slot_get(state.ability, side, i)
    types = active_types(state, side)

    already = slot_get(state.status, side, i) != C.STATUS_NONE
    mask = data["ability_status_immune"][ab]
    by_ability = ((mask >> status.astype(jnp.int32)) & 1).astype(bool)

    # Typing immunities.
    poison = (status == C.PSN) | (status == C.TOX)
    corrosive = jnp.bool_(False) if source_ability is None else \
        (source_ability == A.CORROSION)
    poison_immune = poison & jnp.logical_not(corrosive) & (
        jnp.any(types == C.POISON) | jnp.any(types == C.STEEL))
    burn_immune = (status == C.BRN) & jnp.any(types == C.FIRE)
    para_immune = (status == C.PAR) & jnp.any(types == C.ELECTRIC)
    freeze_immune = (status == C.FRZ) & jnp.any(types == C.ICE)

    # Field protections.
    misty = (state.terrain == C.MISTY_TERRAIN) & is_grounded(state, side)
    electric_sleep = (status == C.SLP) & (state.terrain == C.ELECTRIC_TERRAIN) & \
                     is_grounded(state, side)
    safeguard = state.side_conditions[side, C.SC_SAFEGUARD] > 0
    sun_freeze = (status == C.FRZ) & (
        (effective_weather(state) == C.SUN) | (effective_weather(state) == C.HARSH_SUN))

    return (already | by_ability | poison_immune | burn_immune | para_immune |
            freeze_immune | misty | electric_sleep | safeguard | sun_freeze)


def set_status(data, state, side, status, key, source_ability=None):
    """Try to inflict a major status. Returns `(state, applied)`."""
    i = act(state, side)
    blocked = status_immune(data, state, side, status, key, source_ability)
    ok = jnp.logical_not(blocked) & (status != C.STATUS_NONE) & \
        (slot_get(state.hp, side, i) > 0)

    # Sleep lasts 1-3 turns in Gen 5+; Toxic starts its counter at 1.
    sleep_turns = jax.random.randint(key, (), 2, 5).astype(jnp.int8)
    # Early Bird sleeps for half as long (rounded down, minimum one turn).
    sleep_turns = jnp.where(slot_get(state.ability, side, i) == A.EARLYBIRD,
                            jnp.maximum(sleep_turns // 2, 1), sleep_turns)
    turns = jnp.where(status == C.SLP, sleep_turns,
                      jnp.where(status == C.TOX, jnp.int8(1), jnp.int8(0)))

    cur_status = slot_get(state.status, side, i)
    cur_turns = slot_get(state.status_turns, side, i)
    new_status = jnp.where(ok, status, cur_status).astype(jnp.int8)
    new_turns = jnp.where(ok, turns, cur_turns).astype(jnp.int8)
    return state._replace(
        status=slot_set(state.status, side, i, new_status),
        status_turns=slot_set(state.status_turns, side, i, new_turns),
    ), ok


def synchronize(data, state, side, status, key, when=True):
    """Synchronize passes a burn, paralysis or poison back to whoever caused it.

    `when` additionally gates this (e.g. on whether the status actually landed);
    folded into the mask rather than an outer `lax.cond` for the same reason as
    `cure_status`.
    """
    i = act(state, side)
    other = 1 - side
    passes_back = when & (slot_get(state.ability, side, i) == A.SYNCHRONIZE) & (
        (status == C.BRN) | (status == C.PAR) | (status == C.PSN) | (status == C.TOX))
    masked_status = jnp.where(passes_back, status, jnp.int8(C.STATUS_NONE))
    state, _ = set_status(data, state, other, masked_status, key)
    return state


def cure_status(state, side, slot, when=True):
    """Clear major status. `when=False` is a no-op, without branching on it.

    Takes a mask instead of being wrapped in `lax.cond` at the call site: under
    `vmap`, `lax.cond` computes both branches and selects over the *entire*
    returned pytree, so wrapping a two-field update in it costs as much as
    wrapping a forty-field one. Masking the two fields directly is what makes
    `execute_move` compile in seconds rather than minutes.
    """
    cur_status = slot_get(state.status, side, slot)
    cur_turns = slot_get(state.status_turns, side, slot)
    new_status = jnp.where(when, jnp.int8(C.STATUS_NONE), cur_status)
    new_turns = jnp.where(when, jnp.int8(0), cur_turns)
    return state._replace(
        status=slot_set(state.status, side, slot, new_status),
        status_turns=slot_set(state.status_turns, side, slot, new_turns),
    )


def set_indexed(vector, index, value, when=True):
    """`vector.at[index].set(value)` without a scatter.

    The indices here (which volatile, which side condition, which stat) come from
    the move data and so are traced. A scatter with a traced index is expensive
    for XLA to lower under `vmap`; a comparison against `arange` is a cheap
    elementwise select and compiles in a fraction of the time.
    """
    hit = (jnp.arange(vector.shape[0]) == index) & when
    return jnp.where(hit, jnp.asarray(value, vector.dtype), vector)


def get_indexed(vector, index):
    """`vector[index]` without a gather, for the same reason."""
    return jnp.sum(jnp.where(jnp.arange(vector.shape[0]) == index, vector, 0))


# --- volatiles ---------------------------------------------------------------

def set_volatile(state, side, volatile, turns=1):
    return state._replace(
        volatiles=state.volatiles.at[side, volatile].set(jnp.asarray(turns, jnp.int8)))


def has_volatile(state, side, volatile):
    return state.volatiles[side, volatile] > 0


def clear_volatile(state, side, volatile):
    return state._replace(volatiles=state.volatiles.at[side, volatile].set(jnp.int8(0)))
