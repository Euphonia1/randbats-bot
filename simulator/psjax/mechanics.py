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
from .stats import boost_multiply, chain_modify

M1 = 4096


def m(x: float) -> int:
    return int(x * 4096)


# --- lookups on the active Pokemon -------------------------------------------

def act(state, side):
    """The team index of `side`'s active Pokemon."""
    return state.active[side].astype(jnp.int32)


def active_types(state, side):
    i = act(state, side)
    return current_types(state.types[side, i], state.terastallized[side, i],
                         state.tera_type[side, i])


def is_grounded(state, side) -> jnp.ndarray:
    """Ground moves, hazards and terrain all key off this."""
    i = act(state, side)
    types = active_types(state, side)
    ungrounded = (jnp.any(types == C.FLYING) |
                  (state.ability[side, i] == A.LEVITATE) |
                  (state.item[side, i] == I.AIRBALLOON) |
                  (state.volatiles[side, C.V_MAGNETRISE] > 0) |
                  (state.volatiles[side, C.V_ROOST] > 0))
    forced = (state.gravity > 0) | (state.volatiles[side, C.V_INGRAIN] > 0)
    return forced | jnp.logical_not(ungrounded)


def weather_active(state) -> jnp.ndarray:
    """Cloud Nine / Air Lock on either active Pokemon suppresses the weather."""
    abilities = state.ability[jnp.arange(C.NUM_PLAYERS), state.active.astype(jnp.int32)]
    return jnp.logical_not(jnp.any(abilities == A.NEUTRALIZINGGAS)) & (
        state.weather != C.WEATHER_NONE)


def effective_weather(state):
    return jnp.where(weather_active(state), state.weather, jnp.int8(C.WEATHER_NONE))


def effective_speed(state, side) -> jnp.ndarray:
    """Speed after boosts, abilities, items, Tailwind and paralysis."""
    i = act(state, side)
    spe = boost_multiply(state.stats[side, i, C.SPE], state.boosts[side, C.B_SPE])
    ab, it = state.ability[side, i], state.item[side, i]
    weather = effective_weather(state)
    status = state.status[side, i]

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
    mod = apply(mod, it == I.CHOICESCARF, m(1.5))
    mod = apply(mod, state.side_conditions[side, C.SC_TAILWIND] > 0, m(2.0))
    spe = chain_modify(spe, mod)

    # Paralysis halves Speed from Gen 7 on, and Quick Feet ignores it.
    spe = jnp.where((status == C.PAR) & (ab != A.QUICKFEET), spe // 2, spe)
    return jnp.maximum(spe, 1)


# --- HP ----------------------------------------------------------------------

def damage_pokemon(state, side, slot, amount):
    """Subtract HP, clamped at 0. Fainting is resolved separately."""
    amount = jnp.maximum(amount.astype(jnp.int32), 0)
    hp = state.hp[side, slot].astype(jnp.int32)
    new = jnp.maximum(hp - amount, 0).astype(jnp.int16)
    return state._replace(hp=state.hp.at[side, slot].set(new)), (hp - new)


def heal_pokemon(state, side, slot, amount):
    """Add HP, clamped at max. Healing a fainted Pokemon is a no-op."""
    amount = jnp.maximum(amount.astype(jnp.int32), 0)
    hp = state.hp[side, slot].astype(jnp.int32)
    maxhp = state.maxhp[side, slot].astype(jnp.int32)
    new = jnp.where(hp <= 0, hp, jnp.minimum(hp + amount, maxhp)).astype(jnp.int16)
    return state._replace(hp=state.hp.at[side, slot].set(new)), (new - hp)


def fraction_of_max(state, side, slot, num, den):
    """`floor(maxhp * num / den)`, at least 1 -- Showdown's standard chip amount."""
    maxhp = state.maxhp[side, slot].astype(jnp.int32)
    return jnp.maximum((maxhp * num) // den, 1)


# --- boosts ------------------------------------------------------------------

def boost_immune(state, side, lowering):
    """Abilities that block stat drops from an opponent."""
    i = act(state, side)
    ab = state.ability[side, i]
    blanket = ((ab == A.CLEARBODY) | (ab == A.WHITESMOKE) | (ab == A.FULLMETALBODY))
    item = state.item[side, i] == I.CLEARAMULET
    return lowering & (blanket | item |
                       (state.side_conditions[side, C.SC_MIST] > 0))


def apply_boosts(state, side, delta, from_opponent=False):
    """Apply a `[7]` boost delta, clamped to -6..+6.

    Contrary inverts it; Simple doubles it; Clear Body and friends block drops
    that come from the opponent. Returns the state and the delta actually applied
    (Defiant and Competitive key off that).
    """
    i = act(state, side)
    ab = state.ability[side, i]
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

def status_immune(data, state, side, status, key):
    """True if `side`'s active cannot be given `status` right now."""
    i = act(state, side)
    ab = state.ability[side, i]
    types = active_types(state, side)

    already = state.status[side, i] != C.STATUS_NONE
    mask = data["ability_status_immune"][ab]
    by_ability = ((mask >> status.astype(jnp.int32)) & 1).astype(bool)

    # Typing immunities.
    poison = (status == C.PSN) | (status == C.TOX)
    poison_immune = poison & (jnp.any(types == C.POISON) | jnp.any(types == C.STEEL))
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


def set_status(data, state, side, status, key):
    """Try to inflict a major status. Returns `(state, applied)`."""
    i = act(state, side)
    blocked = status_immune(data, state, side, status, key)
    ok = jnp.logical_not(blocked) & (status != C.STATUS_NONE) & (state.hp[side, i] > 0)

    # Sleep lasts 1-3 turns in Gen 5+; Toxic starts its counter at 1.
    sleep_turns = jax.random.randint(key, (), 2, 5).astype(jnp.int8)
    turns = jnp.where(status == C.SLP, sleep_turns,
                      jnp.where(status == C.TOX, jnp.int8(1), jnp.int8(0)))

    new_status = jnp.where(ok, status, state.status[side, i]).astype(jnp.int8)
    new_turns = jnp.where(ok, turns, state.status_turns[side, i]).astype(jnp.int8)
    return state._replace(
        status=state.status.at[side, i].set(new_status),
        status_turns=state.status_turns.at[side, i].set(new_turns),
    ), ok


def cure_status(state, side, slot):
    return state._replace(
        status=state.status.at[side, slot].set(jnp.int8(C.STATUS_NONE)),
        status_turns=state.status_turns.at[side, slot].set(jnp.int8(0)),
    )


# --- volatiles ---------------------------------------------------------------

def set_volatile(state, side, volatile, turns=1):
    return state._replace(
        volatiles=state.volatiles.at[side, volatile].set(jnp.asarray(turns, jnp.int8)))


def has_volatile(state, side, volatile):
    return state.volatiles[side, volatile] > 0


def clear_volatile(state, side, volatile):
    return state._replace(volatiles=state.volatiles.at[side, volatile].set(jnp.int8(0)))
