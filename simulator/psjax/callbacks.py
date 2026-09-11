"""Implementations of Showdown's per-move JS callbacks.

Each family (`basePowerCallback`, `onBasePower`, `damageCallback`,
`onModifyType`) becomes a `lax.switch` over handlers indexed by the compiled
`move_bp_replace` / `move_bp_modify` / `move_dmg_cb` / `move_type_cb` columns.
Every handler takes the same `CbCtx` and returns a scalar, so the switch is
uniform and cheap to trace.

Formulas are transcribed from `data/moves.ts`, including the integer rounding:
Wring Out and Hard Press in particular use a hand-rolled fixed-point expression
that a naive `power * hp // maxhp` does not reproduce.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import consts as C
from . import effects as E
from .stats import chain_modify, idiv


class CbCtx(NamedTuple):
    """Everything the move callbacks can read."""
    base_power: jnp.ndarray
    move_type: jnp.ndarray
    hit_number: jnp.ndarray          # 1-based index within a multi-hit move
    atk_status: jnp.ndarray
    dfn_status: jnp.ndarray
    atk_hp: jnp.ndarray
    atk_maxhp: jnp.ndarray
    dfn_hp: jnp.ndarray
    dfn_maxhp: jnp.ndarray
    atk_item: jnp.ndarray
    dfn_item: jnp.ndarray
    atk_weight: jnp.ndarray          # kg
    dfn_weight: jnp.ndarray
    atk_speed: jnp.ndarray
    dfn_speed: jnp.ndarray
    atk_boosts: jnp.ndarray          # [7]
    dfn_boosts: jnp.ndarray          # [7]
    moves_first: jnp.ndarray         # user acts before the target this turn
    user_damaged: jnp.ndarray        # user was damaged by the target this turn
    target_damaged: jnp.ndarray      # target already took damage this turn
    weather: jnp.ndarray
    terrain: jnp.ndarray
    grounded_user: jnp.ndarray
    grounded_target: jnp.ndarray
    pp_left: jnp.ndarray
    times_hit: jnp.ndarray
    fainted_count: jnp.ndarray
    terastallized: jnp.ndarray
    tera_type: jnp.ndarray
    fury_multiplier: jnp.ndarray     # consecutive uses of Fury Cutter / Rollout
    level: jnp.ndarray
    last_damage: jnp.ndarray         # damage the user took this turn
    last_damage_category: jnp.ndarray
    off_atk: jnp.ndarray             # boosted but unmodified Attack ...
    off_spa: jnp.ndarray             # ... and Sp. Atk, for the category switches
    user_ability: jnp.ndarray        # for the "-ate" retyping
    last_move_failed: jnp.ndarray    # Stomping Tantrum
    stats_lowered: jnp.ndarray       # Lash Out

M1 = 4096


def _steps(value, thresholds, powers):
    """Piecewise-constant lookup: the power for the highest threshold met."""
    out = jnp.int32(powers[0])
    for t, p in zip(thresholds, powers[1:]):
        out = jnp.where(value >= t, jnp.int32(p), out)
    return out


def _positive_boosts(boosts):
    return jnp.sum(jnp.maximum(boosts[:5].astype(jnp.int32), 0))


# --- basePowerCallback -------------------------------------------------------

def _bp_none(c): return c.base_power
def _bp_acrobatics(c): return jnp.where(c.atk_item == 0, c.base_power * 2, c.base_power)
def _bp_assurance(c): return jnp.where(c.target_damaged, c.base_power * 2, c.base_power)
def _bp_avalanche(c): return jnp.where(c.user_damaged, c.base_power * 2, c.base_power)
def _bp_payback(c): return jnp.where(c.moves_first, c.base_power, c.base_power * 2)
def _bp_boltbeak(c): return jnp.where(c.moves_first, c.base_power * 2, c.base_power)


def _bp_electroball(c):
    # Showdown steps on floor(userSpeed / targetSpeed); comparing against
    # multiples avoids the division entirely.
    d = jnp.maximum(c.dfn_speed.astype(jnp.int32), 1)
    a = c.atk_speed.astype(jnp.int32)
    bp = jnp.int32(40)
    for mult, power in ((1, 60), (2, 80), (3, 120), (4, 150)):
        bp = jnp.where(a >= mult * d, jnp.int32(power), bp)
    return bp


def _bp_gyroball(c):
    power = idiv(25 * c.dfn_speed, c.atk_speed) + 1
    return jnp.clip(power, 1, 150)


def _bp_lowkick(c):
    # Showdown's getWeight() returns hectograms, so 200kg (the 120 BP cutoff)
    # is 2000 in these units.
    w = (c.dfn_weight * 10).astype(jnp.int32)
    return _steps(w, [100, 250, 500, 1000, 2000], [20, 40, 60, 80, 100, 120])


def _bp_heavyslam(c):
    # Same trick as Electro Ball: the steps are on the weight ratio.
    aw = (c.atk_weight * 10).astype(jnp.int32)
    dw = jnp.maximum((c.dfn_weight * 10).astype(jnp.int32), 1)
    bp = jnp.int32(40)
    for mult, power in ((2, 60), (3, 80), (4, 100), (5, 120)):
        bp = jnp.where(aw >= mult * dw, jnp.int32(power), bp)
    return bp


def _bp_eruption(c):
    return jnp.maximum(idiv(c.base_power * c.atk_hp, c.atk_maxhp), 1)


def _hp_fraction_power(base, hp, maxhp):
    """Showdown's Wring Out / Hard Press rounding, transcribed literally."""
    frac = idiv(hp.astype(jnp.int32) * 4096, maxhp)
    bp = (((base * (100 * frac)) + 2048 - 1) // 4096) // 100
    return jnp.maximum(bp, 1)


def _bp_wringout(c): return _hp_fraction_power(120, c.dfn_hp, c.dfn_maxhp)
def _bp_hardpress(c): return _hp_fraction_power(100, c.dfn_hp, c.dfn_maxhp)
def _bp_storedpower(c): return c.base_power + 20 * _positive_boosts(c.atk_boosts)
def _bp_punishment(c): return jnp.minimum(60 + 20 * _positive_boosts(c.dfn_boosts), 200)
def _bp_ragefist(c): return jnp.minimum(50 + 50 * c.times_hit.astype(jnp.int32), 350)
def _bp_lastrespects(c): return 50 + 50 * c.fainted_count.astype(jnp.int32)


def _bp_trumpcard(c):
    table = jnp.array([200, 80, 60, 50], jnp.int32)
    return jnp.where(c.pp_left >= 4, 40, table[jnp.clip(c.pp_left, 0, 3)])


def _bp_terablast(c):
    return jnp.where(c.terastallized & (c.tera_type == C.STELLAR), 100, c.base_power)


def _bp_weatherball(c):
    active = (c.weather != C.WEATHER_NONE) & (c.weather != C.STRONG_WINDS)
    return jnp.where(active, c.base_power * 2, c.base_power)


def _bp_terrainpulse(c):
    on = (c.terrain != C.TERRAIN_NONE) & c.grounded_user
    return jnp.where(on, c.base_power * 2, c.base_power)


def _bp_risingvoltage(c):
    on = (c.terrain == C.ELECTRIC_TERRAIN) & c.grounded_target
    return jnp.where(on, c.base_power * 2, c.base_power)


def _bp_multihit_scaling(c): return c.base_power * jnp.maximum(c.hit_number, 1)


def _bp_furycutter(c):
    return jnp.clip(c.base_power * jnp.maximum(c.fury_multiplier, 1), 1, 160)


# Happiness is not modelled; Showdown's generators leave it at the 255 maximum,
# which pins Return at 102 and Frustration at its floor of 1.
def _bp_happiness(c): return jnp.int32(102)
def _bp_frustration(c): return jnp.int32(1)


BP_REPLACE_FNS = {
    "none": _bp_none, "acrobatics": _bp_acrobatics, "assurance": _bp_assurance,
    "avalanche": _bp_avalanche, "payback": _bp_payback, "boltbeak": _bp_boltbeak,
    "electroball": _bp_electroball, "gyroball": _bp_gyroball, "lowkick": _bp_lowkick,
    "heavyslam": _bp_heavyslam, "eruption": _bp_eruption, "wringout": _bp_wringout,
    "hardpress": _bp_hardpress, "storedpower": _bp_storedpower,
    "punishment": _bp_punishment, "ragefist": _bp_ragefist,
    "lastrespects": _bp_lastrespects, "trumpcard": _bp_trumpcard,
    "terablast": _bp_terablast, "weatherball": _bp_weatherball,
    "terrainpulse": _bp_terrainpulse, "risingvoltage": _bp_risingvoltage,
    "multihit_scaling": _bp_multihit_scaling, "furycutter": _bp_furycutter,
    "happiness": _bp_happiness, "frustration": _bp_frustration,
}


# --- onBasePower -------------------------------------------------------------

def _m_none(c): return jnp.int32(M1)


def _m_facade(c):
    on = (c.atk_status != C.STATUS_NONE) & (c.atk_status != C.SLP)
    return jnp.where(on, 8192, M1)


def _m_hex(c): return jnp.where(c.dfn_status != C.STATUS_NONE, 8192, M1)


def _m_venoshock(c):
    return jnp.where((c.dfn_status == C.PSN) | (c.dfn_status == C.TOX), 8192, M1)


def _m_brine(c): return jnp.where(c.dfn_hp * 2 <= c.dfn_maxhp, 8192, M1)
def _m_knockoff(c): return jnp.where(c.dfn_item != 0, 6144, M1)


def _m_expandingforce(c):
    return jnp.where((c.terrain == C.PSYCHIC_TERRAIN) & c.grounded_user, 6144, M1)


def _m_mistyexplosion(c):
    return jnp.where((c.terrain == C.MISTY_TERRAIN) & c.grounded_user, 6144, M1)


def _m_psyblade(c): return jnp.where(c.terrain == C.ELECTRIC_TERRAIN, 6144, M1)


def _m_solarbeam(c):
    weak = ((c.weather == C.RAIN) | (c.weather == C.HEAVY_RAIN) |
            (c.weather == C.SAND) | (c.weather == C.SNOW))
    return jnp.where(weak, 2048, M1)


def _m_stompingtantrum(c):
    return jnp.where(c.last_move_failed, 8192, M1)


def _m_lashout(c):
    return jnp.where(c.stats_lowered, 8192, M1)


BP_MODIFY_FNS = {
    "none": _m_none, "facade": _m_facade, "hex": _m_hex, "venoshock": _m_venoshock,
    "brine": _m_brine, "knockoff": _m_knockoff, "expandingforce": _m_expandingforce,
    "mistyexplosion": _m_mistyexplosion, "psyblade": _m_psyblade,
    "solarbeam": _m_solarbeam, "stompingtantrum": _m_stompingtantrum,
    "lashout": _m_lashout,
}


# --- damageCallback ----------------------------------------------------------

def _d_none(c): return jnp.int32(-1)          # -1 == "use the damage formula"
def _d_level(c): return c.level.astype(jnp.int32)
def _d_fixed20(c): return jnp.int32(20)
def _d_fixed40(c): return jnp.int32(40)
def _d_halftarget(c): return jnp.maximum(c.dfn_hp.astype(jnp.int32) // 2, 1)
def _d_endeavor(c): return jnp.maximum(c.dfn_hp.astype(jnp.int32) - c.atk_hp.astype(jnp.int32), 0)
def _d_finalgambit(c): return c.atk_hp.astype(jnp.int32)


def _d_psywave(c):
    # Showdown: level * (random(0..100) + 50) / 100; we take the mean roll.
    return jnp.maximum((c.level.astype(jnp.int32) * 100) // 100, 1)


def _d_counter(c):
    ok = c.last_damage_category == C.CAT_PHYSICAL
    return jnp.where(ok, c.last_damage.astype(jnp.int32) * 2, -2)   # -2 == fail


def _d_mirrorcoat(c):
    ok = c.last_damage_category == C.CAT_SPECIAL
    return jnp.where(ok, c.last_damage.astype(jnp.int32) * 2, -2)


def _d_metalburst(c):
    ok = c.last_damage_category >= 0
    return jnp.where(ok, (c.last_damage.astype(jnp.int32) * 3) // 2, -2)


DMG_FNS = {
    "none": _d_none, "level": _d_level, "fixed20": _d_fixed20, "fixed40": _d_fixed40,
    "halftarget": _d_halftarget, "endeavor": _d_endeavor,
    "finalgambit": _d_finalgambit, "psywave": _d_psywave, "counter": _d_counter,
    "mirrorcoat": _d_mirrorcoat, "metalburst": _d_metalburst,
}


# --- onModifyType ------------------------------------------------------------

def _t_none(c): return c.move_type


def _t_weatherball(c):
    t = jnp.int8(C.NORMAL)
    t = jnp.where((c.weather == C.SUN) | (c.weather == C.HARSH_SUN), C.FIRE, t)
    t = jnp.where((c.weather == C.RAIN) | (c.weather == C.HEAVY_RAIN), C.WATER, t)
    t = jnp.where(c.weather == C.SAND, C.ROCK, t)
    t = jnp.where(c.weather == C.SNOW, C.ICE, t)
    return t.astype(jnp.int8)


def _t_terrainpulse(c):
    t = c.move_type
    on = c.grounded_user
    t = jnp.where(on & (c.terrain == C.ELECTRIC_TERRAIN), C.ELECTRIC, t)
    t = jnp.where(on & (c.terrain == C.GRASSY_TERRAIN), C.GRASS, t)
    t = jnp.where(on & (c.terrain == C.MISTY_TERRAIN), C.FAIRY, t)
    t = jnp.where(on & (c.terrain == C.PSYCHIC_TERRAIN), C.PSYCHIC, t)
    return t.astype(jnp.int8)


def _t_terablast(c):
    return jnp.where(c.terastallized, c.tera_type, c.move_type).astype(jnp.int8)


# Judgment / Techno Blast / Multi-Attack read the held plate, drive or memory.
# Those items are not in the modelled item set, so the move keeps its base type;
# `coverage.py` reports this rather than silently pretending otherwise.
def _t_item_typed(c): return c.move_type


TYPE_FNS = {
    "none": _t_none, "weatherball": _t_weatherball, "terrainpulse": _t_terrainpulse,
    "terablast": _t_terablast, "judgment": _t_item_typed,
    "technoblast": _t_item_typed, "multiattack": _t_item_typed,
    "revelationdance": _t_none, "ivycudgel": _t_item_typed,
    "ragingbull": _t_none, "aurawheel": _t_none, "naturalgift": _t_item_typed,
}


# --- weather-dependent accuracy ----------------------------------------------
# Handlers return the accuracy to use, or -1 for "never misses".

def _a_none(c): return jnp.int32(-2)          # -2 == "use the declared accuracy"


def _a_rain_perfect(c):
    rain = (c.weather == C.RAIN) | (c.weather == C.HEAVY_RAIN)
    sun = (c.weather == C.SUN) | (c.weather == C.HARSH_SUN)
    return jnp.where(rain, -1, jnp.where(sun, 50, -2)).astype(jnp.int32)


def _a_snow_perfect(c):
    return jnp.where(c.weather == C.SNOW, -1, -2).astype(jnp.int32)


ACC_FNS = {"none": _a_none, "rain_perfect": _a_rain_perfect,
           "snow_perfect": _a_snow_perfect}


# --- dispatch ----------------------------------------------------------------

def _make_switch(handler_names, fns, name):
    branches = []
    for h in handler_names:
        if h not in fns:
            raise KeyError(f"{name}: no implementation for handler {h!r}")
        branches.append(fns[h])
    branches = tuple(branches)

    def dispatch(index, ctx):
        return jax.lax.switch(jnp.clip(index.astype(jnp.int32), 0, len(branches) - 1),
                              branches, ctx)

    return dispatch


base_power_replace = _make_switch(E.BP_REPLACE_HANDLERS, BP_REPLACE_FNS, "bp_replace")
base_power_modify = _make_switch(E.BP_MODIFY_HANDLERS, BP_MODIFY_FNS, "bp_modify")
fixed_damage = _make_switch(E.DMG_HANDLERS, DMG_FNS, "damage")
modify_type = _make_switch(E.TYPE_HANDLERS, TYPE_FNS, "type")
modify_accuracy = _make_switch(E.ACC_HANDLERS, ACC_FNS, "accuracy")
