"""Executing a single move.

`execute_move` follows Showdown's `runMove` -> `useMove` -> `hitStepMoveHitLoop`
ordering: before-move status checks, PP, accuracy, immunity, the damage/hit loop,
then secondaries, drain and recoil.

Special move behaviour lives in `EFFECT_FNS`, dispatched by the compiled
`move_effect_cb` column. Handlers we have not written are bound to `_noop` and
listed in `UNIMPLEMENTED_EFFECTS`, which `coverage.py` reports -- an unmodelled
move fails silently rather than pretending to work.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import callbacks as cb
from . import consts as C
from . import effects as E
from .damage import (Attacker, Defender, MoveCtx, accuracy_check, calc_damage,
                     crit_chance_stage, current_types, has_flag,
                     resolve_move_ctx, scale_power_for_hit,
                     type_effectiveness, CRIT_RATES)
from .hooks import A, I
from .mechanics import (act, active_types, apply_boosts, cure_status, get_indexed,
                        set_indexed, slot_get, slot_set, synchronize,
                        damage_pokemon, effective_speed, effective_weather,
                        fraction_of_max, heal_pokemon, is_grounded, set_status,
                        set_volatile)
from .stats import boost_multiply, idiv


# --- building the calculation contexts ---------------------------------------

def build_attacker(state, side) -> Attacker:
    i = act(state, side)
    return Attacker(
        level=slot_get(state.level, side, i), stats=slot_get(state.stats, side, i),
        boosts=state.boosts[side], types=active_types(state, side),
        base_types=slot_get(state.types, side, i), ability=slot_get(state.ability, side, i),
        item=slot_get(state.item, side, i), status=slot_get(state.status, side, i),
        hp=slot_get(state.hp, side, i), maxhp=slot_get(state.maxhp, side, i),
        terastallized=slot_get(state.terastallized, side, i),
        tera_type=slot_get(state.tera_type, side, i),
        boosted_stat=state.boosted_stat[side],
    )


def build_defender(data, state, side) -> Defender:
    i = act(state, side)
    return Defender(
        stats=slot_get(state.stats, side, i), status=slot_get(state.status, side, i),
        boosts=state.boosts[side], types=active_types(state, side),
        ability=slot_get(state.ability, side, i), item=slot_get(state.item, side, i),
        hp=slot_get(state.hp, side, i), maxhp=slot_get(state.maxhp, side, i),
        terastallized=slot_get(state.terastallized, side, i),
        tera_type=slot_get(state.tera_type, side, i),
        nfe=data["species_nfe"][slot_get(state.species, side, i)],
        boosted_stat=state.boosted_stat[side],
    )


def build_cb_ctx(data, state, user, target, moves_first, pp_left, hit_number=1):
    ui, ti = act(state, user), act(state, target)
    a_stats, d_stats = slot_get(state.stats, user, ui), slot_get(state.stats, target, ti)
    boosted = lambda stats, boosts, s: boost_multiply(stats[s], boosts[s - 1])
    return cb.CbCtx(
        base_power=jnp.int32(0), move_type=jnp.int8(0),
        hit_number=jnp.asarray(hit_number, jnp.int32),
        atk_status=slot_get(state.status, user, ui), dfn_status=slot_get(state.status, target, ti),
        atk_hp=slot_get(state.hp, user, ui), atk_maxhp=slot_get(state.maxhp, user, ui),
        dfn_hp=slot_get(state.hp, target, ti), dfn_maxhp=slot_get(state.maxhp, target, ti),
        atk_item=slot_get(state.item, user, ui), dfn_item=slot_get(state.item, target, ti),
        atk_weight=data["species_weight"][slot_get(state.species, user, ui)],
        dfn_weight=data["species_weight"][slot_get(state.species, target, ti)],
        atk_speed=effective_speed(state, user), dfn_speed=effective_speed(state, target),
        atk_boosts=state.boosts[user], dfn_boosts=state.boosts[target],
        moves_first=moves_first,
        user_damaged=state.damage_taken[user] > 0,
        target_damaged=state.damage_taken[target] > 0,
        weather=effective_weather(state), terrain=state.terrain,
        grounded_user=is_grounded(state, user),
        grounded_target=is_grounded(state, target),
        pp_left=pp_left.astype(jnp.int32),
        times_hit=state.times_hit[user].astype(jnp.int32),
        fainted_count=state.fainted_count[user].astype(jnp.int32),
        terastallized=slot_get(state.terastallized, user, ui),
        tera_type=slot_get(state.tera_type, user, ui),
        fury_multiplier=jnp.int32(1),
        level=slot_get(state.level, user, ui),
        last_damage=state.damage_taken[user],
        last_damage_category=state.damage_category[user],
        off_atk=boosted(a_stats, state.boosts[user], C.ATK),
        off_spa=boosted(a_stats, state.boosts[user], C.SPA),
        user_ability=slot_get(state.ability, user, ui),
        last_move_failed=state.last_move_failed[user],
        stats_lowered=state.stats_lowered[user],
    )


# --- before-move checks ------------------------------------------------------

def before_move(data, state, user, move_id, key):
    """Can the user act? Returns `(state, can_act)`.

    Order follows Showdown: recharge, then flinch, then sleep/freeze/paralysis,
    then confusion (which can damage the user instead of letting it move).
    """
    ui = act(state, user)
    k_slp, k_frz, k_par, k_cnf, k_hit = jax.random.split(key, 5)

    can = slot_get(state.hp, user, ui) > 0

    # Recharging (Hyper Beam): the turn is spent and the volatile clears.
    recharging = state.volatiles[user, C.V_RECHARGE] > 0
    state = state._replace(
        volatiles=state.volatiles.at[user, C.V_RECHARGE].set(jnp.int8(0)))
    can = can & jnp.logical_not(recharging)

    # Flinch lasts only for the turn it was applied.
    flinched = state.volatiles[user, C.V_FLINCH] > 0
    state = state._replace(
        volatiles=state.volatiles.at[user, C.V_FLINCH].set(jnp.int8(0)))
    can = can & jnp.logical_not(flinched)

    # Sleep: the counter ticks down whenever the Pokemon tries to move.
    asleep = slot_get(state.status, user, ui) == C.SLP
    # Early Bird ticks two turns of sleep off per attempt.
    tick = jnp.where(slot_get(state.ability, user, ui) == A.EARLYBIRD, 2, 1)
    turns = jnp.maximum(slot_get(state.status_turns, user, ui).astype(jnp.int32) - tick, 0)
    wakes = asleep & (turns <= 0)
    state = state._replace(
        status=slot_set(state.status, user, ui, 
            jnp.where(wakes, jnp.int8(C.STATUS_NONE), slot_get(state.status, user, ui))),
        status_turns=slot_set(state.status_turns, user, ui, 
            jnp.where(asleep, turns.astype(jnp.int8), slot_get(state.status_turns, user, ui))))
    sleep_usable = data["move_sleep_usable"][move_id]
    can = can & jnp.logical_not(asleep & jnp.logical_not(wakes) &
                                jnp.logical_not(sleep_usable))

    # Freeze thaws with 20% probability, or immediately on a thawing move.
    frozen = slot_get(state.status, user, ui) == C.FRZ
    thaws = jax.random.uniform(k_frz) < 0.2
    unfreeze = frozen & thaws
    state = state._replace(
        status=slot_set(state.status, user, ui, 
            jnp.where(unfreeze, jnp.int8(C.STATUS_NONE), slot_get(state.status, user, ui))))
    can = can & jnp.logical_not(frozen & jnp.logical_not(thaws))

    # Full paralysis: 25%.
    paralysed = slot_get(state.status, user, ui) == C.PAR
    can = can & jnp.logical_not(paralysed & (jax.random.uniform(k_par) < 0.25))

    # Confusion: tick down, then a 1-in-3 chance of hitting yourself.
    confused = state.volatiles[user, C.V_CONFUSION] > 0
    cnf_left = jnp.maximum(state.volatiles[user, C.V_CONFUSION].astype(jnp.int32) - 1, 0)
    state = state._replace(
        volatiles=state.volatiles.at[user, C.V_CONFUSION].set(
            jnp.where(confused, cnf_left.astype(jnp.int8), jnp.int8(0))))
    still_confused = confused & (cnf_left > 0)
    self_hit = still_confused & (jax.random.uniform(k_cnf) < (1.0 / 3.0))

    # The confusion self-hit is a 40 BP typeless physical hit on yourself.
    atk = build_attacker(state, user)
    conf_dmg = _confusion_damage(state, user, atk, k_hit)
    state, _ = damage_pokemon(state, user, ui,
                              jnp.where(self_hit, conf_dmg, jnp.int32(0)))
    can = can & jnp.logical_not(self_hit)

    # Taunt forbids status moves.
    taunted = (state.volatiles[user, C.V_TAUNT] > 0) & \
              (data["move_category"][move_id] == C.CAT_STATUS)
    can = can & jnp.logical_not(taunted)
    return state, can


def _confusion_damage(state, side, atk: Attacker, key):
    """40 BP physical, no type, no crit -- Showdown's confusion self-hit."""
    level = atk.level.astype(jnp.int32)
    a = boost_multiply(atk.stats[C.ATK], atk.boosts[C.B_ATK - 1])
    d = jnp.maximum(boost_multiply(atk.stats[C.DEF], atk.boosts[C.B_DEF - 1]), 1)
    base = idiv(((2 * level) // 5 + 2) * 40 * a, d)
    base = base // 50 + 2
    roll = jax.random.randint(key, (), 0, 16)
    return jnp.maximum((base * (100 - roll)) // 100, 1)


# --- hit-step helpers --------------------------------------------------------

def move_blocked_by_protect(data, state, target, move_id):
    protected = state.volatiles[target, C.V_PROTECT] > 0
    bypass = data["move_breaks_protect"][move_id] | \
             jnp.logical_not(has_flag(data["move_flags"][move_id], "protect"))
    return protected & jnp.logical_not(bypass)


def ability_absorbs(data, state, target, move_type, category):
    """Water Absorb, Sap Sipper, Flash Fire, ... Returns `(absorbs, heal, boost)`."""
    ti = act(state, target)
    ab = slot_get(state.ability, target, ti)
    absorb_type = data["ability_absorb_type"][ab]
    hits = (absorb_type == move_type) & (absorb_type != C.TYPE_NONE) & \
           (category != C.CAT_STATUS)
    return hits, data["ability_absorb_heal"][ab], \
        (data["ability_absorb_boost_stat"][ab], data["ability_absorb_boost_amt"][ab])


def sound_immune(data, state, target, move_id):
    """Soundproof blocks sound-based moves outright."""
    ti = act(state, target)
    return has_flag(data["move_flags"][move_id], "sound") & \
        (slot_get(state.ability, target, ti) == A.SOUNDPROOF)


def powder_immune(data, state, target, move_id):
    """Powder moves do not affect Grass types, Overcoat or Safety Goggles."""
    ti = act(state, target)
    types = active_types(state, target)
    return has_flag(data["move_flags"][move_id], "powder") & (
        jnp.any(types == C.GRASS) | (slot_get(state.ability, target, ti) == A.OVERCOAT) |
        (slot_get(state.item, target, ti) == I.SAFETYGOGGLES))


def terrain_blocks(data, state, target, move_id, priority):
    """Psychic Terrain blocks priority moves; Misty Terrain blocks Dragon moves."""
    grounded = is_grounded(state, target)
    psychic = (state.terrain == C.PSYCHIC_TERRAIN) & grounded & (priority > 0)
    return psychic


# --- special effects ---------------------------------------------------------
# Signature: (data, state, user, target, key) -> state

def _noop(data, state, user, target, key):
    return state


def _eff_substitute(data, state, user, target, key):
    ui = act(state, user)
    cost = fraction_of_max(state, user, ui, 1, 4)
    enough = slot_get(state.hp, user, ui).astype(jnp.int32) > cost
    already = state.volatiles[user, C.V_SUBSTITUTE] > 0
    ok = enough & jnp.logical_not(already)
    state, _ = damage_pokemon(state, user, ui, jnp.where(ok, cost, 0))
    return state._replace(
        volatiles=state.volatiles.at[user, C.V_SUBSTITUTE].set(
            jnp.where(ok, jnp.int8(1), state.volatiles[user, C.V_SUBSTITUTE])),
        sub_hp=state.sub_hp.at[user].set(
            jnp.where(ok, cost.astype(jnp.int16), state.sub_hp[user])))


def _eff_protect(data, state, user, target, key):
    """Protect and its variants; consecutive uses get exponentially likelier to fail."""
    streak = state.protect_streak[user].astype(jnp.int32)
    # Showdown: succeeds with probability 1/3^streak.
    threshold = jnp.power(3.0, -streak.astype(jnp.float32))
    ok = jax.random.uniform(key) < threshold
    return state._replace(
        volatiles=state.volatiles.at[user, C.V_PROTECT].set(
            jnp.where(ok, jnp.int8(1), jnp.int8(0))),
        protect_streak=state.protect_streak.at[user].set(
            jnp.where(ok, jnp.minimum(streak + 1, 6), 0).astype(jnp.int8)))


def _eff_rest(data, state, user, target, key):
    ui = act(state, user)
    full = slot_get(state.hp, user, ui) >= slot_get(state.maxhp, user, ui)
    ok = jnp.logical_not(full)
    state, _ = heal_pokemon(state, user, ui,
                            jnp.where(ok, slot_get(state.maxhp, user, ui).astype(jnp.int32), 0))
    return state._replace(
        status=slot_set(state.status, user, ui, 
            jnp.where(ok, jnp.int8(C.SLP), slot_get(state.status, user, ui))),
        status_turns=slot_set(state.status_turns, user, ui, 
            jnp.where(ok, jnp.int8(3), slot_get(state.status_turns, user, ui))))


def _eff_haze(data, state, user, target, key):
    """Resets every stat change on both sides."""
    return state._replace(boosts=jnp.zeros_like(state.boosts))


def _eff_leechseed(data, state, user, target, key):
    types = active_types(state, target)
    ok = jnp.logical_not(jnp.any(types == C.GRASS)) & \
         (state.volatiles[target, C.V_LEECHSEED] == 0)
    return state._replace(
        volatiles=state.volatiles.at[target, C.V_LEECHSEED].set(
            jnp.where(ok, jnp.int8(1), state.volatiles[target, C.V_LEECHSEED])))


def _eff_painsplit(data, state, user, target, key):
    ui, ti = act(state, user), act(state, target)
    total = slot_get(state.hp, user, ui).astype(jnp.int32) + slot_get(state.hp, target, ti).astype(jnp.int32)
    half = total // 2
    new_u = jnp.minimum(half, slot_get(state.maxhp, user, ui).astype(jnp.int32))
    new_t = jnp.minimum(half, slot_get(state.maxhp, target, ti).astype(jnp.int32))
    hp = slot_set(state.hp, user, ui, new_u.astype(jnp.int16))
    hp = slot_set(hp, target, ti, new_t.astype(jnp.int16))
    return state._replace(hp=hp)


def _eff_defog(data, state, user, target, key):
    """Clears hazards and screens from both sides, and lowers evasion."""
    keep = jnp.zeros_like(state.side_conditions)
    state = state._replace(side_conditions=keep)
    state, _ = apply_boosts(state, target, jnp.zeros(7, jnp.int32).at[C.B_EVA].set(-1),
                            from_opponent=True)
    return state


def _remove_hazards(state, side):
    sc = state.side_conditions
    for h in (C.SC_STEALTHROCK, C.SC_SPIKES, C.SC_TOXICSPIKES, C.SC_STICKYWEB):
        sc = sc.at[side, h].set(jnp.int8(0))
    return state._replace(side_conditions=sc)


def _eff_rapidspin(data, state, user, target, key):
    state = _remove_hazards(state, user)
    state = state._replace(
        volatiles=state.volatiles.at[user, C.V_PARTIALLYTRAPPED].set(jnp.int8(0)))
    state, _ = apply_boosts(state, user, jnp.zeros(7, jnp.int32).at[C.B_SPE].set(1))
    return state


def _eff_mortalspin(data, state, user, target, key):
    state = _remove_hazards(state, user)
    return set_status(data, state, target, jnp.int8(C.PSN), key)[0]


def _eff_trick(data, state, user, target, key):
    """Swap held items. Fails if either side has nothing to give."""
    ui, ti = act(state, user), act(state, target)
    a, b = slot_get(state.item, user, ui), slot_get(state.item, target, ti)
    ok = (a != 0) | (b != 0)
    item = slot_set(state.item, user, ui, jnp.where(ok, b, a))
    item = slot_set(item, target, ti, jnp.where(ok, a, b))
    return state._replace(item=item)


def _eff_knockoff(data, state, user, target, key):
    ti = act(state, target)
    had_item = slot_get(state.item, target, ti) != 0
    state = state._replace(item=slot_set(state.item, target, ti, jnp.int16(0)))
    # Losing an item is what arms Unburden.
    return state._replace(volatiles=state.volatiles.at[target, C.V_UNBURDEN].set(
        jnp.where(had_item, jnp.int8(1), state.volatiles[target, C.V_UNBURDEN])))


def _eff_bellydrum(data, state, user, target, key):
    ui = act(state, user)
    cost = fraction_of_max(state, user, ui, 1, 2)
    ok = slot_get(state.hp, user, ui).astype(jnp.int32) > cost
    state, _ = damage_pokemon(state, user, ui, jnp.where(ok, cost, 0))
    new = jnp.where(ok, jnp.int8(6), state.boosts[user, C.B_ATK])
    return state._replace(boosts=state.boosts.at[user, C.B_ATK].set(new))


def _eff_roost(data, state, user, target, key):
    ui = act(state, user)
    state, _ = heal_pokemon(state, user, ui, fraction_of_max(state, user, ui, 1, 2))
    return set_volatile(state, user, C.V_ROOST, 1)


def _eff_weather_heal(data, state, user, target, key):
    """Synthesis / Moonlight / Morning Sun: 2/3 in sun, 1/4 in other weather."""
    ui = act(state, user)
    w = effective_weather(state)
    sun = (w == C.SUN) | (w == C.HARSH_SUN)
    other = (w != C.WEATHER_NONE) & jnp.logical_not(sun)
    num = jnp.where(sun, 2, jnp.where(other, 1, 1))
    den = jnp.where(sun, 3, jnp.where(other, 4, 2))
    state, _ = heal_pokemon(state, user, ui, fraction_of_max(state, user, ui, num, den))
    return state


def _eff_shoreup(data, state, user, target, key):
    ui = act(state, user)
    sand = effective_weather(state) == C.SAND
    num = jnp.where(sand, 2, 1)
    den = jnp.where(sand, 3, 2)
    state, _ = heal_pokemon(state, user, ui, fraction_of_max(state, user, ui, num, den))
    return state


def _eff_strengthsap(data, state, user, target, key):
    """Heals by the target's current Attack, then drops it."""
    ui, ti = act(state, user), act(state, target)
    amount = boost_multiply(state.stats[target, ti, C.ATK], state.boosts[target, C.B_ATK])
    state, _ = heal_pokemon(state, user, ui, amount)
    state, _ = apply_boosts(state, target, jnp.zeros(7, jnp.int32).at[C.B_ATK].set(-1),
                            from_opponent=True)
    return state


def _eff_curse(data, state, user, target, key):
    """Ghost types pay half their HP to curse the target; others boost instead."""
    ui = act(state, user)
    ghost = jnp.any(active_types(state, user) == C.GHOST)
    cost = fraction_of_max(state, user, ui, 1, 2)
    state, _ = damage_pokemon(state, user, ui, jnp.where(ghost, cost, 0))
    state = state._replace(
        volatiles=state.volatiles.at[target, C.V_CURSE].set(
            jnp.where(ghost, jnp.int8(1), state.volatiles[target, C.V_CURSE])))
    boosts = jnp.zeros(7, jnp.int32).at[C.B_ATK].set(1).at[C.B_DEF].set(1).at[C.B_SPE].set(-1)
    non_ghost = state
    non_ghost, _ = apply_boosts(non_ghost, user, boosts)
    return jax.tree.map(lambda g, n: jnp.where(ghost, g, n), state, non_ghost)


def _eff_clearsmog(data, state, user, target, key):
    return state._replace(boosts=state.boosts.at[target].set(jnp.zeros(7, jnp.int8)))


def _eff_topsyturvy(data, state, user, target, key):
    return state._replace(boosts=state.boosts.at[target].set(-state.boosts[target]))


def _eff_spectralthief(data, state, user, target, key):
    """Steals the target's positive boosts before dealing damage."""
    stolen = jnp.maximum(state.boosts[target].astype(jnp.int32), 0)
    state = state._replace(boosts=state.boosts.at[target].set(
        jnp.minimum(state.boosts[target].astype(jnp.int32), 0).astype(jnp.int8)))
    state, _ = apply_boosts(state, user, stolen)
    return state


def _eff_refresh(data, state, user, target, key):
    return cure_status(state, user, act(state, user))


def _eff_aromatherapy(data, state, user, target, key):
    return state._replace(
        status=state.status.at[user].set(jnp.zeros(C.TEAM_SIZE, jnp.int8)),
        status_turns=state.status_turns.at[user].set(jnp.zeros(C.TEAM_SIZE, jnp.int8)))


def _eff_psychoshift(data, state, user, target, key):
    ui = act(state, user)
    status = slot_get(state.status, user, ui)
    state, applied = set_status(data, state, target, status, key)
    return cure_status(state, user, ui, when=applied)


def _eff_filletaway(data, state, user, target, key):
    ui = act(state, user)
    cost = fraction_of_max(state, user, ui, 1, 2)
    ok = slot_get(state.hp, user, ui).astype(jnp.int32) > cost
    state, _ = damage_pokemon(state, user, ui, jnp.where(ok, cost, 0))
    boosts = jnp.zeros(7, jnp.int32).at[C.B_ATK].set(2).at[C.B_SPA].set(2).at[C.B_SPE].set(2)
    state, _ = apply_boosts(state, user, jnp.where(ok, boosts, 0))
    return state


def _eff_noretreat(data, state, user, target, key):
    boosts = jnp.full(7, 0, jnp.int32).at[:5].set(1)
    state, _ = apply_boosts(state, user, boosts)
    return state


def _eff_tidyup(data, state, user, target, key):
    state = _remove_hazards(state, user)
    state = _remove_hazards(state, target)
    state = state._replace(
        volatiles=state.volatiles.at[:, C.V_SUBSTITUTE].set(jnp.int8(0)),
        sub_hp=jnp.zeros_like(state.sub_hp))
    boosts = jnp.zeros(7, jnp.int32).at[C.B_ATK].set(1).at[C.B_SPE].set(1)
    state, _ = apply_boosts(state, user, boosts)
    return state


def _eff_courtchange(data, state, user, target, key):
    return state._replace(side_conditions=state.side_conditions[::-1])


def _eff_saltcure(data, state, user, target, key):
    return set_volatile(state, target, C.V_SALTCURE, 1)


def _eff_glaiverush(data, state, user, target, key):
    return set_volatile(state, user, C.V_GLAIVERUSH, 1)


def _eff_screenbreak(data, state, user, target, key):
    """Brick Break / Psychic Fangs / Raging Bull shatter screens before hitting."""
    sc = state.side_conditions
    for screen in (C.SC_REFLECT, C.SC_LIGHTSCREEN, C.SC_AURORAVEIL):
        sc = sc.at[target, screen].set(jnp.int8(0))
    return state._replace(side_conditions=sc)


def _eff_icespinner(data, state, user, target, key):
    """Removes the terrain."""
    return state._replace(terrain=jnp.int8(C.TERRAIN_NONE),
                          terrain_turns=jnp.int8(0))


def _eff_partingshot(data, state, user, target, key):
    """Drops the target's offences, then the user switches out."""
    drop = jnp.zeros(7, jnp.int32).at[C.B_ATK].set(-1).at[C.B_SPA].set(-1)
    state, _ = apply_boosts(state, target, drop, from_opponent=True)
    return state._replace(force_switch=state.force_switch.at[user].set(True))


def _eff_chillyreception(data, state, user, target, key):
    """Sets snow, then the user switches out."""
    state = state._replace(weather=jnp.int8(C.SNOW), weather_turns=jnp.int8(5))
    return state._replace(force_switch=state.force_switch.at[user].set(True))


def _eff_healingwish(data, state, user, target, key):
    """The user faints; its replacement arrives at full HP with no status.

    Modelled as an immediate full heal of the most damaged living team member,
    which is where the wish would land. The user fainting queues the switch.
    """
    ui = act(state, user)
    hp = state.hp[user].astype(jnp.int32)
    alive = (hp > 0) & (jnp.arange(C.TEAM_SIZE) != ui)
    missing = jnp.where(alive, state.maxhp[user].astype(jnp.int32) - hp, -1)
    slot = jnp.argmax(missing)
    any_ally = jnp.any(alive)
    state = state._replace(
        hp=state.hp.at[user, slot].set(
            jnp.where(any_ally, state.maxhp[user, slot], state.hp[user, slot])),
        status=state.status.at[user, slot].set(
            jnp.where(any_ally, jnp.int8(C.STATUS_NONE), state.status[user, slot])))
    state, _ = damage_pokemon(state, user, ui,
                              jnp.where(any_ally, slot_get(state.hp, user, ui).astype(jnp.int32), 0))
    return state


def _eff_revivalblessing(data, state, user, target, key):
    """Revives one fainted team member at half HP."""
    fainted = state.hp[user] <= 0
    slot = jnp.argmax(fainted)
    any_fainted = jnp.any(fainted)
    half = jnp.maximum(state.maxhp[user, slot].astype(jnp.int32) // 2, 1)
    return state._replace(hp=state.hp.at[user, slot].set(
        jnp.where(any_fainted, half.astype(jnp.int16), state.hp[user, slot])))


def _eff_shedtail(data, state, user, target, key):
    """Leaves a Substitute behind, then switches out."""
    state = _eff_substitute(data, state, user, target, key)
    return state._replace(force_switch=state.force_switch.at[user].set(True))


def _eff_burnup(data, state, user, target, key):
    """The user loses the move's own type (Fire for Burn Up, Electric for Double Shock)."""
    ui = act(state, user)
    lost = jnp.where(slot_get(state.types, user, ui) == C.FIRE, jnp.int8(C.TYPE_NONE),
                     slot_get(state.types, user, ui))
    return state._replace(types=slot_set(state.types, user, ui, lost))


def _eff_doubleshock(data, state, user, target, key):
    ui = act(state, user)
    lost = jnp.where(slot_get(state.types, user, ui) == C.ELECTRIC, jnp.int8(C.TYPE_NONE),
                     slot_get(state.types, user, ui))
    return state._replace(types=slot_set(state.types, user, ui, lost))


def _eff_smackdown(data, state, user, target, key):
    """Grounds the target for the rest of the battle."""
    return state._replace(
        volatiles=state.volatiles.at[target, C.V_MAGNETRISE].set(jnp.int8(0)))


def _eff_fakeout(data, state, user, target, key):
    """Fake Out and First Impression only work on the user's first move out.

    The move's flinch/damage has already been applied by the generic path, so
    failing means undoing it; `execute_move` checks `fakeout_ok` before the hit
    instead, and this handler only has to clear the flinch on a late use.
    """
    late = state.moves_since_switch[user] > 0
    return state._replace(volatiles=state.volatiles.at[target, C.V_FLINCH].set(
        jnp.where(late, jnp.int8(0), state.volatiles[target, C.V_FLINCH])))


def _eff_auroraveil(data, state, user, target, key):
    """Aurora Veil only sets while it is snowing."""
    ok = effective_weather(state) == C.SNOW
    return state._replace(side_conditions=state.side_conditions.at[
        user, C.SC_AURORAVEIL].set(
            jnp.where(ok, state.side_conditions[user, C.SC_AURORAVEIL], jnp.int8(0))))


# Handlers we have not implemented yet map to `_noop`; coverage.py lists them.
EFFECT_FNS = {
    "none": _noop,
    "substitute": _eff_substitute, "protect": _eff_protect,
    "protectvariant": _eff_protect, "rest": _eff_rest, "haze": _eff_haze,
    "leechseed": _eff_leechseed, "painsplit": _eff_painsplit, "defog": _eff_defog,
    "rapidspin": _eff_rapidspin, "mortalspin": _eff_mortalspin, "trick": _eff_trick,
    "bellydrum": _eff_bellydrum, "roost": _eff_roost,
    "sunnyday_heal": _eff_weather_heal, "shoreup": _eff_shoreup,
    "strengthsap": _eff_strengthsap, "curse": _eff_curse,
    "clearsmog": _eff_clearsmog, "topsyturvy": _eff_topsyturvy,
    "spectralthief": _eff_spectralthief, "refresh": _eff_refresh,
    "aromatherapy": _eff_aromatherapy, "psychoshift": _eff_psychoshift,
    "filletaway": _eff_filletaway, "noretreat": _eff_noretreat,
    "tidyup": _eff_tidyup, "courtchange": _eff_courtchange,
    "saltcure": _eff_saltcure, "glaiverush": _eff_glaiverush,
    "smackdown": _eff_smackdown, "screenbreak": _eff_screenbreak,
    "icespinner": _eff_icespinner, "partingshot": _eff_partingshot,
    "chillyreception": _eff_chillyreception, "healingwish": _eff_healingwish,
    "revivalblessing": _eff_revivalblessing, "shedtail": _eff_shedtail,
    "burnup": _eff_burnup, "doubleshock": _eff_doubleshock,
    "fakeout": _eff_fakeout, "auroraveil": _eff_auroraveil,
}

UNIMPLEMENTED_EFFECTS = tuple(h for h in E.EFFECT_HANDLERS if h not in EFFECT_FNS)

# vmap lowers `lax.switch` to "evaluate every branch and select", so the switch
# is built over the implemented handlers only and unimplemented ids fold onto the
# no-op. The branches also return just the fields any handler writes rather than
# the whole BattleState: the select tree is proportional to the number of output
# arrays, and 15 is a great deal cheaper to compile than 47.
_IMPLEMENTED_EFFECTS = [h for h in E.EFFECT_HANDLERS if h in EFFECT_FNS]
_COMPACT_EFFECT_ID = jnp.asarray(
    [_IMPLEMENTED_EFFECTS.index(h) if h in EFFECT_FNS else 0
     for h in E.EFFECT_HANDLERS], jnp.int32)

#: Every BattleState field an effect handler can write. Keep in sync with
#: EFFECT_FNS -- a handler writing anything outside this list would have its
#: change silently dropped, so `_check_effect_fields` asserts it at import.
EFFECT_WRITES = (
    "hp", "status", "status_turns", "boosts", "volatiles", "sub_hp",
    "side_conditions", "item", "types", "terrain", "terrain_turns",
    "weather", "weather_turns", "force_switch", "protect_streak",
)


def _project(state):
    return tuple(getattr(state, f) for f in EFFECT_WRITES)


def _wrap(fn):
    def branch(data, state, user, target, key):
        return _project(fn(data, state, user, target, key))
    return branch


_EFFECT_BRANCHES = tuple(_wrap(EFFECT_FNS[h]) for h in _IMPLEMENTED_EFFECTS)


def run_effect(effect_id, data, state, user, target, key, when=True):
    """Dispatch to a move's special-effect handler.

    `when` gates the whole call (e.g. on whether the user could act at all).
    Masked per output field here rather than the caller wrapping this in
    `lax.cond`: that would duplicate-and-select over the entire `BattleState`
    a second time on top of the switch itself, which is exactly the kind of
    stacked full-state branching that made `vmap(execute_move)` uncompilable.
    """
    compact = _COMPACT_EFFECT_ID[jnp.clip(effect_id.astype(jnp.int32), 0,
                                          len(E.EFFECT_HANDLERS) - 1)]
    values = jax.lax.switch(compact, _EFFECT_BRANCHES, data, state, user, target, key)
    updates = {f: jnp.where(when, v, getattr(state, f)) for f, v in
              zip(EFFECT_WRITES, values)}
    return state._replace(**updates)


# Behaviours that are modelled outside `EFFECT_FNS`, so their presence in
# `UNIMPLEMENTED_EFFECTS` would be misleading.
HANDLED_ELSEWHERE = frozenset({
    "freezedry", "flyingpress",   # damage.type_effectiveness
    "photongeyser",               # damage.resolve_move_ctx (category switch)
    "taunt", "encore", "disable", "yawn", "perishsong", "destinybond",
    "poltergeist",                # applied through the declarative volatile column
    "suckerpunch",                # the fail check lives in execute_move
})

MISSING_EFFECTS = tuple(h for h in UNIMPLEMENTED_EFFECTS if h not in HANDLED_ELSEWHERE)


# --- applying a hit ----------------------------------------------------------

def _apply_hit_damage(state, user, target, amount, bypass_sub):
    """Route damage through a Substitute when one is up.

    Returns `(state, damage_dealt_to_the_pokemon)` -- damage absorbed by the
    Substitute must not feed drain, Rocky Helmet or the damage counters.
    """
    ti = act(state, target)
    has_sub = (state.volatiles[target, C.V_SUBSTITUTE] > 0) & jnp.logical_not(bypass_sub)
    sub_hp = state.sub_hp[target].astype(jnp.int32)
    to_sub = jnp.where(has_sub, jnp.minimum(amount, sub_hp), 0)
    new_sub = sub_hp - to_sub
    state = state._replace(
        sub_hp=state.sub_hp.at[target].set(new_sub.astype(jnp.int16)),
        volatiles=state.volatiles.at[target, C.V_SUBSTITUTE].set(
            jnp.where(has_sub & (new_sub <= 0), jnp.int8(0),
                      state.volatiles[target, C.V_SUBSTITUTE])))
    direct = jnp.where(has_sub, 0, amount)
    state, dealt = damage_pokemon(state, target, ti, direct)
    return state, dealt


def _apply_secondaries(data, state, user, target, move_id, key, landed=True):
    """Roll each of the move's up-to-two secondary effects.

    `landed` gates the whole thing rather than the caller wrapping this call in
    `lax.cond`: folding it into each `fires` mask means `set_status` and
    `apply_boosts` run unconditionally and simply no-op, instead of the whole
    function being traced twice (once per `cond` branch) and selected over.
    """
    ti = act(state, target)
    shielded = slot_get(state.ability, target, ti) == A.SHIELDDUST
    covert = slot_get(state.item, target, ti) == I.COVERTCLOAK
    serene = slot_get(state.ability, user, act(state, user)) == A.SERENEGRACE
    blocked = shielded | covert

    for k in range(2):
        chance = data["move_sec_chance"][move_id, k].astype(jnp.int32)
        chance = jnp.where(serene, jnp.minimum(chance * 2, 100), chance)
        key, k_roll, k_status = jax.random.split(key, 3)
        fires = landed & (chance > 0) & \
            (jax.random.randint(k_roll, (), 0, 100) < chance) & jnp.logical_not(blocked)

        status = data["move_sec_status"][move_id, k]
        masked_status = jnp.where(fires, status, jnp.int8(C.STATUS_NONE))
        state, _ = set_status(data, state, target, masked_status, k_status)

        vol = data["move_sec_volatile"][move_id, k]
        # Inner Focus and Steadfast-like abilities cannot be made to flinch.
        flinch_blocked = (vol == C.V_FLINCH) & (
            (slot_get(state.ability, target, ti) == A.INNERFOCUS) |
            (slot_get(state.ability, target, ti) == A.OWNTEMPO))
        state = state._replace(volatiles=state.volatiles.at[target].set(
            set_indexed(state.volatiles[target], vol, jnp.int8(1),
                        when=fires & (vol > 0) & jnp.logical_not(flinch_blocked))))

        tgt_boosts = data["move_sec_boosts"][move_id, k].astype(jnp.int32)
        state, _ = apply_boosts(state, target, jnp.where(fires, tgt_boosts, 0),
                                from_opponent=True)
        self_boosts = data["move_sec_self_boosts"][move_id, k].astype(jnp.int32)
        state, _ = apply_boosts(state, user, jnp.where(fires, self_boosts, 0))
    return state


def execute_move(data, state, user, move_slot, moves_first, key,
                 target_attacking=True):
    """Run one move from `user`'s active Pokemon. Returns the updated state.

    `target_attacking` says whether the opponent is about to use a damaging move
    this turn; Sucker Punch and friends fail when it is false.
    """
    target = 1 - user
    ui = act(state, user)
    move_slot = move_slot.astype(jnp.int32)
    move_id = jnp.maximum(state.moves[user, ui, move_slot], 0).astype(jnp.int32)

    k_before, k_acc, k_crit, k_roll, k_sec, k_eff, k_status, k_hits = \
        jax.random.split(key, 8)

    state, can_act = before_move(data, state, user, move_id, k_before)
    has_pp = state.pp[user, ui, move_slot] > 0
    can_act = can_act & has_pp & (state.moves[user, ui, move_slot] >= 0)

    # Spend PP (Pressure makes the opponent's moves cost two).
    ti0 = act(state, target)
    cost = jnp.where(state.ability[target, ti0] == A.PRESSURE, 2, 1)
    new_pp = jnp.maximum(state.pp[user, ui, move_slot].astype(jnp.int32) -
                         jnp.where(can_act, cost, 0), 0)
    state = state._replace(
        pp=state.pp.at[user, ui, move_slot].set(new_pp.astype(jnp.int8)))

    cb_ctx = build_cb_ctx(data, state, user, target, moves_first,
                          state.pp[user, ui, move_slot])
    mv = resolve_move_ctx(data, move_id, cb_ctx)
    # Resolved once here rather than per hit: only the hit-number scaling below
    # varies within a multi-hit move, and the switches are expensive to trace.
    bp_cb_mod = cb.base_power_modify(data["move_bp_modify"][move_id], cb_ctx)
    atk = build_attacker(state, user)
    dfn = build_defender(data, state, target)

    # --- can the move connect at all? ---
    ti = act(state, target)
    target_alive = slot_get(state.hp, target, ti) > 0
    blocked = move_blocked_by_protect(data, state, target, move_id)
    powder = powder_immune(data, state, target, move_id)
    soundproofed = sound_immune(data, state, target, move_id)
    priority = data["move_priority"][move_id].astype(jnp.int32)
    terrain_block = terrain_blocks(data, state, target, move_id, priority)

    acc_roll = jax.random.randint(k_acc, (), 0, 100)
    weather_acc = cb.modify_accuracy(data["move_acc_cb"][move_id], cb_ctx)
    hits_acc = accuracy_check(data, mv, atk, dfn, state.boosts[user, C.B_ACC],
                              state.boosts[target, C.B_EVA], acc_roll, state.gravity,
                              weather_acc)

    # Moves that simply fail under a condition the generic path cannot express.
    from .effects import EFFECT_HANDLERS
    effect = data["move_effect_cb"][move_id]
    sucker = effect == EFFECT_HANDLERS.index("suckerpunch")
    fakeout = effect == EFFECT_HANDLERS.index("fakeout")
    fails = (sucker & jnp.logical_not(target_attacking & moves_first)) | \
            (fakeout & (state.moves_since_switch[user] > 0))

    def_types = current_types(dfn.types, dfn.terastallized, dfn.tera_type)
    scrappy = (atk.ability == A.SCRAPPY) | (atk.ability == A.MINDSEYE)
    type_exp, type_immune = type_effectiveness(
        data, mv.type, def_types, mv, data["move_ignore_immunity"][move_id],
        dfn.ability, scrappy)
    is_status = mv.category == C.CAT_STATUS
    type_immune = type_immune & jnp.logical_not(is_status)

    absorbs, absorb_heal, (absorb_stat, absorb_amt) = ability_absorbs(
        data, state, target, mv.type, mv.category)

    connects = (can_act & target_alive & hits_acc & jnp.logical_not(blocked) &
                jnp.logical_not(powder) & jnp.logical_not(soundproofed) &
                jnp.logical_not(terrain_block) &
                jnp.logical_not(type_immune) & jnp.logical_not(absorbs) &
                jnp.logical_not(fails))

    # Absorbing abilities heal or boost instead of taking the hit.
    state, _ = heal_pokemon(state, target, ti,
                            jnp.where(absorbs & (absorb_heal > 0),
                                      fraction_of_max(state, target, ti,
                                                      absorb_heal.astype(jnp.int32), 16),
                                      0))
    boost_vec = set_indexed(jnp.zeros(7, jnp.int32), absorb_stat,
                            absorb_amt.astype(jnp.int32),
                            when=(absorb_stat >= 0) & absorbs)
    state, _ = apply_boosts(state, target, boost_vec)

    # --- damage ---
    fixed = cb.fixed_damage(data["move_dmg_cb"][move_id], cb_ctx)
    is_fixed = fixed >= 0
    ohko = data["move_ohko"][move_id]

    n_hits = _roll_hit_count(data, state, user, move_id, k_hits)
    bypass_sub = has_flag(mv.flags, "bypasssub")

    crit_stage = crit_chance_stage(data, mv, atk)
    crit_denom = CRIT_RATES[crit_stage]
    crit_immune = (dfn.ability == A.BATTLEARMOR) | (dfn.ability == A.SHELLARMOR)
    # Probability 1/crit_denom, expressed without a traced modulo: XLA lowers
    # integer division and remainder by a non-constant into a large sequence, and
    # one of those here was enough to stall vmap compilation.
    is_crit = (jax.random.uniform(k_crit) * crit_denom.astype(jnp.float32) < 1.0) & \
              jnp.logical_not(crit_immune) | data["move_will_crit"][move_id]
    is_crit = is_crit & jnp.logical_not(crit_immune)

    # Everything the damage formula reads that cannot change between hits is
    # computed once here. The loop itself carries only scalars -- running HP,
    # Substitute HP and the damage total -- rather than the whole BattleState.
    # Carrying the state meant a batched `while` whose body scattered into 47
    # arrays, and that is what made `vmap(execute_move)` take minutes to compile.
    hit_weather = effective_weather(state)
    hit_terrain = state.terrain
    hit_side_conditions = state.side_conditions[target]
    hit_fainted_count = state.fainted_count[user].astype(jnp.int32)
    hit_analytic = jnp.logical_not(moves_first)
    hit_target_switched = state.switched_this_turn[target]
    target_maxhp = slot_get(state.maxhp, target, ti).astype(jnp.int32)
    can_endure = ((slot_get(state.item, target, ti) == I.FOCUSSASH) |
                  (slot_get(state.ability, target, ti) == A.STURDY))
    start_hp = slot_get(state.hp, target, ti).astype(jnp.int32)
    start_sub = jnp.where(bypass_sub, 0, state.sub_hp[target].astype(jnp.int32))

    # Damage per hit depends only on the roll and (for Triple Kick / Triple
    # Axel) the hit number -- never on the running HP. So the damage for all
    # possible hits is computed in one vectorised pass and only the HP
    # bookkeeping is sequential. Calling `calc_damage` inside a `while` loop
    # instead made `vmap(execute_move)` take minutes to compile.
    MAX_HITS = 10
    hit_index = jnp.arange(MAX_HITS)
    rolls = jax.random.randint(k_roll, (MAX_HITS,), 0, 16)

    def damage_for_hit(roll, i):
        mv_i = scale_power_for_hit(data, move_id, mv, i + 1)
        return calc_damage(
            data, atk, dfn, mv_i, is_crit=is_crit, damage_roll=roll,
            weather=hit_weather, terrain=hit_terrain,
            side_conditions=hit_side_conditions, type_exp=type_exp,
            bp_cb_mod=bp_cb_mod, grounded_user=cb_ctx.grounded_user,
            grounded_target=cb_ctx.grounded_target,
            analytic_ok=hit_analytic, fainted_count=hit_fainted_count,
            target_switched_in=hit_target_switched)

    dmgs = jax.vmap(damage_for_hit)(rolls, hit_index)
    dmgs = jnp.where(is_fixed, fixed, dmgs)
    dmgs = jnp.where(ohko, target_maxhp, dmgs)
    active = (hit_index < n_hits) & connects & jnp.logical_not(is_status)
    dmgs = jnp.where(active, dmgs, 0)

    def accumulate(carry, dmg):
        total, hp, sub = carry
        # A Substitute soaks the hit until it breaks; later hits land directly.
        to_sub = jnp.where(sub > 0, jnp.minimum(dmg, sub), 0)
        sub = sub - to_sub
        direct = jnp.where(to_sub > 0, 0, dmg)
        # Focus Sash / Sturdy leave the target on 1 HP from full.
        endures = can_endure & (hp == target_maxhp) & (direct >= hp)
        direct = jnp.where(endures, hp - 1, direct)
        direct = jnp.minimum(direct, hp)
        return (total + direct, hp - direct, sub), None

    (total_damage, final_hp, final_sub), _ = jax.lax.scan(
        accumulate, (jnp.int32(0), start_hp, start_sub), dmgs)

    # Write the accumulated result back once.
    state = state._replace(
        hp=slot_set(state.hp, target, ti, final_hp.astype(jnp.int16)),
        sub_hp=state.sub_hp.at[target].set(
            jnp.where(bypass_sub, state.sub_hp[target], final_sub.astype(jnp.int16))),
        volatiles=state.volatiles.at[target, C.V_SUBSTITUTE].set(
            jnp.where(jnp.logical_not(bypass_sub) & (final_sub <= 0), jnp.int8(0),
                      state.volatiles[target, C.V_SUBSTITUTE])))

    landed = connects & jnp.logical_not(is_status)
    state = state._replace(
        damage_taken=state.damage_taken.at[target].add(total_damage.astype(jnp.int16)),
        damage_category=state.damage_category.at[target].set(
            jnp.where(landed, mv.category, state.damage_category[target])),
        times_hit=state.times_hit.at[target].add(
            jnp.where(landed, jnp.int8(1), jnp.int8(0))))

    # --- drain, recoil, self-destruct ---
    drain = data["move_drain"][move_id].astype(jnp.int32)
    drained = jnp.where((drain[1] > 0) & landed,
                        idiv(total_damage * drain[0], drain[1]), 0)
    # Liquid Ooze makes draining moves hurt the user instead.
    ooze = slot_get(state.ability, target, ti) == A.LIQUIDOOZE
    state, _ = heal_pokemon(state, user, ui, jnp.where(ooze, 0, drained))
    state, _ = damage_pokemon(state, user, ui, jnp.where(ooze, drained, 0))
    recoil = data["move_recoil"][move_id].astype(jnp.int32)
    no_recoil = (slot_get(state.ability, user, ui) == A.MAGICGUARD) | \
                (slot_get(state.ability, user, ui) == A.ROCKHEAD)
    magic_guard = no_recoil
    state, _ = damage_pokemon(
        state, user, ui,
        jnp.where((recoil[1] > 0) & landed & jnp.logical_not(magic_guard),
                  jnp.maximum(idiv(total_damage * recoil[0], recoil[1]), 1),
                  0))
    # High Jump Kick and friends: half the user's max HP on a miss.
    crashed = data["move_crash_damage"][move_id] & can_act & \
        jnp.logical_not(connects) & target_alive
    state, _ = damage_pokemon(state, user, ui,
                              jnp.where(crashed,
                                        fraction_of_max(state, user, ui, 1, 2), 0))

    selfdestruct = data["move_selfdestruct"][move_id] > 0
    state, _ = damage_pokemon(state, user, ui,
                              jnp.where(selfdestruct & can_act,
                                        slot_get(state.hp, user, ui).astype(jnp.int32), 0))

    # --- status-move payloads (also applied by damaging moves that carry them) ---
    applied = connects
    status = data["move_status"][move_id]
    masked_status = jnp.where(applied, status, jnp.int8(C.STATUS_NONE))
    state, status_landed = set_status(
        data, state, target, masked_status, k_status,
        source_ability=slot_get(state.ability, user, ui))
    # Synchronize bounces the status back at whoever inflicted it.
    state = synchronize(data, state, target, status, k_status, when=status_landed)

    vol = data["move_volatile"][move_id]
    duration = jnp.maximum(data["move_duration"][move_id], 1)
    state = state._replace(volatiles=state.volatiles.at[target].set(
        set_indexed(state.volatiles[target], vol, duration,
                    when=applied & (vol > 0))))
    self_vol = data["move_self_volatile"][move_id]
    state = state._replace(volatiles=state.volatiles.at[user].set(
        set_indexed(state.volatiles[user], self_vol, duration,
                    when=can_act & (self_vol > 0))))

    tgt_boosts = data["move_boosts"][move_id].astype(jnp.int32)
    self_targeting = data["move_target"][move_id] == C.TGT_SELF
    state, _ = apply_boosts(state, user, jnp.where(applied & self_targeting, tgt_boosts, 0))
    state, lowered = apply_boosts(
        state, target, jnp.where(applied & jnp.logical_not(self_targeting), tgt_boosts, 0),
        from_opponent=True)
    state = state._replace(stats_lowered=state.stats_lowered.at[target].set(
        state.stats_lowered[target] | jnp.any(lowered < 0)))
    state, _ = apply_boosts(state, user,
                            jnp.where(can_act, data["move_self_boosts"][move_id].astype(jnp.int32), 0))

    # --- field and side effects ---
    sc = data["move_side_condition"][move_id]
    state = _set_side_condition(state, target, sc, applied & (sc >= 0) &
                                (data["move_target"][move_id] == C.TGT_FOE_SIDE))
    self_sc = data["move_self_side_condition"][move_id]
    state = _set_side_condition(state, user, self_sc, can_act & (self_sc >= 0))
    own_sc = jnp.where(data["move_target"][move_id] == C.TGT_ALLY_SIDE, sc, jnp.int8(-1))
    state = _set_side_condition(state, user, own_sc, applied & (own_sc >= 0))

    weather = data["move_weather"][move_id]
    state = state._replace(
        weather=jnp.where(can_act & (weather > 0), weather, state.weather),
        weather_turns=jnp.where(can_act & (weather > 0), jnp.int8(5), state.weather_turns))
    terrain = data["move_terrain"][move_id]
    state = state._replace(
        terrain=jnp.where(can_act & (terrain > 0), terrain, state.terrain),
        terrain_turns=jnp.where(can_act & (terrain > 0), jnp.int8(5), state.terrain_turns))

    # --- healing moves ---
    heal = data["move_heal"][move_id].astype(jnp.int32)
    state, _ = heal_pokemon(state, user, ui,
                            jnp.where(can_act & (heal[1] > 0),
                                      fraction_of_max(state, user, ui, heal[0], heal[1]), 0))

    # --- contact and on-damage abilities ---
    contact = has_flag(mv.flags, "contact") & landed
    d_ab = slot_get(state.ability, target, ti)
    d_it = slot_get(state.item, target, ti)
    attacker_guarded = slot_get(state.ability, user, ui) == A.MAGICGUARD

    # Rough Skin / Iron Barbs / Rocky Helmet chip the attacker on contact.
    spiky = contact & jnp.logical_not(attacker_guarded) & (
        (d_ab == A.ROUGHSKIN) | (d_ab == A.IRONBARBS) | (d_it == I.ROCKYHELMET))
    state, _ = damage_pokemon(state, user, ui,
                              jnp.where(spiky, fraction_of_max(state, user, ui, 1, 6), 0))

    # Aftermath: fainting to a contact move takes a quarter off the attacker.
    # (`now_hp` is computed below, so this uses the post-damage HP directly.)
    aftermath = contact & (d_ab == A.AFTERMATH) & \
        jnp.logical_not(attacker_guarded) & (slot_get(state.hp, target, ti) <= 0)
    state, _ = damage_pokemon(state, user, ui,
                              jnp.where(aftermath,
                                        fraction_of_max(state, user, ui, 1, 4), 0))

    # Contact abilities with a 30% chance to inflict a status.
    k_contact, k_sec = jax.random.split(k_sec)
    rolls_contact = jax.random.uniform(k_contact) < 0.3
    # At most one of these can match, so they collapse into a single masked call.
    contact_status = jnp.int8(C.STATUS_NONE)
    for ability, status in ((A.STATIC, C.PAR), (A.FLAMEBODY, C.BRN),
                            (A.POISONPOINT, C.PSN)):
        fires = contact & (d_ab == ability) & rolls_contact
        contact_status = jnp.where(fires, jnp.int8(status), contact_status)
    state, _ = set_status(data, state, user, contact_status, k_contact)

    # Abilities that react to taking a hit. The defender has exactly one ability,
    # so at most one of these can fire -- they are accumulated into a single
    # boost vector and applied once. Ten separate `apply_boosts` calls here was
    # enough on its own to tip vmap compilation over a cliff.
    took_damage = landed & (total_damage > 0)
    phys_hit = took_damage & (mv.category == C.CAT_PHYSICAL)
    now_hp = slot_get(state.hp, target, ti).astype(jnp.int32)
    max_hp = slot_get(state.maxhp, target, ti).astype(jnp.int32)
    # Berserk triggers when the hit takes the holder past half HP.
    crossed_half = took_damage & (now_hp * 2 <= max_hp) & \
        ((now_hp + total_damage) * 2 > max_hp) & (now_hp > 0)

    target_boosts = jnp.zeros(7, jnp.int32)
    for ability, boosts, cond in (
            (A.STAMINA, {C.B_DEF: 1}, took_damage),
            (A.WEAKARMOR, {C.B_DEF: -1, C.B_SPE: 2}, phys_hit),
            (A.WATERCOMPACTION, {C.B_DEF: 2}, took_damage & (mv.type == C.WATER)),
            (A.JUSTIFIED, {C.B_ATK: 1}, took_damage & (mv.type == C.DARK)),
            (A.RATTLED, {C.B_SPE: 1}, took_damage &
             ((mv.type == C.BUG) | (mv.type == C.DARK) | (mv.type == C.GHOST))),
            (A.ANGERPOINT, {C.B_ATK: 12}, phys_hit & is_crit),
            (A.BERSERK, {C.B_SPA: 1}, crossed_half)):
        vec = np.zeros(7, np.int32)
        for idx, amount in boosts.items():
            vec[idx] = amount
        target_boosts = jnp.where((d_ab == ability) & cond, jnp.asarray(vec),
                                  target_boosts)
    state, _ = apply_boosts(state, target, target_boosts)

    # Moxie and friends boost the attacker when the hit knocks the target out;
    # folded the same way.
    a_ab = slot_get(state.ability, user, ui)
    ko = landed & (now_hp <= 0)
    user_boosts = jnp.zeros(7, jnp.int32)
    for ability, boost in ((A.MOXIE, C.B_ATK), (A.CHILLINGNEIGH, C.B_ATK),
                           (A.GRIMNEIGH, C.B_SPA)):
        vec = np.zeros(7, np.int32)
        vec[boost] = 1
        user_boosts = jnp.where((a_ab == ability) & ko, jnp.asarray(vec), user_boosts)
    state, _ = apply_boosts(state, user, user_boosts)

    # Poison Touch gives the attacker a 30% chance to poison on contact.
    k_touch, k_sec = jax.random.split(k_sec)
    touch = (slot_get(state.ability, user, ui) == A.POISONTOUCH) & landed & \
        has_flag(mv.flags, "contact") & (jax.random.uniform(k_touch) < 0.3)
    state, _ = set_status(data, state, target,
                          jnp.where(touch, jnp.int8(C.PSN), jnp.int8(C.STATUS_NONE)),
                          k_touch)

    # --- secondaries and the special-effect handler ---
    state = _apply_secondaries(data, state, user, target, move_id, k_sec, landed=landed)
    effect_id = data["move_effect_cb"][move_id]
    state = run_effect(effect_id, data, state, user, target, k_eff,
                       when=can_act & (effect_id > 0))

    # Protect's streak only survives consecutive protecting turns.
    is_stalling = data["move_stalling"][move_id]
    state = state._replace(protect_streak=state.protect_streak.at[user].set(
        jnp.where(is_stalling, state.protect_streak[user], jnp.int8(0))))

    move_failed = can_act & jnp.logical_not(connects) & jnp.logical_not(is_status)
    state = state._replace(
        last_move=state.last_move.at[user].set(
            jnp.where(can_act, move_id.astype(jnp.int16), state.last_move[user])),
        moved_this_turn=state.moved_this_turn.at[user].set(True),
        moves_since_switch=state.moves_since_switch.at[user].add(
            jnp.where(can_act, jnp.int8(1), jnp.int8(0))),
        last_move_failed=state.last_move_failed.at[user].set(move_failed))

    # Choice items lock the user into the move it just used. Compared directly
    # rather than through `data["item_is_choice"]`: a gather whose index is
    # itself a gathered value is a compile cliff under vmap.
    held = slot_get(state.item, user, ui)
    choiced = ((held == I.CHOICESCARF) | (held == I.CHOICEBAND) |
               (held == I.CHOICESPECS))
    state = state._replace(choice_slot=set_indexed(
        state.choice_slot, user, move_slot.astype(jnp.int8),
        when=choiced & can_act))

    # U-turn / Volt Switch and whirlwind-style forced switches. Written as one
    # elementwise update over the two players rather than two scatters.
    self_switch = (data["move_self_switch"][move_id] > 0) & landed
    force_switch = data["move_force_switch"][move_id] & landed
    players = jnp.arange(C.NUM_PLAYERS)
    state = state._replace(
        force_switch=(state.force_switch |
                      ((players == user) & self_switch) |
                      ((players == target) & force_switch)))
    return state


def _set_side_condition(state, side, condition, apply_it):
    """Hazards stack up to their layer cap; screens are set to a turn count."""
    condition = jnp.maximum(condition.astype(jnp.int32), 0)
    current = get_indexed(state.side_conditions[side].astype(jnp.int32), condition)
    caps = np.full(C.NUM_SIDE_CONDITIONS, 5, np.int32)
    for idx, cap in C.SIDE_CONDITION_MAX.items():
        caps[idx] = cap
    cap = get_indexed(jnp.asarray(caps), condition)
    is_hazard = condition < C.SC_REFLECT
    new = jnp.where(is_hazard, jnp.minimum(current + 1, cap), 5)
    return state._replace(side_conditions=state.side_conditions.at[side].set(
        set_indexed(state.side_conditions[side], condition,
                    new.astype(jnp.int8), when=apply_it)))


def _roll_hit_count(data, state, user, move_id, key):
    """Number of hits for a multi-hit move (Loaded Dice and Skill Link raise it)."""
    lo = data["move_multihit"][move_id, 0].astype(jnp.int32)
    hi = data["move_multihit"][move_id, 1].astype(jnp.int32)
    ui = act(state, user)
    # Showdown's 2-5 hit distribution: 2 and 3 at 35%, 4 and 5 at 15%.
    table = jnp.array([2, 2, 2, 3, 3, 3, 4, 5], jnp.int32)
    sampled = table[jax.random.randint(key, (), 0, 8)]
    n = jnp.where(hi > lo, sampled, lo)
    loaded = (slot_get(state.item, user, ui) == I.LOADEDDICE) & (hi > lo)
    n = jnp.where(loaded, jnp.maximum(n, 4), n)
    # Skill Link always rolls the maximum number of hits.
    n = jnp.where((slot_get(state.ability, user, ui) == A.SKILLLINK) & (hi > lo), hi, n)
    return jnp.clip(n, 1, 10)
