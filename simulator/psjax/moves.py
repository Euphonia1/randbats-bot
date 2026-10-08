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

import functools

import jax
import jax.numpy as jnp
import numpy as np

from . import callbacks as cb
from . import consts as C
from . import effects as E
from .damage import (Attacker, Defender, MoveCtx, accuracy_check, calc_damage,
                     crit_chance_stage, has_flag,
                     resolve_move_ctx, scale_power_for_hit,
                     type_effectiveness, CRIT_RATES)
from .data import move_index, species_index
from .hooks import A, I
from .mechanics import (act, active_types, actives_alive, apply_boosts, below,
                        berry_wants, boost_delta, can_lose_item, confuse,
                        cure_status, eat_berry, forme_change, get_indexed, has_type,
                        item_locked, log_event, lose_item, set_indexed, shown_species,
                        slot_get, slot_set,
                        soul_heart, damage_pokemon, effective_speed,
                        effective_weather, fraction_of_max, heal_pokemon,
                        is_grounded, round_fraction, set_status, set_terrain,
                        status_immune,
                        set_volatile, set_weather, species_stats, unnerved, uniform,
                        paradox_update, usable_moves, white_herb)
from .state import add_at, barrier, set_at
from .stats import boost_multiply, chain_modify, floordiv, idiv


# --- random words ------------------------------------------------------------
# Everything random one move can do, one `uint32` word apiece. `run_turn` draws
# a `[MOVE_WORDS]` vector for each action from a single hash, rather than each
# decision hashing a key of its own -- see `mechanics.random_words` for why.

MAX_HITS = 10

(W_THAW, W_FULL_PARALYSIS, W_CONFUSED, W_CONFUSION_ROLL, W_ACCURACY, W_CRIT,
 W_HIT_COUNT, W_STATUS, W_CONTACT, W_POISON_TOUCH, W_EFFECT,
 W_CALL, W_FICKLE, W_ATTRACT, W_CONFUSE) = range(15)
# The defender abilities that roll on being hit (Static, Cursed Body, Effect
# Spore, ...) share one word per hit, `W_CONTACT_HITS` below (W_CONTACT is no
# longer read); W_POISON_TOUCH serves the attacker's (Poison Touch, Toxic
# Chain). A Pokemon has one ability, so at most one of each rolls.
W_SECONDARY = 15                      # two per secondary: the chance, then sleep
W_DAMAGE_ROLL = W_SECONDARY + 2 * 2   # one per hit
W_LOADED_DICE = W_DAMAGE_ROLL + MAX_HITS
W_RAMPAGE = W_LOADED_DICE + 1         # Outrage's 2-3 turns
W_HIT_ACCURACY = W_RAMPAGE + 1        # one per hit (Population Bomb, Triple Axel)
W_CONTACT_HITS = W_HIT_ACCURACY + MAX_HITS   # one per hit: Static, Flame Body, Effect Spore
MOVE_WORDS = W_CONTACT_HITS + MAX_HITS


# --- building the calculation contexts ---------------------------------------

def build_attacker(state, side, data=None) -> Attacker:
    i = act(state, side)
    species_num = 0 if data is None else data["species_num"][slot_get(state.species, side, i)]
    return Attacker(
        species_num=species_num,
        level=slot_get(state.level, side, i), stats=slot_get(state.stats, side, i),
        boosts=state.boosts[side], types=active_types(state, side),
        base_types=slot_get(state.types, side, i), ability=slot_get(state.ability, side, i),
        item=slot_get(state.item, side, i), status=slot_get(state.status, side, i),
        hp=slot_get(state.hp, side, i), maxhp=slot_get(state.maxhp, side, i),
        terastallized=slot_get(state.terastallized, side, i),
        tera_type=slot_get(state.tera_type, side, i),
        boosted_stat=state.boosted_stat[side],
        slow_start=state.volatiles[side, C.V_SLOWSTART] > 0,
        charged=state.volatiles[side, C.V_CHARGE] > 0,
        flash_fire=state.volatiles[side, C.V_FLASHFIRE] > 0,
    )


def build_defender(data, state, side, ability=None) -> Defender:
    """`ability` overrides the real one -- 0 when Mold Breaker is ignoring it."""
    i = act(state, side)
    if ability is None:
        ability = slot_get(state.ability, side, i)
    return Defender(
        stats=slot_get(state.stats, side, i), status=slot_get(state.status, side, i),
        boosts=state.boosts[side], types=active_types(state, side),
        ability=ability, item=slot_get(state.item, side, i),
        hp=slot_get(state.hp, side, i), maxhp=slot_get(state.maxhp, side, i),
        terastallized=slot_get(state.terastallized, side, i),
        tera_type=slot_get(state.tera_type, side, i),
        nfe=data["species_nfe"][slot_get(state.species, side, i)],
        boosted_stat=state.boosted_stat[side],
        glaive_rush=state.volatiles[side, C.V_GLAIVERUSH] > 0,
    )


def weight(data, state, side, ignore_ability=False):
    """The active Pokemon's weight in kg: Heavy Metal doubles it, Light Metal
    halves it (Showdown truncates hectograms)."""
    i = act(state, side)
    kg = data["species_weight"][slot_get(state.species, side, i)]
    ab = jnp.where(ignore_ability, 0, slot_get(state.ability, side, i))
    hg = jnp.round(kg * 10).astype(jnp.int32)
    hg = jnp.where(ab == A.HEAVYMETAL, hg * 2,
                   jnp.where(ab == A.LIGHTMETAL, floordiv(hg, 2), hg))
    return hg.astype(jnp.float32) / 10


def build_cb_ctx(data, state, user, target, moves_first, pp_left, hit_number=1,
                 fickle=False, coin=True):
    ui, ti = act(state, user), act(state, target)
    user_species = slot_get(state.species, user, ui)
    # The target's weight abilities are breakable.
    mold = data["ability_mold_breaker"][slot_get(state.ability, user, ui)]
    a_stats, d_stats = slot_get(state.stats, user, ui), slot_get(state.stats, target, ti)
    boosted = lambda stats, boosts, s: boost_multiply(stats[s], boosts[s - 1])
    return cb.CbCtx(
        base_power=jnp.int32(0), move_type=jnp.int8(0),
        hit_number=jnp.asarray(hit_number, jnp.int32),
        atk_status=slot_get(state.status, user, ui), dfn_status=slot_get(state.status, target, ti),
        atk_hp=slot_get(state.hp, user, ui), atk_maxhp=slot_get(state.maxhp, user, ui),
        dfn_hp=slot_get(state.hp, target, ti), dfn_maxhp=slot_get(state.maxhp, target, ti),
        atk_item=slot_get(state.item, user, ui), dfn_item=slot_get(state.item, target, ti),
        atk_weight=weight(data, state, user),
        dfn_weight=weight(data, state, target, ignore_ability=mold),
        atk_speed=effective_speed(state, user), dfn_speed=effective_speed(state, target),
        atk_boosts=state.boosts[user], dfn_boosts=state.boosts[target],
        moves_first=moves_first,
        user_damaged=state.damage_taken[user] > 0,
        target_damaged=state.damage_taken[target] > 0,
        weather=effective_weather(state), terrain=state.terrain,
        grounded_user=is_grounded(state, user),
        grounded_target=is_grounded(state, target),
        pp_left=pp_left.astype(jnp.int32),
        times_hit=slot_get(state.times_hit, user, ui).astype(jnp.int32),
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
        user_type=active_types(state, user)[0],
        # Filled in once the real effectiveness is known (see execute_move).
        type_exp=jnp.int32(0),
        dfn_def=boosted(d_stats, state.boosts[target], C.DEF),
        dfn_spd=boosted(d_stats, state.boosts[target], C.SPD),
        gravity=state.gravity > 0, fickle=fickle, coin=coin, fusion_last=state.fusion_last,
        target_switched=state.switched_this_turn[target],
        user_base_atk=data["species_base_stats"][user_species, C.ATK],
        user_species=user_species,
        plate_type=data["item_plate_type"][slot_get(state.item, user, ui)],
        dfn_item_locked=item_locked(data, state, target, ti),
    )


# --- before-move checks ------------------------------------------------------

def before_move(data, state, user, move_id, words, move_slot, struggling=False):
    """Can the user act? Returns `(state, can_act)`.

    Showdown's BeforeMove handlers, in their priority order: recharge, sleep and
    freeze, Truant, flinch, Disable, Taunt, confusion, Attract, paralysis. The
    first to stop the move ends the event, so a sleeping Pokemon neither ticks
    its confusion nor rolls a self-hit. `words` is the move's random words (see
    `MOVE_WORDS`).
    """
    ui = act(state, user)
    # Glaive Rush's drawback ends the moment its user next tries to move.
    state = state._replace(volatiles=set_at(state.volatiles, (user, C.V_GLAIVERUSH), 0))
    vol = lambda v: state.volatiles[user, v]

    can = slot_get(state.hp, user, ui) > 0

    # Recharging (Hyper Beam): the turn is spent and the volatile clears. It
    # counts as Truant's idle turn too (Showdown's recharge clears `truant`), so
    # a Slaking acts again straight after recharging from Giga Impact.
    recharging = vol(C.V_RECHARGE) > 0
    state = state._replace(volatiles=set_at(
        set_at(state.volatiles, (user, C.V_RECHARGE), 0), (user, C.V_TRUANT), 0, recharging))
    can = can & jnp.logical_not(recharging)

    # Sleep: the counter ticks down whenever the Pokemon tries to move.
    asleep = can & (slot_get(state.status, user, ui) == C.SLP)
    # Early Bird ticks two turns of sleep off per attempt.
    tick = jnp.where(slot_get(state.ability, user, ui) == A.EARLYBIRD, 2, 1)
    turns = jnp.maximum(slot_get(state.status_turns, user, ui).astype(jnp.int32) - tick, 0)
    wakes = asleep & (turns <= 0)
    attempts = slot_get(state.sleep_attempts, user, ui) + 1
    state = state._replace(
        status=slot_set(state.status, user, ui, jnp.int8(C.STATUS_NONE), wakes),
        status_turns=slot_set(state.status_turns, user, ui, turns, asleep),
        sleep_attempts=slot_set(state.sleep_attempts, user, ui,
                                jnp.where(wakes, 0, attempts).astype(jnp.int8), asleep))
    sleep_usable = data["move_sleep_usable"][move_id]
    can = can & jnp.logical_not(asleep & jnp.logical_not(wakes) &
                                jnp.logical_not(sleep_usable))

    # Freeze thaws with 20% probability, or on any move that thaws its user.
    frozen = can & (slot_get(state.status, user, ui) == C.FRZ)
    thaws = (uniform(words[W_THAW]) < 0.2) | has_flag(data["move_flags"][move_id], "defrost")
    state = state._replace(
        status=slot_set(state.status, user, ui, jnp.int8(C.STATUS_NONE), frozen & thaws))
    can = can & jnp.logical_not(frozen & jnp.logical_not(thaws))

    # Truant loafs on every other move attempt.
    truant = can & (slot_get(state.ability, user, ui) == A.TRUANT)
    loafing = truant & (vol(C.V_TRUANT) > 0)
    state = state._replace(volatiles=set_at(
        state.volatiles, (user, C.V_TRUANT), jnp.where(loafing, 0, 1), truant))
    can = can & jnp.logical_not(loafing)

    # Flinch lasts only for the turn it was applied.
    flinched = vol(C.V_FLINCH) > 0
    state = state._replace(volatiles=set_at(state.volatiles, (user, C.V_FLINCH), 0))
    can = can & jnp.logical_not(flinched)

    # Disable stops the one move; Taunt stops every status move; Throat Chop
    # sound moves; Heal Block healing moves. None of them stops Struggle.
    flags = data["move_flags"][move_id]
    stopped = ((vol(C.V_DISABLE) > 0) & (state.disabled_slot[user] == move_slot)) | \
        ((vol(C.V_TAUNT) > 0) & (data["move_category"][move_id] == C.CAT_STATUS)) | \
        ((vol(C.V_THROATCHOP) > 0) & has_flag(flags, "sound")) | \
        ((vol(C.V_HEALBLOCK) > 0) & has_flag(flags, "heal"))
    can = can & jnp.logical_not(stopped & jnp.logical_not(struggling))

    # Confusion: tick down; while it lasts, a 33% chance of hitting yourself.
    confused = can & (vol(C.V_CONFUSION) > 0)
    cnf_left = jnp.maximum(vol(C.V_CONFUSION).astype(jnp.int32) - 1, 0)
    state = state._replace(volatiles=set_at(
        state.volatiles, (user, C.V_CONFUSION), cnf_left, confused))
    self_hit = confused & (cnf_left > 0) & (below(words[W_CONFUSED], 100) < 33)

    # The confusion self-hit is a 40 BP typeless physical hit on yourself.
    atk = build_attacker(state, user, data)
    conf_dmg = _confusion_damage(state, user, atk, below(words[W_CONFUSION_ROLL], 16))
    state, _ = damage_pokemon(state, user, ui,
                              jnp.where(self_hit, conf_dmg, jnp.int32(0)))
    can = can & jnp.logical_not(self_hit)

    # Attract: immobilised by love half the time.
    can = can & jnp.logical_not((vol(C.V_ATTRACT) > 0) & (uniform(words[W_ATTRACT]) < 0.5))

    # Full paralysis: 25%.
    paralysed = slot_get(state.status, user, ui) == C.PAR
    can = can & jnp.logical_not(paralysed & (uniform(words[W_FULL_PARALYSIS]) < 0.25))
    return state, can


def _confusion_damage(state, side, atk: Attacker, roll):
    """40 BP physical, no type, no crit -- Showdown's confusion self-hit.

    `roll` is the damage roll, 0..15.
    """
    level = atk.level.astype(jnp.int32)
    a = boost_multiply(atk.stats[C.ATK], atk.boosts[C.B_ATK])
    d = jnp.maximum(boost_multiply(atk.stats[C.DEF], atk.boosts[C.B_DEF]), 1)
    base = idiv((floordiv(2 * level, 5) + 2) * 40 * a, d)
    base = floordiv(base, 50) + 2
    return jnp.maximum(floordiv(base * (100 - roll), 100), 1)


# --- hit-step helpers --------------------------------------------------------

def move_blocked_by_protect(data, state, user, target, move_id):
    """Protect and its variants; Mat Block, which only stops damaging moves.

    Unseen Fist's contact moves go straight through both.
    """
    flags = data["move_flags"][move_id]
    protected = (state.volatiles[target, C.V_PROTECT] > 0) | (
        (state.volatiles[target, C.V_MATBLOCK] > 0) &
        (data["move_category"][move_id] != C.CAT_STATUS))
    unseen = (slot_get(state.ability, user, act(state, user)) == A.UNSEENFIST) & \
        has_flag(flags, "contact")
    bypass = data["move_breaks_protect"][move_id] | \
        jnp.logical_not(has_flag(flags, "protect")) | unseen
    return protected & jnp.logical_not(bypass)


def ability_absorbs(data, ab, move_type, category):
    """Water Absorb, Sap Sipper, Flash Fire, ... Returns `(absorbs, heal, boost)`.

    `ab` is the target's ability as the move sees it (Mold Breaker zeroes it).
    Any move of the type aimed at the holder is absorbed, status moves too --
    Spore into Sap Sipper, Thunder Wave into Volt Absorb; the caller checks the
    move is aimed at the holder.
    """
    absorb_type = data["ability_absorb_type"][ab]
    hits = (absorb_type == move_type) & (absorb_type != C.TYPE_NONE)
    return hits, data["ability_absorb_heal"][ab], \
        (data["ability_absorb_boost_stat"][ab], data["ability_absorb_boost_amt"][ab])


def flag_immune(data, ab, move_id, at_foe):
    """Soundproof and Bulletproof block their kind of move aimed at the holder."""
    flags = data["move_flags"][move_id]
    return at_foe & ((has_flag(flags, "sound") & (ab == A.SOUNDPROOF)) |
                     (has_flag(flags, "bullet") & (ab == A.BULLETPROOF)))


def powder_immune(data, state, target, move_id, ab):
    """Powder moves do not affect Grass types, Overcoat or Safety Goggles."""
    ti = act(state, target)
    types = active_types(state, target)
    return has_flag(data["move_flags"][move_id], "powder") & (
        jnp.any(types == C.GRASS) | (ab == A.OVERCOAT) |
        (slot_get(state.item, target, ti) == I.SAFETYGOGGLES))


def move_priority(data, state, side, move_id):
    """A move's priority after Prankster, Triage, Gale Wings and Grassy Glide."""
    i = act(state, side)
    pri = data["move_priority"][move_id].astype(jnp.int32)
    ab = slot_get(state.ability, side, i)
    category = data["move_category"][move_id]
    flags = data["move_flags"][move_id]
    pri = pri + jnp.where((ab == A.PRANKSTER) & (category == C.CAT_STATUS), 1, 0)
    pri = pri + jnp.where((ab == A.TRIAGE) & has_flag(flags, "heal"), 3, 0)
    pri = pri + jnp.where((ab == A.GALEWINGS) & (data["move_type"][move_id] == C.FLYING) &
                          (slot_get(state.hp, side, i) == slot_get(state.maxhp, side, i)), 1, 0)
    glide = (data["move_effect_cb"][move_id] == E.EFFECT_HANDLERS.index("grassyglide")) & \
        (state.terrain == C.GRASSY_TERRAIN) & is_grounded(state, side)
    return pri + jnp.where(glide, 1, 0)


def terrain_blocks(data, state, target, move_id, priority):
    """Psychic Terrain blocks priority moves; Misty Terrain blocks Dragon moves."""
    grounded = is_grounded(state, target)
    psychic = (state.terrain == C.PSYCHIC_TERRAIN) & grounded & (priority > 0)
    return psychic


# --- special effects ---------------------------------------------------------
# Signature: (data, state, user, target, word) -> state, where `word` is the
# move's `W_EFFECT` random word.

def _noop(data, state, user, target, word):
    return state


def _eff_substitute(data, state, user, target, word):
    ui = act(state, user)
    cost = fraction_of_max(state, user, ui, 1, 4)
    enough = slot_get(state.hp, user, ui).astype(jnp.int32) > cost
    already = state.volatiles[user, C.V_SUBSTITUTE] > 0
    ok = enough & jnp.logical_not(already)
    state, _ = damage_pokemon(state, user, ui, jnp.where(ok, cost, 0))
    return state._replace(
        volatiles=set_at(state.volatiles, (user, C.V_SUBSTITUTE), 1, when=ok),
        sub_hp=set_at(state.sub_hp, user, cost, when=ok))


def _eff_protect(data, state, user, target, word):
    """Protect and its variants; consecutive uses get exponentially likelier to fail."""
    streak = state.protect_streak[user].astype(jnp.int32)
    # Showdown: succeeds with probability 1/3^streak.
    threshold = jnp.power(3.0, -streak.astype(jnp.float32))
    ok = uniform(word) < threshold
    return state._replace(
        volatiles=set_at(state.volatiles, (user, C.V_PROTECT), jnp.where(ok, 1, 0)),
        protect_streak=set_at(state.protect_streak, user,
                              jnp.where(ok, jnp.minimum(streak + 1, 6), 0)))


def _eff_rest(data, state, user, target, word):
    """Full heal and three turns' sleep, replacing any other status.

    Fails at full HP, when already asleep, and wherever the user could not fall
    asleep: Insomnia and friends, Leaf Guard in sun, Minior's shell, and Electric
    or Misty Terrain under a grounded user.
    """
    ui = act(state, user)
    full = slot_get(state.hp, user, ui) >= slot_get(state.maxhp, user, ui)
    ab = slot_get(state.ability, user, ui)
    w = effective_weather(state)
    grounded = is_grounded(state, user)
    sleepless = ((data["ability_status_immune"][ab] >> C.SLP) & 1).astype(bool) | \
        ((ab == A.LEAFGUARD) & ((w == C.SUN) | (w == C.HARSH_SUN))) | \
        ((ab == A.SHIELDSDOWN) & (slot_get(state.species, user, ui) ==
                                  species_index("miniormeteor"))) | \
        (grounded & ((state.terrain == C.ELECTRIC_TERRAIN) |
                     (state.terrain == C.MISTY_TERRAIN))) | \
        (slot_get(state.status, user, ui) == C.SLP)
    ok = jnp.logical_not(full) & jnp.logical_not(sleepless)
    state, _ = heal_pokemon(state, user, ui,
                            jnp.where(ok, slot_get(state.maxhp, user, ui).astype(jnp.int32), 0))
    return state._replace(
        status=slot_set(state.status, user, ui, 
            jnp.where(ok, jnp.int8(C.SLP), slot_get(state.status, user, ui))),
        status_turns=slot_set(state.status_turns, user, ui, 
            jnp.where(ok, jnp.int8(3), slot_get(state.status_turns, user, ui))),
        sleep_attempts=slot_set(state.sleep_attempts, user, ui, jnp.int8(0), ok),
        rest_sleep=slot_set(state.rest_sleep, user, ui, True, ok))


def _eff_haze(data, state, user, target, word):
    """Resets every stat change on both sides."""
    return state._replace(boosts=jnp.zeros_like(state.boosts))


def _eff_leechseed(data, state, user, target, word):
    """Grass types cannot be seeded.

    The volatile itself is declarative and has already been applied by the time
    this runs, so the handler's job is to take it back off a Grass type.
    """
    grass = jnp.any(active_types(state, target) == C.GRASS)
    return state._replace(
        volatiles=set_at(state.volatiles, (target, C.V_LEECHSEED), 0, when=grass))


def _eff_painsplit(data, state, user, target, word):
    ui, ti = act(state, user), act(state, target)
    total = slot_get(state.hp, user, ui).astype(jnp.int32) + slot_get(state.hp, target, ti).astype(jnp.int32)
    half = floordiv(total, 2)
    new_u = jnp.minimum(half, slot_get(state.maxhp, user, ui).astype(jnp.int32))
    new_t = jnp.minimum(half, slot_get(state.maxhp, target, ti).astype(jnp.int32))
    hp = slot_set(state.hp, user, ui, new_u.astype(jnp.int16))
    hp = slot_set(hp, target, ti, new_t.astype(jnp.int16))
    return state._replace(hp=hp)


def _eff_defog(data, state, user, target, word):
    """Lowers the target's evasion (not through its Substitute), clears hazards
    from both sides and screens, Safeguard and Mist from the target's, and ends
    any terrain. The user's own screens and Tailwind stay."""
    sc = state.side_conditions
    target_side = (jnp.arange(C.NUM_PLAYERS) == target)[:, None]
    clears = jnp.where(target_side, jnp.asarray(_DEFOGGED), jnp.asarray(_HAZARDS)[None, :])
    state = state._replace(side_conditions=jnp.where(clears, jnp.int8(0), sc),
                           terrain=jnp.int8(C.TERRAIN_NONE), terrain_turns=jnp.int8(0))
    shielded = (state.volatiles[target, C.V_SUBSTITUTE] > 0) & \
        (slot_get(state.ability, user, act(state, user)) != A.INFILTRATOR)
    state, _ = apply_boosts(state, target,
                            jnp.where(shielded, 0, boost_delta((C.B_EVA, -1))),
                            from_opponent=True)
    return state


_HAZARDS = np.isin(np.arange(C.NUM_SIDE_CONDITIONS),
                   [C.SC_STEALTHROCK, C.SC_SPIKES, C.SC_TOXICSPIKES, C.SC_STICKYWEB])
_DEFOGGED = _HAZARDS | np.isin(np.arange(C.NUM_SIDE_CONDITIONS),
                               [C.SC_REFLECT, C.SC_LIGHTSCREEN, C.SC_AURORAVEIL,
                                C.SC_SAFEGUARD, C.SC_MIST])


def _remove_hazards(state, side):
    return state._replace(side_conditions=set_at(
        state.side_conditions, side, jnp.where(_HAZARDS, 0, state.side_conditions)))


def _spin_free(state, user):
    """Rapid Spin and Mortal Spin clear hazards, binding and Leech Seed from the
    user's side -- if the user survived the hit."""
    alive = slot_get(state.hp, user, act(state, user)) > 0
    spun = _remove_hazards(state, user)
    vols = set_at(state.volatiles, (user, C.V_PARTIALLYTRAPPED), 0, alive)
    vols = set_at(vols, (user, C.V_LEECHSEED), 0, alive)
    return state._replace(
        volatiles=vols,
        side_conditions=jnp.where(alive, spun.side_conditions, state.side_conditions))


def _eff_rapidspin(data, state, user, target, word):
    # The Speed boost is a 100%-chance secondary in the move data, so it is
    # already applied by `_apply_secondaries`; doing it here too doubled it.
    return _spin_free(state, user)


def _eff_mortalspin(data, state, user, target, word):
    # Its poison goes through the one `set_status` call in `execute_move`.
    return _spin_free(state, user)


def _mold_breaker(data, state, user):
    """Does the user's ability ignore its target's breakable abilities?"""
    return data["ability_mold_breaker"][slot_get(state.ability, user, act(state, user))]


def _eff_trick(data, state, user, target, word):
    """Swap held items.

    Fails if neither has anything to give, against Sticky Hold, and if either
    item cannot leave its holder (a plate on Arceus).
    """
    ui, ti = act(state, user), act(state, target)
    a, b = slot_get(state.item, user, ui), slot_get(state.item, target, ti)
    mold = _mold_breaker(data, state, user)
    sticky = (slot_get(state.ability, target, ti) == A.STICKYHOLD) & jnp.logical_not(mold)
    ok = ((a != 0) | (b != 0)) & jnp.logical_not(sticky) & \
        ((a == 0) | can_lose_item(data, state, user, ui, by_opponent=False)) & \
        ((b == 0) | can_lose_item(data, state, target, ti, ignore_ability=mold))
    item = slot_set(state.item, user, ui, jnp.where(ok, b, a))
    item = slot_set(item, target, ti, jnp.where(ok, a, b))
    # Whoever ends up empty-handed has lost an item, which arms Unburden.
    vols = set_at(state.volatiles, (user, C.V_UNBURDEN), 1, ok & (b == 0))
    vols = set_at(vols, (target, C.V_UNBURDEN), 1, ok & (a == 0))
    # Whoever receives an item is freed of any Choice lock -- the item's onStart
    # clears it, even when it is the same Choice item it gave away.
    choice = set_at(state.choice_slot, user, -1, ok & (b != 0))
    choice = set_at(choice, target, -1, ok & (a != 0))
    return state._replace(item=item, volatiles=vols, choice_slot=choice)


def _eff_knockoff(data, state, user, target, word):
    """Knock the target's item away -- if its holder can lose it and the user
    survived to do it (Rocky Helmet can faint the attacker first)."""
    ui, ti = act(state, user), act(state, target)
    ok = can_lose_item(data, state, target, ti, ignore_ability=_mold_breaker(data, state, user)) & \
        (slot_get(state.hp, user, ui) > 0)
    return lose_item(state, target, ti, ok)


def _eff_bellydrum(data, state, user, target, word):
    ui = act(state, user)
    cost = fraction_of_max(state, user, ui, 1, 2)
    # Fails at half HP or less, or with Attack already maxed.
    ok = (slot_get(state.hp, user, ui).astype(jnp.int32) > cost) & \
        (state.boosts[user, C.B_ATK] < 6)
    state, _ = damage_pokemon(state, user, ui, jnp.where(ok, cost, 0))
    return state._replace(boosts=set_at(state.boosts, (user, C.B_ATK), 6, when=ok))


def _eff_roost(data, state, user, target, word):
    """Nothing beyond the declared data: the half heal and the `roost` self
    volatile are applied by `execute_move`, which also fails them both at full
    HP. Doing either here as well healed twice."""
    return state


def _eff_weather_heal(data, state, user, target, word):
    """Synthesis / Moonlight / Morning Sun: 2/3 in sun, 1/4 in other weather.

    Showdown heals `modify(maxhp, factor)`, which quantises the factor to
    4096ths -- 0.667 becomes 2731/4096, not exactly two thirds -- so a plain
    fraction is off by a point.
    """
    ui = act(state, user)
    w = effective_weather(state)
    sun = (w == C.SUN) | (w == C.HARSH_SUN)
    other = (w != C.WEATHER_NONE) & jnp.logical_not(sun)
    factor = jnp.where(sun, 2731, jnp.where(other, 1024, 2048))
    maxhp = slot_get(state.maxhp, user, ui).astype(jnp.int32)
    state, _ = heal_pokemon(state, user, ui, chain_modify(maxhp, factor))
    return state


def _eff_shoreup(data, state, user, target, word):
    """Two thirds in sand, half otherwise -- same 4096ths rounding as above."""
    ui = act(state, user)
    sand = effective_weather(state) == C.SAND
    factor = jnp.where(sand, 2731, 2048)
    maxhp = slot_get(state.maxhp, user, ui).astype(jnp.int32)
    state, _ = heal_pokemon(state, user, ui, chain_modify(maxhp, factor))
    return state


def _eff_strengthsap(data, state, user, target, word):
    """Heals by the target's current Attack, then drops it. Fails outright on a
    target already at -6 Attack."""
    ui, ti = act(state, user), act(state, target)
    amount = boost_multiply(state.stats[target, ti, C.ATK], state.boosts[target, C.B_ATK])
    amount = jnp.where(state.boosts[target, C.B_ATK] <= -6, 0, amount)
    # Liquid Ooze turns the drink into damage.
    ooze = slot_get(state.ability, target, ti) == A.LIQUIDOOZE
    state, _ = heal_pokemon(state, user, ui, jnp.where(ooze, 0, amount))
    state, _ = damage_pokemon(state, user, ui, jnp.where(ooze, amount, 0))
    state, _ = apply_boosts(state, target, boost_delta((C.B_ATK, -1)),
                            from_opponent=True)
    return state


def _eff_curse(data, state, user, target, word):
    """Ghost types pay half their HP to curse the target; others boost instead."""
    ui = act(state, user)
    ghost = jnp.any(active_types(state, user) == C.GHOST)
    cost = fraction_of_max(state, user, ui, 1, 2)
    state, _ = damage_pokemon(state, user, ui, jnp.where(ghost, cost, 0))
    state = state._replace(
        volatiles=set_at(state.volatiles, (target, C.V_CURSE), 1, when=ghost))
    boosts = boost_delta((C.B_ATK, 1), (C.B_DEF, 1), (C.B_SPE, -1))
    state, _ = apply_boosts(state, user, jnp.where(ghost, 0, boosts))
    return state


def _eff_clearsmog(data, state, user, target, word):
    return state._replace(boosts=set_at(state.boosts, target, 0))


def _eff_topsyturvy(data, state, user, target, word):
    return state._replace(boosts=set_at(state.boosts, target, -state.boosts))


def _eff_spectralthief(data, state, user, target, word):
    """Steals the target's positive boosts before dealing damage."""
    stolen = jnp.maximum(state.boosts[target].astype(jnp.int32), 0)
    state = state._replace(boosts=set_at(state.boosts, target,
                                         jnp.minimum(state.boosts, 0)))
    state, _ = apply_boosts(state, user, stolen)
    return state


def _eff_refresh(data, state, user, target, word):
    return cure_status(state, user, act(state, user))


def _eff_aromatherapy(data, state, user, target, word):
    return state._replace(status=set_at(state.status, user, C.STATUS_NONE),
                          status_turns=set_at(state.status_turns, user, 0))


def _eff_filletaway(data, state, user, target, word):
    """Pays half its max HP for the +2s, which are declarative (applying them
    here too doubled them); the move already failed at half HP or less."""
    ui = act(state, user)
    return damage_pokemon(state, user, ui, fraction_of_max(state, user, ui, 1, 2))[0]


def _eff_noretreat(data, state, user, target, word):
    """Nothing beyond the declared data: the +1 to every stat and the trapping
    `noretreat` volatile are both declarative (applying the boosts here too
    doubled them), and `execute_move` fails the move if it is already up."""
    return state


def _eff_tidyup(data, state, user, target, word):
    state = _remove_hazards(state, user)
    state = _remove_hazards(state, target)
    state = state._replace(
        volatiles=set_at(state.volatiles, (slice(None), C.V_SUBSTITUTE), 0),
        sub_hp=jnp.zeros_like(state.sub_hp))
    state, _ = apply_boosts(state, user, boost_delta((C.B_ATK, 1), (C.B_SPE, 1)))
    return state


def _eff_courtchange(data, state, user, target, word):
    return state._replace(side_conditions=state.side_conditions[::-1])


def _eff_saltcure(data, state, user, target, word):
    return set_volatile(state, target, C.V_SALTCURE, 1)


def _eff_trickroom(data, state, user, target, word):
    """Trick Room twists the dimensions for five turns; used again, it ends them."""
    return state._replace(
        trick_room=jnp.where(state.trick_room > 0, jnp.int8(0), jnp.int8(5)))


def _eff_glaiverush(data, state, user, target, word):
    return set_volatile(state, user, C.V_GLAIVERUSH, 1)


_SCREENS = np.isin(np.arange(C.NUM_SIDE_CONDITIONS),
                   [C.SC_REFLECT, C.SC_LIGHTSCREEN, C.SC_AURORAVEIL])


def _eff_screenbreak(data, state, user, target, word):
    """Brick Break / Psychic Fangs / Raging Bull shatter screens before hitting."""
    return state._replace(side_conditions=set_at(
        state.side_conditions, target, jnp.where(_SCREENS, 0, state.side_conditions)))


def _eff_icespinner(data, state, user, target, word):
    """Removes the terrain."""
    return state._replace(terrain=jnp.int8(C.TERRAIN_NONE),
                          terrain_turns=jnp.int8(0))


def _eff_partingshot(data, state, user, target, word):
    """Drops the target's offences. The switch out is the move's `selfSwitch`,
    which `execute_move` grants only if a stat actually changed."""
    drop = boost_delta((C.B_ATK, -1), (C.B_SPA, -1))
    state, _ = apply_boosts(state, target, drop, from_opponent=True)
    return state


def _eff_chillyreception(data, state, user, target, word):
    """Sets snow, then the user switches out."""
    state = set_weather(state, jnp.int8(C.SNOW))
    return state._replace(force_switch=set_at(state.force_switch, user, True))


def _eff_healingwish(data, state, user, target, word):
    """The user faints, and the first replacement to arrive hurt or statused is
    restored in full (`arrive` keeps the wish until then). Fails with no one to
    switch to. Lunar Dance shares this, and restores PP as well.
    """
    ui = act(state, user)
    any_ally = jnp.any((state.hp[user] > 0) & (jnp.arange(C.TEAM_SIZE) != ui))
    state, _ = damage_pokemon(state, user, ui,
                              jnp.where(any_ally, slot_get(state.hp, user, ui).astype(jnp.int32), 0))
    # `last_move` already holds the move being used.
    lunar = state.last_move[user] == move_index("lunardance")
    return state._replace(healing_wish=set_at(state.healing_wish, user,
                                              jnp.where(lunar, 2, 1), any_ally))


def _eff_revivalblessing(data, state, user, target, word):
    """Asks the player which fainted team member to revive, as a self-switch asks
    for a replacement: `pass_mode` 4 marks the prompt, the answer revives that
    Pokemon at half HP (`engine.revive`), and the user stays in. The move has
    already failed if nobody has fainted."""
    return state._replace(force_switch=set_at(state.force_switch, user, True),
                          pass_mode=set_at(state.pass_mode, user, C.PASS_REVIVE))


def _eff_shedtail(data, state, user, target, word):
    """Pays half its max HP (rounded up) for a Substitute it leaves behind.

    The Substitute itself is the usual quarter of max HP; `switch_to` hands it
    to the replacement. Fails without the HP to pay or a Pokemon to pass to.
    """
    ui = act(state, user)
    maxhp = slot_get(state.maxhp, user, ui).astype(jnp.int32)
    cost = maxhp - floordiv(maxhp, 2)
    bench = jnp.any((state.hp[user] > 0) & (jnp.arange(C.TEAM_SIZE) != ui))
    ok = (slot_get(state.hp, user, ui).astype(jnp.int32) > cost) & bench &         (state.volatiles[user, C.V_SUBSTITUTE] <= 0)
    state, _ = damage_pokemon(state, user, ui, jnp.where(ok, cost, 0))
    return state._replace(
        volatiles=set_at(state.volatiles, (user, C.V_SUBSTITUTE), 1, ok),
        sub_hp=set_at(state.sub_hp, user, floordiv(maxhp, 4), ok),
        force_switch=set_at(state.force_switch, user, True, ok))


def _eff_burnup(data, state, user, target, word):
    """The user loses the move's own type (Fire for Burn Up, Electric for Double
    Shock) -- unless it is Terastallized: Showdown's `setType` refuses to touch
    a Terastallized Pokemon's types, so it keeps the type, and the 2x STAB of a
    Tera type it already had."""
    ui = act(state, user)
    lost = jnp.where(slot_get(state.types, user, ui) == C.FIRE, jnp.int8(C.TYPE_NONE),
                     slot_get(state.types, user, ui))
    return state._replace(types=slot_set(state.types, user, ui, lost,
                                         jnp.logical_not(slot_get(state.terastallized, user, ui))))


def _eff_doubleshock(data, state, user, target, word):
    ui = act(state, user)
    lost = jnp.where(slot_get(state.types, user, ui) == C.ELECTRIC, jnp.int8(C.TYPE_NONE),
                     slot_get(state.types, user, ui))
    return state._replace(types=slot_set(state.types, user, ui, lost,
                                         jnp.logical_not(slot_get(state.terastallized, user, ui))))


def _eff_smackdown(data, state, user, target, word):
    """Grounds an airborne target for as long as it stays in.

    Showdown's condition only sticks to something that was off the ground --
    a Flying type, Levitate or Magnet Rise -- and knocks Magnet Rise away.
    """
    ti = act(state, target)
    airborne = jnp.any(active_types(state, target) == C.FLYING) | \
        (slot_get(state.ability, target, ti) == A.LEVITATE) | \
        (state.volatiles[target, C.V_MAGNETRISE] > 0)
    anchored = (state.volatiles[target, C.V_INGRAIN] > 0) | (state.gravity > 0)
    # The move's declarative volatile has already gone on; take it back off
    # anything it does not stick to.
    sticks = airborne & jnp.logical_not(anchored)
    vols = set_at(state.volatiles, (target, C.V_SMACKDOWN), jnp.where(sticks, 1, 0))
    vols = set_at(vols, (target, C.V_MAGNETRISE), 0)
    return state._replace(volatiles=vols)


def _eff_fakeout(data, state, user, target, word):
    """Fake Out and First Impression only work on the user's first move out.

    The move's flinch/damage has already been applied by the generic path, so
    failing means undoing it; `execute_move` checks `fakeout_ok` before the hit
    instead, and this handler only has to clear the flinch on a late use.
    """
    late = state.moves_since_switch[user] > 0
    return state._replace(
        volatiles=set_at(state.volatiles, (target, C.V_FLINCH), 0, when=late))


def _eff_auroraveil(data, state, user, target, word):
    """Nothing here: Aurora Veil fails without snow, which `execute_move`
    checks before anything is set. (Undoing the veil here instead wiped out
    one already up.)"""
    return state


def transform_into(data, state, side, when=True):
    """`side`'s active becomes a copy of the opposing one (Transform, Imposter).

    Showdown's `transformInto`: species, types (as they were before any Tera),
    stats other than HP, stat stages, ability and moves -- each move with 5 PP.
    The Pokemon's own moves wait in `tf_*` until it leaves the field. Fails on
    a Substitute, if either side is already Transformed or disguised by
    Illusion, and around a Terastallized Terapagos. Returns `(state, ok)`.
    """
    other = 1 - side
    i, oi = act(state, side), act(state, other)
    species = slot_get(state.species, other, oi)
    terapagos = ((species == species_index("terapagos")) |
                 (species == species_index("terapagosterastal")) |
                 (species == species_index("terapagosstellar")))
    my_tera = slot_get(state.terastallized, side, i)
    stellar = my_tera & (slot_get(state.tera_type, side, i) == C.STELLAR)
    ok = when & (slot_get(state.hp, other, oi) > 0) & (slot_get(state.hp, side, i) > 0) & \
        jnp.logical_not(state.transformed[side] | state.transformed[other]) & \
        (state.volatiles[other, C.V_SUBSTITUTE] == 0) & \
        (state.illusion[side] < 0) & (state.illusion[other] < 0) & \
        jnp.logical_not(terapagos & (my_tera | slot_get(state.terastallized, other, oi))) & \
        jnp.logical_not(stellar)

    moves = slot_get(state.moves, other, oi)
    pp = jnp.where(moves >= 0,
                   jnp.minimum(5, data["move_pp"][jnp.maximum(moves, 0)].astype(jnp.int32)), 0)
    own_stats = slot_get(state.stats, side, i)
    stats = jnp.concatenate([own_stats[:1], slot_get(state.stats, other, oi)[1:]])
    return state._replace(
        tf_moves=set_at(state.tf_moves, side, slot_get(state.moves, side, i), ok),
        tf_pp=set_at(state.tf_pp, side, slot_get(state.pp, side, i), ok),
        tf_maxpp=set_at(state.tf_maxpp, side, slot_get(state.maxpp, side, i), ok),
        transformed=set_at(state.transformed, side, True, ok),
        species=slot_set(state.species, side, i, species, ok),
        types=slot_set(state.types, side, i, slot_get(state.types, other, oi), ok),
        stats=slot_set(state.stats, side, i, stats, ok),
        boosts=set_at(state.boosts, side, state.boosts[other], ok),
        ability=slot_set(state.ability, side, i, slot_get(state.ability, other, oi), ok),
        moves=slot_set(state.moves, side, i, moves, ok),
        pp=slot_set(state.pp, side, i, pp, ok),
        maxpp=slot_set(state.maxpp, side, i, pp, ok),
        # The critical-hit volatiles come along too.
        volatiles=set_at(state.volatiles, (side, C.V_FOCUSENERGY),
                         state.volatiles[other, C.V_FOCUSENERGY], ok),
        # A Choice lock is on Transform, which the new moves lack, so Showdown
        # drops it at the next choice.
        choice_slot=set_at(state.choice_slot, side, -1, ok),
    ), ok


def _eff_transform(data, state, user, target, word):
    return transform_into(data, state, user)[0]


def _eff_sparklingaria(data, state, user, target, word):
    """Cures the target's burn.

    It is the move's secondary effect, so Shield Dust, Covert Cloak and the
    user's Sheer Force all stop it.
    """
    ti = act(state, target)
    shielded = (slot_get(state.ability, target, ti) == A.SHIELDDUST) | \
        (slot_get(state.item, target, ti) == I.COVERTCLOAK) | \
        (slot_get(state.ability, user, act(state, user)) == A.SHEERFORCE)
    burned = slot_get(state.status, target, ti) == C.BRN
    return cure_status(state, target, ti, when=burned & jnp.logical_not(shielded))


def _lay_hazard(state, user, target, condition):
    """Ceaseless Edge and Stone Axe: a hazard on the target's side after the hit,
    if the user is still standing and Sheer Force has not eaten the effect."""
    ui = act(state, user)
    ok = (slot_get(state.hp, user, ui) > 0) & \
        (slot_get(state.ability, user, ui) != A.SHEERFORCE)
    return _set_side_condition(state, target, jnp.int32(condition), ok)


def _eff_ceaselessedge(data, state, user, target, word):
    return _lay_hazard(state, user, target, C.SC_SPIKES)


def _eff_stoneaxe(data, state, user, target, word):
    return _lay_hazard(state, user, target, C.SC_STEALTHROCK)


def _eff_takeheart(data, state, user, target, word):
    """+1 Sp. Atk and Sp. Def, and cures the user's status."""
    state = cure_status(state, user, act(state, user))
    state, _ = apply_boosts(state, user, boost_delta((C.B_SPA, 1), (C.B_SPD, 1)))
    return state


def _eff_clangoroussoul(data, state, user, target, word):
    """Pays a third of max HP for its +1 to every stat (the boosts are declarative).

    `directDamage(maxhp * 33 / 100)`, truncated; the move already failed if the
    user could not afford it.
    """
    ui = act(state, user)
    cost = jnp.maximum(floordiv(slot_get(state.maxhp, user, ui).astype(jnp.int32) * 33, 100), 1)
    return damage_pokemon(state, user, ui, cost)[0]


def _eff_wish(data, state, user, target, word):
    """Heals whoever holds the slot at the end of next turn by half the wisher's
    max HP. A slot takes one Wish at a time."""
    ui = act(state, user)
    ok = state.wish_turns[user] == 0
    half = floordiv(slot_get(state.maxhp, user, ui).astype(jnp.int32), 2)
    return state._replace(wish_turns=set_at(state.wish_turns, user, 2, ok),
                          wish_hp=set_at(state.wish_hp, user, half, ok))


def _eff_matblock(data, state, user, target, word):
    """Shields the user's side from damaging moves for the rest of the turn."""
    return set_volatile(state, user, C.V_MATBLOCK, 1)


def _last_move_slot(state, side):
    """`(slot, found)`: which of `side`'s moves it used last."""
    moves = slot_get(state.moves, side, act(state, side))
    hit = (moves == state.last_move[side]) & (state.last_move[side] >= 0)
    return jnp.argmax(hit).astype(jnp.int8), jnp.any(hit)


def disable_move(data, state, side, when, will_not_move):
    """Disable `side`'s last move (the Disable move, Cursed Body).

    Showdown's condition lasts 5 residuals, one fewer if the Pokemon is yet to
    move this turn -- or is moving right now, as when Cursed Body disables the
    move that hit it. It fails on a move with no PP left; Aroma Veil is the
    caller's to check, since Mold Breaker gets past it.
    """
    i = act(state, side)
    slot, found = _last_move_slot(state, side)
    pp = get_indexed(slot_get(state.pp, side, i), slot)
    ok = when & found & (pp > 0) & (state.volatiles[side, C.V_DISABLE] <= 0)
    turns = jnp.where(will_not_move, 5, 4)
    return state._replace(
        volatiles=set_at(state.volatiles, (side, C.V_DISABLE), turns, ok),
        disabled_slot=set_at(state.disabled_slot, side, slot, ok))


def _will_not_move(state, side):
    """True if `side` has no move still to come this turn (`!queue.willMove`)."""
    return state.moved_this_turn[side] | state.switched_this_turn[side]


def _eff_disable(data, state, user, target, word):
    ti = act(state, target)
    veiled = (slot_get(state.ability, target, ti) == A.AROMAVEIL) & \
        jnp.logical_not(_mold_breaker(data, state, user))
    return disable_move(data, state, target, jnp.logical_not(veiled),
                        _will_not_move(state, target))


def _eff_encore(data, state, user, target, word):
    """Locks the target into its last move for 3 turns (4 if it already moved).

    Fails on a move with no PP left, one flagged `failencore`, a target already
    encored, and through Aroma Veil.
    """
    ti = act(state, target)
    slot, found = _last_move_slot(state, target)
    last = jnp.maximum(state.last_move[target], 0)
    pp = get_indexed(slot_get(state.pp, target, ti), slot)
    veiled = (slot_get(state.ability, target, ti) == A.AROMAVEIL) & \
        jnp.logical_not(_mold_breaker(data, state, user))
    ok = found & (pp > 0) & jnp.logical_not(has_flag(data["move_flags"][last], "failencore")) & \
        (state.volatiles[target, C.V_ENCORE] <= 0) & jnp.logical_not(veiled)
    turns = jnp.where(_will_not_move(state, target), 4, 3)
    return state._replace(
        volatiles=set_at(state.volatiles, (target, C.V_ENCORE), turns, ok),
        encore_slot=set_at(state.encore_slot, target, slot, ok))


# Handlers we have not implemented yet map to `_noop`; coverage.py lists them.
EFFECT_FNS = {
    "none": _noop,
    "substitute": _eff_substitute, "protect": _eff_protect,
    "protectvariant": _eff_protect, "rest": _eff_rest, "haze": _eff_haze,
    "leechseed": _eff_leechseed, "painsplit": _eff_painsplit, "defog": _eff_defog,
    "rapidspin": _eff_rapidspin, "mortalspin": _eff_mortalspin, "trick": _eff_trick,
    "knockoff": _eff_knockoff, "bellydrum": _eff_bellydrum, "roost": _eff_roost,
    "sunnyday_heal": _eff_weather_heal, "shoreup": _eff_shoreup,
    "strengthsap": _eff_strengthsap, "curse": _eff_curse,
    "clearsmog": _eff_clearsmog, "topsyturvy": _eff_topsyturvy,
    "spectralthief": _eff_spectralthief, "refresh": _eff_refresh,
    "aromatherapy": _eff_aromatherapy,
    "filletaway": _eff_filletaway, "noretreat": _eff_noretreat,
    "tidyup": _eff_tidyup, "courtchange": _eff_courtchange,
    "saltcure": _eff_saltcure, "glaiverush": _eff_glaiverush,
    "trickroom": _eff_trickroom,
    "smackdown": _eff_smackdown, "thousandarrows": _eff_smackdown,
    "screenbreak": _eff_screenbreak,
    "icespinner": _eff_icespinner, "partingshot": _eff_partingshot,
    "chillyreception": _eff_chillyreception, "healingwish": _eff_healingwish,
    "revivalblessing": _eff_revivalblessing, "shedtail": _eff_shedtail,
    "burnup": _eff_burnup, "doubleshock": _eff_doubleshock,
    "fakeout": _eff_fakeout, "auroraveil": _eff_auroraveil,
    "encore": _eff_encore, "disable": _eff_disable,
    "transform": _eff_transform, "sparklingaria": _eff_sparklingaria,
    "ceaselessedge": _eff_ceaselessedge, "stoneaxe": _eff_stoneaxe,
    "takeheart": _eff_takeheart, "clangoroussoul": _eff_clangoroussoul,
    "wish": _eff_wish, "matblock": _eff_matblock,
}

UNIMPLEMENTED_EFFECTS = tuple(h for h in E.EFFECT_HANDLERS if h not in EFFECT_FNS)

#: Handlers that only act if the move reached its target: past Protect,
#: accuracy, immunities -- and, for a damaging move, past any Substitute (Knock
#: Off does not knock anything off a Substitute).
EFFECT_ON_HIT = frozenset({
    "defog",                      # all of it in `onHit`: Protect stops the lot
    "knockoff", "trick", "painsplit", "leechseed", "clearsmog", "spectralthief",
    "partingshot", "saltcure", "smackdown",
    "thousandarrows", "strengthsap", "topsyturvy",
    "burnup", "doubleshock", "sparklingaria", "transform", "encore", "disable",
})
#: Handlers that act once a damaging move connects, even into a Substitute:
#: Showdown runs them from `onTryHit` or `onAfterSubDamage` as well.
EFFECT_ON_CONNECT = frozenset({"ceaselessedge", "stoneaxe", "screenbreak",
                               "glaiverush",    # a `self` effect: needs a hit
                               # hazards / terrain cleared from onAfterSubDamage too
                               "rapidspin", "mortalspin", "icespinner"})

#: Every BattleState field an effect handler can write. Keep in sync with
#: EFFECT_FNS -- a handler writing anything outside this list would have its
#: change silently dropped, which `test_effect_handlers_only_write_declared_fields`
#: guards against.
EFFECT_WRITES = (
    "hp", "status", "status_turns", "sleep_attempts", "rest_sleep", "boosts",
    "volatiles", "sub_hp",
    "side_conditions", "item", "types", "terrain", "terrain_turns",
    "weather", "weather_turns", "force_switch", "protect_streak", "trick_room",
    "species", "stats", "ability", "moves", "pp", "maxpp", "transformed",
    "tf_moves", "tf_pp", "tf_maxpp", "wish_turns", "wish_hp", "healing_wish", "encore_slot",
    "pass_mode", "last_faint", "stats_raised", "stats_lowered", "choice_slot",
    "disabled_slot", "last_item", "cud_berry", "cud_turns",
)

def _gate_of(name):
    return "hit" if name in EFFECT_ON_HIT else (
        "connected" if name in EFFECT_ON_CONNECT else "when")


# (handler ids, gate, handler) for the implemented ones; any other id is a
# no-op. Handlers that share a function and a gate (Protect and its variants,
# Smack Down and Thousand Arrows) are traced once.
_EFFECT_GROUPS = {}
for _h in E.EFFECT_HANDLERS:
    if _h in EFFECT_FNS:
        _EFFECT_GROUPS.setdefault((EFFECT_FNS[_h], _gate_of(_h)), []).append(
            E.EFFECT_HANDLERS.index(_h))
_EFFECT_HANDLERS_BY_ID = tuple((tuple(ids), gate, fn)
                               for (fn, gate), ids in _EFFECT_GROUPS.items())


def run_effect(effect_id, data, state, user, target, word, when=True, connected=True,
               hit=True):
    """Dispatch to a move's special-effect handler.

    `word` is the handler's random word. `when` gates the whole call (e.g. on
    whether the user could act at all); handlers in `EFFECT_ON_HIT` and
    `EFFECT_ON_CONNECT` are gated on `hit` and `connected` instead.

    Not a `lax.switch`. Under vmap a switch evaluates every branch and then
    selects each output across all of them: 15 fields through a 43-way select
    is a tree of some 1,300 operations, nearly all choosing between copies of a
    field the handler never touched. Every handler is evaluated here too, but
    each field selects only among the handlers that write it -- a handler that
    leaves a field alone hands back the very same array, which is checked at
    trace time.
    """
    effect_id = effect_id.astype(jnp.int32)
    fields = {f: getattr(state, f) for f in EFFECT_WRITES}
    gates = {"hit": hit, "connected": connected, "when": when}
    for ids, gate, fn in _EFFECT_HANDLERS_BY_ID:
        out = fn(data, state, user, target, word)
        picked = functools.reduce(jnp.logical_or, [effect_id == i for i in ids])
        chosen = gates[gate] & picked
        for f in EFFECT_WRITES:
            new = getattr(out, f)
            if new is not getattr(state, f):
                fields[f] = jnp.where(chosen, new, fields[f])
    return state._replace(**fields)


# Behaviours that are modelled outside `EFFECT_FNS`, so their presence in
# `UNIMPLEMENTED_EFFECTS` would be misleading.
HANDLED_ELSEWHERE = frozenset({
    "freezedry", "flyingpress",   # damage.type_effectiveness
    "photongeyser", "shellsidearm",   # resolve_move_ctx (category switch)
    "taunt", "yawn", "perishsong", "destinybond",
    "poltergeist", "suckerpunch",  # their fail checks live in execute_move
    "grassyglide",                # move_priority
    # execute_move: fail checks, and moves that replace or defer the hit
    "teleport", "hyperspacefury", "magnetrise", "focuspunch", "chargeboost",
    "sleeptalk", "futuresight", "beakblast", "relicsong", "psychoshift",
    "bugbite", "stuffcheeks",     # berries and statuses: one call per side
    "batonpass",                  # switch_to, keyed on the move's `pass_mode`
    "syrupbomb",                  # a secondary volatile with its own duration
    # Pollen Puff only does something different to an ally, and singles has none.
    "pollenpuff",
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
        sub_hp=set_at(state.sub_hp, target, new_sub),
        volatiles=set_at(state.volatiles, (target, C.V_SUBSTITUTE), 0,
                         when=has_sub & (new_sub <= 0)))
    direct = jnp.where(has_sub, 0, amount)
    state, dealt = damage_pokemon(state, target, ti, direct)
    return state, dealt


def _apply_secondaries(data, state, user, target, move_id, words, landed=True,
                       target_ability=None, connected=None, target_grounded=None,
                       target_raised=None):
    """Roll each of the move's up-to-two secondary effects.

    `words` is the move's random words; each secondary reads two from
    `W_SECONDARY`. `landed` means the move reached the Pokemon itself -- a
    Substitute soaks secondary effects. It gates the whole thing rather than the
    caller wrapping this call in `lax.cond`: folding it into each `fires` mask
    means `set_status` and `apply_boosts` run unconditionally and simply no-op,
    instead of the whole function being traced twice (once per `cond` branch)
    and selected over.

    Sheer Force deletes the secondaries outright. Shield Dust and Covert Cloak
    keep only the parts aimed at the user (Showdown filters to `effect.self`).

    Returns `(state, status, word, drops)`: a status a secondary wants to
    inflict, the random word for its sleep length, and the stat changes it
    makes to the target. The caller folds the status into the one `set_status`
    call a move makes on its target and the drops into its one `apply_boosts`,
    which is also what lets Defiant answer a secondary's drop.
    """
    ti = act(state, target)
    ui = act(state, user)
    t_ab = slot_get(state.ability, target, ti) if target_ability is None else target_ability
    shielded = (t_ab == A.SHIELDDUST) | (slot_get(state.item, target, ti) == I.COVERTCLOAK)
    u_ab = slot_get(state.ability, user, ui)
    serene = u_ab == A.SERENEGRACE
    sheer = (u_ab == A.SHEERFORCE) & data["move_has_secondary"][move_id]
    # The user's own secondary boosts (Trailblaze's Speed) come with a hit on a
    # Substitute too; the target's effects need the Pokemon itself.
    connected = landed if connected is None else connected
    connected = connected & jnp.logical_not(sheer)
    landed = landed & jnp.logical_not(sheer)

    sec_status, sec_word = jnp.int8(C.STATUS_NONE), jnp.uint32(0)
    drops = jnp.zeros(C.NUM_BOOSTS, jnp.int32)
    # Spirit Shackle's and Anchor Shot's trap does not hold a Ghost type.
    target_ghost = has_type(state, target, C.GHOST)
    # Burning Jealousy and Alluring Voice only work on a target whose stats
    # rose this turn.
    raised = state.stats_raised[target] if target_raised is None else target_raised
    envious = jnp.logical_not(data["move_sec_if_raised"][move_id]) | raised
    for k in range(2):
        chance = data["move_sec_chance"][move_id, k].astype(jnp.int32)
        chance = jnp.where(serene, jnp.minimum(chance * 2, 100), chance)
        roll_word = words[W_SECONDARY + 2 * k]
        sleep_word = words[W_SECONDARY + 2 * k + 1]
        rolled = landed & (chance > 0) & (below(roll_word, 100) < chance)
        fires = rolled & jnp.logical_not(shielded) & envious

        status = data["move_sec_status"][move_id, k]
        if k == 0:
            # Tri Attack and Dire Claw pick theirs from three at random, with the
            # roll word a second secondary would have used (they have none).
            picks = data["move_sec_status_pick"][move_id]
            pick = picks[below(words[W_SECONDARY + 2], 3)]
            status = jnp.where(picks[0] > 0, pick, status)
        takes = fires & (status != C.STATUS_NONE) & (sec_status == C.STATUS_NONE)
        sec_status = jnp.where(takes, status, sec_status)
        sec_word = jnp.where(takes, sleep_word, sec_word)

        vol = data["move_sec_volatile"][move_id, k]
        # Inner Focus cannot be made to flinch; confusion rolls its own length
        # (and Own Tempo refuses it); Syrup Bomb lasts four residuals.
        flinch_blocked = ((vol == C.V_FLINCH) & (t_ab == A.INNERFOCUS)) | \
            ((vol == C.V_TRAPPED) & target_ghost)
        confusing = vol == C.V_CONFUSION
        state = confuse(state, target, sleep_word, fires & confusing, target_grounded)
        # Throat Chop and Psychic Noise's Heal Block last two turns and are not
        # renewed by a second hit.
        two_turns = (vol == C.V_THROATCHOP) | (vol == C.V_HEALBLOCK)
        renewing = (two_turns & (state.volatiles[target, jnp.maximum(vol, 0)] > 0)) | \
            ((vol == C.V_HEALBLOCK) & (t_ab == A.AROMAVEIL))
        state = state._replace(volatiles=set_at(
            state.volatiles, (target, vol),
            jnp.where(vol == C.V_SYRUPBOMB, 4, jnp.where(two_turns, 2, 1)),
            when=fires & (vol >= 0) & jnp.logical_not(flinch_blocked | confusing | renewing)))

        tgt_boosts = data["move_sec_boosts"][move_id, k].astype(jnp.int32)
        drops = drops + jnp.where(fires, tgt_boosts, 0)
        self_boosts = data["move_sec_self_boosts"][move_id, k].astype(jnp.int32)
        self_rolled = connected & (chance > 0) & (below(roll_word, 100) < chance)
        state, _ = apply_boosts(state, user, jnp.where(self_rolled, self_boosts, 0))
    return state, sec_status, sec_word, drops


def _at_foe(data, move_id):
    """Does the move target the opposing Pokemon (rather than itself or a side)?"""
    tgt = data["move_target"][move_id]
    return jnp.any(jnp.stack([tgt == t for t in C.FOE_TARGETS]))


def _beat_up_powers(data, state, user):
    """Beat Up's per-hit base powers and hit count.

    One hit for the user and one for each other party member that is neither
    fainted nor statused, each at 5 + the base Attack / 10 of the species its
    set names (not a forme it has since taken, nor a Transform), in Showdown's
    party order -- which puts the user, being active, first.
    """
    ui = act(state, user)
    slots = jnp.arange(C.TEAM_SIZE)
    joins = ((state.hp[user] > 0) & (state.status[user] == C.STATUS_NONE)) | (slots == ui)
    key = state.party_pos[user]
    rank = jnp.sum(joins[None, :] & (key[None, :] < key[:, None]), axis=1)
    base = jnp.where(state.base_species[user] >= 0, state.base_species[user], state.species[user])
    power = 5 + floordiv(data["species_base_stats"][data["species_set"][base], C.ATK]
                         .astype(jnp.int32), 10)
    hits = jnp.arange(MAX_HITS)
    per_hit = jnp.sum(jnp.where(joins[None, :] & (rank[None, :] == hits[:, None]),
                                power[None, :], 0), axis=1)
    return per_hit, jnp.sum(joins).astype(jnp.int32)


def execute_move(data, state, user, move_slot, moves_first, words,
                 target_attacking=True, target_acting=True, struggling=None):
    """Run one move from `user`'s active Pokemon. Returns the updated state.

    `words` is a `[MOVE_WORDS]` uint32 vector of random words, one per random
    decision the move can make (the `W_*` indices). `target_attacking` says
    whether the opponent is about to use a damaging move this turn; Sucker Punch
    and friends fail when it is false.
    """
    target = 1 - user
    ui = act(state, user)
    ti = act(state, target)
    fx = lambda effect, name: effect == E.EFFECT_HANDLERS.index(name)
    # A Choice item locks in the move used while holding it -- not one that
    # Tricked it into the user's hands.
    held_at_start = slot_get(state.item, user, ui)
    t_held_at_start = slot_get(state.item, target, ti)
    # Burning Jealousy and Alluring Voice ask about rises before this move:
    # Showdown runs the secondaries ahead of DamagingHit (Stamina, Weak Armor).
    t_raised_at_start = state.stats_raised[target]
    # The field as the move found it, to tell a field move that did nothing
    # (Sunny Day in the sun) -- a failure, for Stomping Tantrum.
    field_before = (state.weather, state.terrain, state.side_conditions)
    alive_before = actives_alive(state)
    # Who the opponent sees using the move: an Illusion broken by an earlier
    # hit this turn is already gone, and a Transform the move makes is not yet.
    shown_at_start = shown_species(state, user, ui)

    # A move the user is locked into (the second turn of Solar Beam), or encored
    # into, replaces whatever was chosen. The action mask normally forces it
    # already; this covers an Encore landing mid-turn, before the target moves.
    move_slot = move_slot.astype(jnp.int32)
    encored = (state.volatiles[user, C.V_ENCORE] > 0) & (state.encore_slot[user] >= 0)
    move_slot = jnp.where(encored, state.encore_slot[user].astype(jnp.int32), move_slot)
    locked = state.locked_slot[user] >= 0
    move_slot = jnp.where(locked, state.locked_slot[user].astype(jnp.int32), move_slot)
    # With no move it can use, the Pokemon struggles: a typeless 50 BP hit that
    # spends no PP and costs a quarter of its max HP. Showdown decides that when
    # the move is chosen, so `run_turn` passes in what was usable then -- a move
    # Disabled mid-turn fails rather than turning into Struggle.
    if struggling is None:
        struggling = jnp.logical_not(jnp.any(usable_moves(data, state, user)))
    struggling = struggling & jnp.logical_not(locked)
    chosen_id = jnp.where(struggling, move_index("struggle"),
                          jnp.maximum(state.moves[user, ui, move_slot], 0)).astype(jnp.int32)

    state, can_act = before_move(data, state, user, chosen_id, words, move_slot, struggling)
    has_pp = state.pp[user, ui, move_slot] > 0
    # The second turn of a charge move spends no PP and needs none.
    can_act = can_act & (has_pp | locked | struggling) & \
        ((state.moves[user, ui, move_slot] >= 0) | struggling)
    # Focus Punch loses its focus if the user was hit first.
    lost_focus = fx(data["move_effect_cb"][chosen_id], "focuspunch") & \
        (state.volatiles[user, C.V_FOCUSPUNCH] >= 2)
    can_act = can_act & jnp.logical_not(lost_focus)

    # Destiny Bond holds until its user next tries to move: any other move, or
    # not managing to move at all, ends it -- and a second Destiny Bond in a
    # row fails, removing it (Showdown's onPrepareHit).
    bonded = state.volatiles[user, C.V_DESTINYBOND] > 0
    rebond = bonded & (chosen_id == move_index("destinybond")) & can_act
    state = state._replace(volatiles=set_at(state.volatiles, (user, C.V_DESTINYBOND), 0,
                                            bonded))

    # Sleep Talk uses one of the user's other moves at random, still asleep.
    u_ab = slot_get(state.ability, user, ui)
    my_moves = slot_get(state.moves, user, ui)
    my_flags = data["move_flags"][jnp.maximum(my_moves, 0)]
    callable_ = (my_moves >= 0) & jnp.logical_not(has_flag(my_flags, "nosleeptalk")) & \
        jnp.logical_not(has_flag(my_flags, "charge"))
    n_callable = jnp.sum(callable_)
    nth = jnp.floor(uniform(words[W_CALL]) * n_callable.astype(jnp.float32)).astype(jnp.int32)
    pick = jnp.argmax(callable_ & (jnp.cumsum(callable_) - 1 == nth))
    asleep = (slot_get(state.status, user, ui) == C.SLP) | (u_ab == A.COMATOSE)
    calls = fx(data["move_effect_cb"][chosen_id], "sleeptalk") & can_act & asleep & \
        (n_callable > 0)
    move_id = jnp.where(calls, get_indexed(my_moves, pick), chosen_id).astype(jnp.int32)
    effect = data["move_effect_cb"][move_id]

    # Spend PP. Pressure charges one more for a move whose targets include it:
    # anything aimed at the foe or at the whole field, but not a move aimed at
    # the user, its own side, or the foe's side (bar the hazards, which carry
    # `mustpressure`) -- and nothing when the foe has fainted and the move has
    # no target at all. Curse aims at the user unless the user is a Ghost. A
    # move Sleep Talk calls charges its extra PP to Sleep Talk.
    self_curse = (move_id == move_index("curse")) & \
        jnp.logical_not(jnp.any(active_types(state, user) == C.GHOST))
    pressured = (_at_foe(data, move_id) & jnp.logical_not(self_curse)) | \
        (data["move_target"][move_id] == C.TGT_ALL) | data["move_must_pressure"][move_id]
    cost = jnp.where((slot_get(state.ability, target, ti) == A.PRESSURE) & pressured &
                     (slot_get(state.hp, target, ti) > 0), 2, 1)
    new_pp = jnp.maximum(state.pp[user, ui, move_slot].astype(jnp.int32) -
                         jnp.where(can_act & jnp.logical_not(locked | struggling), cost, 0), 0)
    state = state._replace(pp=set_at(state.pp, (user, ui, move_slot), new_pp))

    # Charge moves spend their first turn charging -- unless sun (Solar Beam),
    # rain (Electro Shot) or a Power Herb lets them fire at once. Meteor Beam
    # and Electro Shot raise Sp. Atk as they start, whichever way it goes.
    u_item = slot_get(state.item, user, ui)
    second_turn = state.volatiles[user, C.V_TWOTURN] > 0
    w = effective_weather(state)
    sunny = (w == C.SUN) | (w == C.HARSH_SUN)
    rainy = (w == C.RAIN) | (w == C.HEAVY_RAIN)
    solar = data["move_bp_modify"][move_id] == E.BP_MODIFY_HANDLERS.index("solarbeam")
    boosts_first = fx(effect, "chargeboost")
    instant = (solar & sunny) | (boosts_first & (data["move_type"][move_id] == C.ELECTRIC) & rainy)
    starting = can_act & data["move_is_charge"][move_id] & jnp.logical_not(second_turn)
    herb = starting & jnp.logical_not(instant) & (u_item == I.POWERHERB)
    charging = starting & jnp.logical_not(instant | herb)
    state, _ = apply_boosts(state, user, jnp.where(starting & boosts_first,
                                                   boost_delta((C.B_SPA, 1)), 0))
    state = lose_item(state, user, ui, herb)
    # The charge is spent on the second turn whether or not the user can act.
    state = state._replace(
        volatiles=set_at(state.volatiles, (user, C.V_TWOTURN), jnp.where(charging, 1, 0),
                         charging | second_turn),
        locked_slot=set_at(state.locked_slot, user, jnp.where(charging, move_slot, -1),
                           charging | second_turn))
    attacks = can_act & jnp.logical_not(charging)

    cb_ctx = build_cb_ctx(data, state, user, target, moves_first,
                          state.pp[user, ui, move_slot],
                          fickle=below(words[W_FICKLE], 10) < 3,
                          # (Fickle Beam and Shell Side Arm never share a move.)
                          coin=below(words[W_FICKLE], 2) == 0)
    grounded_of = lambda side: jnp.where(side == user, cb_ctx.grounded_user,
                                         cb_ctx.grounded_target)
    mv = resolve_move_ctx(data, move_id, cb_ctx)
    is_status = mv.category == C.CAT_STATUS

    # Protean / Libero: the user becomes the move's type before using it, once
    # per entry. Not while Terastallized, not for Future Sight, and not for a
    # move with nobody to aim at (U-turn after the foe fainted), which fails
    # before Showdown's PrepareHit.
    my_types = slot_get(state.types, user, ui)
    has_target = (slot_get(state.hp, target, ti) > 0) | \
        jnp.logical_not(_at_foe(data, move_id) & jnp.logical_not(self_curse))
    protean = can_act & has_target & ((u_ab == A.PROTEAN) | (u_ab == A.LIBERO)) & \
        jnp.logical_not(state.type_changed[user]) & \
        jnp.logical_not(slot_get(state.terastallized, user, ui)) & \
        jnp.logical_not(has_flag(mv.flags, "futuremove")) & \
        jnp.logical_not((my_types[0] == mv.type) & (my_types[1] == C.TYPE_NONE))
    state = state._replace(
        types=slot_set(state.types, user, ui,
                       jnp.stack([mv.type, jnp.int8(C.TYPE_NONE)]).astype(jnp.int8), protean),
        type_changed=set_at(state.type_changed, user, True, protean))
    atk = build_attacker(state, user, data)

    # Mold Breaker (and moves such as Sunsteel Strike, and Mycelium Might's
    # status moves) ignore the target's ability for the move, if it is breakable.
    t_ab_raw = slot_get(state.ability, target, ti)
    breaks = data["ability_mold_breaker"][u_ab] | data["move_ignore_ability"][move_id] | \
        ((u_ab == A.MYCELIUMMIGHT) & is_status)
    t_ab = jnp.where(breaks & data["ability_breakable"][t_ab_raw], 0, t_ab_raw).astype(
        t_ab_raw.dtype)
    dfn = build_defender(data, state, target, ability=t_ab)

    # --- can the move connect at all? ---
    target_alive = slot_get(state.hp, target, ti) > 0
    # A non-Ghost Curse aims at its user: Good as Gold, Protect and Magic
    # Bounce have nothing to say about it.
    at_foe = _at_foe(data, move_id) & jnp.logical_not(self_curse)
    blocked = move_blocked_by_protect(data, state, user, target, move_id) & \
        jnp.logical_not(self_curse)
    # Only a move aimed at the foe needs one standing there: Swords Dance, a
    # screen, Stealth Rock or Trick Room works after the foe has fainted.
    reached = attacks & (target_alive | jnp.logical_not(at_foe)) & jnp.logical_not(blocked)
    powder = powder_immune(data, state, target, move_id, t_ab)
    flagged = flag_immune(data, t_ab, move_id, at_foe)
    priority = move_priority(data, state, user, move_id)
    terrain_block = terrain_blocks(data, state, target, move_id, priority) & at_foe

    acc_roll = below(words[W_ACCURACY], 100)
    weather_acc = cb.modify_accuracy(data["move_acc_cb"][move_id], cb_ctx)
    hits_acc = accuracy_check(data, mv, atk, dfn, state.boosts[user, C.B_ACC],
                              state.boosts[target, C.B_EVA], acc_roll, state.gravity,
                              weather_acc)
    # Since Gen 8 a Poison type's Toxic cannot miss; nothing misses a Pokemon
    # that used Glaive Rush.
    hits_acc = hits_acc | ((move_id == move_index("toxic")) &
                           jnp.any(active_types(state, user) == C.POISON)) | \
        (state.volatiles[target, C.V_GLAIVERUSH] > 0)

    # Moves that simply fail under a condition the generic path cannot express.
    # Protect and friends fail when nothing is left to act after them; Shed
    # Tail without more than half its HP, a Substitute already up, or someone
    # to pass to; Healing Wish without someone to switch to.
    species = slot_get(state.species, user, ui)
    hp_now = slot_get(state.hp, user, ui).astype(jnp.int32)
    maxhp_now = slot_get(state.maxhp, user, ui).astype(jnp.int32)
    bench = jnp.any((state.hp[user] > 0) & (jnp.arange(C.TEAM_SIZE) != ui))
    shed_ok = (hp_now > maxhp_now - floordiv(maxhp_now, 2)) & bench & \
        (state.volatiles[user, C.V_SUBSTITUTE] <= 0)
    grounded_by = (state.volatiles[user, C.V_SMACKDOWN] > 0) | \
        (state.volatiles[user, C.V_INGRAIN] > 0) | (state.gravity > 0)
    damp = (data["move_selfdestruct"][move_id] == 1) & ((u_ab == A.DAMP) | (t_ab == A.DAMP))
    # The primal weathers stop the opposing type's attacks before they start.
    washed_out = jnp.logical_not(is_status) & (
        ((w == C.HARSH_SUN) & (mv.type == C.WATER)) | ((w == C.HEAVY_RAIN) & (mv.type == C.FIRE)))
    future = fx(effect, "futuresight")
    fails = (fx(effect, "suckerpunch") & jnp.logical_not(target_attacking & moves_first)) | \
        (fx(effect, "fakeout") & (state.moves_since_switch[user] > 0)) | \
        (fx(effect, "hyperspacefury") & (species != species_index("hoopaunbound"))) | \
        (fx(effect, "magnetrise") & grounded_by) | \
        (fx(effect, "clangoroussoul") & ((hp_now * 100 <= maxhp_now * 33) | (maxhp_now == 1))) | \
        (fx(effect, "stuffcheeks") & jnp.logical_not(data["item_is_berry"][u_item])) | \
        (fx(effect, "matblock") & ((state.moves_since_switch[user] > 0) |
                                   jnp.logical_not(moves_first))) | \
        (future & (state.future_turns[target] > 0)) | damp | washed_out | \
        (fx(effect, "noretreat") & (state.volatiles[user, C.V_NORETREAT] > 0)) | \
        (data["move_stalling"][move_id] & jnp.logical_not(target_acting)) | \
        (fx(effect, "shedtail") & jnp.logical_not(shed_ok)) | \
        (fx(effect, "healingwish") & jnp.logical_not(bench)) | \
        (fx(effect, "filletaway") & (hp_now * 2 <= maxhp_now)) | \
        (fx(effect, "revivalblessing") & jnp.logical_not(jnp.any(state.hp[user] <= 0))) | \
        (fx(effect, "poltergeist") & (slot_get(state.item, target, ti) == 0)) | \
        (fx(effect, "auroraveil") & (w != C.SNOW)) | \
        (fx(effect, "doubleshock") & jnp.logical_not(has_type(state, user, C.ELECTRIC))) | \
        (fx(effect, "burnup") & jnp.logical_not(has_type(state, user, C.FIRE))) | rebond | \
        (cb.fixed_damage(data["move_dmg_cb"][move_id], cb_ctx) == -2)   # Counter / Endeavor
    # Queenly Majesty stops priority moves aimed at its holder; Prankster's
    # boosted status moves fail against Dark types; Good as Gold is immune to
    # the opponent's status moves.
    fails = fails | ((t_ab == A.QUEENLYMAJESTY) & (priority > 0) & at_foe) | \
        ((u_ab == A.PRANKSTER) & is_status & at_foe & jnp.any(dfn.types == C.DARK)) | \
        ((t_ab == A.GOODASGOLD) & is_status & at_foe)

    # On the ground as this move sees it: Mold Breaker gets past Levitate, but
    # not past wings, an Air Balloon or Magnet Rise.
    lifted = jnp.any(dfn.types == C.FLYING) | (dfn.item == I.AIRBALLOON) | \
        (state.volatiles[target, C.V_MAGNETRISE] > 0)
    move_grounded = cb_ctx.grounded_target | (
        (t_ab_raw == A.LEVITATE) & (t_ab != A.LEVITATE) & jnp.logical_not(lifted))
    type_exp, type_immune = type_effectiveness(
        data, mv.type, dfn.types, mv, data["move_ignore_immunity"][move_id],
        t_ab, (u_ab == A.SCRAPPY) | (u_ab == A.MINDSEYE),
        def_terastallized=dfn.terastallized, def_grounded=move_grounded,
        def_full_hp=dfn.hp >= dfn.maxhp)
    # Struggle is typeless: it hits everything neutrally.
    type_exp = jnp.where(struggling, 0, type_exp)
    type_immune = type_immune & jnp.logical_not(struggling)

    # Collision Course and friends key off the effectiveness, so the base-power
    # callback is resolved now that it is known. Resolved once rather than per
    # hit: only the hit-number scaling varies, and the switch is costly to trace.
    cb_ctx = cb_ctx._replace(type_exp=type_exp)
    bp_cb_mod = cb.base_power_modify(data["move_bp_modify"][move_id], cb_ctx)

    # Absorbing abilities take the hit instead: Water Absorb heals, Sap Sipper
    # boosts, Wind Rider eats wind moves for +1 Attack. They answer before
    # accuracy is checked, but only to a move that got past Protect.
    absorbs, absorb_heal, (absorb_stat, absorb_amt) = ability_absorbs(
        data, t_ab, mv.type, mv.category)
    wind = (t_ab == A.WINDRIDER) & has_flag(mv.flags, "wind") & at_foe
    absorbs = ((absorbs & at_foe) | wind) & reached & jnp.logical_not(fails)
    absorb_stat = jnp.where(wind, C.B_ATK, absorb_stat)
    absorb_amt = jnp.where(wind, 1, absorb_amt)

    # A Substitute stops status moves aimed through it, unless they bypass it.
    bypass_sub = has_flag(mv.flags, "bypasssub") | (u_ab == A.INFILTRATOR)
    had_sub = (state.volatiles[target, C.V_SUBSTITUTE] > 0) & jnp.logical_not(bypass_sub)

    # Magic Bounce reflects a reflectable status move back at its user.
    # (Only from a Pokemon still standing: Spikes aimed at a side whose
    # Hatterene just fell to them is not bounced back.)
    bounces = (t_ab == A.MAGICBOUNCE) & has_flag(mv.flags, "reflectable") & reached & \
        jnp.logical_not(fails) & target_alive

    connects = (reached & hits_acc & jnp.logical_not(powder) & jnp.logical_not(flagged) &
                jnp.logical_not(terrain_block) & jnp.logical_not(type_immune) &
                jnp.logical_not(absorbs) & jnp.logical_not(fails) &
                jnp.logical_not(is_status & at_foe & had_sub) & jnp.logical_not(bounces))
    # The reflected move has to land on its user in turn: past powder immunity,
    # its Substitute, its own Good as Gold, and (Thunder Wave) Ground typing.
    u_types = active_types(state, user)
    bounce_lands = bounces & hits_acc & (slot_get(state.hp, user, ui) > 0) & \
        jnp.logical_not(powder_immune(data, state, user, move_id, u_ab)) & \
        jnp.logical_not((state.volatiles[user, C.V_SUBSTITUTE] > 0) & jnp.logical_not(bypass_sub)) & \
        (u_ab != A.GOODASGOLD) & jnp.logical_not(
            (mv.type == C.ELECTRIC) & (data["move_ignore_immunity"][move_id] == 0) &
            jnp.any(u_types == C.GROUND))

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
    state = state._replace(volatiles=set_at(state.volatiles, (target, C.V_FLASHFIRE), 1,
                                            absorbs & (t_ab == A.FLASHFIRE)))

    # Phase boundary: everything from here on reads whether the move connected.
    connects, bounce_lands, type_exp, bp_cb_mod, fails, attacks, is_status = barrier(
        connects, bounce_lands, type_exp, bp_cb_mod, fails, attacks, is_status)

    # --- Future Sight: the move is spent now and hits two turns later ---
    deferred = future & attacks & jnp.logical_not(fails)
    state = state._replace(
        future_turns=set_at(state.future_turns, target, 3, deferred),
        future_move=set_at(state.future_move, target, move_id, deferred),
        future_source=set_at(state.future_source, target, ui, deferred))
    connects = connects & jnp.logical_not(future)

    # --- damage ---
    fixed = cb.fixed_damage(data["move_dmg_cb"][move_id], cb_ctx)
    is_fixed = fixed >= 0
    ohko = data["move_ohko"][move_id]

    n_hits = _roll_hit_count(data, state, user, move_id, words[W_HIT_COUNT],
                             words[W_LOADED_DICE])
    beat_up = data["move_bp_replace"][move_id] == E.BP_REPLACE_HANDLERS.index("beatup")
    beat_up_powers, beat_up_hits = _beat_up_powers(data, state, user)
    n_hits = jnp.where(beat_up, beat_up_hits, n_hits)

    crit_stage = crit_chance_stage(data, mv, atk)
    crit_denom = CRIT_RATES[crit_stage]
    crit_immune = (dfn.ability == A.BATTLEARMOR) | (dfn.ability == A.SHELLARMOR)
    # Probability 1/crit_denom, expressed without a traced modulo: XLA lowers
    # integer division and remainder by a non-constant into a large sequence, and
    # one of those here was enough to stall vmap compilation.
    is_crit = (uniform(words[W_CRIT]) * crit_denom.astype(jnp.float32) < 1.0) & \
              jnp.logical_not(crit_immune) | data["move_will_crit"][move_id]
    is_crit = is_crit & jnp.logical_not(crit_immune)

    # Everything the damage formula reads that cannot change between hits is
    # computed once here. The loop itself carries only scalars -- running HP,
    # Substitute HP and the damage total -- rather than the whole BattleState.
    # Carrying the state meant a batched `while` whose body scattered into 47
    # arrays, and that is what made `vmap(execute_move)` take minutes to compile.
    hit_weather = effective_weather(state)
    hit_terrain = state.terrain
    # Brick Break, Psychic Fangs and Raging Bull shatter the screens in
    # onTryHit, before their own damage is worked out.
    hit_side_conditions = jnp.where(jnp.asarray(_SCREENS) & fx(effect, "screenbreak"), 0,
                                    state.side_conditions[target])
    hit_fainted_count = state.overlord_count[user].astype(jnp.int32)
    hit_analytic = jnp.logical_not(moves_first)
    hit_target_switched = state.switched_this_turn[target]
    target_maxhp = slot_get(state.maxhp, target, ti).astype(jnp.int32)
    can_endure = ((slot_get(state.item, target, ti) == I.FOCUSSASH) |
                  (dfn.ability == A.STURDY))
    start_hp = slot_get(state.hp, target, ti).astype(jnp.int32)
    start_sub = jnp.where(bypass_sub, 0, state.sub_hp[target].astype(jnp.int32))
    # A resist berry only halves the hit if it gets eaten: not behind a
    # Substitute, and not with Unnerve across the field.
    berry_ok = jnp.logical_not(had_sub) & jnp.logical_not(unnerved(state, target))
    # Disguise and Ice Face (physical hits only) take the first hit that reaches
    # the Pokemon for it.
    t_species = slot_get(state.species, target, ti)
    shield = jnp.logical_not(is_status) & (
        ((dfn.ability == A.DISGUISE) & (t_species == species_index("mimikyu"))) |
        ((dfn.ability == A.ICEFACE) & (t_species == species_index("eiscue")) &
         (mv.category == C.CAT_PHYSICAL)))

    # Damage per hit depends only on the roll and (for Triple Kick / Triple
    # Axel / Beat Up) the hit number -- never on the running HP. So the damage
    # for all possible hits is computed in one vectorised pass and only the HP
    # bookkeeping is sequential. Calling `calc_damage` inside a `while` loop
    # instead made `vmap(execute_move)` take minutes to compile.
    hit_index = jnp.arange(MAX_HITS)
    rolls = below(words[W_DAMAGE_ROLL:W_DAMAGE_ROLL + MAX_HITS], 16)

    # Struggle's missing type means no STAB for anyone.
    typeless = atk._replace(base_types=jnp.full(2, C.TYPE_NONE, jnp.int8),
                            terastallized=jnp.bool_(False))
    atk_hit = jax.tree.map(lambda a, b: jnp.where(struggling, a, b), typeless, atk)

    # Weak Armor, Stamina and Water Compaction change the target's Defense with
    # every hit that reaches it, and the later hits of the move meet the new
    # stage. Counted as if each earlier hit got through -- exact unless a
    # Substitute is in the way, when the stage is left alone.
    def_step = jnp.where(
        t_ab_raw == A.STAMINA, 1,
        jnp.where((t_ab_raw == A.WEAKARMOR) & (mv.category == C.CAT_PHYSICAL), -1,
                  jnp.where((t_ab_raw == A.WATERCOMPACTION) & (mv.type == C.WATER), 2, 0)))
    def_step = jnp.where(had_sub, 0, def_step)
    # Ice Face takes the first hit and breaks into Noice Eiscue before the
    # next, which meets Noice's Defense. (Behind a Substitute, left alone.)
    noice_def = species_stats(data, species_index("eiscuenoice"), slot_get(state.level, target, ti),
                              slot_get(state.spread_zero, target, ti))[C.DEF]
    ice_breaks = shield & (dfn.ability == A.ICEFACE) & jnp.logical_not(had_sub)
    # Seed Sower lays Grassy Terrain at the first hit, under the rest.
    sows = (t_ab_raw == A.SEEDSOWER) & jnp.logical_not(had_sub)

    # Which hits happen does not depend on their damage: Population Bomb and
    # the Triple kicks roll accuracy again for every hit after the first and
    # stop at a miss -- unless Loaded Dice or Skill Link.
    multi_acc = data["move_multiaccuracy"][move_id] & \
        (slot_get(state.item, user, ui) != I.LOADEDDICE) & (u_ab != A.SKILLLINK)
    hit_lands = accuracy_check(
        data, mv, atk, dfn, state.boosts[user, C.B_ACC], state.boosts[target, C.B_EVA],
        below(words[W_HIT_ACCURACY:W_HIT_ACCURACY + MAX_HITS], 100), state.gravity,
        weather_acc) | (state.volatiles[target, C.V_GLAIVERUSH] > 0)
    keeps_hitting = jnp.cumprod((hit_index == 0) | jnp.logical_not(multi_acc) | hit_lands)
    active = (hit_index < n_hits) & connects & jnp.logical_not(is_status) & \
        keeps_hitting.astype(bool)

    # Static, Flame Body, Effect Spore and Cursed Body roll on every hit that
    # reaches the Pokemon. A Flame Body burn halves the rest of a physical
    # multi-hit move: counted, like the Defense changes, as if each earlier hit
    # got through, and not with a Substitute in the way.
    contact_words = words[W_CONTACT_HITS:W_CONTACT_HITS + MAX_HITS]
    contact_rolls = uniform(contact_words) < 0.3
    u_types_now = active_types(state, user)
    burnable = (slot_get(state.status, user, ui) == C.STATUS_NONE) & \
        jnp.logical_not(jnp.any(u_types_now == C.FIRE)) & \
        (u_ab != A.WATERVEIL) & (u_ab != A.WATERBUBBLE) & (u_ab != A.THERMALEXCHANGE) & \
        (u_ab != A.COMATOSE) & (u_ab != A.PURIFYINGSALT) & \
        (state.side_conditions[user, C.SC_SAFEGUARD] <= 0)
    flame = active & contact_rolls & has_flag(mv.flags, "contact") & \
        (t_ab_raw == A.FLAMEBODY) & burnable & jnp.logical_not(had_sub)
    burned_by_hit = (atk.status == C.BRN) | \
        (jnp.cumsum(flame.astype(jnp.int32)) - flame.astype(jnp.int32) > 0)

    def damage_for_hit(roll, i, beat_up_power, burned_now):
        mv_i = scale_power_for_hit(data, move_id, mv, i + 1)
        mv_i = mv_i._replace(base_power=jnp.where(beat_up, beat_up_power, mv_i.base_power))
        return calc_damage(
            data, atk_hit, dfn, mv_i, is_crit=is_crit, damage_roll=roll,
            weather=hit_weather, terrain=hit_terrain,
            side_conditions=hit_side_conditions, type_exp=type_exp,
            bp_cb_mod=bp_cb_mod, grounded_user=cb_ctx.grounded_user,
            grounded_target=cb_ctx.grounded_target,
            analytic_ok=hit_analytic, fainted_count=hit_fainted_count,
            target_switched_in=hit_target_switched,
            technician_power=mv.base_power, berry_ok=berry_ok,
            def_stage_delta=def_step * i,
            def_raw=jnp.where(ice_breaks & (i > 0), noice_def, dfn.stats[C.DEF]),
            sown=sows & (i > 0),
            # Only the first hit finds the target at full HP (Multiscale) --
            # unless a Substitute is taking them.
            full_hp=(dfn.hp == dfn.maxhp) & ((i == 0) | had_sub),
            burned=burned_now)

    dmgs = jax.vmap(damage_for_hit)(rolls, hit_index, beat_up_powers, burned_by_hit)
    dmgs = jnp.where(is_fixed, fixed, dmgs)
    dmgs = jnp.where(ohko, target_maxhp, dmgs)
    dmgs = jnp.where(active, dmgs, 0)

    # Rough Skin, Iron Barbs and Rocky Helmet hurt the attacker on every contact
    # hit that reaches the Pokemon, and a multi-hit move stops once they have
    # knocked the attacker out.
    t_item = slot_get(state.item, target, ti)
    contact_move = has_flag(mv.flags, "contact") & jnp.logical_not(u_ab == A.MAGICGUARD)
    barb = jnp.where(contact_move & ((t_ab_raw == A.ROUGHSKIN) | (t_ab_raw == A.IRONBARBS)),
                     fraction_of_max(state, user, ui, 1, 8), 0)
    helmet = jnp.where(contact_move & (t_item == I.ROCKYHELMET),
                       fraction_of_max(state, user, ui, 1, 6), 0)
    chip = barb + helmet

    # The hits in order, as prefix sums over all ten rather than a loop: every
    # quantity carried from hit to hit is a running total or "the first hit
    # that ...". (Unrolled, the loop's chain of seven carries was a quarter of
    # `vmap(execute_move)`'s run time.)
    before = lambda x: jnp.cumsum(x) - x            # sum over the earlier hits
    first = lambda x: x & (before(x.astype(jnp.int32)) == 0)
    # A Substitute soaks each hit whole until it breaks -- what is left of it is
    # its HP less every earlier hit, floored at 0 -- and later hits land.
    to_sub = (jnp.maximum(start_sub - before(dmgs), 0) > 0) & (dmgs > 0)
    direct = jnp.where(to_sub, 0, dmgs)
    # Disguise / Ice Face take the first hit that gets through.
    absorbed = shield & first(direct > 0)
    direct = jnp.where(absorbed, 0, direct)
    # Focus Sash / Sturdy leave the target on 1 HP from full, which only the
    # first hit to reach it can find it at.
    endures = can_endure & (start_hp == target_maxhp) & first(direct > 0) & \
        (direct >= start_hp)
    direct = jnp.where(endures, start_hp - 1, direct)
    # A healing berry is eaten the moment a hit takes its holder past its
    # threshold -- Sitrus and Oran at half, Figy at a quarter, Cheek Pouch on
    # top -- and the later hits of the move land on the healed HP. Only taken
    # here when more hits follow; after the last hit, the berry step below
    # (which also keeps Cud Chew's count) eats it as for any single hit.
    t_berry = slot_get(state.item, target, ti)
    half_berry = (t_berry == I.SITRUSBERRY) | (t_berry == I.ORANBERRY)
    quarter_berry = t_berry == I.FIGYBERRY
    berry_heal = jnp.where(t_berry == I.SITRUSBERRY, fraction_of_max(state, target, ti, 1, 4),
                           jnp.where(t_berry == I.ORANBERRY, 10,
                                     fraction_of_max(state, target, ti, 1, 3)))
    berry_heal = berry_heal + jnp.where(t_ab_raw == A.CHEEKPOUCH,
                                        fraction_of_max(state, target, ti, 1, 3), 0)
    hp_after = start_hp - jnp.cumsum(direct)
    crossed = jnp.where(quarter_berry, hp_after * 4 <= target_maxhp, hp_after * 2 <= target_maxhp)
    more_hits = (jnp.cumsum(active[::-1].astype(jnp.int32))[::-1] -
                 active.astype(jnp.int32)) > 0
    can_eat = (half_berry | quarter_berry) & jnp.logical_not(unnerved(state, target)) & \
        (state.volatiles[target, C.V_HEALBLOCK] <= 0)
    eat_now = first(can_eat & (direct > 0) & (hp_after > 0) & crossed & more_hits)
    eat_at = jnp.argmax(eat_now)
    ate_mid = jnp.any(eat_now)
    mid_heal = jnp.minimum(berry_heal, target_maxhp - hp_after[eat_at])
    bonus = jnp.where(ate_mid & (hit_index > eat_at), mid_heal, 0)
    direct = jnp.minimum(direct, jnp.maximum(start_hp + bonus - before(direct), 0))
    # Rough Skin, Iron Barbs and Rocky Helmet: nothing more lands once the
    # attacker has fallen to them. The hits that do land are a prefix, and
    # cutting off the rest changes nothing before it.
    start_user_hp = slot_get(state.hp, user, ui).astype(jnp.int32)
    standing = start_user_hp - chip * before((direct > 0).astype(jnp.int32)) > 0
    direct = jnp.where(standing, direct, 0)
    # (A berry hit cut off by the attacker's fall is the last one after all.)
    ate_mid = ate_mid & jnp.any(standing & active & (hit_index > eat_at))
    bonus = jnp.where(ate_mid, bonus, 0)
    total_damage = jnp.sum(direct)
    final_hp = start_hp + jnp.where(ate_mid, mid_heal, 0) - total_damage
    final_sub = jnp.maximum(start_sub - jnp.sum(jnp.where(standing, dmgs, 0)), 0)
    final_shield = shield & jnp.logical_not(jnp.any(absorbed & standing))
    endured = jnp.any(endures & standing)
    reached_vec = direct > 0                        # the hits that reached the Pokemon
    hits_reached = jnp.sum(reached_vec.astype(jnp.int32))
    # Every hit made, a Substitute's included, until either side falls (Showdown's
    # hit loop): what Rage Fist counts.
    hits_made = jnp.sum((active & standing &
                         (start_hp + bonus - before(direct) > 0)).astype(jnp.int32))
    contact_chip = chip * hits_reached
    busted = shield & jnp.logical_not(final_shield)

    # Write the accumulated result back once.
    state = state._replace(
        last_faint=jnp.where((start_hp > 0) & (final_hp <= 0), target,
                             state.last_faint).astype(jnp.int8),
        hp=slot_set(state.hp, target, ti, final_hp),
        sub_hp=set_at(state.sub_hp, target, final_sub, when=jnp.logical_not(bypass_sub)),
        volatiles=set_at(state.volatiles, (target, C.V_SUBSTITUTE), 0,
                         when=jnp.logical_not(bypass_sub) & (final_sub <= 0)))
    # Destiny Bond: knocking out its user takes the attacker down with it (not
    # Future Sight). The attacker faints second, which settles a double
    # knockout in its side's favour, as in Showdown.
    bonded_ko = (start_hp > 0) & (final_hp <= 0) & \
        (state.volatiles[target, C.V_DESTINYBOND] > 0) & jnp.logical_not(future)
    state, _ = damage_pokemon(state, user, ui, jnp.where(
        bonded_ko, slot_get(state.hp, user, ui).astype(jnp.int32), 0))

    # Focus Sash is used up by the hit it saves its holder from. Sturdy, which
    # Showdown asks first, saves it instead and keeps the Sash.
    state = lose_item(state, target, ti, endured & (dfn.ability != A.STURDY) &
                      (slot_get(state.item, target, ti) == I.FOCUSSASH))
    # The berry eaten between hits is gone (and remembered, for Harvest), and
    # Cud Chew will eat it again a turn later, as after any bite.
    state = lose_item(state, target, ti, ate_mid)
    chews = ate_mid & (t_ab_raw == A.CUDCHEW)
    state = state._replace(
        last_item=slot_set(state.last_item, target, ti, t_berry, ate_mid),
        cud_berry=set_at(state.cud_berry, target, t_berry, chews),
        cud_turns=set_at(state.cud_turns, target, 2, chews))

    landed = connects & jnp.logical_not(is_status)
    # Whether the hit reached the Pokemon rather than stopping at its Substitute:
    # secondaries, contact effects and on-hit abilities all need that.
    hit_pokemon = landed & (jnp.logical_not(had_sub) | (total_damage > 0) | busted)
    state = state._replace(
        damage_taken=add_at(state.damage_taken, target, total_damage),
        damage_category=set_at(state.damage_category, target, mv.category, when=landed),
        times_hit=add_at(state.times_hit, (target, ti), hits_made, when=landed))

    took_damage = hit_pokemon & (total_damage > 0)
    d_ab = t_ab_raw
    attacker_guarded = u_ab == A.MAGICGUARD

    # Drain heals with the damage itself, before anything answers the hit
    # (Gulp Missile's spit, Rough Skin's chip at the end of the move). Showdown
    # drains a hit on a Substitute rounding up, any other rounding to nearest.
    sub_damage = jnp.where(bypass_sub, 0, start_sub - final_sub)
    drain = data["move_drain"][move_id].astype(jnp.int32)
    drained = jnp.where((drain[0] > 0) & landed,
                        round_fraction(total_damage, drain[0], drain[1]) +
                        idiv(sub_damage * drain[0] + drain[1] - 1, drain[1]), 0)
    # Liquid Ooze makes draining moves hurt the user instead.
    ooze = t_ab_raw == A.LIQUIDOOZE
    state, _ = heal_pokemon(state, user, ui, jnp.where(ooze, 0, drained))
    state, _ = damage_pokemon(state, user, ui, jnp.where(ooze, drained, 0))

    # The target's forme changes from being hit -- at most one applies, so they
    # share a call. Disguise breaks into Busted Mimikyu, which costs it an
    # eighth of its max HP; Ice Face breaks into Noice Eiscue (both permanent).
    # Gulp Missile spits its catch at the attacker -- a quarter of the
    # attacker's max HP, then -1 Defense (Arrokuda) or paralysis (Pikachu) --
    # and Cramorant goes back to its base forme.
    mimikyu = busted & (dfn.ability == A.DISGUISE)
    iced = busted & (dfn.ability == A.ICEFACE)
    gulping = t_species == species_index("cramorantgulping")
    gorging = t_species == species_index("cramorantgorging")
    spits = took_damage & (d_ab == A.GULPMISSILE) & (gulping | gorging) & \
        (slot_get(state.hp, user, ui) > 0)
    state = forme_change(data, state, target, ti, jnp.where(
        mimikyu, species_index("mimikyubusted"), jnp.where(
            iced, species_index("eiscuenoice"), species_index("cramorant"))),
        when=mimikyu | iced | spits, permanent=mimikyu | iced)
    state, _ = damage_pokemon(state, target, ti,
                              jnp.where(mimikyu, fraction_of_max(state, target, ti, 1, 8), 0))
    state, _ = damage_pokemon(state, user, ui, jnp.where(
        spits & jnp.logical_not(attacker_guarded), fraction_of_max(state, user, ui, 1, 4), 0))

    # A resist berry that halved the hit has been eaten (in the berry step below).
    berry_type = data["item_resist_type"][slot_get(state.item, target, ti)]
    ate_resist = hit_pokemon & berry_ok & (berry_type == mv.type) & \
        (berry_type != C.TYPE_NONE) & ((type_exp > 0) | (berry_type == C.NORMAL))

    # Phase boundary: the hit has landed; the rest reacts to it.
    total_damage, hit_pokemon, landed, took_damage, busted = barrier(
        total_damage, hit_pokemon, landed, took_damage, busted)

    # --- drain, recoil, self-destruct ---
    # Both count what a Substitute soaked as well as what reached the Pokemon.
    # Showdown drains a hit on a Substitute rounding up, any other rounding to
    # nearest, and rounds recoil to nearest.
    # (Drain itself came earlier, with the damage.)
    recoil = data["move_recoil"][move_id].astype(jnp.int32)
    magic_guard = (u_ab == A.MAGICGUARD) | (u_ab == A.ROCKHEAD)
    dealt = total_damage + sub_damage
    state, _ = damage_pokemon(
        state, user, ui,
        jnp.where((recoil[0] > 0) & landed & (dealt > 0) & jnp.logical_not(magic_guard),
                  jnp.maximum(round_fraction(dealt, recoil[0], recoil[1]), 1),
                  0))
    # An Air Balloon pops at the first hit, even one its holder's Substitute takes.
    state = lose_item(state, target, ti, landed & ((total_damage > 0) | (sub_damage > 0)) &
                      (slot_get(state.item, target, ti) == I.AIRBALLOON))

    # Struggle's recoil is a quarter of max HP, rounded -- direct damage that
    # Rock Head and Magic Guard do nothing about.
    state, _ = damage_pokemon(state, user, ui, jnp.where(
        struggling & landed,
        jnp.maximum(round_fraction(slot_get(state.maxhp, user, ui), 1, 4), 1), 0))

    # High Jump Kick and friends: half the user's max HP on a miss.
    crashed = data["move_crash_damage"][move_id] & attacks & \
        jnp.logical_not(connects) & target_alive
    state, _ = damage_pokemon(state, user, ui,
                              jnp.where(crashed,
                                        fraction_of_max(state, user, ui, 1, 2), 0))

    # Explosion and friends faint the user even on a miss -- unless Damp
    # stopped them before they started.
    mode = data["move_selfdestruct"][move_id]
    dies = ((mode == 1) & attacks & jnp.logical_not(damp)) | ((mode == 2) & connects)
    last = state.last_faint
    state, _ = damage_pokemon(state, user, ui,
                              jnp.where(dies, slot_get(state.hp, user, ui).astype(jnp.int32), 0))
    # Showdown faints an exploding user before the blast lands, so a target it
    # takes down with it is the one that fainted last.
    state = state._replace(last_faint=jnp.where(
        dies & (mode == 1) & (start_hp > 0) & (final_hp <= 0), last, state.last_faint))

    # --- status-move payloads (also applied by damaging moves that carry them) ---
    # A bounced move delivers its payload to its user, credited to the bouncer.
    applied = connects | bounce_lands
    recv = jnp.where(bounce_lands, user, target)
    src = jnp.where(bounce_lands, target, user)
    src_ab = jnp.where(bounce_lands, t_ab_raw, u_ab)
    recv_ab = slot_get(state.ability, recv, act(state, recv))
    recv_ab = jnp.where(breaks & jnp.logical_not(bounce_lands) &
                        data["ability_breakable"][recv_ab], 0, recv_ab)
    # (The move's own status is inflicted below, in the one `set_status` call
    # the recipient gets, together with a secondary's or Poison Touch's.)

    # Volatiles land on the user for a move aimed at itself (Magnet Rise, Charge,
    # Focus Energy), otherwise on the target. Encore and Disable are applied by
    # their handlers, which know which move to lock (and Substitute and Shed
    # Tail by theirs, which pay for it); confusion rolls its own
    # length; Attract needs opposite genders.
    self_targeting = data["move_target"][move_id] == C.TGT_SELF
    vol = data["move_volatile"][move_id]
    vol = jnp.where((vol == C.V_ENCORE) | (vol == C.V_DISABLE) | (vol == C.V_SUBSTITUTE) |
                    (vol == C.V_CURSE), -1, vol)
    vol_side = jnp.where(self_targeting, user, recv)
    vi = act(state, vol_side)
    genders = (slot_get(state.gender, vol_side, vi), slot_get(state.gender, 1 - vol_side,
                                                                act(state, 1 - vol_side)))
    w_now = effective_weather(state)
    # Oblivious cures Taunt and Attract on its next Update, so Mold Breaker
    # does not get them to stick; it does get past Aroma Veil.
    oblivious = slot_get(state.ability, recv, act(state, recv)) == A.OBLIVIOUS
    vol_blocked = (
        ((recv_ab == A.AROMAVEIL) & ((vol == C.V_ATTRACT) | (vol == C.V_TAUNT) |
                                     (vol == C.V_TORMENT))) |
        (oblivious & ((vol == C.V_ATTRACT) | (vol == C.V_TAUNT))) |
        ((vol == C.V_ATTRACT) & jnp.logical_not(
            (genders[0] != C.GENDER_NONE) & (genders[1] != C.GENDER_NONE) &
            (genders[0] != genders[1]))) |
        # Ghost types cannot be trapped (Mean Look, Block, Thousand Waves).
        ((vol == C.V_TRAPPED) &
         jnp.any(jnp.where(vol_side == user, u_types, dfn.types) == C.GHOST)) |
        # Yawn fails on anything that would refuse the sleep itself -- a status,
        # Insomnia and friends, Electric or Misty Terrain, Safeguard -- though
        # not on Sleep Clause, which only stops the sleep when it comes, nor on
        # Comatose, which only refuses the sleep itself (it has no status, and
        # nothing like Insomnia's block on the Yawn).
        ((vol == C.V_YAWN) & (
            (status_immune(data, state, vol_side, jnp.int8(C.SLP),
                           ignore_ability=breaks & jnp.logical_not(bounce_lands),
                           self_inflicted=True, grounded=grounded_of(vol_side)) &
             (recv_ab != A.COMATOSE)) |
            (state.side_conditions[vol_side, C.SC_SAFEGUARD] > 0))))
    duration = jnp.maximum(data["move_duration"][move_id], 1)
    # Binding moves hold on for Showdown's `random(5, 7)` residuals.
    duration = jnp.where(vol == C.V_PARTIALLYTRAPPED, 5 + below(words[W_EFFECT], 2), duration)
    # Taunt counts the turn it lands on only if its target has yet to move.
    # (A replacement that came in this turn after a U-turn has not: Showdown
    # also asks for `activeTurns`. A Taunt bounced back by Magic Bounce lands on
    # the Pokemon using it, which is moving now.)
    already_moved = state.moved_this_turn[vol_side] | (vol_side == user)
    duration = jnp.where((vol == C.V_TAUNT) & already_moved &
                         jnp.logical_not(state.switched_this_turn[vol_side]),
                         duration + 1, duration)
    state = confuse(state, vol_side, words[W_CONFUSE], applied & (vol == C.V_CONFUSION),
                    grounded_of(vol_side))
    state = state._replace(volatiles=set_at(
        state.volatiles, (vol_side, vol), duration,
        when=applied & (vol >= 0) & (vol != C.V_CONFUSION) &
        jnp.logical_not(vol_blocked) &
        (state.volatiles[vol_side, jnp.maximum(vol, 0)] <= 0)))
    # A healing move fails outright at full HP, taking its side effects with it:
    # a Roost that heals nothing does not shed the Flying type.
    heal = data["move_heal"][move_id].astype(jnp.int32)
    heal_fails = (heal[0] > 0) & (slot_get(state.hp, user, ui) >= slot_get(state.maxhp, user, ui))
    self_vol = data["move_self_volatile"][move_id]
    rampage_move = self_vol == C.V_LOCKEDMOVE
    self_vol = jnp.where(rampage_move, -1, self_vol)
    # A Terastallized Pokemon's Roost heals but leaves its type alone: there is
    # no `roost` volatile for it (Showdown's condition refuses to start).
    self_vol = jnp.where((self_vol == C.V_ROOST) & slot_get(state.terastallized, user, ui),
                         -1, self_vol)
    # A damaging move's `self` effect comes with a hit: Hyper Beam that misses
    # or is blocked leaves nothing to recharge from.
    state = state._replace(volatiles=set_at(
        state.volatiles, (user, self_vol), duration,
        when=jnp.where(is_status, attacks, connects) & (self_vol >= 0) &
        jnp.logical_not(heal_fails)))

    # A move's boosts land on its user when it targets itself. When it targets
    # the foe they are applied further down, together with any secondary's.
    tgt_boosts = data["move_boosts"][move_id].astype(jnp.int32)
    state, _ = apply_boosts(state, user, jnp.where(applied & self_targeting, tgt_boosts, 0))
    # `self` boosts (Close Combat's drops, Hyperspace Fury's) only come with a
    # move that connected, and Sheer Force deletes them along with secondaries.
    self_chance = data["move_self_chance"][move_id].astype(jnp.int32)
    self_chance = jnp.where(u_ab == A.SERENEGRACE, jnp.minimum(self_chance * 2, 100),
                            self_chance)
    self_rolls = (self_chance == 0) | (below(words[W_SECONDARY], 100) < self_chance)
    state, _ = apply_boosts(state, user,
                            jnp.where(connects & self_rolls & jnp.logical_not(
                                (u_ab == A.SHEERFORCE) & data["move_has_secondary"][move_id]),
                                data["move_self_boosts"][move_id].astype(jnp.int32), 0))

    # Dancer copies a dance the opponent performs: the self-boosting ones onto
    # itself, Feather Dance's drop onto the dancer's target. (Damaging dances
    # would need the whole move run again, and are not copied.)
    dancer_alive = slot_get(state.hp, target, ti) > 0
    dances = connects & is_status & has_flag(mv.flags, "dance") & \
        (t_ab_raw == A.DANCER) & dancer_alive
    state, _ = apply_boosts(state, target, jnp.where(dances & self_targeting, tgt_boosts, 0))
    # Stat drops the attacker suffers from the target's side collect here and
    # land in one call: a copied Feather Dance, Gulp Missile's spit, Gooey. Only
    # one of those abilities can be the target's.
    user_drops = jnp.where(dances & jnp.logical_not(self_targeting), tgt_boosts, 0) + \
        jnp.where(spits & gulping, jnp.asarray(boost_delta((C.B_DEF, -1))), 0)
    state = confuse(state, user, words[W_CONFUSE], dances & (vol == C.V_CONFUSION),
                    cb_ctx.grounded_user)
    state, _ = damage_pokemon(state, target, ti, jnp.where(
        dances & fx(effect, "clangoroussoul"),
        jnp.maximum(floordiv(target_maxhp * 33, 100), 1), 0))

    # --- field and side effects ---
    sc = data["move_side_condition"][move_id]
    state = _set_side_condition(state, recv, sc, applied & (sc >= 0) &
                                (data["move_target"][move_id] == C.TGT_FOE_SIDE))
    self_sc = data["move_self_side_condition"][move_id]
    state = _set_side_condition(state, user, self_sc, attacks & (self_sc >= 0))
    own_sc = jnp.where(data["move_target"][move_id] == C.TGT_ALLY_SIDE, sc, jnp.int8(-1))
    tailwind_before = state.side_conditions[user, C.SC_TAILWIND] > 0
    state = _set_side_condition(state, user, own_sc, applied & (own_sc >= 0),
                                light_clay=slot_get(state.item, user, ui) == I.LIGHTCLAY)
    # Wind Rider: a Tailwind starting on its own side is +1 Attack -- not one
    # already blowing, which makes the move fail.
    tailwind = applied & ((own_sc == C.SC_TAILWIND) | (self_sc == C.SC_TAILWIND)) & \
        (u_ab == A.WINDRIDER) & jnp.logical_not(tailwind_before)
    state, _ = apply_boosts(state, user, jnp.where(tailwind, boost_delta((C.B_ATK, 1)), 0))

    state = set_weather(state, data["move_weather"][move_id], attacks)
    state = set_terrain(state, data["move_terrain"][move_id], attacks)

    # --- healing moves (Showdown rounds the fraction) ---
    state, _ = heal_pokemon(state, user, ui, jnp.where(
        attacks & (heal[0] > 0) & jnp.logical_not(heal_fails),
        round_fraction(slot_get(state.maxhp, user, ui), heal[0], heal[1]), 0))

    # --- contact and on-damage abilities ---
    contact = has_flag(mv.flags, "contact") & hit_pokemon
    d_it = slot_get(state.item, target, ti)

    # Rough Skin and Iron Barbs take an eighth, Rocky Helmet a sixth, for each
    # contact hit -- counted in the hit loop above.
    state, _ = damage_pokemon(state, user, ui, contact_chip)

    # Aftermath: fainting to a contact move takes a quarter off the attacker
    # (unless Damp is on the field).
    aftermath = contact & (d_ab == A.AFTERMATH) & jnp.logical_not(attacker_guarded) & \
        (slot_get(state.hp, target, ti) <= 0) & (u_ab != A.DAMP)
    state, _ = damage_pokemon(state, user, ui,
                              jnp.where(aftermath,
                                        fraction_of_max(state, user, ui, 1, 4), 0))

    # Abilities that inflict something on whoever hit them. The defender has one
    # ability, so these share a single random word; the status they settle on
    # goes into the one `set_status` call the attacker gets, further down.
    # Each hit that reached the Pokemon rolls its own word; the first to come up
    # decides (a status, once given, is not given again).
    rolls_contact = jnp.any(reached_vec & contact_rolls)
    spore_k = below(contact_words, 100)
    spore_hit = reached_vec & (spore_k < 30)
    spore = jnp.where(jnp.any(spore_hit), spore_k[jnp.argmax(spore_hit)], 100)
    u_types = active_types(state, user)
    spore_immune = jnp.any(u_types == C.GRASS) | (u_ab == A.OVERCOAT) | \
        (slot_get(state.item, user, ui) == I.SAFETYGOGGLES)
    contact_status = jnp.int8(C.STATUS_NONE)
    for ability, status_ in ((A.STATIC, C.PAR), (A.FLAMEBODY, C.BRN),
                             (A.POISONPOINT, C.PSN)):
        fires = contact & (d_ab == ability) & rolls_contact
        contact_status = jnp.where(fires, jnp.int8(status_), contact_status)
    sporing = contact & (d_ab == A.EFFECTSPORE) & jnp.logical_not(spore_immune)
    contact_status = jnp.where(sporing & (spore < 30), jnp.int8(C.PSN), contact_status)
    contact_status = jnp.where(sporing & (spore < 21), jnp.int8(C.PAR), contact_status)
    contact_status = jnp.where(sporing & (spore < 11), jnp.int8(C.SLP), contact_status)
    # Beak Blast, while it heats up, burns anything that makes contact with it.
    contact_status = jnp.where(contact & (state.volatiles[target, C.V_BEAKBLAST] > 0),
                               jnp.int8(C.BRN), contact_status)
    contact_status = jnp.where(spits & gorging, jnp.int8(C.PAR), contact_status)

    # Cute Charm infatuates on contact, given opposite genders.
    ug, tg = slot_get(state.gender, user, ui), slot_get(state.gender, target, ti)
    charmed = contact & (d_ab == A.CUTECHARM) & rolls_contact & \
        (ug != C.GENDER_NONE) & (tg != C.GENDER_NONE) & (ug != tg) & \
        (u_ab != A.OBLIVIOUS) & (u_ab != A.AROMAVEIL)
    state = set_volatile(state, user, C.V_ATTRACT, jnp.where(
        charmed, 1, state.volatiles[user, C.V_ATTRACT]))
    # Cursed Body disables the move that hit it (Aroma Veil protects).
    cursed = took_damage & (d_ab == A.CURSEDBODY) & rolls_contact & (u_ab != A.AROMAVEIL) & \
        jnp.logical_not(future)
    state = state._replace(last_move=set_at(state.last_move, user, move_id, can_act))
    state = disable_move(data, state, user, cursed, False)
    # Gooey and Tangling Hair slow a contact attacker; with the other drops the
    # attacker takes from the target's side, that is one call.
    # Both answer every contact hit: Triple Axel into Tangling Hair is -3.
    user_drops = user_drops + jnp.where(
        contact & ((d_ab == A.GOOEY) | (d_ab == A.TANGLINGHAIR)),
        jnp.asarray(boost_delta((C.B_SPE, -1))) * jnp.maximum(hits_reached, 1), 0)
    state, _ = apply_boosts(state, user, user_drops, from_opponent=True)
    # Toxic Debris scatters Toxic Spikes on a physical hit; Seed Sower raises
    # Grassy Terrain; Electromorphosis charges up.
    # Toxic Debris answers every physical hit, up to the two layers.
    physical_hit = took_damage & (mv.category == C.CAT_PHYSICAL)
    for k in range(2):
        layers = state.side_conditions[user, C.SC_TOXICSPIKES]
        state = _set_side_condition(state, user, jnp.int32(C.SC_TOXICSPIKES),
                                    physical_hit & (d_ab == A.TOXICDEBRIS) & (layers < 2) &
                                    (hits_reached > k))
    state = set_terrain(state, jnp.int8(C.GRASSY_TERRAIN), took_damage & (d_ab == A.SEEDSOWER))
    state = state._replace(volatiles=set_at(
        state.volatiles, (target, C.V_CHARGE), 1, took_damage & (d_ab == A.ELECTROMORPHOSIS)))
    # Taking a damaging hit breaks Illusion and makes a Focus Punch user lose
    # its focus.
    state = state._replace(
        illusion=set_at(state.illusion, target, -1, took_damage),
        volatiles=set_at(state.volatiles, (target, C.V_FOCUSPUNCH), 2,
                         took_damage & (state.volatiles[target, C.V_FOCUSPUNCH] > 0)))

    # Abilities that react to taking a hit. The defender has exactly one ability,
    # so at most one of these can fire -- they are accumulated into a single
    # boost vector and applied once. Ten separate `apply_boosts` calls here was
    # enough on its own to tip vmap compilation over a cliff.
    phys_hit = took_damage & (mv.category == C.CAT_PHYSICAL)
    now_hp = slot_get(state.hp, target, ti).astype(jnp.int32)
    max_hp = slot_get(state.maxhp, target, ti).astype(jnp.int32)
    # Berserk and Anger Shell trigger when the hit takes the holder past half
    # HP -- but not off a Sheer Force move, whose after-effects never happen.
    sheer_move = (u_ab == A.SHEERFORCE) & data["move_has_secondary"][move_id]
    crossed_half = took_damage & (now_hp * 2 <= max_hp) & \
        ((now_hp + total_damage) * 2 > max_hp) & (now_hp > 0) & jnp.logical_not(sheer_move)

    target_boosts = jnp.zeros(7, jnp.int32)
    for ability, boosts, cond in (
            (A.STAMINA, {C.B_DEF: 1}, took_damage),
            (A.WEAKARMOR, {C.B_DEF: -1, C.B_SPE: 2}, phys_hit),
            (A.WATERCOMPACTION, {C.B_DEF: 2}, took_damage & (mv.type == C.WATER)),
            (A.JUSTIFIED, {C.B_ATK: 1}, took_damage & (mv.type == C.DARK)),
            (A.THERMALEXCHANGE, {C.B_ATK: 1}, took_damage & (mv.type == C.FIRE)),
            (A.RATTLED, {C.B_SPE: 1}, took_damage &
             ((mv.type == C.BUG) | (mv.type == C.DARK) | (mv.type == C.GHOST))),
            (A.ANGERPOINT, {C.B_ATK: 12}, phys_hit & is_crit),
            (A.BERSERK, {C.B_SPA: 1}, crossed_half),
            (A.ANGERSHELL, {C.B_ATK: 1, C.B_SPA: 1, C.B_SPE: 1, C.B_DEF: -1, C.B_SPD: -1},
             crossed_half)):
        vec = np.zeros(7, np.int32)
        for idx, amount in boosts.items():
            vec[idx] = amount
        target_boosts = jnp.where((d_ab == ability) & cond, jnp.asarray(vec),
                                  target_boosts)
    # Stamina, Weak Armor, Water Compaction, Justified and Rattled answer every
    # hit of a multi-hit move. (The damage of the later hits does not see the
    # Defense they change.)
    per_hit = (d_ab == A.STAMINA) | (d_ab == A.WEAKARMOR) | (d_ab == A.WATERCOMPACTION) | \
        (d_ab == A.JUSTIFIED) | (d_ab == A.RATTLED) | (d_ab == A.THERMALEXCHANGE)
    target_boosts = jnp.where(per_hit, target_boosts * jnp.maximum(hits_reached, 1),
                              target_boosts)
    # Weakness Policy: +2 Attack and Sp. Atk from a super-effective hit it
    # survives, unless the damage was fixed (Seismic Toss, Super Fang).
    policy = took_damage & (type_exp > 0) & jnp.logical_not(is_fixed) & (now_hp > 0) & \
        (slot_get(state.item, target, ti) == I.WEAKNESSPOLICY)
    target_boosts = target_boosts + jnp.where(
        policy, jnp.asarray(boost_delta((C.B_ATK, 2), (C.B_SPA, 2))), 0)
    state = lose_item(state, target, ti, policy)
    state, _ = apply_boosts(state, target, target_boosts)

    # Moxie and friends boost the attacker when the hit knocks the target out;
    # folded the same way. Battle Bond does it once a battle, and only while the
    # opponent still has something left to send in.
    ko = landed & (now_hp <= 0)
    user_boosts = jnp.zeros(7, jnp.int32)
    for ability, boost in ((A.MOXIE, C.B_ATK), (A.CHILLINGNEIGH, C.B_ATK),
                           (A.ASONEGLASTRIER, C.B_ATK), (A.GRIMNEIGH, C.B_SPA),
                           (A.ASONESPECTRIER, C.B_SPA)):
        vec = np.zeros(7, np.int32)
        vec[boost] = 1
        user_boosts = jnp.where((u_ab == ability) & ko, jnp.asarray(vec), user_boosts)
    bond = ko & (u_ab == A.BATTLEBOND) & (species == species_index("greninjabond")) & \
        jnp.logical_not(slot_get(state.bond_used, user, ui)) & \
        (slot_get(state.hp, user, ui) > 0) & jnp.any(state.hp[target] > 0)
    user_boosts = jnp.where(bond, jnp.asarray(boost_delta((C.B_ATK, 1), (C.B_SPA, 1),
                                                          (C.B_SPE, 1))), user_boosts)
    state = state._replace(bond_used=slot_set(state.bond_used, user, ui, True, bond))
    # Throat Spray: +1 Sp. Atk after a sound move succeeds -- after the move's
    # secondaries, which Sheer Force deletes, so not off one of those.
    spray = connects & has_flag(mv.flags, "sound") & jnp.logical_not(sheer_move) & \
        (slot_get(state.item, user, ui) == I.THROATSPRAY) & (slot_get(state.hp, user, ui) > 0)
    user_boosts = user_boosts + jnp.where(spray, jnp.asarray(boost_delta((C.B_SPA, 1))), 0)
    state = lose_item(state, user, ui, spray)
    state, _ = apply_boosts(state, user, user_boosts)

    # The user's forme changes from its own move, which never coincide: Gulp
    # Missile catches something when Cramorant uses Surf (an Arrokuda above half
    # HP, a Pikachu at or below it), and Relic Song flips Meloetta between Aria
    # and Pirouette Forme.
    # (Not a Ditto that has transformed into Cramorant.)
    catches = connects & (u_ab == A.GULPMISSILE) & (species == species_index("cramorant")) & \
        (move_id == move_index("surf")) & jnp.logical_not(state.transformed[user])
    aria, pirouette = species_index("meloetta"), species_index("meloettapirouette")
    sings = hit_pokemon & fx(effect, "relicsong") & ((species == aria) | (species == pirouette)) & \
        jnp.logical_not(state.transformed[user]) & (slot_get(state.hp, user, ui) > 0)
    state = forme_change(data, state, user, ui, jnp.where(
        sings, jnp.where(species == aria, pirouette, aria),
        jnp.where(hp_now * 2 <= maxhp_now, species_index("cramorantgorging"),
                  species_index("cramorantgulping"))), when=catches | sings)

    # --- secondaries and the special-effect handler ---
    state, sec_status, sec_word, sec_drops = _apply_secondaries(
        data, state, user, target, move_id, words, landed=hit_pokemon, target_ability=t_ab,
        connected=landed, target_grounded=cb_ctx.grounded_target,
        target_raised=t_raised_at_start)
    # The recipient's stat changes -- the move's own (Growl, or a bounced one)
    # and its secondaries' (Icy Wind; not on a target the hit knocked out, so
    # no Mirror Armor bounce either) -- in one call, and Defiant and
    # Competitive answer whatever the opponent lowered.
    sec_drops = jnp.where(slot_get(state.hp, target, ti) > 0, sec_drops, 0)
    state, lowered = apply_boosts(
        state, recv,
        jnp.where(applied & jnp.logical_not(self_targeting), tgt_boosts, 0) + sec_drops,
        from_opponent=True,
        ignore_ability=breaks & jnp.logical_not(bounce_lands) &
        data["ability_breakable"][t_ab_raw])
    # A move Magic Bounce reflects is used by the bouncer on the original user:
    # Strength Sap, Leech Seed's Grass check and Defog run the other way round.
    boosts_before = state.boosts
    # Choice items lock the user into the move it is using, if it was holding
    # one as the move began -- set here, at the start of the move as Showdown's
    # onModifyMove does, so that a Trick below can clear it again. Compared
    # directly rather than through `data["item_is_choice"]`: a gather whose
    # index is itself a gathered value is a compile cliff under vmap.
    held = held_at_start
    choiced = ((held == I.CHOICESCARF) | (held == I.CHOICEBAND) |
               (held == I.CHOICESPECS))
    state = state._replace(choice_slot=set_indexed(
        state.choice_slot, user, move_slot.astype(jnp.int8),
        when=choiced & can_act & jnp.logical_not(struggling)))
    # Every handler indexes with these, and they sit at the end of the bounce
    # chain: materialise them once rather than in each handler's kernels.
    src, recv = barrier(src, recv)
    state = run_effect(effect, data, state, src, recv, words[W_EFFECT],
                       when=attacks & jnp.logical_not(fails) & (effect > 0),
                       connected=connects | bounce_lands,
                       hit=jnp.where(is_status, connects | bounce_lands, hit_pokemon))
    # Whether Parting Shot changed a stat -- read now, before a White Herb
    # puts the drops back.
    stats_moved = jnp.any(state.boosts != boosts_before)


    # --- the status each side ends up with: one `set_status` call apiece ---
    # The recipient: the move's own status (Toxic, or a bounced one), Psycho
    # Shift passing the user's, Mortal Spin's poison, a secondary's (Scald's
    # burn), or the attacker's Poison Touch / Toxic Chain -- the latter two
    # blocked by Shield Dust and Covert Cloak although they are not secondaries.
    none = jnp.int8(C.STATUS_NONE)
    main = jnp.where(applied, data["move_status"][move_id], none)
    shift = fx(effect, "psychoshift") & connects
    main = jnp.where(shift, slot_get(state.status, user, ui), main)
    main = jnp.where(fx(effect, "mortalspin") & hit_pokemon & (main == C.STATUS_NONE),
                     jnp.int8(C.PSN), main)
    by_move = jnp.where(main != C.STATUS_NONE, main, sec_status)
    shielded = (d_ab == A.SHIELDDUST) | (d_it == I.COVERTCLOAK)
    poisons = took_damage & jnp.logical_not(shielded) & (uniform(words[W_POISON_TOUCH]) < 0.3)
    touch = poisons & (u_ab == A.POISONTOUCH) & has_flag(mv.flags, "contact")
    chain = poisons & (u_ab == A.TOXICCHAIN)
    status = jnp.where(by_move != C.STATUS_NONE, by_move,
                       jnp.where(chain, jnp.int8(C.TOX), jnp.where(touch, jnp.int8(C.PSN), none)))
    state, status_landed = set_status(
        data, state, recv, status,
        jnp.where(main != C.STATUS_NONE, words[W_STATUS], sec_word), source_ability=src_ab,
        ignore_ability=breaks & jnp.logical_not(bounce_lands), grounded=grounded_of(recv))
    state = cure_status(state, user, ui, when=shift & status_landed)
    # Poison Puppeteer: poison from Pecharunt's moves confuses as well.
    puppet = status_landed & (by_move != C.STATUS_NONE) & \
        ((status == C.PSN) | (status == C.TOX)) & (src_ab == A.POISONPUPPETEER) & \
        (slot_get(state.species, src, act(state, src)) == species_index("pecharunt"))
    state = confuse(state, recv, words[W_CONFUSE], puppet, grounded_of(recv))
    # The other side: Synchronize passing a burn, paralysis or poison straight
    # back, or else whatever a contact ability (Static, Effect Spore, Beak
    # Blast's heat) inflicts on the attacker.
    r_ab_now = slot_get(state.ability, recv, act(state, recv))
    synced = status_landed & (r_ab_now == A.SYNCHRONIZE) & (
        (status == C.BRN) | (status == C.PAR) | (status == C.PSN) | (status == C.TOX))
    state, _ = set_status(data, state, 1 - recv,
                          jnp.where(synced, status, contact_status), words[W_STATUS],
                          grounded=grounded_of(1 - recv))


    # Item theft after the hit. Magician takes the target's item when the user
    # has none; Pickpocket takes the attacker's when a contact move hits it.
    # Neither happens off a move that is switching its user out.
    switching_out = data["move_self_switch"][move_id] > 0
    ui_item = slot_get(state.item, user, ui)
    ti_item = slot_get(state.item, target, ti)
    magician = took_damage & (u_ab == A.MAGICIAN) & (ui_item == 0) & \
        jnp.logical_not(switching_out) & (slot_get(state.hp, user, ui) > 0) & \
        can_lose_item(data, state, target, ti, ignore_ability=breaks)
    pickpocket = contact & (d_ab == A.PICKPOCKET) & (ti_item == 0) & \
        jnp.logical_not(switching_out) & (slot_get(state.hp, target, ti) > 0) & \
        can_lose_item(data, state, user, ui)
    state = lose_item(state, target, ti, magician)
    state = lose_item(state, user, ui, pickpocket)
    state = state._replace(item=slot_set(slot_set(state.item, user, ui, ti_item, magician),
                                         target, ti, ui_item, pickpocket))

    # Charge is spent by the next Electric move the user tries to use.
    state = state._replace(volatiles=set_at(
        state.volatiles, (user, C.V_CHARGE), 0,
        can_act & (mv.type == C.ELECTRIC) & (self_vol != C.V_CHARGE) & (vol != C.V_CHARGE)))

    # Life Orb takes a tenth of its holder's max HP after any damaging move
    # that connected -- a Substitute counts -- unless Magic Guard spares it or
    # Sheer Force paid for the move by deleting its secondaries.
    orb = landed & (slot_get(state.item, user, ui) == I.LIFEORB) & \
        (u_ab != A.MAGICGUARD) & jnp.logical_not(sheer_move) & \
        (slot_get(state.hp, user, ui) > 0)
    state, _ = damage_pokemon(state, user, ui,
                              jnp.where(orb, fraction_of_max(state, user, ui, 1, 10), 0))

    # Soul-Heart gains Sp. Atk for every Pokemon that fainted during the move.
    state = soul_heart(state, alive_before)

    # Outrage, Petal Dance, Thrash, Raging Fury: a hit locks the user in for two
    # or three uses in all, spending no PP after the first. Showdown's
    # `lockedmove`, counted in uses: a use that does not connect before the
    # last ends the lock quietly; the last use, connecting or not, leaves the
    # user confused from fatigue -- unless it fell asleep, which just ends it.
    # That happens as the move ends, ahead of the berries below: a Lum Berry
    # cures the fatigue at once.
    remaining = state.volatiles[user, C.V_LOCKEDMOVE].astype(jnp.int32)
    rampaging = remaining > 0
    starts = rampage_move & connects & jnp.logical_not(rampaging)
    left = jnp.where(starts, 1 + below(words[W_RAMPAGE], 2),
                     jnp.where(connects, remaining - 1, 0))
    goes_on = (starts | rampaging) & (left > 0)
    asleep_now = slot_get(state.status, user, ui) == C.SLP
    fatigue = rampaging & jnp.logical_not(goes_on) & (remaining == 1) & \
        jnp.logical_not(asleep_now)
    state = state._replace(
        volatiles=set_at(state.volatiles, (user, C.V_LOCKEDMOVE),
                         jnp.where(goes_on, left, 0), starts | rampaging),
        locked_slot=set_at(state.locked_slot, user,
                           jnp.where(goes_on, move_slot, -1), starts | rampaging))
    # A berry the user already wants before the fatigue (Lum, for a Flame Body
    # burn from this very hit) is eaten on Showdown's Update after the hit --
    # so the confusion that comes as the move ends stays.
    early_bite = berry_wants(state, user)
    state = confuse(state, user, words[W_CONFUSE], fatigue, cb_ctx.grounded_user)
    fatigue_confusion = state.volatiles[user, C.V_CONFUSION]

    # --- berries: at most one bite per side per move, whatever the reason ---
    # Bug Bite takes the target's berry first and eats it on the spot.
    stolen = slot_get(state.item, target, ti)
    bites = fx(effect, "bugbite") & hit_pokemon & data["item_is_berry"][stolen] & \
        (slot_get(state.hp, user, ui) > 0) & \
        can_lose_item(data, state, target, ti, ignore_ability=breaks)
    state = lose_item(state, target, ti, bites)
    # The target: its berry's own trigger (Sitrus under half, Lum on a new
    # status), or a resist berry that just halved the hit.
    state = eat_berry(data, state, target, slot_get(state.item, target, ti),
                      when=berry_wants(state, target) | ate_resist)
    # The user: the stolen berry (which spends nothing of its own, and which
    # Cud Chew does not count), Stuff Cheeks' forced bite after its +2 Defense,
    # or its own berry's trigger.
    cheeks = fx(effect, "stuffcheeks") & attacks & jnp.logical_not(fails)
    state, _ = apply_boosts(state, user, jnp.where(cheeks, boost_delta((C.B_DEF, 2)), 0))
    state = eat_berry(data, state, user,
                      jnp.where(bites, stolen, slot_get(state.item, user, ui)),
                      when=bites | cheeks | berry_wants(state, user),
                      consume=jnp.logical_not(bites), cud=jnp.where(bites, 0, 2))
    state = state._replace(volatiles=set_at(state.volatiles, (user, C.V_CONFUSION),
                                            fatigue_confusion, early_bite & fatigue))

    # Protect's streak only survives consecutive protecting turns.
    # Protect's streak only survives a stalling move that went ahead (its
    # handler raised or reset it); anything else, or one that failed, clears it.
    is_stalling = data["move_stalling"][move_id]
    state = state._replace(protect_streak=set_at(
        state.protect_streak, user, 0,
        when=jnp.logical_not(is_stalling & attacks & jnp.logical_not(fails))))

    # Fusion Flare and Fusion Bolt remember whether they were the last move to
    # succeed this turn.
    fusion = jnp.where(data["move_bp_modify"][move_id] ==
                       E.BP_MODIFY_HANDLERS.index("fusionflare"), 1,
                       jnp.where(data["move_bp_modify"][move_id] ==
                                 E.BP_MODIFY_HANDLERS.index("fusionbolt"), 2, 0))
    state = state._replace(fusion_last=jnp.where(connects, fusion, state.fusion_last)
                           .astype(jnp.int8))

    # Stomping Tantrum's "failed": missed, immune, fizzled, or a status move
    # whose status did not take (Spore into Sleep Clause) -- but not stopped by
    # Protect or bounced back, which Showdown records as NOT_FAIL, not false.
    # A field move that changed nothing failed too: a weather or terrain
    # already up, a screen or hazard already at its limit.
    weather_before, terrain_before, sc_before = field_before
    field_move = (data["move_weather"][move_id] > 0) | (data["move_terrain"][move_id] > 0) | \
        (data["move_side_condition"][move_id] >= 0) | \
        (data["move_self_side_condition"][move_id] >= 0)
    field_unchanged = (state.weather == weather_before) & (state.terrain == terrain_before) & \
        jnp.all(state.side_conditions == sc_before)
    move_failed = attacks & jnp.logical_not(deferred) & jnp.logical_not(blocked) & \
        jnp.logical_not(bounces) & (
            jnp.logical_not(connects) |
            (is_status & (data["move_status"][move_id] != C.STATUS_NONE) &
             jnp.logical_not(status_landed)) |
            (is_status & field_move & field_unchanged))
    state = state._replace(
        last_move=set_at(state.last_move, user, chosen_id, when=can_act),
        moved_this_turn=set_at(state.moved_this_turn, user, True),
        moves_since_switch=add_at(state.moves_since_switch, user, 1, when=can_act),
        last_move_failed=set_at(state.last_move_failed, user, move_failed))
    state = log_event(state, user, ui, shown_at_start, chosen_id, when=can_act)

    # A Pokemon that comes by a new item -- Trick, Switcheroo, Pickpocket --
    # is freed of any Choice lock: a Choice item's `onStart` clears it, so even
    # Tricking one Choice item for another leaves the user unlocked.
    for side, idx, before in ((user, ui, held_at_start), (target, ti, t_held_at_start)):
        now = slot_get(state.item, side, idx)
        state = state._replace(choice_slot=set_at(
            state.choice_slot, side, -1, (now != before) & (now != 0)))

    # Two different kinds of switch come out of a move, and they behave
    # differently. A self-switch (U-turn, Volt Switch, Parting Shot) lets its
    # user *choose* a replacement, so it suspends the turn -- `force_switch`
    # asks the player. Being phazed (Whirlwind, Dragon Tail) drags in a *random*
    # Pokemon with no choice, so the engine resolves it immediately.
    # Gated on `connects`, not `landed`: `landed` excludes status moves, and
    # Whirlwind, Roar, Parting Shot and Teleport are all status moves. Baton
    # Pass and Shed Tail record what they hand over in `pass_mode`.

    # Beak Blast only burns attackers while it heats up: once its user has
    # moved, a contact hit is safe again (Showdown's onAfterMove).
    state = state._replace(volatiles=set_at(state.volatiles, (user, C.V_BEAKBLAST), 0,
                                            attacks))

    # White Herb answers any stat drop once the move is over, on either side,
    # and Protosynthesis and Quark Drive answer any change of weather or terrain.
    state = white_herb(state)
    state = paradox_update(state)

    mode = data["move_self_switch"][move_id]
    # A user that fainted on the way (Rocky Helmet, Life Orb) is not switched
    # out: its replacement arrives after the turn, like any fainted Pokemon's.
    # Parting Shot only switches its user out if a stat actually changed --
    # not against Clear Body, or a target already at -6 (Mirror Armor's
    # bounce counts).
    parting_ok = jnp.logical_not(fx(effect, "partingshot")) | stats_moved
    self_switch = (mode > 0) & connects & (slot_get(state.hp, user, ui) > 0) & \
        jnp.logical_not(fx(effect, "revivalblessing")) & parting_ok
    # Dragon Tail and Circle Throw drag out only a target they hit -- one
    # behind a Substitute stays in.
    # A Roar reflected by Magic Bounce drags out its user instead.
    phazing = data["move_force_switch"][move_id] & \
        jnp.where(is_status, applied, hit_pokemon) & \
        (slot_get(state.hp, recv, act(state, recv)) > 0)
    players = jnp.arange(C.NUM_PLAYERS)
    state = state._replace(
        force_switch=state.force_switch | ((players == user) & self_switch),
        phazed=state.phazed | ((players == recv) & phazing),
        pass_mode=set_at(state.pass_mode, user, mode, self_switch))
    return state


def _set_side_condition(state, side, condition, apply_it, light_clay=False):
    """Hazards stack up to their layer cap; screens are set to a turn count.

    Screens and the like last five turns (eight for a screen from a Light Clay
    holder; Tailwind four), and setting one that is already up fails rather
    than restarting it.
    """
    condition = jnp.maximum(condition.astype(jnp.int32), 0)
    current = get_indexed(state.side_conditions[side].astype(jnp.int32), condition)
    caps = np.full(C.NUM_SIDE_CONDITIONS, 5, np.int32)
    for idx, cap in C.SIDE_CONDITION_MAX.items():
        caps[idx] = cap
    cap = get_indexed(jnp.asarray(caps), condition)
    is_hazard = condition < C.SC_REFLECT
    screen = (condition == C.SC_REFLECT) | (condition == C.SC_LIGHTSCREEN) | \
        (condition == C.SC_AURORAVEIL)
    turns = jnp.where(condition == C.SC_TAILWIND, 4,
                      jnp.where(jnp.asarray(light_clay) & screen, 8, 5))
    new = jnp.where(is_hazard, jnp.minimum(current + 1, cap),
                    jnp.where(current > 0, current, turns))
    return state._replace(side_conditions=set_at(
        state.side_conditions, (side, condition), new, when=apply_it))


def _roll_hit_count(data, state, user, move_id, word, dice_word):
    """Number of hits for a multi-hit move (Loaded Dice and Skill Link raise it)."""
    lo = data["move_multihit"][move_id, 0].astype(jnp.int32)
    hi = data["move_multihit"][move_id, 1].astype(jnp.int32)
    ui = act(state, user)
    # Showdown's Gen 5+ 2-5 hit distribution, sampled from the same 20-entry
    # table: 2 and 3 at 35%, 4 and 5 at 15%.
    table = jnp.array([2] * 7 + [3] * 7 + [4] * 3 + [5] * 3, jnp.int32)
    sampled = table[below(word, 20)]
    n = jnp.where(hi > lo, sampled, lo)
    # Loaded Dice re-rolls a 2-5 hitter that came up short to four or five
    # hits, and takes a ten-hit move (Population Bomb) to 4-10.
    loaded = slot_get(state.item, user, ui) == I.LOADEDDICE
    n = jnp.where(loaded & (hi > lo) & (n < 4), 5 - below(dice_word, 2), n)
    n = jnp.where(loaded & (n == 10), 10 - below(dice_word, 7), n)
    # Skill Link always rolls the maximum number of hits.
    n = jnp.where((slot_get(state.ability, user, ui) == A.SKILLLINK) & (hi > lo), hi, n)
    return jnp.clip(n, 1, 10)
