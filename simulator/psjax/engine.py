"""Turn resolution: ordering, switching, residuals, and the public `step`.

The battle is a state machine over three phases:

    PHASE_MOVE    both players submit a move or a switch; the engine resolves the
                  turn (both actions, then end-of-turn residuals)
    PHASE_SWITCH  one or both players owe a replacement
    PHASE_END     the battle is over; `winner` says who took it

There are three ways a Pokemon leaves the field, and they are not the same:

* **Fainting.** The replacement is chosen after the turn finishes, residuals and
  all, which is what Showdown does in singles.
* **A self-switch** (U-turn, Volt Switch, Parting Shot, Teleport). The user picks
  a replacement *during* the turn, so the turn suspends: `PHASE_SWITCH` is
  entered with `pending_side`/`pending_action` holding the opponent's
  already-locked move, which then runs against whoever arrives.
* **Being phazed** (Whirlwind, Roar, Dragon Tail). A random Pokemon is dragged in
  with no choice to make, so `resolve_phazing` settles it inline and no decision
  point is created.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import consts as C
from . import effects as E
from .damage import (Attacker, MoveCtx, CRIT_RATES, calc_damage, current_types,
                     type_effectiveness)
from .data import load_data, move_index, species_index
from .hooks import A, ABILITY_IDX, I
from .mechanics import (_actives, act, active_types, actives_alive, apply_boosts, below,
                        berry_update, berry_wants, boost_delta, cure_status, eat_berry,
                        forme_change, revert_on_switch_out, slot_get, slot_set,
                        soul_heart, damage_pokemon, effective_speed,
                        effective_weather, fraction_of_max, heal_pokemon,
                        is_grounded, log_event, random_words, set_status, set_terrain,
                        set_weather, shown_species, uniform, unnerved, paradox_update,
                        usable_moves, white_herb, lose_item)
from .moves import (MOVE_WORDS, build_defender, execute_move, move_priority,
                    transform_into)
from .stats import boost_multiply
from .state import BattleState, barrier, reset_slot_state, select_state, set_at


# --- action decoding ---------------------------------------------------------

def is_switch(action):
    return action >= C.ACTION_SWITCH_BASE


def switch_slot(action):
    return (action - C.ACTION_SWITCH_BASE).astype(jnp.int32)


def move_slot(action):
    """Move index for both the plain and the terastallize encodings."""
    return jnp.where(action >= C.ACTION_TERA_BASE,
                     action - C.ACTION_TERA_BASE, action).astype(jnp.int32)


def wants_tera(action):
    return (action >= C.ACTION_TERA_BASE) & (action < C.ACTION_SWITCH_BASE)


# --- ordering ----------------------------------------------------------------

def action_priority(data, state, side, action):
    """Twice the priority, less one for Mycelium Might's fractional -0.1.

    Switches outrank every move; otherwise move priority plus ability bonuses
    (`moves.move_priority`). Doubling keeps it an integer while letting Mycelium
    Might's status moves go last within their bracket. A move the user is
    locked into (charging Solar Beam) or encored into is what counts, and so
    is Struggle for a Pokemon with nothing usable -- whatever sits in slot 0.
    A Custap Berry ready to go adds one, ahead of its own bracket.
    """
    ai = act(state, side)
    mid = _action_move(data, state, side, action)
    pri = 2 * move_priority(data, state, side, mid)
    mycelium = (state.ability[side, ai] == A.MYCELIUMMIGHT) & \
        (data["move_category"][mid] == C.CAT_STATUS)
    pri = pri - jnp.where(mycelium, 1, 0) + jnp.where(custap_fires(data, state, side, action), 1, 0)
    return jnp.where(is_switch(action), jnp.int32(20), pri)


def _action_move(data, state, side, action):
    """The move a move action will use: a locked or encored move overrides the
    chosen slot, and Struggle stands in when nothing is usable."""
    ai = act(state, side)
    slot = move_slot(action)
    locked = state.locked_slot[side] >= 0
    slot = jnp.where(locked, state.locked_slot[side].astype(jnp.int32),
                     jnp.where((state.volatiles[side, C.V_ENCORE] > 0) &
                               (state.encore_slot[side] >= 0),
                               state.encore_slot[side].astype(jnp.int32), slot))
    struggles = jnp.logical_not(jnp.any(usable_moves(data, state, side))) & \
        jnp.logical_not(locked)
    return jnp.where(struggles, move_index("struggle"),
                     jnp.maximum(state.moves[side, ai, slot], 0))


def custap_fires(data, state, side, action):
    """Custap Berry: at a quarter HP or less, a move of priority 0 or lower goes
    first in its bracket, and the berry is eaten as the order is settled
    (Showdown's onFractionalPriority). Unnerve keeps it uneaten."""
    ai = act(state, side)
    hp = state.hp[side, ai].astype(jnp.int32)
    mid = _action_move(data, state, side, action)
    return (state.item[side, ai] == I.CUSTAPBERRY) & (hp > 0) & \
        (hp * 4 <= state.maxhp[side, ai].astype(jnp.int32)) & \
        jnp.logical_not(unnerved(state, side)) & jnp.logical_not(is_switch(action)) & \
        (action >= 0) & (move_priority(data, state, side, mid) <= 0)


def turn_order(data, state, actions, key):
    """Which side acts first: priority, then Speed, with Trick Room and ties."""
    p0 = action_priority(data, state, 0, actions[0])
    p1 = action_priority(data, state, 1, actions[1])
    s0 = effective_speed(state, 0)
    s1 = effective_speed(state, 1)
    # Trick Room inverts the Speed comparison but not priority.
    faster0 = jnp.where(state.trick_room > 0, s0 < s1, s0 > s1)
    tie = s0 == s1
    coin = jax.random.bernoulli(key)
    first = jnp.where(p0 != p1, p0 > p1, jnp.where(tie, coin, faster0))
    return jnp.where(first, 0, 1).astype(jnp.int32)


# --- switching ---------------------------------------------------------------

def speed_order(state):
    """The side whose active goes first by Speed alone (Trick Room reverses
    it), p1 on a tie: the order of entry effects after simultaneous switches."""
    s0, s1 = effective_speed(state, 0), effective_speed(state, 1)
    faster0 = jnp.where(state.trick_room > 0, s0 < s1, s0 > s1)
    return jnp.where(faster0 | (s0 == s1), 0, 1).astype(jnp.int32)


def apply_entry_hazards(data, state, side):
    """Stealth Rock, Spikes, Toxic Spikes and Sticky Web on the incoming Pokemon."""
    i = act(state, side)
    sc = state.side_conditions[side]
    types = active_types(state, side)
    boots = state.item[side, i] == I.HEAVYDUTYBOOTS
    grounded = is_grounded(state, side)
    magic_guard = state.ability[side, i] == A.MAGICGUARD

    # Stealth Rock scales with the Rock matchup: 1/8 at neutral.
    rock_eff = data["type_eff"][C.ROCK, types]
    present = types != C.TYPE_NONE
    mult = jnp.prod(jnp.where(present, rock_eff, 1.0))
    sr_damage = (state.maxhp[side, i].astype(jnp.float32) * mult / 8.0).astype(jnp.int32)
    sr_damage = jnp.where((sc[C.SC_STEALTHROCK] > 0) & jnp.logical_not(boots) &
                          jnp.logical_not(magic_guard),
                          jnp.maximum(sr_damage, 1), 0)
    state, _ = damage_pokemon(state, side, i, sr_damage)

    # Spikes: 1/8, 1/6, 1/4 by layer count, grounded only.
    layers = sc[C.SC_SPIKES].astype(jnp.int32)
    denom = jnp.array([8, 8, 6, 4], jnp.int32)[jnp.clip(layers, 0, 3)]
    spike_damage = jnp.where((layers > 0) & grounded & jnp.logical_not(boots) &
                             jnp.logical_not(magic_guard),
                             fraction_of_max(state, side, i, 1, denom), 0)
    state, _ = damage_pokemon(state, side, i, spike_damage)

    # Toxic Spikes: poison (or badly poison at two layers); Poison types absorb
    # them -- if the Stealth Rock and Spikes before them left one standing.
    tspikes = sc[C.SC_TOXICSPIKES].astype(jnp.int32)
    is_poison = jnp.any(types == C.POISON)
    absorb = (tspikes > 0) & grounded & is_poison & (state.hp[side, i] > 0)
    state = state._replace(side_conditions=set_at(
        state.side_conditions, (side, C.SC_TOXICSPIKES), 0, when=absorb))
    poison_status = jnp.where(tspikes >= 2, jnp.int8(C.TOX), jnp.int8(C.PSN))
    should_poison = (tspikes > 0) & grounded & jnp.logical_not(is_poison) & \
                    jnp.logical_not(boots)
    state, _ = set_status(
        data, state, side,
        jnp.where(should_poison, poison_status, jnp.int8(C.STATUS_NONE)), jnp.uint32(0))

    # Sticky Web drops Speed on a grounded arrival.
    web = (sc[C.SC_STICKYWEB] > 0) & grounded & jnp.logical_not(boots)
    state, _ = apply_boosts(state, side,
                            jnp.where(web, boost_delta((C.B_SPE, -1)), 0),
                            from_opponent=True)
    return state


def apply_switch_in_ability(data, state, side):
    """Entry abilities: weather and terrain setters, Intimidate, Download, ..."""
    i = act(state, side)
    ab = state.ability[side, i]
    other = 1 - side
    oi = act(state, other)
    foe_out = state.hp[other, oi] > 0
    species = slot_get(state.species, side, i)

    # Formes that are settled on arrival -- at most one applies, so they share
    # a call. Tera Shift: Terapagos takes its Terastal Form for good, and with
    # it Tera Shell (and a new max HP, keeping the damage it has taken). Shields
    # Down: Minior arrives in its Meteor shell above half HP. Ice Face rebuilds
    # itself if it is snowing.
    shift = (ab == A.TERASHIFT) & (species == species_index("terapagos"))
    hp = slot_get(state.hp, side, i).astype(jnp.int32)
    maxhp = slot_get(state.maxhp, side, i).astype(jnp.int32)
    shell = (ab == A.SHIELDSDOWN) & (data["species_num"][species] == 774) & \
        jnp.logical_not(state.transformed[side]) & (hp * 2 > maxhp)
    refreeze = (ab == A.ICEFACE) & (species == species_index("eiscuenoice")) & \
        (effective_weather(state) == C.SNOW)
    state = forme_change(
        data, state, side, i,
        jnp.where(shift, species_index("terapagosterastal"),
                  jnp.where(shell, species_index("miniormeteor"), species_index("eiscue"))),
        when=shift | shell | refreeze, permanent=shift | refreeze,
        ability=jnp.where(shift, A.TERASHELL, ab))
    # Trace copies the foe's ability, and Imposter the whole foe; either way the
    # copy's own entry effect then runs (a traced Intimidate intimidates).
    o_ab = state.ability[other, oi]
    traced = (ab == A.TRACE) & foe_out & (o_ab != 0) & \
        jnp.logical_not(data["ability_notrace"][o_ab])
    state = state._replace(ability=slot_set(state.ability, side, i, o_ab, traced))
    state, _ = transform_into(data, state, side, when=ab == A.IMPOSTER)
    ab = slot_get(state.ability, side, i)

    state = set_weather(state, data["ability_weather"][ab])
    state = set_terrain(state, data["ability_terrain"][ab])

    # Intimidate drops the opponent's Attack -- except, from Gen 8, the Attack of
    # an Oblivious, Own Tempo, Inner Focus or Scrappy Pokemon.
    shrugs = (o_ab == A.OBLIVIOUS) | (o_ab == A.OWNTEMPO) | (o_ab == A.INNERFOCUS) | \
        (o_ab == A.SCRAPPY)
    # A Substitute keeps Intimidate out altogether.
    intimidate = (ab == A.INTIMIDATE) & foe_out & jnp.logical_not(shrugs) & \
        (state.volatiles[other, C.V_SUBSTITUTE] <= 0)
    state, applied = apply_boosts(
        state, other,
        jnp.where(intimidate, boost_delta((C.B_ATK, -1)), 0),
        from_opponent=True)
    # Rattled answers Intimidate with +1 Speed (Gen 9); Defiant and Competitive
    # answer it inside `apply_boosts`, as they do any drop.
    rattled = (applied[C.B_ATK] != 0) & (state.ability[other, oi] == A.RATTLED)
    state, _ = apply_boosts(state, other, jnp.where(rattled, boost_delta((C.B_SPE, 1)), 0))

    # Slow Start halves Attack and Speed for five turns; the residuals count
    # them down, skipping a turn the holder switched in on.
    state = state._replace(volatiles=set_at(
        state.volatiles, (side, C.V_SLOWSTART), 5, ab == A.SLOWSTART))

    # Intrepid Sword / Dauntless Shield boost the arriving Pokemon once, and so
    # do Download (whichever attacking stat the foe's defences leave open),
    # Wind Rider (a Tailwind already blowing on its side) and a Terastallized
    # Ogerpon's Embody Aspect, which boosts again on every entry. One boost at
    # most.
    o_stats = slot_get(state.stats, other, oi)
    o_def = boost_multiply(o_stats[C.DEF], state.boosts[other, C.B_DEF])
    o_spd = boost_multiply(o_stats[C.SPD], state.boosts[other, C.B_SPD])
    entry = jnp.zeros(C.NUM_BOOSTS, jnp.int32)
    # Since Gen 9 Intrepid Sword and Dauntless Shield work once a battle.
    sword = ((ab == A.INTREPIDSWORD) | (ab == A.DAUNTLESSSHIELD)) & \
        jnp.logical_not(slot_get(state.bond_used, side, i))
    state = state._replace(bond_used=slot_set(state.bond_used, side, i, True, sword))
    for cond, delta in (
            (sword & (ab == A.INTREPIDSWORD), boost_delta((C.B_ATK, 1))),
            (sword & (ab == A.DAUNTLESSSHIELD), boost_delta((C.B_DEF, 1))),
            ((ab == A.DOWNLOAD) & foe_out & (o_def >= o_spd), boost_delta((C.B_SPA, 1))),
            ((ab == A.DOWNLOAD) & foe_out & (o_def < o_spd), boost_delta((C.B_ATK, 1))),
            ((ab == A.WINDRIDER) & (state.side_conditions[side, C.SC_TAILWIND] > 0),
             boost_delta((C.B_ATK, 1)))) + tuple(
            ((ab == ABILITY_IDX[aspect]) & slot_get(state.terastallized, side, i),
             boost_delta((stage, 1))) for _, _, aspect, stage in _OGERPON_TERA):
        entry = jnp.where(cond, jnp.asarray(delta), entry)
    state, _ = apply_boosts(state, side, entry)
    # The arrival may have brought sun or Electric Terrain (or Booster Energy).
    return paradox_update(state)


_BATON_PASS_MASK = np.isin(np.arange(C.NUM_VOLATILES), C.BATON_PASS_VOLATILES)
_SHED_TAIL_MASK = np.arange(C.NUM_VOLATILES) == C.V_SUBSTITUTE
_SOURCE_BOUND_MASK = np.isin(np.arange(C.NUM_VOLATILES),
                             [C.V_ATTRACT, C.V_SYRUPBOMB, C.V_PARTIALLYTRAPPED, C.V_TRAPPED])


def switch_to(data, state, side, slot):
    """Bring `slot` in for `side`, clearing slot state and running entry effects."""
    state, valid = swap_in(data, state, side, slot)
    return arrive(data, state, side, valid)


def swap_in(data, state, side, slot):
    """The switch itself: the old Pokemon leaves and `slot` takes its place.

    Returns `(state, valid)`. Entry effects -- hazards, abilities -- are
    `arrive`'s: when both sides switch at once, Showdown makes both switches
    before either newcomer's entry effects run.
    """
    requested = slot.astype(jnp.int32)
    slot = jnp.clip(requested, 0, C.TEAM_SIZE - 1)
    valid = ((requested == slot) & (slot_get(state.hp, side, slot) > 0) &
             (slot != act(state, side)))
    slot = jnp.where(valid, slot, act(state, side))

    alive_before = actives_alive(state)
    # Regenerator heals the departing Pokemon by a third.
    old = act(state, side)
    regen = state.ability[side, old] == A.REGENERATOR
    state, _ = heal_pokemon(state, side, old,
                            jnp.where(regen & valid,
                                      fraction_of_max(state, side, old, 1, 3), 0),
                            blockable=False)
    # Natural Cure clears status on the way out.
    natural = state.ability[side, old] == A.NATURALCURE
    state = cure_status(state, side, old, when=natural & valid)
    # Zero to Hero: Palafin leaves as Palafin-Hero, and stays that way. Making
    # the Hero its base forme is enough -- reverting on the way out below
    # rebuilds its stats as that forme (its HP is the same either way).
    hero = valid & (state.hp[side, old] > 0) & (state.ability[side, old] == A.ZEROTOHERO) & \
        (slot_get(state.species, side, old) == species_index("palafin"))
    state = state._replace(base_species=slot_set(
        state.base_species, side, old, species_index("palafinhero"), hero))

    # Baton Pass hands over boosts and most volatiles, Shed Tail its
    # Substitute; they have to be read off the slot before it is cleared.
    mode = state.pass_mode[side]
    passed_boosts = state.boosts[side]
    passed_vols = state.volatiles[side]
    passed_sub = state.sub_hp[side]
    state = revert_on_switch_out(data, state, side, old, when=valid)
    state = reset_slot_state(state, side)
    baton = valid & (mode == 2)
    keeps = jnp.where(baton, jnp.asarray(_BATON_PASS_MASK),
                      jnp.where(valid & (mode == 3), jnp.asarray(_SHED_TAIL_MASK), False))
    state = state._replace(
        boosts=set_at(state.boosts, side, passed_boosts, baton),
        volatiles=set_at(state.volatiles, side,
                         jnp.where(keeps, passed_vols, state.volatiles[side])),
        sub_hp=set_at(state.sub_hp, side, passed_sub, keeps[C.V_SUBSTITUTE]),
        pass_mode=set_at(state.pass_mode, side, 0))

    # Whatever the departing Pokemon was holding the foe with lets go: Attract,
    # Syrup Bomb, binding moves and Mean Look's trap all end with their source.
    other = 1 - side
    state = state._replace(volatiles=set_at(
        state.volatiles, other,
        jnp.where(jnp.asarray(_SOURCE_BOUND_MASK) & valid, 0, state.volatiles[other])))

    # Party order: the newcomer swaps places with the Pokemon it replaces, and
    # Illusion disguises it as the last healthy Pokemon behind it in that order.
    pos = state.party_pos[side].astype(jnp.int32)
    new_pos, old_pos = pos[slot], pos[old]
    behind = (pos > new_pos) & (state.hp[side] > 0) & (jnp.arange(C.TEAM_SIZE) != slot)
    disguise = jnp.where(jnp.any(behind), jnp.argmax(jnp.where(behind, pos, -1)), -1)
    pos = jnp.where(jnp.arange(C.TEAM_SIZE) == slot, old_pos,
                    jnp.where(jnp.arange(C.TEAM_SIZE) == old, new_pos, pos))

    # Toxic's count starts over each time its holder comes in.
    entering_tox = valid & (slot_get(state.status, side, slot) == C.TOX)
    state = state._replace(
        status_turns=slot_set(state.status_turns, side, slot, jnp.int8(1), entering_tox),
        active=set_at(state.active, side, slot),
        switched_this_turn=set_at(state.switched_this_turn, side, True),
        force_switch=set_at(state.force_switch, side, False),
        party_pos=set_at(state.party_pos, side, pos, valid),
        illusion=set_at(state.illusion, side, disguise,
                        slot_get(state.ability, side, slot) == A.ILLUSION))
    state = log_event(state, side, slot, shown_species(state, side, slot), -1, when=valid)
    return state, valid


def arrive(data, state, side, when=True):
    """A newcomer's entry effects: Healing Wish, hazards, its ability, berries.

    `when=False` leaves the state alone.
    """
    alive_before = actives_alive(state)
    arrived = _arrive(data, state, side, when)
    return select_state(when, soul_heart(arrived, alive_before), state)


def _arrive(data, state, side, valid):
    # Showdown runs Update after the switch itself, before anything of the
    # newcomer's starts: a foe kept from its berry by the Unnerve that just
    # left eats it now, whatever the newcomer's own ability.
    state = berry_update(data, state, 1 - side, when=valid, foe_started=False)
    # A Healing Wish waiting on this side restores the newcomer in full if it
    # is hurt or statused -- before the hazards, as Showdown runs slot
    # conditions ahead of side conditions. Lunar Dance restores its PP too,
    # and waits for one that is hurt, statused or short of PP.
    i = act(state, side)
    lunar = state.healing_wish[side] == 2
    short_of_pp = jnp.any(state.pp[side, i] < state.maxpp[side, i])
    restores = valid & (state.healing_wish[side] > 0) & (
        (state.hp[side, i] < state.maxhp[side, i]) | (state.status[side, i] != C.STATUS_NONE) |
        (lunar & short_of_pp))
    state = state._replace(
        hp=set_at(state.hp, (side, i), state.maxhp[side, i], restores),
        status=set_at(state.status, (side, i), jnp.int8(C.STATUS_NONE), restores),
        pp=set_at(state.pp, (side, i), state.maxpp[side, i], restores & lunar),
        healing_wish=set_at(state.healing_wish, side, 0, restores))
    # Tera Shift (switch-in priority 2) takes Terapagos to its Terastal Form,
    # and its larger max HP, before the hazards measure it.
    shift = valid & (state.ability[side, i] == A.TERASHIFT) & \
        (state.species[side, i] == species_index("terapagos"))
    state = forme_change(data, state, side, i, species_index("terapagosterastal"),
                         when=shift, permanent=True, ability=A.TERASHELL)
    state = apply_entry_hazards(data, state, side)
    # A newcomer the hazards knocked out does nothing on arrival (no Intimidate).
    state = select_state(state.hp[side, i] > 0,
                         apply_switch_in_ability(data, state, side), state)
    # Supreme Overlord counts the allies down as its holder comes in.
    state = state._replace(overlord_count=set_at(
        state.overlord_count, side, jnp.sum(state.hp[side] <= 0).astype(jnp.int8)))
    state = berry_update(data, state, side)
    # White Herb answers a drop from the arrival (Intimidate, Sticky Web).
    return trace_update(data, white_herb(state))


def trace_update(data, state):
    """Trace keeps looking until it has something to copy (Showdown's
    `onUpdate`): a Trace user that came in facing nobody, or an ability it
    could not copy, takes the next foe's. Both sides at once. Only the ability
    is copied here; its entry effect, if it has one, does not run."""
    foe = jnp.arange(C.NUM_PLAYERS)[::-1]
    ab = _actives(state, state.ability)
    foe_ab = ab[foe]
    foe_alive = (_actives(state, state.hp) > 0)[foe]
    tracing = (ab == A.TRACE) & (_actives(state, state.hp) > 0) & foe_alive & \
        (foe_ab != 0) & jnp.logical_not(data["ability_notrace"][foe_ab])
    on = (jnp.arange(C.TEAM_SIZE)[None, :] == state.active.astype(jnp.int32)[:, None]) & \
        tracing[:, None]
    return state._replace(ability=jnp.where(on, foe_ab[:, None], state.ability))


# --- residuals ---------------------------------------------------------------

def residuals(data, state, key, when=True):
    """End-of-turn effects, in Showdown's residual order.

    `when=False` makes the whole thing a no-op, for a turn that was suspended
    partway through so a self-switch could pick its replacement. The old value
    is selected back for every field the residuals wrote -- found at trace
    time, since an untouched field is the very same array -- which is cheaper
    than branching around the whole function, which under vmap would select
    over the entire BattleState. (A hand-kept list of the written fields once
    missed `boosted_stat`: a replacement's Protosynthesis was switched off by a
    sun that, in the residuals that were then thrown away, had just ended.)
    """
    original = state
    state = _residuals(data, state, key)
    return original._replace(**{
        f: jnp.where(when, getattr(state, f), getattr(original, f))
        for f in state._fields if getattr(state, f) is not getattr(original, f)})


def future_hit(data, state, side, when, words):
    """Future Sight or Doom Desire lands on whoever now holds `side`'s slot.

    The damage is calculated with the user's stats as they are now, even from
    the bench (then without its stat stages). The move ignores Protect, is
    stopped by a Substitute, and -- unlike the move as used -- respects type
    immunities, so a Dark type takes nothing from Future Sight.
    """
    src_side = 1 - side
    src = state.future_source[side].astype(jnp.int32)
    ti = act(state, side)
    src_out = act(state, src_side) == src
    move = state.future_move[side].astype(jnp.int32)
    atk = Attacker(
        level=slot_get(state.level, src_side, src), stats=slot_get(state.stats, src_side, src),
        boosts=jnp.where(src_out, state.boosts[src_side], 0),
        types=current_types(slot_get(state.types, src_side, src),
                            slot_get(state.terastallized, src_side, src),
                            slot_get(state.tera_type, src_side, src)),
        base_types=slot_get(state.types, src_side, src),
        ability=slot_get(state.ability, src_side, src), item=slot_get(state.item, src_side, src),
        status=slot_get(state.status, src_side, src), hp=slot_get(state.hp, src_side, src),
        maxhp=slot_get(state.maxhp, src_side, src),
        terastallized=slot_get(state.terastallized, src_side, src),
        tera_type=slot_get(state.tera_type, src_side, src),
        boosted_stat=jnp.where(src_out, state.boosted_stat[src_side], jnp.int8(-1)),
        slow_start=jnp.bool_(False),
        species_num=data["species_num"][slot_get(state.species, src_side, src)])
    dfn = build_defender(data, state, side)
    mv = MoveCtx(id=move, type=data["move_type"][move], category=data["move_category"][move],
                 base_power=data["move_base_power"][move].astype(jnp.int32),
                 flags=data["move_flags"][move], crit_ratio=data["move_crit_ratio"][move],
                 effect_cb=jnp.int8(0))
    type_exp, immune = type_effectiveness(data, mv.type, dfn.types, mv, jnp.int32(0),
                                          dfn.ability, jnp.bool_(False),
                                          def_terastallized=dfn.terastallized,
                                          def_grounded=is_grounded(state, side),
                                          def_full_hp=dfn.hp >= dfn.maxhp)
    crit = uniform(words[0]) * CRIT_RATES[0].astype(jnp.float32) < 1.0
    crit = crit & (dfn.ability != A.BATTLEARMOR) & (dfn.ability != A.SHELLARMOR)
    dmg = calc_damage(data, atk, dfn, mv, is_crit=crit, damage_roll=below(words[1], 16),
                      weather=effective_weather(state), terrain=state.terrain,
                      side_conditions=state.side_conditions[side], type_exp=type_exp,
                      grounded_target=is_grounded(state, side))
    lands = when & (state.hp[side, ti] > 0) & jnp.logical_not(immune)
    # Materialise the damage once. Without the barrier XLA fuses the whole
    # damage chain into every kernel that reads it, and recomputes it in each:
    # that alone made the residuals ten times slower batched.
    dmg = jax.lax.optimization_barrier(jnp.where(lands, dmg, 0))
    # A Substitute soaks it; Focus Sash and Sturdy hold at 1 HP from full.
    sub = state.sub_hp[side].astype(jnp.int32)
    has_sub = state.volatiles[side, C.V_SUBSTITUTE] > 0
    to_sub = jnp.where(has_sub, jnp.minimum(dmg, sub), 0)
    hp = state.hp[side, ti].astype(jnp.int32)
    direct = jnp.where(has_sub, 0, dmg)
    endures = ((dfn.item == I.FOCUSSASH) | (dfn.ability == A.STURDY)) & (hp == dfn.maxhp) & \
        (direct >= hp)
    direct = jnp.where(endures, hp - 1, direct)
    state = state._replace(
        sub_hp=set_at(state.sub_hp, side, sub - to_sub),
        volatiles=set_at(state.volatiles, (side, C.V_SUBSTITUTE), 0,
                         has_sub & (sub - to_sub <= 0)))
    return damage_pokemon(state, side, ti, direct)[0]


def _residuals(data, state, key):
    # Per side: Shed Skin, Harvest, and Future Sight's crit and damage roll.
    residual_words = random_words(key, (C.NUM_PLAYERS, 4))
    alive_before = actives_alive(state)
    dec = lambda v: jnp.maximum(v.astype(jnp.int32) - 1, 0).astype(v.dtype)

    # The weather counts down first, at its own residual step: on its last turn
    # it ends there, before it chips anyone or heals Rain Dish and friends.
    weather_turns = dec(state.weather_turns)
    state = state._replace(
        weather=jnp.where(weather_turns <= 0, jnp.int8(C.WEATHER_NONE), state.weather),
        weather_turns=weather_turns)

    # Showdown runs the residuals effect by effect -- each one for both sides,
    # fastest first -- and stops the moment a side has nobody left. So they are
    # split into phases by Showdown's residual order, each a `fori_loop` over
    # the two sides in speed order (one compiled copy apiece), and a side's turn
    # in a phase is skipped once the battle is decided. Running all of one
    # side's residuals before the other's had Bad Dreams land before the
    # victim's Leftovers, and one side's poison knock it out before its Leech
    # Seed had finished the other off.
    first = speed_order(state)

    def phased(phase, state):
        def body(k, st):
            side = jnp.where(k == 0, first, 1 - first)
            i = act(st, side)
            live = jnp.logical_not(_battle_over(st))
            return phase(side, i, (st.hp[side, i] > 0) & live, st)
        return jax.lax.fori_loop(0, C.NUM_PLAYERS, body, state)

    # Order 1-4: the weather's chip and the abilities that answer it, then
    # Future Sight / Doom Desire and Wish (slot conditions).
    def weather_and_slots(side, i, alive, state):
        ab = state.ability[side, i]
        magic_guard = ab == A.MAGICGUARD
        types = active_types(state, side)
        w = effective_weather(state)
        words = residual_words[side]
        # Sandstorm chips everything but Rock, Ground and Steel.
        sand_immune = (jnp.any(types == C.ROCK) | jnp.any(types == C.GROUND) |
                       jnp.any(types == C.STEEL) | (ab == A.SANDVEIL) |
                       (ab == A.SANDRUSH) | (ab == A.SANDFORCE) |
                       (ab == A.OVERCOAT) | (state.item[side, i] == I.SAFETYGOGGLES))
        sand = (w == C.SAND) & alive & jnp.logical_not(sand_immune) & \
            jnp.logical_not(magic_guard)
        state, _ = damage_pokemon(state, side, i,
                                  jnp.where(sand, fraction_of_max(state, side, i, 1, 16), 0))
        # Ice Body heals a sixteenth in snow and Rain Dish in rain; Dry Skin
        # heals an eighth in rain and loses one in sun, as Solar Power does.
        sunny = (w == C.SUN) | (w == C.HARSH_SUN)
        rainy = (w == C.RAIN) | (w == C.HEAVY_RAIN)
        sixteenth = ((ab == A.ICEBODY) & (w == C.SNOW)) | ((ab == A.RAINDISH) & rainy)
        wet = (ab == A.DRYSKIN) & rainy
        scorched = (((ab == A.DRYSKIN) | (ab == A.SOLARPOWER)) & sunny) & \
            jnp.logical_not(magic_guard)
        state, _ = heal_pokemon(state, side, i, jnp.where(
            alive & (sixteenth | wet),
            fraction_of_max(state, side, i, 1, jnp.where(wet, 8, 16)), 0))
        state, _ = damage_pokemon(state, side, i, jnp.where(
            alive & scorched, fraction_of_max(state, side, i, 1, 8), 0))
        # Showdown runs Update after the weather's turn (the only residual that
        # does), so a berry the chip has triggered is eaten now -- before
        # Harvest, which can grow it back the same turn.
        state = berry_update(data, state, side, during_residual=True,
                             when=alive & (state.weather != C.WEATHER_NONE))
        # Future Sight lands, then Wish heals, counting down as they go.
        future = state.future_turns[side]
        state = future_hit(data, state, side, (future == 1) & jnp.logical_not(
            _battle_over(state)), words[2:])
        wish = state.wish_turns[side]
        state, _ = heal_pokemon(state, side, i, jnp.where(
            (wish == 1) & alive & (state.hp[side, i] > 0), state.wish_hp[side], 0))
        return state._replace(
            future_turns=set_at(state.future_turns, side, jnp.maximum(future - 1, 0)),
            wish_turns=set_at(state.wish_turns, side, jnp.maximum(wish - 1, 0)))

    # Order 5-7: Grassy Terrain, Shed Skin / Hydration, Leftovers, Aqua Ring and
    # Ingrain.
    def healing(side, i, alive, state):
        ab = state.ability[side, i]
        w = effective_weather(state)
        words = residual_words[side]
        grassy = (state.terrain == C.GRASSY_TERRAIN) & alive & is_grounded(state, side)
        state, _ = heal_pokemon(state, side, i,
                                jnp.where(grassy, fraction_of_max(state, side, i, 1, 16), 0))
        # Shed Skin / Hydration (5.3) come before Leftovers and the poison or
        # burn damage the status would have dealt.
        shed = (ab == A.SHEDSKIN) & alive & (slot_get(state.status, side, i) != 0) & \
            (uniform(words[0]) < (1.0 / 3.0))
        hydrated = (ab == A.HYDRATION) & alive & ((w == C.RAIN) | (w == C.HEAVY_RAIN))
        state = cure_status(state, side, i, when=shed | hydrated)
        lefto = (state.item[side, i] == I.LEFTOVERS) & alive
        state, _ = heal_pokemon(state, side, i,
                                jnp.where(lefto, fraction_of_max(state, side, i, 1, 16), 0))
        for vol in (C.V_AQUARING, C.V_INGRAIN):
            on = (state.volatiles[side, vol] > 0) & alive
            state, _ = heal_pokemon(state, side, i,
                                    jnp.where(on, fraction_of_max(state, side, i, 1, 16), 0))
        return state

    # Order 8: Leech Seed drains to the opponent's active slot.
    def leech_seed(side, i, alive, state):
        ab = state.ability[side, i]
        other = 1 - side
        oi = act(state, other)
        seeded = (state.volatiles[side, C.V_LEECHSEED] > 0) & alive & \
            (ab != A.MAGICGUARD) & (state.hp[other, oi] > 0)
        drain = jnp.where(seeded, fraction_of_max(state, side, i, 1, 8), 0)
        state, dealt = damage_pokemon(state, side, i, drain)
        # Liquid Ooze turns the seeder's healing into damage.
        ooze = ab == A.LIQUIDOOZE
        state, _ = heal_pokemon(state, other, oi, jnp.where(ooze, 0, dealt))
        state, _ = damage_pokemon(state, other, oi, jnp.where(ooze, dealt, 0))
        return state

    # Order 9-10: poison, then burn. Toxic ramps by a sixteenth a turn.
    def status_damage(side, i, alive, state):
        ab = state.ability[side, i]
        magic_guard = ab == A.MAGICGUARD
        status = state.status[side, i]
        counter = state.status_turns[side, i].astype(jnp.int32)
        poison_heal = ab == A.POISONHEAL
        burn = (status == C.BRN) & alive & jnp.logical_not(magic_guard)
        psn = (status == C.PSN) & alive & jnp.logical_not(magic_guard) & \
            jnp.logical_not(poison_heal)
        tox = (status == C.TOX) & alive & jnp.logical_not(magic_guard) & \
            jnp.logical_not(poison_heal)
        # Heatproof halves the burn's sixteenth.
        burn_dmg = jnp.where(burn, fraction_of_max(state, side, i, 1,
                                                   jnp.where(ab == A.HEATPROOF, 32, 16)), 0)
        psn_dmg = jnp.where(psn, fraction_of_max(state, side, i, 1, 8), 0)
        # Showdown floors the sixteenth first and then multiplies by the stage.
        tox_dmg = jnp.where(tox, fraction_of_max(state, side, i, 1, 16) *
                            jnp.clip(counter, 1, 15), 0)
        state, _ = damage_pokemon(state, side, i, burn_dmg + psn_dmg + tox_dmg)
        # Poison Heal turns poison into recovery.
        state, _ = heal_pokemon(
            state, side, i,
            jnp.where(poison_heal & ((status == C.PSN) | (status == C.TOX)) & alive,
                      fraction_of_max(state, side, i, 1, 8), 0))
        # The count climbs even when Poison Heal or Magic Guard takes the damage.
        return state._replace(status_turns=set_at(
            state.status_turns, (side, i), jnp.minimum(counter + 1, 16),
            when=(status == C.TOX) & alive))

    # Order 12-14: Curse, binding moves, Salt Cure, Syrup Bomb.
    def lingering(side, i, alive, state):
        magic_guard = state.ability[side, i] == A.MAGICGUARD
        types = active_types(state, side)
        cursed = (state.volatiles[side, C.V_CURSE] > 0) & alive & jnp.logical_not(magic_guard)
        state, _ = damage_pokemon(state, side, i,
                                  jnp.where(cursed, fraction_of_max(state, side, i, 1, 4), 0))
        # Binding moves chip an eighth each residual -- Showdown counts the
        # turn down first and ends it there, so not on the last.
        alive = state.hp[side, i] > 0
        # (Not once the binder has fainted, earlier in the residuals.)
        binder_up = state.hp[1 - side, act(state, 1 - side)] > 0
        bound = (state.volatiles[side, C.V_PARTIALLYTRAPPED] > 1) & alive & binder_up & \
            jnp.logical_not(magic_guard)
        state, _ = damage_pokemon(state, side, i,
                                  jnp.where(bound, fraction_of_max(state, side, i, 1, 8), 0))
        # Salt Cure: 1/4 against Water and Steel, else 1/8.
        alive = state.hp[side, i] > 0
        salted = (state.volatiles[side, C.V_SALTCURE] > 0) & alive & \
            jnp.logical_not(magic_guard)
        weak = jnp.any(types == C.WATER) | jnp.any(types == C.STEEL)
        state, _ = damage_pokemon(
            state, side, i,
            jnp.where(salted, fraction_of_max(state, side, i, 1, jnp.where(weak, 4, 8)), 0))
        # Syrup Bomb takes a stage of Speed each residual it has left.
        syrup = (state.volatiles[side, C.V_SYRUPBOMB] > 1) & (state.hp[side, i] > 0)
        state, _ = apply_boosts(state, side, jnp.where(syrup, boost_delta((C.B_SPE, -1)), 0),
                                from_opponent=True)
        return state

    # Order 16-26: Encore ending on an empty move, Yawn, Speed Boost.
    def late_volatiles(side, i, alive, state):
        ab = state.ability[side, i]
        words = residual_words[side]
        encore_pp = state.pp[side, i, jnp.maximum(state.encore_slot[side], 0)]
        state = state._replace(volatiles=set_at(
            state.volatiles, (side, C.V_ENCORE), 0,
            (state.volatiles[side, C.V_ENCORE] > 0) & (encore_pp <= 0)))
        # Yawn puts its target to sleep as it runs out.
        drowsy = (state.volatiles[side, C.V_YAWN] == 1) & alive
        state, _ = set_status(data, state, side,
                              jnp.where(drowsy, jnp.int8(C.SLP), jnp.int8(C.STATUS_NONE)),
                              words[1])
        # Speed Boost, though not on the turn its holder came in.
        state, _ = apply_boosts(
            state, side,
            jnp.where((ab == A.SPEEDBOOST) & alive & jnp.logical_not(
                state.switched_this_turn[side]),
                boost_delta((C.B_SPE, 1)), 0))
        return state

    # Order 28-29: Bad Dreams, Harvest and the berries, Flame / Toxic Orb, and
    # the formes that follow the turn.
    def end_of_turn(side, i, alive, state):
        ab = state.ability[side, i]
        w = effective_weather(state)
        words = residual_words[side]
        # Bad Dreams: a sleeping foe loses an eighth.
        other = 1 - side
        oi = act(state, other)
        nightmare = (ab == A.BADDREAMS) & alive & (state.hp[other, oi] > 0) & \
            ((state.status[other, oi] == C.SLP) | (state.ability[other, oi] == A.COMATOSE)) & \
            (state.ability[other, oi] != A.MAGICGUARD)
        state, _ = damage_pokemon(state, other, oi, jnp.where(
            nightmare, fraction_of_max(state, other, oi, 1, 8), 0))
        # Harvest regrows the last berry eaten: always in sun, else half the
        # time. Cud Chew eats the same berry again, a turn after the first time.
        alive = state.hp[side, i] > 0
        sunny = (w == C.SUN) | (w == C.HARSH_SUN)
        regrows = (ab == A.HARVEST) & alive & (state.item[side, i] == 0) & \
            (state.last_item[side, i] != 0) & (sunny | (uniform(words[1]) < 0.5))
        state = state._replace(
            item=set_at(state.item, (side, i), state.last_item[side, i], regrows),
            last_item=set_at(state.last_item, (side, i), 0, regrows))
        # Either way, a side eats at most once here: Cud Chew's second helping,
        # or its held berry if the residuals have met its trigger (Sitrus after
        # poison) -- Showdown's Update after them, so after Harvest too. Only a
        # berry first eaten here restarts Cud Chew at one turn.
        chewing = (state.cud_turns[side] > 0) & alive
        chewed = chewing & (state.cud_turns[side] == 1)
        second_helping = state.cud_berry[side]
        state = state._replace(
            cud_turns=set_at(state.cud_turns, side, state.cud_turns[side] - 1, chewing),
            cud_berry=set_at(state.cud_berry, side, 0, chewed))
        state = eat_berry(data, state, side,
                          jnp.where(chewed, second_helping, state.item[side, i]),
                          when=chewed | berry_wants(state, side),
                          consume=jnp.logical_not(chewed), cud=jnp.where(chewed, 0, 1))
        # Flame Orb and Toxic Orb, after the burn and poison damage, so the holder
        # takes none until the next turn. Safeguard does not stop a status the
        # holder gives itself.
        held = state.item[side, i]
        orb = jnp.where(held == I.FLAMEORB, jnp.int8(C.BRN),
                        jnp.where(held == I.TOXICORB, jnp.int8(C.TOX), jnp.int8(C.STATUS_NONE)))
        state, _ = set_status(data, state, side,
                              jnp.where(state.hp[side, i] > 0, orb, jnp.int8(C.STATUS_NONE)),
                              words[1], self_inflicted=True)
        # Morpeko's Hunger Switch flips between Full Belly and Hangry (not once
        # Terastallized); Minior's Shields Down drops its shell at half HP and
        # grows it back above.
        alive = state.hp[side, i] > 0
        species = state.species[side, i]
        morpeko = (ab == A.HUNGERSWITCH) & alive & jnp.logical_not(state.terastallized[side, i]) & (
            (species == species_index("morpeko")) | (species == species_index("morpekohangry")))
        base = state.base_species[side, i]
        minior = (ab == A.SHIELDSDOWN) & alive & (data["species_num"][species] == 774) & \
            jnp.logical_not(state.transformed[side])
        shell = state.hp[side, i].astype(jnp.int32) * 2 > state.maxhp[side, i].astype(jnp.int32)
        return forme_change(data, state, side, i, jnp.where(
            morpeko, jnp.where(species == species_index("morpeko"),
                               species_index("morpekohangry"), species_index("morpeko")),
            jnp.where(shell, species_index("miniormeteor"), jnp.where(base >= 0, base, species))),
            when=morpeko | minior)

    for phase in (weather_and_slots, healing, leech_seed, status_damage, lingering,
                  late_volatiles):
        state = phased(phase, state)

    # Tick down timed field effects and per-slot volatiles -- Showdown's orders
    # 15-27, so after the Yawn and Speed Boost above and before the berries and
    # Harvest below: a Sitrus Berry held back by Heal Block is eaten the moment
    # the block ends. Terrain, unlike the
    # weather, ends after its last turn's Grassy Terrain healing.
    terrain_turns = dec(state.terrain_turns)
    state = state._replace(
        terrain=jnp.where(terrain_turns <= 0, jnp.int8(C.TERRAIN_NONE), state.terrain),
        terrain_turns=terrain_turns,
        trick_room=dec(state.trick_room), gravity=dec(state.gravity),
        side_conditions=jnp.where(
            jnp.arange(C.NUM_SIDE_CONDITIONS) >= C.SC_REFLECT,
            dec(state.side_conditions), state.side_conditions),
    )

    # Volatiles that count down on their own; Protect and Flinch last one turn.
    timed = np.zeros(C.NUM_VOLATILES, bool)
    # (A rampage's lock counts uses, not turns: `execute_move` keeps it.)
    timed[[C.V_TAUNT, C.V_ENCORE, C.V_DISABLE, C.V_YAWN, C.V_MAGNETRISE,
           C.V_THROATCHOP, C.V_TORMENT, C.V_PERISHSONG,
           C.V_PARTIALLYTRAPPED, C.V_SYRUPBOMB, C.V_HEALBLOCK]] = True
    state = state._replace(volatiles=jnp.where(timed, dec(state.volatiles),
                                               state.volatiles))
    # Slow Start counts only turns its holder began on the field (Showdown's
    # `activeTurns`): not the turn it switched in on, but a lead's first turn
    # and a replacement's first full turn both count.
    counting = (jnp.arange(C.NUM_VOLATILES) == C.V_SLOWSTART)[None, :] & \
        jnp.logical_not(state.switched_this_turn)[:, None]
    state = state._replace(volatiles=jnp.where(counting, dec(state.volatiles),
                                               state.volatiles))
    single_turn = np.zeros(C.NUM_VOLATILES, bool)
    single_turn[[C.V_PROTECT, C.V_FLINCH, C.V_ROOST, C.V_HELPINGHAND, C.V_ENDURE,
                 C.V_FOCUSPUNCH, C.V_BEAKBLAST, C.V_MATBLOCK]] = True
    state = state._replace(volatiles=jnp.where(single_turn, jnp.int8(0),
                                               state.volatiles))
    # A rampage ends at any residual its user is asleep for (Showdown's
    # `lockedmove`), with no fatigue: Yawn mid-Outrage frees the moves.
    asleep = jnp.stack([slot_get(state.status, s, act(state, s)) == C.SLP
                        for s in range(C.NUM_PLAYERS)])
    raging_asleep = asleep & (state.volatiles[:, C.V_LOCKEDMOVE] > 0)
    state = state._replace(
        volatiles=jnp.where((jnp.arange(C.NUM_VOLATILES) == C.V_LOCKEDMOVE)[None, :] &
                            raging_asleep[:, None], jnp.int8(0), state.volatiles),
        locked_slot=jnp.where(raging_asleep, jnp.int8(-1), state.locked_slot))

    state = phased(end_of_turn, state)

    # White Herb checks once more at the end of the turn (a Syrup Bomb drop),
    # and the sun or terrain may have ended.
    state = white_herb(state)
    state = paradox_update(state)

    # Soul-Heart counts whatever fainted during the residuals.
    return soul_heart(state, alive_before)


def resolve_phazing(data, state, key):
    """Drag in a random replacement for anyone Whirlwind-style forced out.

    Unlike a self-switch there is no choice to make, so this happens inline
    rather than becoming a decision point. A side with nothing left on the bench
    simply stays in.

    Written as a `fori_loop` rather than a Python loop over the two sides: the
    body holds a `switch_to`, which is a large piece of program (hazards, entry
    abilities), and the loop compiles it once instead of once per side. The
    random scores are drawn before the loop so that no hash sits inside it.
    """
    all_scores = jax.random.uniform(key, (C.NUM_PLAYERS, C.TEAM_SIZE))

    def one_side(side, st):
        active = st.active[side].astype(jnp.int32)
        eligible = (st.hp[side] > 0) & (jnp.arange(C.TEAM_SIZE) != active)
        # Uniform over the eligible slots: score them randomly and take the best.
        scores = jnp.where(eligible, all_scores[side], -1.0)
        slot = jnp.argmax(scores)
        return select_state(st.phazed[side] & jnp.any(eligible),
                            switch_to(data, st, side, slot), st)

    state = jax.lax.fori_loop(0, C.NUM_PLAYERS, one_side, state)
    return state._replace(phazed=jnp.zeros((C.NUM_PLAYERS,), bool))


# --- win condition -----------------------------------------------------------

def check_winner(state):
    alive = jnp.sum(state.hp > 0, axis=1)
    p0_out = alive[0] == 0
    p1_out = alive[1] == 0
    # Both out at once: Showdown (Gen 5+) gives it to the side that fainted last.
    winner = jnp.where(p0_out & p1_out,
                       jnp.where(state.last_faint >= 0, state.last_faint, 2).astype(jnp.int8),
                       jnp.where(p1_out, jnp.int8(0),
                                 jnp.where(p0_out, jnp.int8(1), jnp.int8(-1))))
    return winner


# --- one action --------------------------------------------------------------

def is_attacking_action(data, state, side, action):
    """True if `side` is about to use a damaging move (not a switch or status move).

    Sucker Punch needs this about its target. A Pokemon recharging is not.
    """
    ai = act(state, side)
    mid = jnp.maximum(state.moves[side, ai, move_slot(action)], 0)
    return jnp.logical_not(is_switch(action)) & \
        (data["move_category"][mid] != C.CAT_STATUS) & \
        (state.volatiles[side, C.V_RECHARGE] <= 0)


def run_action(data, state, side, action, moves_first, words, target_attacking=True,
               target_acting=True, struggling=None):
    """Use one side's chosen move, if its Pokemon is still able to act.

    Switches are not actions here: they all resolve before any move, in the
    switch-in loop of `run_turn`, and a switch action is a pass. So is a negative
    one -- used for the opponent of a self-switch, whose locked-in move is held
    over until the replacement is in. `words` is the move's `[MOVE_WORDS]` random
    words (see `execute_move`).
    """
    i = act(state, side)
    able = (slot_get(state.hp, side, i) > 0) & (action >= 0) & \
        jnp.logical_not(is_switch(action))
    moved = execute_move(data, state, side, move_slot(action), moves_first, words,
                         target_attacking, target_acting, struggling)

    # Computed either way and selected, as a batched `lax.cond` would do; see
    # `state.select_state` for why it is not one.
    return select_state(able, moved, state)


# Ogerpon Terastallizes into its "-Tera" forme, whose ability, Embody Aspect,
# raises one stat as it does: (forme, Tera forme, ability, stage raised).
_OGERPON_TERA = (
    ("ogerpon", "ogerpontealtera", "embodyaspectteal", C.B_SPE),
    ("ogerponwellspring", "ogerponwellspringtera", "embodyaspectwellspring", C.B_SPD),
    ("ogerponhearthflame", "ogerponhearthflametera", "embodyaspecthearthflame", C.B_ATK),
    ("ogerponcornerstone", "ogerponcornerstonetera", "embodyaspectcornerstone", C.B_DEF),
)


def terastallize(data, state, side, when):
    """Terastallize `side`'s active Pokemon, with the formes that come with it.

    Showdown resolves Terastallization as its own action, after switches and
    before any move, so a Pokemon that moves second is already its Tera type
    when it is hit -- and still spends the side's Terastallization if it is
    knocked out before it moves.
    """
    i = act(state, side)
    tera = when & jnp.logical_not(state.terastallized[side, i]) & \
        (slot_get(state.hp, side, i) > 0)
    state = state._replace(terastallized=set_at(state.terastallized, (side, i), True,
                                                when=tera),
                           tera_used=set_at(state.tera_used, side, True, when=tera))
    species = state.species[side, i]
    # Terapagos-Terastal becomes its Stellar Form, whose Teraform Zero clears the
    # weather and terrain.
    stellar = tera & (species == species_index("terapagosterastal"))
    target = jnp.int16(species_index("terapagosstellar"))
    ability = jnp.int16(A.TERAFORMZERO)
    embody = jnp.zeros(C.NUM_BOOSTS, jnp.int32)
    ogerpon = jnp.bool_(False)
    for base, tera_forme, aspect, stage in _OGERPON_TERA:
        hit = tera & (species == species_index(base))
        ogerpon = ogerpon | hit
        target = jnp.where(hit, jnp.int16(species_index(tera_forme)), target)
        ability = jnp.where(hit, jnp.int16(ABILITY_IDX[aspect]), ability)
        embody = jnp.where(hit, jnp.asarray(boost_delta((stage, 1))), embody)
    state = forme_change(data, state, side, i, target, when=stellar | ogerpon,
                         permanent=True, ability=ability)
    state, _ = apply_boosts(state, side, embody)
    # A Morpeko that Terastallizes Hangry stays Hangry for good, on the bench
    # too: Showdown makes the forme its base species.
    hangry = tera & (species == species_index("morpekohangry")) & \
        jnp.logical_not(state.transformed[side])
    state = state._replace(
        base_species=slot_set(state.base_species, side, i, species, hangry),
        weather=jnp.where(stellar, jnp.int8(C.WEATHER_NONE), state.weather),
        weather_turns=jnp.where(stellar, jnp.int8(0), state.weather_turns),
        terrain=jnp.where(stellar, jnp.int8(C.TERRAIN_NONE), state.terrain),
        terrain_turns=jnp.where(stellar, jnp.int8(0), state.terrain_turns))
    return paradox_update(state)


# --- the public step ---------------------------------------------------------

def start_turn(state, when=True):
    """Clear the per-turn scratch fields. `when=False` leaves them alone.

    Resuming a suspended turn must not reset these -- the first half of the turn
    already happened.
    """
    pick = lambda new, old: jnp.where(when, new, old)
    # Stat changes from the leads' entry (Intimidate) still count on turn 1:
    # Showdown only clears those flags from the second turn on.
    stats_pick = lambda new, old: jnp.where(when & (state.turn >= 1), new, old)
    return state._replace(
        moved_this_turn=pick(jnp.zeros((C.NUM_PLAYERS,), bool), state.moved_this_turn),
        switched_this_turn=pick(jnp.zeros((C.NUM_PLAYERS,), bool),
                                state.switched_this_turn),
        damage_taken=pick(jnp.zeros((C.NUM_PLAYERS,), jnp.int16), state.damage_taken),
        damage_category=pick(jnp.full((C.NUM_PLAYERS,), -1, jnp.int8),
                             state.damage_category),
        stats_lowered=stats_pick(jnp.zeros((C.NUM_PLAYERS,), bool), state.stats_lowered),
        stats_raised=stats_pick(jnp.zeros((C.NUM_PLAYERS,), bool), state.stats_raised),
        fusion_last=pick(jnp.int8(0), state.fusion_last),
        pass_mode=pick(jnp.zeros((C.NUM_PLAYERS,), jnp.int8), state.pass_mode),
        turn=pick(state.turn + 1, state.turn),
    )


def run_turn(data, state, actions, key, resuming):
    """Resolve a turn, or the second half of one that a self-switch suspended.

      fresh turn   slot A = the first mover, slot B = the second mover
      resuming     slot A does nothing, slot B runs the move held over from
                   before the switch

    `resuming` is traced, so both readings are always built and selected between.

    Switching comes first. A switch outranks every move, so the switchers go in
    turn order ahead of both slots, and on a resume the replacements that were
    owed come in before the held-over move; one loop covers both. The two slots
    then run through a second `fori_loop` rather than two calls. `run_action`
    contains the whole move engine -- by far the largest thing in the program --
    and the loop puts one copy in the compiled graph instead of two. That is
    what keeps compilation tractable; see the note in `psjax/__init__.py`.
    """
    k_order, k_act, k_res, k_ph = jax.random.split(key, 4)
    # Both slots' random words in one hash, drawn outside the loop.
    move_words = random_words(k_act, (2, MOVE_WORDS))
    state = start_turn(state, when=jnp.logical_not(resuming))
    first = turn_order(data, state, actions, k_order)
    second = 1 - first
    # A Custap Berry that moved its holder up is eaten as the order is settled
    # (and remembered, for Harvest).
    for side in range(C.NUM_PLAYERS):
        ai = act(state, side)
        custap = jnp.logical_not(resuming) & custap_fires(data, state, side, actions[side])
        state = lose_item(state, side, ai, custap)
        state = state._replace(last_item=slot_set(state.last_item, side, ai,
                                                  jnp.int16(I.CUSTAPBERRY), custap))

    # Struggle is settled when the move is chosen: whoever has no usable move
    # now struggles this turn, whatever happens to its moves before it acts.
    struggles = jnp.stack([jnp.logical_not(jnp.any(usable_moves(data, state, s)))
                           for s in range(C.NUM_PLAYERS)])

    # Sucker Punch checks whether its target is about to attack.
    attacks = jnp.stack([is_attacking_action(data, state, 0, actions[0]),
                         is_attacking_action(data, state, 1, actions[1])])
    pending = resuming & (state.pending_side >= 0)

    # Switches come in two kinds, and Showdown orders their entry effects
    # differently. A switch chosen as the turn's action is made and its
    # newcomer's entry effects run before the next switch (so the faster
    # switcher's Intimidate hits the slower side's outgoing Pokemon). Replacements
    # answering a prompt (Showdown's `instaswitch`) are all made first, then the
    # newcomers arrive fastest first. With p1's replacement swapped in ahead of
    # it, one loop body -- one swap, one arrival -- covers both in two steps:
    #   chosen        (swap first, arrive first) (swap second, arrive second)
    #   replacements  [swap p1] (swap p2, arrive faster) (arrive slower)
    # (Arrivals are the costly part; a third step for the replacements' would
    # be one more on every turn.)
    def switch_in(st, came, side, when):
        action = actions[side]
        owed = resuming & st.force_switch[side]
        chosen = jnp.logical_not(resuming) & is_switch(action) & \
            (slot_get(st.hp, side, act(st, side)) > 0)
        reviving = when & owed & (st.pass_mode[side] == C.PASS_REVIVE)
        moving = when & (owed | chosen) & jnp.logical_not(reviving)
        swapped, valid = swap_in(data, st, side, switch_slot(action))
        st = select_state(moving, swapped, st)
        st = select_state(reviving, revive(st, side, switch_slot(action)), st)
        came = jnp.where(jnp.arange(C.NUM_PLAYERS) == side, came | (moving & valid), came)
        return st, came

    def switch_step(k, carry):
        st, came, arrival_first = carry
        side = jnp.where(resuming, 1, jnp.where(k == 0, first, second))
        st, came = switch_in(st, came, side, jnp.logical_not(resuming) | (k == 0))
        # Replacements arrive once both are in, fastest first (Trick Room
        # reversing it); a chosen switch's newcomer arrives straight away.
        arrival_first = jnp.where(resuming & (k == 0), speed_order(st), arrival_first)
        coming = jnp.where(resuming, jnp.where(k == 0, arrival_first, 1 - arrival_first), side)
        return arrive(data, st, coming, came[coming]), came, arrival_first

    state, came = switch_in(state, jnp.zeros(C.NUM_PLAYERS, bool), 0, resuming)
    state, _, _ = jax.lax.fori_loop(0, 2, switch_step, (state, came, jnp.int32(0)))

    # Terastallization, in turn order, after the switches and before any move.
    # A held-over move was Terastallized for when the turn began.
    def tera_in(j, st):
        side = jnp.where(j == 0, first, second)
        wants = jnp.logical_not(resuming) & wants_tera(actions[side])
        return terastallize(data, st, side, wants)

    state = jax.lax.fori_loop(0, C.NUM_PLAYERS, tera_in, state)

    # Focus Punch and Beak Blast start their charge before anyone moves.
    for side in range(C.NUM_PLAYERS):
        ai = act(state, side)
        mid = jnp.maximum(state.moves[side, ai, move_slot(actions[side])], 0)
        eff = data["move_effect_cb"][mid]
        using = jnp.logical_not(resuming) & jnp.logical_not(is_switch(actions[side]))
        for name, vol in (("focuspunch", C.V_FOCUSPUNCH), ("beakblast", C.V_BEAKBLAST)):
            state = state._replace(volatiles=set_at(
                state.volatiles, (side, vol), 1, using & (eff == E.EFFECT_HANDLERS.index(name))))

    def slot(i, carry):
        st, suspend, suspend_b = carry
        is_a = i == 0

        # Slot B is the second mover, or the move held over from a suspension.
        # A switch already happened above, so it is a pass here.
        b_side = jnp.where(pending, st.pending_side.astype(jnp.int32), second)
        # Roared or Dragon Tailed out by the first mover (both used -6
        # priority, say), the second mover never gets to act: Showdown drags
        # it out on the spot. The drag itself happens in `resolve_phazing`.
        dragged = st.phazed[second] & _has_bench(st)[second]
        b_action = jnp.where(
            resuming,
            jnp.where(pending, st.pending_action.astype(jnp.int32), jnp.int32(-1)),
            jnp.where(suspend | dragged, jnp.int32(-1), actions[second]))
        side = jnp.where(is_a, first, b_side)
        action = jnp.where(is_a, jnp.where(resuming, jnp.int32(-1), actions[first]),
                           b_action)
        # Whoever holds a move over has already seen its opponent act, so from
        # its point of view nothing is "about to attack".
        target_attacks = jnp.where(is_a, attacks[second],
                                   jnp.where(resuming, jnp.bool_(False), attacks[first]))
        # Whether anything acts after this (Protect fails if not): only slot A
        # can be followed, by a second mover that chose a move.
        target_acts = is_a & jnp.logical_not(resuming) & jnp.logical_not(suspend) & \
            jnp.logical_not(is_switch(actions[second]))
        # Once a side has nobody left, the battle is over and nothing else acts.
        action = jnp.where(_battle_over(st), jnp.int32(-1), action)
        st = run_action(data, st, side, action, is_a, move_words[i], target_attacks,
                        target_acts, struggles[side])

        # A self-switch suspends the turn: its user picks a replacement before
        # the opponent's already-locked move resolves, so that move lands on
        # whoever comes in rather than on the Pokemon that left. Decided after
        # slot A has run, and read by slot B on the next iteration.
        after_a = jnp.logical_not(resuming) & st.force_switch[first] & \
            _has_bench(st)[first] & jnp.logical_not(_battle_over(st))
        # A self-switch by slot B suspends what is left of the turn too: its
        # replacement comes in before the residuals, which then fall on it.
        after_b = (action >= 0) & st.force_switch[side] & _has_bench(st)[side] & \
            jnp.logical_not(_battle_over(st))
        return (st, jnp.where(is_a, after_a, suspend),
                jnp.where(is_a, jnp.bool_(False), after_b))

    state, suspend, suspend_b = jax.lax.fori_loop(
        0, 2, slot, (state, jnp.bool_(False), jnp.bool_(False)))
    b_side = jnp.where(pending, state.pending_side.astype(jnp.int32), second)

    # Resolved once, after both actions: every phazing move has negative
    # priority, so its target has already moved and the ordering is the same.
    state = resolve_phazing(data, state, k_ph)

    # Residuals close out a turn: skipped when one is suspended, and skipped on a
    # resume that was only bringing in a fainted Pokemon's replacement, since
    # that turn already ended.
    # A suspension after slot B holds no move over, only the residuals: it
    # records slot B's side with action -1, which the resume runs as a pass.
    # (And no residuals once the battle is decided: Showdown stops at the KO.)
    finishing = jnp.where(resuming, pending, jnp.logical_not(suspend)) & \
        jnp.logical_not(suspend_b) & jnp.logical_not(_battle_over(state))
    # A binder knocked out during the turn lets go before its last chip.
    state = _release_bound(state)
    state = residuals(data, state, k_res, when=finishing)

    state = state._replace(
        pending_side=jnp.where(suspend, second,
                               jnp.where(suspend_b, b_side, -1)).astype(jnp.int8),
        pending_action=jnp.where(suspend, actions[second],
                                 jnp.where(suspend_b, -1, 0)).astype(jnp.int8))
    state = _release_bound(state)

    # Mid-turn, only the self-switcher picks a replacement. A Pokemon knocked out
    # in the meantime is replaced once the turn is over, as Showdown does.
    return _request_replacements(state, when=jnp.logical_not(suspend | suspend_b))


def _release_bound(state):
    """A Pokemon that has fainted lets go of whatever it held its foe with --
    Mean Look's trap, Attract, a binding move -- as Showdown clears linked
    volatiles on fainting, rather than once its replacement comes in."""
    source_down = jnp.logical_not(actives_alive(state))[::-1]
    return state._replace(volatiles=jnp.where(
        jnp.asarray(_SOURCE_BOUND_MASK)[None, :] & source_down[:, None], jnp.int8(0),
        state.volatiles))


def _battle_over(state):
    """Has either side run out of Pokemon?"""
    return jnp.any(jnp.sum(state.hp > 0, axis=1) == 0)


def _has_bench(state):
    """[P] bool: can this side answer its prompt? A healthy Pokemon that is not
    already out to switch to -- or, for Revival Blessing, a fainted one to revive."""
    benched = jnp.arange(C.TEAM_SIZE)[None, :] != state.active.astype(jnp.int32)[:, None]
    reviving = (state.pass_mode == C.PASS_REVIVE)[:, None]
    wanted = jnp.where(reviving, state.hp <= 0, state.hp > 0)
    return jnp.sum(wanted & benched, axis=1) > 0


def revive(state, side, slot):
    """Revival Blessing's answer: `slot`, if fainted, comes back at half HP with
    no status -- otherwise the first fainted Pokemon does. The user stays in.
    Fainting ended any Terastallization (Showdown deletes it then), though the
    side's is still spent."""
    slot = jnp.clip(slot, 0, C.TEAM_SIZE - 1)
    fainted = state.hp[side] <= 0
    slot = jnp.where(fainted[slot], slot, jnp.argmax(fainted))
    half = jnp.maximum(state.maxhp[side, slot].astype(jnp.int32) // 2, 1)
    return state._replace(
        hp=set_at(state.hp, (side, slot), half),
        status=set_at(state.status, (side, slot), jnp.int8(C.STATUS_NONE)),
        terastallized=set_at(state.terastallized, (side, slot), False),
        force_switch=set_at(state.force_switch, side, False),
        pass_mode=set_at(state.pass_mode, side, 0))


def _request_replacements(state, when=True):
    """Ask for a replacement from anyone who needs one and can provide one.

    `when=False` asks only those who already owe one (a self-switch), leaving
    fainted Pokemon to be replaced when the turn is over.
    """
    fainted = state.hp[jnp.arange(C.NUM_PLAYERS), state.active.astype(jnp.int32)] <= 0
    return state._replace(
        force_switch=(state.force_switch | (fainted & when)) & _has_bench(state),
        fainted_count=jnp.sum(state.hp <= 0, axis=1).astype(jnp.int8))


def step(state: BattleState, actions, data=None) -> BattleState:
    """Advance the battle by one decision point.

    `actions` is a `[2]` int array using the `ACTION_*` encoding. Its meaning
    depends on `state.phase`: a move or switch during `PHASE_MOVE`, a replacement
    slot during `PHASE_SWITCH`.
    """
    if data is None:
        data = load_data()
    actions = jnp.asarray(actions, jnp.int32)
    key, subkey = jax.random.split(state.key)
    state = state._replace(key=key, events=jnp.full_like(state.events, -1))

    # A switch prompt means the chosen replacements come in before anything else;
    # `run_turn` then either finishes a suspended turn or does nothing further.
    resuming = state.phase == C.PHASE_SWITCH
    state = run_turn(data, state, actions, subkey, resuming)

    winner = check_winner(state)
    needs_switch = jnp.any(state.force_switch)
    phase = jnp.where(winner >= 0, jnp.int8(C.PHASE_END),
                      jnp.where(needs_switch, jnp.int8(C.PHASE_SWITCH),
                                jnp.int8(C.PHASE_MOVE)))
    return state._replace(winner=winner, phase=phase)
