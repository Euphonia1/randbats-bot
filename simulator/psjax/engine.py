"""Turn resolution: ordering, switching, residuals, and the public `step`.

The battle is a two-phase state machine:

    PHASE_MOVE    both players submit a move or a switch; the engine resolves the
                  whole turn (both actions, then end-of-turn residuals)
    PHASE_SWITCH  one or both players owe a replacement for a fainted Pokemon
    PHASE_END     the battle is over; `winner` says who took it

Known deviation: a self-switch (U-turn, Volt Switch) or a forced switch
(Whirlwind) is resolved at the *end* of the turn rather than immediately. In a
real battle the replacement arrives before the opponent moves; here the opponent
attacks the Pokemon that used the move. Everything else in the turn follows
Showdown's ordering.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import consts as C
from .data import load_data
from .hooks import A, I
from .mechanics import (act, active_types, apply_boosts, cure_status,
                        damage_pokemon, effective_speed, effective_weather,
                        fraction_of_max, heal_pokemon, is_grounded, set_status)
from .moves import execute_move
from .state import BattleState, reset_slot_state


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
    """Switches outrank every move; otherwise move priority plus ability bonuses."""
    ai = act(state, side)
    slot = move_slot(action)
    mid = jnp.maximum(state.moves[side, ai, slot], 0)
    pri = data["move_priority"][mid].astype(jnp.int32)

    ab = state.ability[side, ai]
    category = data["move_category"][mid]
    flags = data["move_flags"][mid]
    is_heal = (flags & (1 << C.FLAG_BITS["heal"])) != 0
    pri = pri + jnp.where((ab == A.PRANKSTER) & (category == C.CAT_STATUS), 1, 0)
    pri = pri + jnp.where((ab == A.TRIAGE) & is_heal, 3, 0)
    pri = pri + jnp.where((ab == A.GALEWINGS) &
                          (data["move_type"][mid] == C.FLYING) &
                          (state.hp[side, ai] == state.maxhp[side, ai]), 1, 0)
    return jnp.where(is_switch(action), jnp.int32(10), pri)


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

    # Toxic Spikes: poison (or badly poison at two layers); Poison types absorb.
    tspikes = sc[C.SC_TOXICSPIKES].astype(jnp.int32)
    is_poison = jnp.any(types == C.POISON)
    absorb = (tspikes > 0) & grounded & is_poison
    state = state._replace(side_conditions=state.side_conditions.at[
        side, C.SC_TOXICSPIKES].set(jnp.where(absorb, jnp.int8(0), sc[C.SC_TOXICSPIKES])))
    poison_status = jnp.where(tspikes >= 2, jnp.int8(C.TOX), jnp.int8(C.PSN))
    should_poison = (tspikes > 0) & grounded & jnp.logical_not(is_poison) & \
                    jnp.logical_not(boots)
    state, _ = jax.lax.cond(
        should_poison,
        lambda s: set_status(data, s, side, poison_status, jax.random.PRNGKey(0)),
        lambda s: (s, jnp.bool_(False)), state)

    # Sticky Web drops Speed on a grounded arrival.
    web = (sc[C.SC_STICKYWEB] > 0) & grounded & jnp.logical_not(boots)
    state, _ = apply_boosts(state, side,
                            jnp.where(web, jnp.zeros(7, jnp.int32).at[C.B_SPE].set(-1), 0),
                            from_opponent=True)
    return state


def apply_switch_in_ability(data, state, side):
    """Entry abilities: weather and terrain setters, Intimidate, Download, ..."""
    i = act(state, side)
    ab = state.ability[side, i]
    other = 1 - side

    weather = data["ability_weather"][ab]
    state = state._replace(
        weather=jnp.where(weather > 0, weather, state.weather),
        weather_turns=jnp.where(weather > 0, jnp.int8(5), state.weather_turns))
    terrain = data["ability_terrain"][ab]
    state = state._replace(
        terrain=jnp.where(terrain > 0, terrain, state.terrain),
        terrain_turns=jnp.where(terrain > 0, jnp.int8(5), state.terrain_turns))

    # Intimidate drops the opponent's Attack.
    oi = act(state, other)
    intimidate = (ab == A.INTIMIDATE) & (state.hp[other, oi] > 0)
    state, applied = apply_boosts(
        state, other,
        jnp.where(intimidate, jnp.zeros(7, jnp.int32).at[C.B_ATK].set(-1), 0),
        from_opponent=True)

    # Defiant and Competitive answer any stat drop from the opponent.
    o_ab = state.ability[other, oi]
    dropped = jnp.any(applied < 0)
    state, _ = apply_boosts(
        state, other,
        jnp.where(dropped & (o_ab == A.DEFIANT),
                  jnp.zeros(7, jnp.int32).at[C.B_ATK].set(2), 0))
    state, _ = apply_boosts(
        state, other,
        jnp.where(dropped & (o_ab == A.COMPETITIVE),
                  jnp.zeros(7, jnp.int32).at[C.B_SPA].set(2), 0))

    # Intrepid Sword / Dauntless Shield boost the arriving Pokemon once.
    state, _ = apply_boosts(
        state, side,
        jnp.where(ab == A.INTREPIDSWORD, jnp.zeros(7, jnp.int32).at[C.B_ATK].set(1), 0))
    state, _ = apply_boosts(
        state, side,
        jnp.where(ab == A.DAUNTLESSSHIELD, jnp.zeros(7, jnp.int32).at[C.B_DEF].set(1), 0))
    return state


def switch_to(data, state, side, slot):
    """Bring `slot` in for `side`, clearing slot state and running entry effects."""
    slot = slot.astype(jnp.int32)
    valid = (state.hp[side, slot] > 0) & (slot != act(state, side))
    slot = jnp.where(valid, slot, act(state, side))

    # Regenerator heals the departing Pokemon by a third.
    old = act(state, side)
    regen = state.ability[side, old] == A.REGENERATOR
    state, _ = heal_pokemon(state, side, old,
                            jnp.where(regen & valid,
                                      fraction_of_max(state, side, old, 1, 3), 0))
    # Natural Cure clears status on the way out.
    natural = state.ability[side, old] == A.NATURALCURE
    state = jax.lax.cond(natural & valid,
                         lambda s: cure_status(s, side, old), lambda s: s, state)

    state = reset_slot_state(state, side)
    state = state._replace(
        active=state.active.at[side].set(slot.astype(jnp.int8)),
        switched_this_turn=state.switched_this_turn.at[side].set(True),
        force_switch=state.force_switch.at[side].set(False))
    state = apply_entry_hazards(data, state, side)
    state = apply_switch_in_ability(data, state, side)
    return state


# --- residuals ---------------------------------------------------------------

def residuals(data, state, key):
    """End-of-turn effects, in Showdown's residual order."""
    order = jnp.array([0, 1])

    def side_residual(state, side):
        i = act(state, side)
        alive = state.hp[side, i] > 0
        ab = state.ability[side, i]
        magic_guard = ab == A.MAGICGUARD
        types = active_types(state, side)
        w = effective_weather(state)

        # 1. Sandstorm chips everything but Rock, Ground and Steel.
        sand_immune = (jnp.any(types == C.ROCK) | jnp.any(types == C.GROUND) |
                       jnp.any(types == C.STEEL) | (ab == A.SANDVEIL) |
                       (ab == A.SANDRUSH) | (ab == A.SANDFORCE) |
                       (ab == A.OVERCOAT) | (state.item[side, i] == I.SAFETYGOGGLES))
        sand = (w == C.SAND) & alive & jnp.logical_not(sand_immune) & \
               jnp.logical_not(magic_guard)
        state, _ = damage_pokemon(state, side, i,
                                  jnp.where(sand, fraction_of_max(state, side, i, 1, 16), 0))

        # 2. Grassy Terrain heals grounded Pokemon.
        grassy = (state.terrain == C.GRASSY_TERRAIN) & alive & is_grounded(state, side)
        state, _ = heal_pokemon(state, side, i,
                                jnp.where(grassy, fraction_of_max(state, side, i, 1, 16), 0))

        # 3. Leftovers.
        lefto = (state.item[side, i] == I.LEFTOVERS) & alive
        state, _ = heal_pokemon(state, side, i,
                                jnp.where(lefto, fraction_of_max(state, side, i, 1, 16), 0))

        # 4. Aqua Ring and Ingrain.
        for vol in (C.V_AQUARING, C.V_INGRAIN):
            on = (state.volatiles[side, vol] > 0) & alive
            state, _ = heal_pokemon(state, side, i,
                                    jnp.where(on, fraction_of_max(state, side, i, 1, 16), 0))

        # 5. Leech Seed drains to the opponent's active slot.
        other = 1 - side
        oi = act(state, other)
        seeded = (state.volatiles[side, C.V_LEECHSEED] > 0) & alive & \
                 jnp.logical_not(magic_guard) & (state.hp[other, oi] > 0)
        drain = jnp.where(seeded, fraction_of_max(state, side, i, 1, 8), 0)
        state, dealt = damage_pokemon(state, side, i, drain)
        state, _ = heal_pokemon(state, other, oi, dealt)

        # 6. Burn and poison damage. Toxic ramps by 1/16 per turn.
        status = state.status[side, i]
        counter = state.status_turns[side, i].astype(jnp.int32)
        poison_heal = ab == A.POISONHEAL
        burn = (status == C.BRN) & alive & jnp.logical_not(magic_guard)
        psn = (status == C.PSN) & alive & jnp.logical_not(magic_guard) & \
              jnp.logical_not(poison_heal)
        tox = (status == C.TOX) & alive & jnp.logical_not(magic_guard) & \
              jnp.logical_not(poison_heal)
        burn_dmg = jnp.where(burn, fraction_of_max(state, side, i, 1, 16), 0)
        psn_dmg = jnp.where(psn, fraction_of_max(state, side, i, 1, 8), 0)
        tox_dmg = jnp.where(tox, fraction_of_max(state, side, i,
                                                 jnp.clip(counter, 1, 15), 16), 0)
        state, _ = damage_pokemon(state, side, i, burn_dmg + psn_dmg + tox_dmg)
        # Poison Heal turns poison into recovery.
        state, _ = heal_pokemon(
            state, side, i,
            jnp.where(poison_heal & ((status == C.PSN) | (status == C.TOX)) & alive,
                      fraction_of_max(state, side, i, 1, 8), 0))
        state = state._replace(status_turns=state.status_turns.at[side, i].set(
            jnp.where(tox, jnp.minimum(counter + 1, 15).astype(jnp.int8),
                      state.status_turns[side, i])))

        # 7. Salt Cure: 1/4 against Water and Steel, else 1/8.
        salted = (state.volatiles[side, C.V_SALTCURE] > 0) & alive & \
                 jnp.logical_not(magic_guard)
        weak = jnp.any(types == C.WATER) | jnp.any(types == C.STEEL)
        state, _ = damage_pokemon(
            state, side, i,
            jnp.where(salted, fraction_of_max(state, side, i, 1, jnp.where(weak, 4, 8)), 0))

        # 8. Curse.
        cursed = (state.volatiles[side, C.V_CURSE] > 0) & alive & \
                 jnp.logical_not(magic_guard)
        state, _ = damage_pokemon(state, side, i,
                                  jnp.where(cursed, fraction_of_max(state, side, i, 1, 4), 0))
        return state

    state = side_residual(state, 0)
    state = side_residual(state, 1)

    # Tick down timed field effects and per-slot volatiles.
    dec = lambda v: jnp.maximum(v.astype(jnp.int32) - 1, 0).astype(v.dtype)
    weather_turns = dec(state.weather_turns)
    terrain_turns = dec(state.terrain_turns)
    state = state._replace(
        weather=jnp.where(weather_turns <= 0, jnp.int8(C.WEATHER_NONE), state.weather),
        weather_turns=weather_turns,
        terrain=jnp.where(terrain_turns <= 0, jnp.int8(C.TERRAIN_NONE), state.terrain),
        terrain_turns=terrain_turns,
        trick_room=dec(state.trick_room), gravity=dec(state.gravity),
        side_conditions=jnp.where(
            jnp.arange(C.NUM_SIDE_CONDITIONS) >= C.SC_REFLECT,
            dec(state.side_conditions), state.side_conditions),
    )

    # Volatiles that count down on their own; Protect and Flinch last one turn.
    timed = jnp.zeros(C.NUM_VOLATILES, bool)
    for v in (C.V_TAUNT, C.V_ENCORE, C.V_DISABLE, C.V_YAWN, C.V_MAGNETRISE,
              C.V_THROATCHOP, C.V_TORMENT, C.V_SLOWSTART, C.V_PERISHSONG,
              C.V_PARTIALLYTRAPPED, C.V_LOCKEDMOVE):
        timed = timed.at[v].set(True)
    state = state._replace(volatiles=jnp.where(timed, dec(state.volatiles),
                                               state.volatiles))
    single_turn = jnp.zeros(C.NUM_VOLATILES, bool)
    for v in (C.V_PROTECT, C.V_FLINCH, C.V_ROOST, C.V_HELPINGHAND, C.V_ENDURE):
        single_turn = single_turn.at[v].set(True)
    state = state._replace(volatiles=jnp.where(single_turn, jnp.int8(0),
                                               state.volatiles))
    return state


# --- win condition -----------------------------------------------------------

def check_winner(state):
    alive = jnp.sum(state.hp > 0, axis=1)
    p0_out = alive[0] == 0
    p1_out = alive[1] == 0
    winner = jnp.where(p0_out & p1_out, jnp.int8(2),
                       jnp.where(p1_out, jnp.int8(0),
                                 jnp.where(p0_out, jnp.int8(1), jnp.int8(-1))))
    return winner


# --- one action --------------------------------------------------------------

def run_action(data, state, side, action, moves_first, key):
    """Execute one side's chosen action, if its Pokemon is still able to act."""
    i = act(state, side)
    alive = state.hp[side, i] > 0

    def do_switch(s):
        return switch_to(data, s, side, switch_slot(action))

    def do_move(s):
        # Terastallize first: it changes the type line before the move resolves.
        tera = wants_tera(action) & jnp.logical_not(s.terastallized[side, i])
        ai = act(s, side)
        s = s._replace(terastallized=s.terastallized.at[side, ai].set(
            s.terastallized[side, ai] | tera))
        return execute_move(data, s, side, move_slot(action), moves_first, key)

    return jax.lax.cond(alive,
                        lambda s: jax.lax.cond(is_switch(action), do_switch, do_move, s),
                        lambda s: s, state)


# --- the public step ---------------------------------------------------------

def start_turn(state):
    """Clear the per-turn scratch fields."""
    return state._replace(
        moved_this_turn=jnp.zeros((C.NUM_PLAYERS,), bool),
        switched_this_turn=jnp.zeros((C.NUM_PLAYERS,), bool),
        damage_taken=jnp.zeros((C.NUM_PLAYERS,), jnp.int16),
        damage_category=jnp.full((C.NUM_PLAYERS,), -1, jnp.int8),
        turn=state.turn + 1,
    )


def run_turn(data, state, actions, key):
    k_order, k_a, k_b, k_res = jax.random.split(key, 4)
    state = start_turn(state)
    first = turn_order(data, state, actions, k_order)

    state = run_action(data, state, first, actions[first], jnp.bool_(True), k_a)
    state = run_action(data, state, 1 - first, actions[1 - first],
                       jnp.bool_(False), k_b)
    state = residuals(data, state, k_res)

    # Anyone whose active fainted owes a replacement.
    fainted = state.hp[jnp.arange(C.NUM_PLAYERS), state.active.astype(jnp.int32)] <= 0
    has_bench = jnp.sum(state.hp > 0, axis=1) > 0
    state = state._replace(
        force_switch=(state.force_switch | fainted) & has_bench,
        fainted_count=jnp.sum(state.hp <= 0, axis=1).astype(jnp.int8))
    return state


def apply_forced_switches(data, state, actions):
    """Resolve pending replacements for whichever sides owe one."""
    for side in range(C.NUM_PLAYERS):
        state = jax.lax.cond(
            state.force_switch[side],
            lambda s: switch_to(data, s, side, switch_slot(actions[side])),
            lambda s: s, state)
    return state


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
    state = state._replace(key=key)

    state = jax.lax.cond(
        state.phase == C.PHASE_SWITCH,
        lambda s: apply_forced_switches(data, s, actions),
        lambda s: run_turn(data, s, actions, subkey),
        state)

    winner = check_winner(state)
    needs_switch = jnp.any(state.force_switch)
    phase = jnp.where(winner >= 0, jnp.int8(C.PHASE_END),
                      jnp.where(needs_switch, jnp.int8(C.PHASE_SWITCH),
                                jnp.int8(C.PHASE_MOVE)))
    return state._replace(winner=winner, phase=phase)
