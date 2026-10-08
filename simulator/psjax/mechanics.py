"""Shared in-battle derived quantities: speed, grounding, boosts, status, HP.

These sit between the raw `BattleState` and the turn logic. Everything takes and
returns a `BattleState`, and every index may be a traced value, so the whole
module composes under `jit` and `vmap`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from . import consts as C
from .damage import current_types, has_flag
from .data import species_index
from .hooks import A, I
from .state import barrier, set_at
from .stats import (boost_multiply, chain_modify, compute_all_stats, compute_hp,
                    floordiv, idiv)

M1 = 4096


def m(x: float) -> int:
    return int(x * 4096)


# --- randomness --------------------------------------------------------------
# Every `jax.random` call is a Threefry hash inlined into the program, about a
# hundred XLA operations each, and `randint` is three of them. Drawing a key per
# decision put ~56 hashes inside the move loop -- 40% of its operations, and the
# largest single thing XLA:GPU's fusion pass had to wade through. So randomness
# is drawn in bulk instead: `run_turn` hashes once for all the words a turn can
# need, and each consumer takes a `uint32` word rather than a key.

def random_words(key, shape):
    """Independent uniform `uint32` words, all from a single hash of `key`."""
    return jax.random.bits(key, shape, jnp.uint32)


def uniform(word):
    """A float uniform in [0, 1), from the top 24 bits of a random word."""
    return (word >> 8).astype(jnp.float32) * (1.0 / (1 << 24))


def below(word, n: int):
    """An integer uniform in [0, n), from a random word (bias under n / 2**32)."""
    return (word % jnp.uint32(n)).astype(jnp.int32)


# --- lookups on the active Pokemon -------------------------------------------

def act(state, side):
    """The team index of `side`'s active Pokemon."""
    return state.active[side].astype(jnp.int32)


def active_types(state, side):
    """The active Pokemon's types right now: Tera, and Roost's lost Flying."""
    i = act(state, side)
    types = current_types(slot_get(state.types, side, i),
                          slot_get(state.terastallized, side, i),
                          slot_get(state.tera_type, side, i))
    # Roost sheds the Flying type for the rest of the turn; a pure Flying type
    # is left Normal (Showdown's Gen 5+ rule for an empty type list).
    roosting = state.volatiles[side, C.V_ROOST] > 0
    types = jnp.where(roosting & (types == C.FLYING), jnp.int8(C.TYPE_NONE), types)
    emptied = roosting & jnp.all(types == C.TYPE_NONE)
    return jnp.where(emptied, jnp.array([C.NORMAL, C.TYPE_NONE], jnp.int8), types)


def is_grounded(state, side) -> jnp.ndarray:
    """Ground moves, hazards and terrain all key off this."""
    i = act(state, side)
    types = active_types(state, side)
    # Roost is already reflected in `active_types`: it grounds a Flying type
    # by taking the type away, not by lifting anything else.
    ungrounded = (jnp.any(types == C.FLYING) |
                  (slot_get(state.ability, side, i) == A.LEVITATE) |
                  (slot_get(state.item, side, i) == I.AIRBALLOON) |
                  (state.volatiles[side, C.V_MAGNETRISE] > 0))
    forced = ((state.gravity > 0) | (state.volatiles[side, C.V_INGRAIN] > 0) |
              (state.volatiles[side, C.V_SMACKDOWN] > 0))
    return forced | jnp.logical_not(ungrounded)


def weather_active(state) -> jnp.ndarray:
    """Air Lock or Cloud Nine on either active Pokemon suppresses the weather --
    while it stands: a fainted holder's ability has ended."""
    sides = jnp.arange(C.NUM_PLAYERS)
    abilities = state.ability[sides, state.active.astype(jnp.int32)]
    standing = state.hp[sides, state.active.astype(jnp.int32)] > 0
    suppressed = jnp.any(((abilities == A.AIRLOCK) | (abilities == A.CLOUDNINE)) & standing)
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
    spe = jnp.where((status == C.PAR) & (ab != A.QUICKFEET), floordiv(spe, 2), spe)
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
    """`field.at[side, slot].set(value)` without a scatter (see `state.set_at`)."""
    return set_at(field, (side, slot), value, when)


def shown_species(state, side, slot):
    """The species `side`'s Pokemon in `slot` appears to be to the opponent:
    the party member it impersonates, while an active Illusion holds."""
    disguise = state.illusion[side].astype(jnp.int32)
    shown = jnp.where((disguise >= 0) & (slot == act(state, side)), disguise, slot)
    return slot_get(state.species, side, shown)


def log_event(state, side, slot, species, move, when=True):
    """Append a row to `state.events`: `side`'s Pokemon in `slot`, seen as
    `species`, used `move` -- or switched in, for move -1. A full log drops it."""
    row = jnp.stack([jnp.asarray(x, jnp.int32) for x in (side, slot, species, move)])
    n = jnp.sum(state.events[:, C.EV_SIDE] >= 0)
    return state._replace(events=set_at(state.events, n, row.astype(jnp.int16), when))


# --- HP ----------------------------------------------------------------------

def damage_pokemon(state, side, slot, amount):
    """Subtract HP, clamped at 0. Fainting is resolved separately."""
    amount = jnp.maximum(amount.astype(jnp.int32), 0)
    hp = slot_get(state.hp, side, slot).astype(jnp.int32)
    new = jnp.maximum(hp - amount, 0).astype(jnp.int16)
    fainted = (hp > 0) & (new <= 0)
    return state._replace(
        hp=slot_set(state.hp, side, slot, new),
        last_faint=jnp.where(fainted, side, state.last_faint).astype(jnp.int8)), (hp - new)


def heal_pokemon(state, side, slot, amount, blockable=True):
    """Add HP, clamped at max. Healing a fainted Pokemon is a no-op.

    Heal Block stops all of it for the active Pokemon -- Leftovers, drain, a
    healing move, a berry -- except what Showdown does not route through its
    heal event (Regenerator passes `blockable=False`).
    """
    amount = jnp.maximum(amount.astype(jnp.int32), 0)
    if blockable:
        blocked = (state.volatiles[side, C.V_HEALBLOCK] > 0) & (act(state, side) == slot)
        amount = jnp.where(blocked, 0, amount)
    hp = slot_get(state.hp, side, slot).astype(jnp.int32)
    maxhp = slot_get(state.maxhp, side, slot).astype(jnp.int32)
    new = jnp.where(hp <= 0, hp, jnp.minimum(hp + amount, maxhp)).astype(jnp.int16)
    return state._replace(hp=slot_set(state.hp, side, slot, new)), (new - hp)


def fraction_of_max(state, side, slot, num, den):
    """`floor(maxhp * num / den)`, at least 1 -- Showdown's standard chip amount."""
    maxhp = slot_get(state.maxhp, side, slot).astype(jnp.int32)
    return jnp.maximum(idiv(maxhp * num, den), 1)


def round_fraction(amount, num, den):
    """`Math.round(amount * num / den)` for non-negative integers.

    Showdown rounds, not floors, for a move's own healing (Recover's half),
    drain and recoil -- a point off either way whenever the fraction is a half
    or more.
    """
    amount = jnp.asarray(amount, jnp.int32)
    return idiv(2 * amount * num + den, 2 * den)


def _strong_weather(weather):
    return (weather == C.HARSH_SUN) | (weather == C.HEAVY_RAIN) | (weather == C.STRONG_WINDS)


def set_weather(state, weather, when=True):
    """Start `weather` for five turns, as Showdown's `setWeather` does.

    Setting the weather that is already up fails rather than restarting its
    count, and nothing but another primal weather replaces a primal one. A
    `weather` of 0 or less sets nothing.
    """
    ok = jnp.asarray(when) & (weather > 0) & (weather != state.weather) & \
        (jnp.logical_not(_strong_weather(state.weather)) | _strong_weather(weather))
    return state._replace(
        weather=jnp.where(ok, weather, state.weather).astype(state.weather.dtype),
        weather_turns=jnp.where(ok, jnp.int8(5), state.weather_turns))


def set_terrain(state, terrain, when=True):
    """Start `terrain` for five turns; re-setting the active terrain fails."""
    ok = jnp.asarray(when) & (terrain > 0) & (terrain != state.terrain)
    return state._replace(
        terrain=jnp.where(ok, terrain, state.terrain).astype(state.terrain.dtype),
        terrain_turns=jnp.where(ok, jnp.int8(5), state.terrain_turns))


# --- boosts ------------------------------------------------------------------

def boost_delta(*changes):
    """A constant `[7]` boost delta from `(stage, amount)` pairs."""
    delta = np.zeros(C.NUM_BOOSTS, np.int32)
    for stage, amount in changes:
        delta[stage] = amount
    return delta


def boost_immune(state, side, lowering, ignore_ability=False):
    """Abilities that block stat drops from an opponent (unless a Mold Breaker
    is ignoring them -- Full Metal Body it cannot), Clear Amulet and Mist."""
    i = act(state, side)
    ab = slot_get(state.ability, side, i)
    blanket = ((((ab == A.CLEARBODY) | (ab == A.WHITESMOKE)) & jnp.logical_not(ignore_ability)) |
               (ab == A.FULLMETALBODY))
    item = slot_get(state.item, side, i) == I.CLEARAMULET
    return lowering & (blanket | item |
                       (state.side_conditions[side, C.SC_MIST] > 0))


def has_type(state, side, typ):
    """Does `side`'s active have `typ` right now (Tera included)?

    Cheaper than `active_types` for any type but Flying, which is the only one
    Roost takes away.
    """
    i = act(state, side)
    types = current_types(slot_get(state.types, side, i),
                          slot_get(state.terastallized, side, i),
                          slot_get(state.tera_type, side, i))
    return jnp.any(types == typ)


def _boost_row(state, side, old, delta, from_opponent, bounce, ignore_ability=False):
    """`delta` applied to `side`'s boost row `old`, which may not have been
    written back to `state` yet. Returns the new row, the change actually made,
    and what Mirror Armor sends back.

    `from_opponent` is a Python bool, so a Pokemon boosting itself skips every
    drop-blocking check at trace time rather than computing and discarding them.
    `ignore_ability` is a Mold Breaker attacker getting past `side`'s ability
    (the caller has checked it is breakable).
    """
    i = act(state, side)
    raw_ab = slot_get(state.ability, side, i)
    ab = jnp.where(ignore_ability, 0, raw_ab)
    delta = delta.astype(jnp.int32)
    delta = jnp.where(ab == A.CONTRARY, -delta, delta)
    delta = jnp.where(ab == A.SIMPLE, delta * 2, delta)

    # Showdown caps the change before the TryBoost handlers see it, so a drop
    # at -6 is nothing -- Mirror Armor has nothing to send back.
    delta = jnp.clip(old + delta, -6, 6) - old
    reflected = jnp.zeros_like(delta)
    if from_opponent:
        lowering = delta < 0
        stage = jnp.arange(C.NUM_BOOSTS)
        # Clear Body and friends; the targeted protections; and Flower Veil,
        # which shields a Grass-type holder from every drop an opponent causes.
        blocked = boost_immune(state, side, lowering, ignore_ability) | (lowering & (
            ((ab == A.HYPERCUTTER) & (stage == C.B_ATK)) |
            ((ab == A.BIGPECKS) & (stage == C.B_DEF)) |
            (((ab == A.KEENEYE) | (ab == A.MINDSEYE)) & (stage == C.B_ACC)) |
            ((ab == A.FLOWERVEIL) & has_type(state, side, C.GRASS))))
        if bounce:
            # Mirror Armor sends the drops back where they came from instead.
            bounced = lowering & (ab == A.MIRRORARMOR) & jnp.logical_not(blocked)
            reflected = jnp.where(bounced, delta, 0)
            blocked = blocked | bounced
        delta = jnp.where(blocked, 0, delta)
    return old + delta, delta, reflected


def _boost_one(state, side, delta, from_opponent, bounce):
    """`apply_boosts` for one side; also returns what Mirror Armor sends back."""
    new, delta, reflected = _boost_row(state, side, state.boosts[side].astype(jnp.int32),
                                       delta, from_opponent, bounce)
    return state._replace(boosts=set_at(state.boosts, side, new)), delta, reflected


def apply_boosts(state, side, delta, from_opponent=False, ignore_ability=False):
    """Apply a `[7]` boost delta, clamped to -6..+6.

    Contrary inverts it; Simple doubles it; Clear Body and friends block drops
    that come from the opponent, and Mirror Armor reflects them onto the
    opponent. Returns the state and the delta actually applied (Defiant and
    Competitive key off that). Any stat that rises or falls is noted for the
    turn (`stats_raised`, `stats_lowered`), as Showdown's `boost` does.
    `ignore_ability` is a Mold Breaker attacker getting past `side`'s
    (breakable) ability: Clear Body, Contrary, Mirror Armor and the like.
    """
    assert isinstance(from_opponent, bool), "from_opponent is decided at trace time"
    if not from_opponent:
        state, applied, _ = _boost_one(state, side, delta, False, False)
        return _note_changes(state, side == jnp.arange(C.NUM_PLAYERS), applied), applied
    # A drop, Defiant's answer, Mirror Armor's bounce and the answer to that
    # change nothing but the two boost rows, so the rows are worked out first
    # and written back once: four rounds of rewriting the state, each re-read
    # by the next, made this a sixth of `vmap(execute_move)`'s run time.
    other = 1 - side
    row, applied, reflected = _boost_row(state, side, state.boosts[side].astype(jnp.int32),
                                         delta, True, True, ignore_ability)
    row, answered = _answer_drops(state, side, row, applied)
    # The reflected drops land on the opponent as though it had lowered its own
    # stats through Mirror Armor -- which a second Mirror Armor does not return.
    alive = slot_get(state.hp, other, act(state, other)) > 0
    row_o, bounced, _ = _boost_row(state, other, state.boosts[other].astype(jnp.int32),
                                   jnp.where(alive, reflected, 0), True, False)
    row_o, answered_o = _answer_drops(state, other, row_o, bounced)
    is_side = jnp.arange(C.NUM_PLAYERS) == side
    rows = jnp.where(is_side[:, None], row[None], row_o[None])
    # The changes sit at the end of a long chain, and every kernel reading one
    # recomputes the chain: materialise them once (`barrier`), and fold the
    # turn's raised / lowered flags into one write. Four separate updates made
    # this twelve times slower.
    applied, answered, bounced, answered_o = barrier(applied, answered, bounced, answered_o)
    up = jnp.stack([jnp.any(applied > 0) | jnp.any(answered > 0),
                    jnp.any(bounced > 0) | jnp.any(answered_o > 0)])
    down = jnp.stack([jnp.any(applied < 0) | jnp.any(answered < 0),
                      jnp.any(bounced < 0) | jnp.any(answered_o < 0)])
    mine = jnp.where(is_side, 0, 1)
    state = state._replace(boosts=rows.astype(state.boosts.dtype),
                           stats_raised=state.stats_raised | up[mine],
                           stats_lowered=state.stats_lowered | down[mine])
    return state, applied


def _note_changes(state, sides, changes):
    """Mark `sides` ([P] bool) as having had a stat raised / lowered this turn,
    from one batch of `changes` actually made."""
    return state._replace(
        stats_raised=state.stats_raised | (sides & jnp.any(changes > 0)),
        stats_lowered=state.stats_lowered | (sides & jnp.any(changes < 0)))


def _answer_drops(state, side, row, applied):
    """Defiant (+2 Attack) and Competitive (+2 Sp. Atk) answer any drop the
    opponent causes -- a move, Intimidate, Sticky Web, Gooey, Syrup Bomb -- once
    for every stat lowered (Showdown's `onAfterEachBoost`: Parting Shot's two
    drops earn +4). Returns `side`'s new boost row and the change made."""
    ab = slot_get(state.ability, side, act(state, side))
    dropped = jnp.sum((applied < 0).astype(jnp.int32))
    answer = jnp.where(ab == A.DEFIANT, dropped * jnp.asarray(boost_delta((C.B_ATK, 2))),
                       jnp.where(ab == A.COMPETITIVE,
                                 dropped * jnp.asarray(boost_delta((C.B_SPA, 2))), 0))
    new, answered, _ = _boost_row(state, side, row, answer, False, False)
    return new, answered


# --- status ------------------------------------------------------------------

def status_immune(data, state, side, status, source_ability=None, ignore_ability=False,
                  self_inflicted=False, grounded=None):
    """True if `side`'s active cannot be given `status` right now.

    `source_ability` is the ability of whoever is inflicting it, so Corrosion can
    poison Steel and Poison types. `ignore_ability` is Mold Breaker's effect on
    the target's breakable abilities (Insomnia, Leaf Guard, ...).
    `self_inflicted` (Flame Orb, Toxic Orb) gets past Safeguard, which only
    keeps out what the other side inflicts. `grounded` is for the terrains; the
    move engine passes in its own rather than have it recomputed (see `confuse`).
    """
    i = act(state, side)
    ab = slot_get(state.ability, side, i)
    # Roost only ever removes Flying, which grants no status immunity, so the
    # plain type line (Tera included) is enough here.
    types = current_types(slot_get(state.types, side, i),
                          slot_get(state.terastallized, side, i),
                          slot_get(state.tera_type, side, i))
    w = effective_weather(state)
    sunny = (w == C.SUN) | (w == C.HARSH_SUN)

    already = slot_get(state.status, side, i) != C.STATUS_NONE
    mask = data["ability_status_immune"][ab]
    by_ability = ((mask >> status.astype(jnp.int32)) & 1).astype(bool)
    # Leaf Guard in sun; Flower Veil on a Grass-type holder (from anyone but the
    # holder itself, which is every caller here); Minior's Meteor shell.
    by_ability = by_ability | ((ab == A.LEAFGUARD) & sunny) | (
        (ab == A.FLOWERVEIL) & jnp.any(types == C.GRASS)) | (
        (ab == A.SHIELDSDOWN) &
        (slot_get(state.species, side, i) == species_index("miniormeteor")))
    # Mold Breaker gets a status past a breakable ability -- but Insomnia and the
    # like cure it again on the next Update, so for them it never sticks.
    cures_itself = (ab == A.IMMUNITY) | (ab == A.LIMBER) | (ab == A.INSOMNIA) | \
        (ab == A.VITALSPIRIT) | (ab == A.WATERVEIL) | (ab == A.MAGMAARMOR) | \
        (ab == A.THERMALEXCHANGE) | (ab == A.WATERBUBBLE)
    by_ability = by_ability & jnp.logical_not(
        jnp.asarray(ignore_ability) & data["ability_breakable"][ab] &
        jnp.logical_not(cures_itself))

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
    grounded = is_grounded(state, side) if grounded is None else grounded
    misty = (state.terrain == C.MISTY_TERRAIN) & grounded
    electric_sleep = (status == C.SLP) & (state.terrain == C.ELECTRIC_TERRAIN) & grounded
    safeguard = (state.side_conditions[side, C.SC_SAFEGUARD] > 0) & \
        jnp.logical_not(self_inflicted)
    sun_freeze = (status == C.FRZ) & sunny
    # Sleep Clause Mod, a rule of the Random Battle format: the opponent cannot
    # put a second Pokemon to sleep while one it put to sleep still is. A Rest
    # does not count.
    sleep_clause = (status == C.SLP) & jnp.logical_not(self_inflicted) & jnp.any(
        (state.status[side] == C.SLP) & (state.hp[side] > 0) &
        jnp.logical_not(state.rest_sleep[side]))

    return (already | by_ability | poison_immune | burn_immune | para_immune |
            freeze_immune | misty | electric_sleep | safeguard | sun_freeze |
            sleep_clause)


def set_status(data, state, side, status, word, source_ability=None,
               ignore_ability=False, self_inflicted=False, grounded=None):
    """Try to inflict a major status. Returns `(state, applied)`.

    `word` is a random `uint32` (see `random_words`); only sleep reads it.
    """
    i = act(state, side)
    blocked = status_immune(data, state, side, status, source_ability, ignore_ability,
                            self_inflicted, grounded)
    ok = jnp.logical_not(blocked) & (status != C.STATUS_NONE) & \
        (slot_get(state.hp, side, i) > 0)

    # Sleep lasts 1-3 turns in Gen 5+; Toxic starts its counter at 1. Early
    # Bird does not shorten the roll: it ticks the counter twice per attempt
    # to move (see `can_move`), as Showdown does.
    sleep_turns = (2 + below(word, 3)).astype(jnp.int8)
    turns = jnp.where(status == C.SLP, sleep_turns,
                      jnp.where(status == C.TOX, jnp.int8(1), jnp.int8(0)))

    cur_status = slot_get(state.status, side, i)
    cur_turns = slot_get(state.status_turns, side, i)
    new_status = jnp.where(ok, status, cur_status).astype(jnp.int8)
    new_turns = jnp.where(ok, turns, cur_turns).astype(jnp.int8)
    falls_asleep = ok & (status == C.SLP)
    return state._replace(
        status=slot_set(state.status, side, i, new_status),
        status_turns=slot_set(state.status_turns, side, i, new_turns),
        sleep_attempts=slot_set(state.sleep_attempts, side, i, jnp.int8(0), falls_asleep),
        rest_sleep=slot_set(state.rest_sleep, side, i, False, falls_asleep),
    ), ok


def synchronize(data, state, side, status, when=True):
    """Synchronize passes a burn, paralysis or poison back to whoever caused it.

    `when` additionally gates this (e.g. on whether the status actually landed);
    folded into the mask rather than an outer `lax.cond` for the same reason as
    `cure_status`. Sleep never passes back, so no randomness is involved.
    """
    i = act(state, side)
    other = 1 - side
    passes_back = when & (slot_get(state.ability, side, i) == A.SYNCHRONIZE) & (
        (status == C.BRN) | (status == C.PAR) | (status == C.PSN) | (status == C.TOX))
    masked_status = jnp.where(passes_back, status, jnp.int8(C.STATUS_NONE))
    state, _ = set_status(data, state, other, masked_status, jnp.uint32(0))
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
    return state._replace(volatiles=set_at(state.volatiles, (side, volatile), turns))


def has_volatile(state, side, volatile):
    return state.volatiles[side, volatile] > 0


def clear_volatile(state, side, volatile):
    return state._replace(volatiles=set_at(state.volatiles, (side, volatile), 0))


def actives_alive(state):
    """`[P]` bool: is each side's active Pokemon still standing?"""
    return jnp.stack([slot_get(state.hp, p, act(state, p)) > 0
                      for p in range(C.NUM_PLAYERS)])


def soul_heart(state, alive_before):
    """Soul-Heart: +1 Sp. Atk for every Pokemon that fainted since `alive_before`.

    Only the actives can faint mid-turn, so comparing them is enough. Written
    straight into the boost table: nothing that can hold Soul-Heart has Contrary
    or Simple, so `apply_boosts` would be a long way round.
    """
    alive_now = actives_alive(state)
    fainted = jnp.sum(alive_before & jnp.logical_not(alive_now)).astype(jnp.int32)
    boosts = state.boosts
    for side in range(C.NUM_PLAYERS):
        holder = (slot_get(state.ability, side, act(state, side)) == A.SOULHEART) & \
            alive_now[side]
        raised = jnp.minimum(boosts[side, C.B_SPA].astype(jnp.int32) + fainted, 6)
        boosts = set_at(boosts, (side, C.B_SPA), raised, holder)
    return state._replace(boosts=boosts)


def confuse(state, side, word, when=True, grounded=None):
    """Confuse `side` for 2-5 turns (Showdown's `random(2, 6)`).

    Counted in move attempts, not turns: `before_move` ticks it. A Pokemon that
    is already confused keeps its old count, and Own Tempo is immune, as is a
    grounded one on Misty Terrain. The move engine passes in `grounded` from
    its context: recomputing it from the state at every call site cost a fifth
    of `vmap(execute_move)`'s run time.
    """
    i = act(state, side)
    turns = (2 + below(word, 4)).astype(jnp.int8)
    grounded = is_grounded(state, side) if grounded is None else grounded
    misty = (state.terrain == C.MISTY_TERRAIN) & grounded
    ok = when & (state.volatiles[side, C.V_CONFUSION] <= 0) & jnp.logical_not(misty) & \
        (slot_get(state.ability, side, i) != A.OWNTEMPO) & (slot_get(state.hp, side, i) > 0)
    return state._replace(volatiles=set_at(state.volatiles, (side, C.V_CONFUSION), turns, ok))


def usable_moves(data, state, side):
    """`[4]` bool: the active Pokemon's moves it may choose this turn.

    Everything Showdown's request disables: no PP, a Choice lock (while it
    still holds the Choice item), Disable, Taunt (status moves), Torment (a
    repeat), Encore (all but the encored move), Assault Vest (status moves),
    Throat Chop (sound moves) and Heal Block (healing moves). A move it is
    locked into -- a charge move's second turn, a rampage -- is the only
    choice. With none left the Pokemon struggles.
    """
    i = act(state, side)
    slots = jnp.arange(C.MOVES_PER_POKEMON)
    moves = state.moves[side, i]
    vol = state.volatiles[side]
    flags = data["move_flags"][jnp.maximum(moves, 0)]
    status_move = data["move_category"][jnp.maximum(moves, 0)] == C.CAT_STATUS
    held = state.item[side, i]
    ok = (moves >= 0) & (state.pp[side, i] > 0)
    choiced = (state.choice_slot[side] >= 0) & (
        (held == I.CHOICESCARF) | (held == I.CHOICEBAND) | (held == I.CHOICESPECS))
    ok = ok & jnp.where(choiced, slots == state.choice_slot[side], True)
    ok = ok & ~((vol[C.V_DISABLE] > 0) & (slots == state.disabled_slot[side]))
    ok = ok & ~((vol[C.V_TAUNT] > 0) & status_move)
    ok = ok & ~((vol[C.V_TORMENT] > 0) & (moves == state.last_move[side]))
    ok = ok & ~((held == I.ASSAULTVEST) & status_move)
    # Blood Moon and Gigaton Hammer cannot be used twice in a row.
    ok = ok & ~(data["move_cant_use_twice"][jnp.maximum(moves, 0)] &
                (moves == state.last_move[side]))
    ok = ok & ~((vol[C.V_THROATCHOP] > 0) & has_flag(flags, "sound"))
    ok = ok & ~((vol[C.V_HEALBLOCK] > 0) & has_flag(flags, "heal"))
    encored = (vol[C.V_ENCORE] > 0) & (state.encore_slot[side] >= 0)
    ok = ok & jnp.where(encored, slots == state.encore_slot[side], True)
    locked = state.locked_slot[side] >= 0
    return jnp.where(locked, slots == state.locked_slot[side], ok)


def is_trapped(state, side):
    """True if `side`'s active cannot choose to switch out.

    A move it is locked into (charging Solar Beam) or recharging from traps it
    outright. Shadow Tag, Arena Trap, Magnet Pull, Ingrain, binding moves and
    Mean Look's kind go through Showdown's `tryTrap`, which Ghost types are
    immune to from Gen 6 on.
    """
    i = act(state, side)
    other = 1 - side
    oi = act(state, other)
    types = active_types(state, side)
    ab = slot_get(state.ability, side, i)
    o_ab = slot_get(state.ability, other, oi)
    foe_out = slot_get(state.hp, other, oi) > 0
    by_ability = foe_out & (
        ((o_ab == A.SHADOWTAG) & (ab != A.SHADOWTAG)) |
        ((o_ab == A.ARENATRAP) & is_grounded(state, side)) |
        ((o_ab == A.MAGNETPULL) & jnp.any(types == C.STEEL)))
    held = (state.volatiles[side, C.V_PARTIALLYTRAPPED] > 0) | \
        (state.volatiles[side, C.V_INGRAIN] > 0) | (state.volatiles[side, C.V_NORETREAT] > 0) | \
        (state.volatiles[side, C.V_TRAPPED] > 0)
    ghost = jnp.any(types == C.GHOST)
    locked = (state.locked_slot[side] >= 0) | (state.volatiles[side, C.V_RECHARGE] > 0)
    return locked | (jnp.logical_not(ghost) & (by_ability | held))


# --- formes, Transform, and switching back -----------------------------------

def species_stats(data, species, level, spread_zero):
    """The `[6]` stat line `species` has at this Pokemon's level and EV/IV spread."""
    base = data["species_base_stats"][species].astype(jnp.int32)
    zeroed = ((spread_zero.astype(jnp.int32) >> jnp.arange(C.NUM_STATS)) & 1).astype(bool)
    return compute_all_stats(base, level.astype(jnp.int32),
                             jnp.where(zeroed, 0, 31), jnp.where(zeroed, 0, 85))


def forme_change(data, state, side, slot, species, when=True, permanent=False,
                 ability=None):
    """Showdown's `formeChange`: new species, types and stats.

    Max HP is untouched unless the change is `permanent`, which also makes the
    new forme what the Pokemon reverts to and keeps the damage it has taken
    (`updateMaxHp`). `ability`, when given, replaces the current ability -- and
    the one it reverts to, if permanent.
    """
    species = jnp.asarray(species, jnp.int16)
    level = slot_get(state.level, side, slot)
    spread = slot_get(state.spread_zero, side, slot)
    old_species = slot_get(state.species, side, slot)
    fresh = species_stats(data, species, level, spread)
    stats = slot_get(state.stats, side, slot).astype(jnp.int32)
    stats = jnp.concatenate([stats[:1], fresh[1:].astype(jnp.int32)])
    state = state._replace(
        species=slot_set(state.species, side, slot, species, when),
        types=slot_set(state.types, side, slot, data["species_types"][species], when),
        stats=slot_set(state.stats, side, slot, stats, when))
    if permanent is not False:
        # `permanent` may be traced, so that several possible forme changes can
        # share one call.
        perm = when & permanent
        # Max HP moves by the difference between the two formes' HP. The mask
        # does not record HP EVs, which the generator sometimes trims, but the
        # difference is exact whenever the base HP is shared (Mimikyu, Eiscue)
        # and Terapagos, whose base HP is not, always has the full spread.
        old_max = slot_get(state.maxhp, side, slot).astype(jnp.int32)
        was = compute_hp(data["species_base_stats"][old_species, C.HP], level,
                         jnp.where(spread & 1, 0, 31), jnp.where(spread & 1, 0, 85))
        new_max = old_max + fresh[C.HP].astype(jnp.int32) - was.astype(jnp.int32)
        hp = slot_get(state.hp, side, slot).astype(jnp.int32)
        hp = jnp.where(hp <= 0, 0, jnp.maximum(1, new_max - (old_max - hp)))
        state = state._replace(
            base_species=slot_set(state.base_species, side, slot, species, perm),
            maxhp=slot_set(state.maxhp, side, slot, new_max, perm),
            hp=slot_set(state.hp, side, slot, hp, perm))
    if ability is not None:
        state = state._replace(ability=slot_set(state.ability, side, slot, ability, when))
        if permanent is not False:
            state = state._replace(base_ability=slot_set(
                state.base_ability, side, slot, ability, when & permanent))
    return state


def revert_on_switch_out(data, state, side, slot, when=True):
    """Undo everything that does not survive leaving the field.

    Showdown's `clearVolatile`: Transform ends (its own moves and PP come back),
    the ability returns to the base one (Trace, Transform), and the species is
    reset to the base forme, which also resets its types (Protean, Burn Up,
    Roost) and stats (Meloetta, Minior, Transform). Called on the Pokemon that
    is leaving, before the slot state is cleared.
    """
    species = slot_get(state.species, side, slot)
    base = slot_get(state.base_species, side, slot)
    base = jnp.where(base >= 0, base, species)
    base_ab = slot_get(state.base_ability, side, slot)
    base_ab = jnp.where(base_ab >= 0, base_ab, slot_get(state.ability, side, slot))
    transformed = state.transformed[side] & when
    state = forme_change(data, state, side, slot, base,
                         when=when & (transformed | (species != base)))
    return state._replace(
        types=slot_set(state.types, side, slot, data["species_types"][base], when),
        ability=slot_set(state.ability, side, slot, base_ab, when),
        moves=slot_set(state.moves, side, slot, state.tf_moves[side], transformed),
        pp=slot_set(state.pp, side, slot, state.tf_pp[side], transformed),
        maxpp=slot_set(state.maxpp, side, slot, state.tf_maxpp[side], transformed))


# --- held items --------------------------------------------------------------

def item_locked(data, state, side, slot):
    """True if the held item belongs to its holder's species (a plate on Arceus,
    Booster Energy on a Paradox Pokemon).

    Nothing can take such an item, and Knock Off gets no bonus for trying.
    """
    item = slot_get(state.item, side, slot)
    species = slot_get(state.species, side, slot)
    num = data["item_locked_num"][item]
    return ((num > 0) & (data["species_num"][species] == num)) | \
        ((item == I.BOOSTERENERGY) & data["species_paradox"][species])


def can_lose_item(data, state, side, slot, by_opponent=True, ignore_ability=False):
    """Whether an item can be taken from `side` (Knock Off, Trick, Bug Bite, ...).

    Sticky Hold keeps it out of an opponent's hands; Mold Breaker gets past that.
    """
    ab = slot_get(state.ability, side, slot)
    sticky = by_opponent & (ab == A.STICKYHOLD) & jnp.logical_not(ignore_ability) & \
        (slot_get(state.hp, side, slot) > 0)
    return (slot_get(state.item, side, slot) != 0) & \
        jnp.logical_not(item_locked(data, state, side, slot)) & jnp.logical_not(sticky)


def lose_item(state, side, slot, when=True):
    """Remove the held item. Losing one is what arms Unburden."""
    had = when & (slot_get(state.item, side, slot) != 0)
    return state._replace(
        item=slot_set(state.item, side, slot, jnp.int16(0), had),
        volatiles=set_at(state.volatiles, (side, C.V_UNBURDEN), 1,
                         had & (act(state, side) == slot)))


def _actives(state, field):
    """`[P,...]` values of the active Pokemon, as a masked sum (no gather)."""
    on = jnp.arange(C.TEAM_SIZE)[None, :] == state.active.astype(jnp.int32)[:, None]
    on = on.reshape(on.shape + (1,) * (field.ndim - 2))
    return jnp.sum(jnp.where(on, field, 0), axis=1).astype(field.dtype)


def _use_active_items(state, used):
    """Both sides' active items used up where `used` ([P] bool), arming Unburden."""
    on = (jnp.arange(C.TEAM_SIZE)[None, :] == state.active.astype(jnp.int32)[:, None]) & \
        used[:, None]
    return state._replace(
        item=jnp.where(on, jnp.int16(0), state.item),
        volatiles=jnp.where((jnp.arange(C.NUM_VOLATILES) == C.V_UNBURDEN)[None, :] &
                            used[:, None], jnp.int8(1), state.volatiles))


def white_herb(state):
    """White Herb restores its holder's lowered stats, and is used up doing so.

    Showdown checks it after every move, after every switch-in and in the
    residuals, so the callers are those three places. Both sides at once: one
    copy in the compiled program per call site rather than two.
    """
    lowered = state.boosts < 0
    fires = (_actives(state, state.item) == I.WHITEHERB) & jnp.any(lowered, axis=1) & \
        (_actives(state, state.hp) > 0)
    state = state._replace(boosts=jnp.where(lowered & fires[:, None], jnp.int8(0),
                                            state.boosts))
    return _use_active_items(state, fires)


def paradox_update(state):
    """Protosynthesis (sun) and Quark Drive (Electric Terrain), as Showdown runs
    them on every change of weather or terrain.

    While the sun or terrain lasts, the ability raises the holder's best stat --
    stat stages counted, ties to the earlier stat. Booster Energy wakes it when
    neither is up, is used up doing so, and keeps it awake after the weather or
    terrain ends. Called after anything that can change either: a switch-in, a
    move, Terastallization, the residuals.
    """
    ab = _actives(state, state.ability)
    w = effective_weather(state)
    proto, quark = ab == A.PROTOSYNTHESIS, ab == A.QUARKDRIVE
    natural = (proto & ((w == C.SUN) | (w == C.HARSH_SUN))) | \
        (quark & (state.terrain == C.ELECTRIC_TERRAIN))
    alive = _actives(state, state.hp) > 0
    active = state.boosted_stat >= 0
    stop = active & jnp.logical_not(natural) & jnp.logical_not(state.paradox_booster)
    active = active & jnp.logical_not(stop)
    booster = (proto | quark) & jnp.logical_not(natural) & alive & \
        jnp.logical_not(state.transformed) & \
        (_actives(state, state.item) == I.BOOSTERENERGY)
    start = jnp.logical_not(active) & alive & (natural | booster)
    stats = _actives(state, state.stats)
    boosted = boost_multiply(stats[:, C.ATK:C.SPE + 1], state.boosts[:, C.B_ATK:C.B_SPE + 1])
    best = (jnp.argmax(boosted, axis=1) + C.ATK).astype(jnp.int8)
    state = state._replace(
        boosted_stat=jnp.where(start, best, jnp.where(stop, jnp.int8(-1), state.boosted_stat)),
        paradox_booster=jnp.where(start, booster,
                                  jnp.where(stop, False, state.paradox_booster)))
    return _use_active_items(state, start & booster)


# --- berries -----------------------------------------------------------------

def unnerved(state, side):
    """Unnerve (and As One) on the other side stops `side` eating berries."""
    other = 1 - side
    oi = act(state, other)
    ab = slot_get(state.ability, other, oi)
    return (slot_get(state.hp, other, oi) > 0) & (
        (ab == A.UNNERVE) | (ab == A.ASONEGLASTRIER) | (ab == A.ASONESPECTRIER))


def eat_berry(data, state, side, berry, when=True, consume=True, cud=2):
    """`side`'s active eats `berry`, which it need not be holding (Bug Bite).

    `consume` spends the held item (and remembers it for Harvest); `cud` is how
    many residuals until Cud Chew eats it again, 0 for never -- Showdown's
    counter is 2, or 1 when the berry went down during the residual phase.
    Both may be traced, so every reason a side eats during one move can share a
    single call.
    """
    i = act(state, side)
    berry = jnp.asarray(berry, jnp.int16)
    when = when & (berry != 0) & (slot_get(state.hp, side, i) > 0)
    ab = slot_get(state.ability, side, i)

    heal = jnp.where(berry == I.SITRUSBERRY, fraction_of_max(state, side, i, 1, 4), 0)
    heal = jnp.where(berry == I.ORANBERRY, 10, heal)
    heal = jnp.where(berry == I.FIGYBERRY, fraction_of_max(state, side, i, 1, 3), heal)
    # Cheek Pouch tops up a third, whatever the berry was.
    heal = heal + jnp.where(ab == A.CHEEKPOUCH, fraction_of_max(state, side, i, 1, 3), 0)
    state, _ = heal_pokemon(state, side, i, jnp.where(when, heal, 0))

    status = slot_get(state.status, side, i)
    cures = when & ((berry == I.LUMBERRY) | ((berry == I.CHESTOBERRY) & (status == C.SLP)))
    state = cure_status(state, side, i, when=cures)
    state = state._replace(volatiles=set_at(
        state.volatiles, (side, C.V_CONFUSION), 0, when & (berry == I.LUMBERRY)))

    boost = jnp.zeros(C.NUM_BOOSTS, jnp.int32)
    for item, stage in ((I.SALACBERRY, C.B_SPE), (I.PETAYABERRY, C.B_SPA),
                        (I.LIECHIBERRY, C.B_ATK)):
        boost = jnp.where(when & (berry == item), jnp.asarray(boost_delta((stage, 1))), boost)
    state, _ = apply_boosts(state, side, boost)

    # Leppa restores 10 PP to the first empty move, else the first one missing any.
    moves = slot_get(state.moves, side, i)
    pp = slot_get(state.pp, side, i).astype(jnp.int32)
    maxpp = slot_get(state.maxpp, side, i).astype(jnp.int32)
    empty = (moves >= 0) & (pp == 0)
    short = (moves >= 0) & (pp < maxpp)
    pick = jnp.where(jnp.any(empty), jnp.argmax(empty), jnp.argmax(short))
    restore = when & (berry == I.LEPPABERRY) & jnp.any(short)
    new_pp = jnp.where(jnp.arange(C.MOVES_PER_POKEMON) == pick,
                       jnp.minimum(pp + 10, maxpp), pp)
    state = state._replace(pp=slot_set(state.pp, side, i, new_pp, restore))

    spent = when & consume
    state = lose_item(state, side, i, spent)
    state = state._replace(last_item=slot_set(state.last_item, side, i, berry, spent))
    chew = when & (ab == A.CUDCHEW) & (jnp.asarray(cud) > 0)
    return state._replace(cud_berry=set_at(state.cud_berry, side, berry, chew),
                          cud_turns=set_at(state.cud_turns, side, cud, chew))


def berry_update(data, state, side, during_residual=False, when=True, foe_started=True):
    """Eat the held berry if its trigger is met -- Showdown's berry `onUpdate`.

    Run wherever Showdown runs Update: after each move, after the residuals and
    after a switch-in.
    """
    return eat_berry(data, state, side, slot_get(state.item, side, act(state, side)),
                     when=when & berry_wants(state, side, foe_started),
                     cud=1 if during_residual else 2)


def berry_wants(state, side, foe_started=True):
    """True if the held berry's own trigger is met and nothing stops the eating.

    `foe_started=False` is the moment a foe has switched in but its ability
    has not started, so its Unnerve does not count yet.
    """
    i = act(state, side)
    it = slot_get(state.item, side, i)
    hp = slot_get(state.hp, side, i).astype(jnp.int32)
    maxhp = slot_get(state.maxhp, side, i).astype(jnp.int32)
    status = slot_get(state.status, side, i)
    half, quarter = hp * 2 <= maxhp, hp * 4 <= maxhp
    moves = slot_get(state.moves, side, i)
    # A healing berry is not even eaten under Heal Block (its `onTryEatItem`
    # asks `TryHeal` first); it waits for the block to end.
    can_heal = state.volatiles[side, C.V_HEALBLOCK] <= 0
    wants = (
        (((it == I.SITRUSBERRY) | (it == I.ORANBERRY)) & half & can_heal) |
        ((it == I.FIGYBERRY) & quarter & can_heal) |
        (((it == I.SALACBERRY) | (it == I.PETAYABERRY) | (it == I.LIECHIBERRY)) & quarter) |
        ((it == I.LUMBERRY) & ((status != C.STATUS_NONE) |
                               (state.volatiles[side, C.V_CONFUSION] > 0))) |
        ((it == I.CHESTOBERRY) & (status == C.SLP)) |
        ((it == I.LEPPABERRY) & jnp.any((moves >= 0) & (slot_get(state.pp, side, i) == 0))))
    return wants & (hp > 0) & jnp.logical_not(unnerved(state, side) & foe_started)
