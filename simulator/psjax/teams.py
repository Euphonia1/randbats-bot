"""Gen 9 Random Battle team generation, on device.

Species, sets, movepools, abilities and Tera types come straight from Showdown's
`data/random-battles/gen9/sets.json`, so a generated team is drawn from the same
pool a real Random Battle uses.

What is *approximated*: held items. Showdown picks those in generator code
(`teams.ts`) full of move- and species-specific special cases. We use a simple
role-driven policy instead -- documented in `ITEM_BY_ROLE` -- which produces
plausible items but will not match Showdown set-for-set.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from . import consts as C
from .data import load_data, load_team_pool
from .hooks import A, I
from .mechanics import is_trapped, usable_moves
from .state import empty_state
from .stats import compute_all_stats, compute_stat

# Role -> held item. An approximation of Showdown's generator; see the note above.
ITEM_BY_ROLE = {
    "AV Pivot": I.ASSAULTVEST,
    "Fast Attacker": I.LIFEORB,
    "Wallbreaker": I.LIFEORB,
    "Setup Sweeper": I.LIFEORB,
    "Fast Bulky Setup": I.LEFTOVERS,
    "Bulky Setup": I.LEFTOVERS,
    "Bulky Attacker": I.LEFTOVERS,
    "Bulky Support": I.LEFTOVERS,
    "Fast Support": I.HEAVYDUTYBOOTS,
    "Tera Blast user": I.LIFEORB,
}


# Numpy, and built at import rather than on first use. A `jnp` array created
# inside a jit trace and then cached across traces stays bound to the trace that
# made it, so the second distinct trace to read the cache raises
# UnexpectedTracerError -- which is what happens the moment a process builds two
# `BattleEnv` instances. A numpy constant is trace-agnostic; the `jnp.asarray`
# at the call site is a fresh constant per trace and XLA folds the duplicates.
_ROLE_ITEMS = np.asarray(
    [ITEM_BY_ROLE.get(r, I.LEFTOVERS) for r in C.ROLES], np.int16)


def _role_items():
    return jnp.asarray(_ROLE_ITEMS)


def _choose_moves(key, movepool, pool_len):
    """Pick up to four distinct moves from a padded movepool.

    Sorting random keys keeps this a fixed-shape, jit-friendly sample without
    replacement; padding slots are pushed to the end by giving them a large key.
    """
    scores = jax.random.uniform(key, movepool.shape)
    valid = jnp.arange(movepool.shape[0]) < pool_len
    scores = jnp.where(valid, scores, 2.0)
    order = jnp.argsort(scores)
    picked = movepool[order[:C.MOVES_PER_POKEMON]]
    enough = jnp.arange(C.MOVES_PER_POKEMON) < pool_len
    return jnp.where(enough, picked, -1).astype(jnp.int16)


def build_pokemon(data, key, entry):
    """Roll one Pokemon from randbats entry index `entry`."""
    k_set, k_moves, k_ability, k_tera = jax.random.split(key, 4)

    species = data["rb_species"][entry]
    level = data["rb_level"][entry]
    n_sets = jnp.maximum(data["rb_num_sets"][entry].astype(jnp.int32), 1)
    s = jax.random.randint(k_set, (), 0, n_sets)

    movepool = data["rb_movepool"][entry, s]
    moves = _choose_moves(k_moves, movepool, data["rb_movepool_len"][entry, s])

    n_ab = jnp.maximum(data["rb_abilities_len"][entry, s].astype(jnp.int32), 1)
    ability = data["rb_abilities"][entry, s, jax.random.randint(k_ability, (), 0, n_ab)]

    n_tera = jnp.maximum(data["rb_tera_len"][entry, s].astype(jnp.int32), 1)
    tera = data["rb_tera"][entry, s, jax.random.randint(k_tera, (), 0, n_tera)]

    item = _role_items()[data["rb_role"][entry, s]]

    base = data["species_base_stats"][species]
    # Showdown zeroes Attack on sets with no physical move and Speed on Trick
    # Room sets; we reproduce the Attack case, which is the one that affects
    # damage (via Foul Play and confusion) rather than just Speed order.
    categories = jnp.where(moves >= 0, data["move_category"][jnp.maximum(moves, 0)],
                           C.CAT_STATUS)
    has_physical = jnp.any(categories == C.CAT_PHYSICAL)
    evs = jnp.full(6, 85, jnp.int32).at[C.ATK].set(jnp.where(has_physical, 85, 0))
    ivs = jnp.full(6, 31, jnp.int32).at[C.ATK].set(jnp.where(has_physical, 31, 0))
    stats = compute_all_stats(base[None], level[None], ivs[None], evs[None])[0]

    pp = jnp.where(moves >= 0,
                   (data["move_pp"][jnp.maximum(moves, 0)].astype(jnp.int32) * 8) // 5,
                   0).astype(jnp.int8)
    types = data["species_types"][species]
    return dict(species=species, level=level, stats=stats, types=types, moves=moves,
                pp=pp, item=item, ability=ability, tera_type=tera)


def team_from_pool(pool, key):
    """Draw one team that Showdown's own generator produced.

    Team generation is not a JAX problem. Showdown's generator runs to a few
    thousand lines of sequential logic -- per-species move enforcement, state
    that accumulates as the team is built, item rules keyed on the set's role
    and its final move list -- and is not expressible as a fixed-shape traced
    computation. Rather than approximate it, `tools/dump_teams.js` calls the
    real thing and this samples the result, which makes items and movesets
    exact rather than merely plausible.
    """
    n = pool["pool_species"].shape[0]
    i = jax.random.randint(key, (), 0, n)
    return {name: pool[f"pool_{name}"][i] for name in (
        "species", "level", "stats", "types", "moves", "pp",
        "item", "ability", "tera_type")}


def random_team(data, key, pool=None):
    """Six distinct species drawn from the Random Battle pool.

    Uses the Showdown-generated pool when one is available; the procedural
    generator below is the fallback for a checkout that has not built one.
    """
    if pool:
        return team_from_pool(pool, key)
    n_entries = data["rb_species"].shape[0]
    k_species, k_mons = jax.random.split(key)
    entries = jax.random.choice(k_species, n_entries, (C.TEAM_SIZE,), replace=False)
    keys = jax.random.split(k_mons, C.TEAM_SIZE)
    return jax.vmap(build_pokemon, in_axes=(None, 0, 0))(data, keys, entries)


def new_battle(key, data=None, pool=None):
    """A fresh Gen 9 Random Battle with both teams rolled and both leads sent out.

    `pool` defaults to the Showdown-generated team pool if one has been built.
    Pass `False` to force the procedural generator.
    """
    from .engine import apply_switch_in_ability, apply_entry_hazards
    if data is None:
        data = load_data()
    if pool is None:
        pool = load_team_pool()
    k_p0, k_p1, k_state = jax.random.split(key, 3)
    t0 = random_team(data, k_p0, pool)
    t1 = random_team(data, k_p1, pool)
    team = jax.tree.map(lambda a, b: jnp.stack([a, b]), t0, t1)

    state = empty_state(k_state)
    maxhp = team["stats"][..., C.HP].astype(jnp.int16)
    state = state._replace(
        species=team["species"].astype(jnp.int16),
        level=team["level"].astype(jnp.int8),
        hp=maxhp, maxhp=maxhp,
        stats=team["stats"].astype(jnp.int16),
        types=team["types"].astype(jnp.int8),
        moves=team["moves"].astype(jnp.int16),
        pp=team["pp"].astype(jnp.int8), maxpp=team["pp"].astype(jnp.int8),
        item=team["item"].astype(jnp.int16),
        ability=team["ability"].astype(jnp.int16),
        tera_type=team["tera_type"].astype(jnp.int8),
        active=jnp.zeros(C.NUM_PLAYERS, jnp.int8),
    )
    state = finish_teams(data, state, jax.random.fold_in(k_state, 1))
    from .mechanics import log_event, shown_species
    for side in range(C.NUM_PLAYERS):
        state = log_event(state, side, 0, shown_species(state, side, 0), -1)
    # Leads arrive: entry abilities fire, fastest first (it decides a weather
    # war), but there are no hazards on turn one.
    from .engine import speed_order
    from .state import select_state
    p0_first = apply_switch_in_ability(data, apply_switch_in_ability(data, state, 0), 1)
    p1_first = apply_switch_in_ability(data, apply_switch_in_ability(data, state, 1), 0)
    state = select_state(speed_order(state) == 0, p0_first, p1_first)
    # A White Herb answers a lead's Intimidate before the first turn.
    from .mechanics import white_herb
    return white_herb(state)


def finish_teams(data, state, key):
    """Fill in what a battle derives from its teams rather than stores in them.

    What each Pokemon reverts to on leaving the field; which of its stats were
    built with 0 IVs and EVs (Showdown's generator zeroes only Attack and Speed,
    so a stat below its full-spread value is one of those), which lets forme
    changes rebuild the stats exactly; its gender, fixed by species or else an
    even coin flip as Showdown does; and a lead's Illusion.
    """
    base = data["species_base_stats"][state.species].astype(jnp.int32)
    level = state.level.astype(jnp.int32)[..., None]
    full = compute_stat(base, level)
    zeroed = state.stats.astype(jnp.int32) < full
    bits = jnp.sum(jnp.where(zeroed[..., 1:], 1 << jnp.arange(1, C.NUM_STATS), 0), axis=-1)

    fixed = data["species_gender"][state.species]
    coin = jax.random.bernoulli(key, 0.5, fixed.shape)
    gender = jnp.where(data["species_random_gender"][state.species],
                       jnp.where(coin, C.GENDER_M, C.GENDER_F), fixed).astype(jnp.int8)

    return lead_illusion(state._replace(
        base_species=state.species, base_ability=state.ability,
        spread_zero=bits.astype(jnp.int8), gender=gender))


def lead_illusion(state):
    """A lead with Illusion disguises itself as the last healthy party member.

    The leads are at the front of the party (slot 0), so that is the highest
    healthy slot. Later entries go through `engine.switch_to`.
    """
    slots = jnp.arange(C.TEAM_SIZE)
    last = jnp.max(jnp.where((state.hp > 0) & (slots > 0), slots, -1), axis=1)
    lead_ability = state.ability[:, 0]
    return state._replace(
        illusion=jnp.where(lead_ability == A.ILLUSION, last, -1).astype(jnp.int8))


def legal_action_mask(data, state):
    """`[2, NUM_ACTIONS]` bool mask of the actions each player may pick.

    During `PHASE_SWITCH` only replacement switches are legal; otherwise a player
    may use any move with PP left (respecting a Choice lock) or switch to any
    healthy benched Pokemon.
    """
    rows = []
    slots = jnp.arange(C.MOVES_PER_POKEMON)
    for side in range(C.NUM_PLAYERS):
        i = state.active[side].astype(jnp.int32)
        move_ok = usable_moves(data, state, side)
        # Recharging (Hyper Beam) leaves one choice, Showdown's "Recharge":
        # slot 0 stands for it, with no Terastallizing and no switching.
        recharging = state.volatiles[side, C.V_RECHARGE] > 0
        # Locked into a move (a charge move's second turn, a rampage) or with
        # nothing usable -- and so struggling -- it cannot Terastallize.
        free = jnp.any(move_ok) & (state.locked_slot[side] < 0) & jnp.logical_not(recharging)
        # With nothing usable the Pokemon struggles; slot 0 stands for that.
        move_ok = jnp.where(jnp.any(move_ok) & jnp.logical_not(recharging), move_ok,
                            jnp.arange(C.MOVES_PER_POKEMON) == 0)

        # Transformed into Ogerpon or Terapagos (Imposter), it cannot
        # Terastallize until it switches out.
        num = data["species_num"][state.species[side, i]]
        tera_locked = state.transformed[side] & ((num == 1017) | (num == 1024))
        can_tera = free & jnp.logical_not(state.tera_used[side]) & \
                   jnp.logical_not(tera_locked) & (state.tera_type[side, i] != C.TYPE_NONE)
        switch_ok = (state.hp[side] > 0) & (jnp.arange(C.TEAM_SIZE) != i)
        # Revival Blessing's prompt picks a fainted Pokemon to bring back instead.
        revive_ok = (state.hp[side] <= 0) & (jnp.arange(C.TEAM_SIZE) != i)
        # Shadow Tag, Arena Trap, Magnet Pull, binding moves and move locks keep
        # the active in -- but never stop a replacement for a fainted Pokemon.
        free_switch = switch_ok & jnp.logical_not(is_trapped(state, side))

        row = jnp.concatenate([move_ok, move_ok & can_tera, free_switch])
        # Replacing a fainted Pokemon: switches only.
        forced = state.force_switch[side]
        answer = jnp.where(state.pass_mode[side] == C.PASS_REVIVE, revive_ok, switch_ok)
        row = jnp.where(forced,
                        jnp.concatenate([jnp.zeros(8, bool), answer]), row)
        # A player with nothing to do still needs one legal action. While being
        # asked for a replacement that fallback has to decode to a switch, not to
        # move slot 0 -- action 0 would decode as switch-to-slot-minus-eight.
        stuck = np.arange(C.NUM_ACTIONS) == C.ACTION_SWITCH_BASE
        idle = np.arange(C.NUM_ACTIONS) == 0
        row = jnp.where(jnp.any(row), row, jnp.where(forced, stuck, idle))
        rows.append(row)
    return jnp.stack(rows)
