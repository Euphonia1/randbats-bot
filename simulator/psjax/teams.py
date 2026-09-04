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

from . import consts as C
from .data import load_data
from .hooks import I
from .state import empty_state
from .stats import compute_all_stats

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


@functools.lru_cache(maxsize=1)
def _role_items():
    return jnp.asarray([ITEM_BY_ROLE.get(r, I.LEFTOVERS) for r in C.ROLES], jnp.int16)


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


def random_team(data, key):
    """Six distinct species drawn from the Random Battle pool."""
    n_entries = data["rb_species"].shape[0]
    k_species, k_mons = jax.random.split(key)
    entries = jax.random.choice(k_species, n_entries, (C.TEAM_SIZE,), replace=False)
    keys = jax.random.split(k_mons, C.TEAM_SIZE)
    return jax.vmap(build_pokemon, in_axes=(None, 0, 0))(data, keys, entries)


def new_battle(key, data=None):
    """A fresh Gen 9 Random Battle with both teams rolled and both leads sent out."""
    from .engine import apply_switch_in_ability, apply_entry_hazards
    if data is None:
        data = load_data()
    k_p0, k_p1, k_state = jax.random.split(key, 3)
    t0 = random_team(data, k_p0)
    t1 = random_team(data, k_p1)
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
    # Leads arrive: entry abilities fire, but there are no hazards on turn one.
    state = apply_switch_in_ability(data, state, 0)
    state = apply_switch_in_ability(data, state, 1)
    return state


def legal_action_mask(data, state):
    """`[2, NUM_ACTIONS]` bool mask of the actions each player may pick.

    During `PHASE_SWITCH` only replacement switches are legal; otherwise a player
    may use any move with PP left (respecting a Choice lock) or switch to any
    healthy benched Pokemon.
    """
    mask = jnp.zeros((C.NUM_PLAYERS, C.NUM_ACTIONS), bool)
    for side in range(C.NUM_PLAYERS):
        i = state.active[side].astype(jnp.int32)
        has_move = (state.moves[side, i] >= 0) & (state.pp[side, i] > 0)
        # A Choice item locks the holder into the move it last used.
        locked = state.choice_slot[side] >= 0
        choice_ok = jnp.arange(C.MOVES_PER_POKEMON) == state.choice_slot[side]
        move_ok = has_move & jnp.where(locked, choice_ok, True)
        # If nothing is usable the Pokemon would use Struggle; allow slot 0.
        move_ok = jnp.where(jnp.any(move_ok), move_ok,
                            jnp.arange(C.MOVES_PER_POKEMON) == 0)

        can_tera = jnp.logical_not(jnp.any(state.terastallized[side])) & \
                   (state.tera_type[side, i] != C.TYPE_NONE)
        switch_ok = (state.hp[side] > 0) & (jnp.arange(C.TEAM_SIZE) != i)

        row = jnp.concatenate([move_ok, move_ok & can_tera, switch_ok])
        # Replacing a fainted Pokemon: switches only.
        forced = state.force_switch[side]
        row = jnp.where(forced,
                        jnp.concatenate([jnp.zeros(8, bool), switch_ok]), row)
        # A player with nothing to do passes; slot 0 keeps the action space total.
        row = jnp.where(jnp.any(row), row, jnp.zeros(C.NUM_ACTIONS, bool).at[0].set(True))
        mask = mask.at[side].set(row)
    return mask
