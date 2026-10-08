"""Mechanics the Showdown differential harness cannot observe.

`test_effects.py` compares what a turn does to the battle. These pin the rest:
which actions a player is offered (trapping, Disable, Encore, Taunt, a charging
move), what a Pokemon turns back into when it leaves the field, what the team
builder derives from a team, what Illusion shows an opponent, and the public
sleep count.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from psjax import consts as C
from psjax.data import load_data, load_team_pool, names
from psjax.engine import step, switch_to
from psjax.env import BattleEnv
from psjax.fog import FogOfWarEnv, FogState, empty_history
from psjax.mechanics import set_status, species_stats
from psjax.teams import legal_action_mask, new_battle
from scenario import build

DATA = load_data()
N = names()
JIT_STEP = jax.jit(step)


def mon(species, ability, moves=("splash",), **kw):
    return {"species": species, "ability": ability, "moves": list(moves), **kw}


def battle(p1, p2, p1team=None, p2team=None, **kw):
    case = {"p1": p1, "p2": p2, **kw}
    if p1team:
        case["p1team"] = p1team
    if p2team:
        case["p2team"] = p2team
    return build(case, jax.random.PRNGKey(0))


BENCH = mon("Blissey", "Natural Cure")


def switches_offered(state, side):
    mask = legal_action_mask(DATA, state)[side]
    return bool(jnp.any(mask[C.ACTION_SWITCH_BASE:]))


def moves_offered(state, side):
    mask = legal_action_mask(DATA, state)[side]
    return [int(i) for i in np.flatnonzero(np.asarray(mask[:C.ACTION_TERA_BASE]))]


# --- trapping -------------------------------------------------------------------

def test_shadow_tag_traps_the_foe_but_not_another_shadow_tag():
    state = battle(mon("Clefable", "Unaware"), mon("Gothitelle", "Shadow Tag"),
                   p1team=[mon("Clefable", "Unaware"), BENCH],
                   p2team=[mon("Gothitelle", "Shadow Tag"), BENCH])
    assert not switches_offered(state, 0), "Shadow Tag's foe must stay in"
    assert switches_offered(state, 1), "Shadow Tag does not trap its holder"
    tagged = battle(mon("Gothitelle", "Shadow Tag"), mon("Gothitelle", "Shadow Tag"),
                    p1team=[mon("Gothitelle", "Shadow Tag"), BENCH])
    assert switches_offered(tagged, 0), "Shadow Tag does not trap another Shadow Tag"


def test_arena_trap_only_traps_the_grounded():
    team = lambda lead: [lead, BENCH]
    grounded = battle(mon("Clefable", "Unaware"), mon("Dugtrio", "Arena Trap"),
                      p1team=team(mon("Clefable", "Unaware")))
    flying = battle(mon("Corviknight", "Pressure"), mon("Dugtrio", "Arena Trap"),
                    p1team=team(mon("Corviknight", "Pressure")))
    floating = battle(mon("Rotom-Wash", "Levitate"), mon("Dugtrio", "Arena Trap"),
                      p1team=team(mon("Rotom-Wash", "Levitate")))
    assert not switches_offered(grounded, 0)
    assert switches_offered(flying, 0), "a Flying type is not grounded"
    assert switches_offered(floating, 0), "Levitate is not grounded"


def test_magnet_pull_only_traps_steel_types():
    steel = battle(mon("Corviknight", "Pressure"), mon("Magnezone", "Magnet Pull"),
                   p1team=[mon("Corviknight", "Pressure"), BENCH])
    other = battle(mon("Clefable", "Unaware"), mon("Magnezone", "Magnet Pull"),
                   p1team=[mon("Clefable", "Unaware"), BENCH])
    assert not switches_offered(steel, 0)
    assert switches_offered(other, 0)


def test_ghost_types_slip_any_trap():
    state = battle(mon("Gengar", "Cursed Body"), mon("Gothitelle", "Shadow Tag"),
                   p1team=[mon("Gengar", "Cursed Body"), BENCH])
    assert switches_offered(state, 0)


def test_a_trap_never_blocks_replacing_a_fainted_pokemon():
    state = battle(mon("Clefable", "Unaware"), mon("Gothitelle", "Shadow Tag"),
                   p1team=[mon("Clefable", "Unaware"), BENCH])
    state = state._replace(hp=state.hp.at[0, 0].set(0),
                           force_switch=state.force_switch.at[0].set(True),
                           phase=jnp.int8(C.PHASE_SWITCH))
    assert switches_offered(state, 0)


# --- move restrictions ----------------------------------------------------------

FOUR = ["tackle", "swordsdance", "seismictoss", "protect"]


def test_disable_takes_one_move_off_the_menu():
    state = battle(mon("Clefable", "Unaware", FOUR), mon("Blissey", "Natural Cure"))
    state = state._replace(volatiles=state.volatiles.at[0, C.V_DISABLE].set(3),
                           disabled_slot=state.disabled_slot.at[0].set(2))
    assert moves_offered(state, 0) == [0, 1, 3]


def test_encore_leaves_only_the_encored_move():
    state = battle(mon("Clefable", "Unaware", FOUR), mon("Blissey", "Natural Cure"))
    state = state._replace(volatiles=state.volatiles.at[0, C.V_ENCORE].set(3),
                           encore_slot=state.encore_slot.at[0].set(1))
    assert moves_offered(state, 0) == [1]


def test_taunt_removes_status_moves():
    state = battle(mon("Clefable", "Unaware", FOUR), mon("Blissey", "Natural Cure"))
    state = state._replace(volatiles=state.volatiles.at[0, C.V_TAUNT].set(3))
    assert moves_offered(state, 0) == [0, 2]


def test_a_charging_move_is_the_only_choice_and_traps():
    """Turn one of Solar Beam leaves the user locked in, and unable to switch."""
    state = battle(mon("Venusaur", "Overgrow", ["solarbeam", "tackle"]), BENCH,
                   p1team=[mon("Venusaur", "Overgrow", ["solarbeam", "tackle"]), BENCH])
    state = JIT_STEP(state, jnp.array([0, 0], jnp.int32))
    assert int(state.locked_slot[0]) == 0
    assert moves_offered(state, 0) == [0]
    assert not switches_offered(state, 0)
    # The second turn fires it and releases the lock.
    state = JIT_STEP(state, jnp.array([1, 0], jnp.int32))
    assert int(state.locked_slot[0]) == -1
    assert int(state.hp[1, 0]) < int(state.maxhp[1, 0])


# --- leaving the field ----------------------------------------------------------

def test_transform_ends_on_switching_out():
    ditto = mon("Ditto", "Imposter")
    state = battle(ditto, mon("Gyarados", "Moxie", ["waterfall", "dragondance"]),
                   p1team=[ditto, BENCH])
    assert bool(state.transformed[0])
    assert int(state.species[0, 0]) == N.species_id("gyarados")
    assert int(state.moves[0, 0, 0]) == N.move_id("waterfall")
    assert int(state.pp[0, 0, 0]) == 5, "a Transformed move has 5 PP"
    own = species_stats(DATA, jnp.int16(N.species_id("ditto")), jnp.int8(100), jnp.int8(0))

    state = switch_to(DATA, state, 0, jnp.int32(1))
    assert not bool(state.transformed[0])
    assert int(state.species[0, 0]) == N.species_id("ditto")
    assert int(state.ability[0, 0]) == N.ability_id("imposter")
    assert int(state.moves[0, 0, 0]) == N.move_id("splash")
    assert int(state.pp[0, 0, 0]) == 16
    assert bool(jnp.all(state.stats[0, 0, 1:] == own[1:]))


def test_a_battle_forme_reverts_on_switching_out():
    """Relic Song's Pirouette Forme and Protean's typing do not outlast the switch."""
    meloetta = mon("Meloetta", "Serene Grace", ["relicsong"])
    state = battle(meloetta, BENCH, p1team=[meloetta, BENCH])
    state = JIT_STEP(state, jnp.array([0, 0], jnp.int32))
    assert int(state.species[0, 0]) == N.species_id("meloettapirouette")
    state = switch_to(DATA, state, 0, jnp.int32(1))
    assert int(state.species[0, 0]) == N.species_id("meloetta")
    aria = species_stats(DATA, jnp.int16(N.species_id("meloetta")), jnp.int8(100), jnp.int8(0))
    assert bool(jnp.all(state.stats[0, 0, 1:] == aria[1:]))


def test_a_permanent_forme_survives_switching_out():
    mimikyu = mon("Mimikyu", "Disguise")
    state = battle(mimikyu, BENCH, p1team=[mimikyu, BENCH])
    state = state._replace(species=state.species.at[0, 0].set(N.species_id("mimikyubusted")),
                           base_species=state.base_species.at[0, 0].set(
                               N.species_id("mimikyubusted")))
    state = switch_to(DATA, state, 0, jnp.int32(1))
    assert int(state.species[0, 0]) == N.species_id("mimikyubusted")


# --- the team builder -------------------------------------------------------------

def test_derived_spreads_rebuild_every_pool_pokemon_exactly():
    """The zeroed-stat mask must reproduce each Pokemon's stats from its species.

    Formes and Transform's revert rebuild stats this way, so a mismatch here
    would quietly change a Pokemon's stats the first time its forme changed.
    """
    if load_team_pool() is None:
        return
    rebuild = jax.jit(jax.vmap(jax.vmap(
        lambda sp, lv, z: species_stats(DATA, sp, lv, z))))
    for seed in range(8):
        state = new_battle(jax.random.PRNGKey(seed), DATA)
        # Leads may have changed forme or Transformed on entry; judge them by
        # the species they started as.
        rebuilt = rebuild(state.base_species, state.level, state.spread_zero)
        benched = jnp.arange(C.TEAM_SIZE)[None, :] > 0
        same = jnp.all(rebuilt[..., 1:] == state.stats[..., 1:], axis=-1)
        assert bool(jnp.all(same | ~benched)), f"seed {seed}"


def test_genders_follow_species_or_a_coin_flip():
    genders = []
    for seed in range(16):
        state = new_battle(jax.random.PRNGKey(seed), DATA)
        fixed = np.asarray(DATA["species_gender"][state.species])
        random = np.asarray(DATA["species_random_gender"][state.species])
        g = np.asarray(state.gender)
        assert np.all(np.where(random, g != C.GENDER_NONE, g == fixed))
        genders.extend(g[random].tolist())
    counts = np.bincount(genders, minlength=3)
    assert counts[C.GENDER_M] > 0 and counts[C.GENDER_F] > 0


# --- Illusion under fog of war -------------------------------------------------------

def test_illusion_shows_the_disguise_until_hit():
    zoroark = mon("Zoroark-Hisui", "Illusion", ["tackle"])
    state = battle(zoroark, mon("Clefable", "Unaware", ["moonblast"]),
                   p1team=[zoroark, BENCH, mon("Corviknight", "Pressure")])
    assert int(state.illusion[0]) == 2, "Illusion copies the last healthy party member"

    env = FogOfWarEnv(BattleEnv(DATA))
    fs = FogState(battle=state, revealed=jnp.zeros((2, 6), bool),
                  revealed_moves=jnp.zeros((2, 6, 4), bool), history=empty_history())
    seen = env._censor(fs, 0)
    assert bool(jnp.all(seen.types[0, 0] == state.types[0, 2])), \
        "the opponent sees Corviknight's typing"
    assert bool(jnp.all(env._censor(fs, 1).types[0, 0] == state.types[0, 0])), \
        "the player itself sees the truth"

    # A damaging hit breaks it.
    state = JIT_STEP(state, jnp.array([0, 0], jnp.int32))
    assert int(state.illusion[0]) == -1


# --- sleep -----------------------------------------------------------------------

def _asleep(ability, roll):
    """A Dodrio put to sleep with roll 0, 1 or 2 (2-4 turns on the counter)."""
    state = battle(mon("Dodrio", ability), BENCH)
    state, ok = set_status(DATA, state, 0, jnp.int8(C.SLP), jnp.uint32(roll))
    assert bool(ok)
    return state


def _attempts_until_awake(state):
    """`sleep_attempts` after each turn the sleeper tries to move, until it wakes."""
    seen = []
    for _ in range(5):
        state = JIT_STEP(state, jnp.array([0, 0], jnp.int32))
        if int(state.status[0, 0]) != C.SLP:
            assert int(state.sleep_attempts[0, 0]) == 0, "waking clears the count"
            return seen
        seen.append(int(state.sleep_attempts[0, 0]))
    raise AssertionError("never woke")


def test_sleep_counts_each_attempt_that_stays_asleep():
    """Sleep lasts 1-3 failed attempts; the waking attempt is not one of them."""
    for roll, failed in [(0, 1), (1, 2), (2, 3)]:
        assert _attempts_until_awake(_asleep("Run Away", roll)) == list(range(1, failed + 1))


def test_early_bird_ticks_twice_without_shortening_the_roll():
    """Showdown's Early Bird only doubles the tick, so it wakes after 0, 1 or 1 failed
    attempts -- not always on the first."""
    for roll, turns in [(0, 2), (1, 3), (2, 4)]:
        assert int(_asleep("Early Bird", roll).status_turns[0, 0]) == turns
    for roll, failed in [(0, 0), (1, 1), (2, 1)]:
        assert len(_attempts_until_awake(_asleep("Early Bird", roll))) == failed


def test_rest_sleeps_through_exactly_two_attempts():
    state = battle(mon("Snorlax", "Thick Fat", ["rest"]), BENCH)
    state = state._replace(hp=state.hp.at[0, 0].set(1))
    state = JIT_STEP(state, jnp.array([0, 0], jnp.int32))
    assert int(state.status[0, 0]) == C.SLP
    assert bool(state.rest_sleep[0, 0]) and int(state.sleep_attempts[0, 0]) == 0
    assert _attempts_until_awake(state) == [1, 2]


def test_the_sleep_count_survives_switching_and_restarts_with_a_new_sleep():
    dodrio = mon("Dodrio", "Run Away")
    state = battle(dodrio, BENCH, p1team=[dodrio, BENCH])
    state, _ = set_status(DATA, state, 0, jnp.int8(C.SLP), jnp.uint32(2))
    state = JIT_STEP(state, jnp.array([0, 0], jnp.int32))
    assert int(state.sleep_attempts[0, 0]) == 1
    state = switch_to(DATA, state, 0, jnp.int32(1))
    state = switch_to(DATA, state, 0, jnp.int32(0))
    assert int(state.sleep_attempts[0, 0]) == 1, "the count waits on the bench, as the sleep does"

    # A fresh sleep starts from zero and is not a Rest, whatever came before.
    state = state._replace(status=state.status.at[0, 0].set(C.STATUS_NONE),
                           rest_sleep=state.rest_sleep.at[0, 0].set(True))
    state, _ = set_status(DATA, state, 0, jnp.int8(C.SLP), jnp.uint32(0))
    assert int(state.sleep_attempts[0, 0]) == 0 and not bool(state.rest_sleep[0, 0])


def test_the_opponent_sees_how_long_a_pokemon_has_slept_but_not_how_long_is_left():
    state = _asleep("Run Away", 2)
    env = FogOfWarEnv(BattleEnv(DATA))
    fs = FogState(battle=state, revealed=jnp.zeros((2, 6), bool),
                  revealed_moves=jnp.zeros((2, 6, 4), bool), history=empty_history())
    seen = lambda st: env.observe(fs._replace(battle=st))[1]   # player 1's view
    base = seen(state)
    assert not bool(jnp.allclose(base, seen(state._replace(
        sleep_attempts=state.sleep_attempts.at[0, 0].set(2)))))
    assert not bool(jnp.allclose(base, seen(state._replace(
        rest_sleep=state.rest_sleep.at[0, 0].set(True)))))
    assert bool(jnp.allclose(base, seen(state._replace(
        status_turns=state.status_turns.at[0, 0].set(1))))), "turns left stay hidden"


# --- found replaying whole battles against Showdown -------------------------------

def _hp_lost(before, after, side):
    return int(before.hp[side, before.active[side]]) - int(after.hp[side, after.active[side]])


def test_beat_up_reads_each_sets_species_in_party_order():
    """5 + base Attack / 10 of the species the set names -- Terapagos (65), not the
    Terastal Form (95) it has become -- user first, then by party position."""
    from psjax.moves import _beat_up_powers
    team = [mon("Fezandipiti", "Technician", ["beatup"]), mon("Haxorus", "Mold Breaker"),
            mon("Terapagos", "Tera Shift")]
    state = battle(team[0], BENCH, p1team=team)
    terastal = N.species_id("terapagosterastal")
    state = state._replace(species=state.species.at[0, 2].set(terastal),
                           base_species=state.base_species.at[0, 2].set(terastal),
                           party_pos=state.party_pos.at[0].set(jnp.array([0, 2, 1, 3, 4, 5],
                                                                         jnp.int8)))
    powers, hits = _beat_up_powers(DATA, state, 0)
    assert int(hits) == 3
    assert [int(p) for p in powers[:3]] == [5 + 91 // 10, 5 + 65 // 10, 5 + 147 // 10]


def test_hits_after_ice_face_breaks_meet_noice_defense():
    cloyster = mon("Cloyster", "Skill Link", ["iciclespear"])
    dealt = {}
    for species in ("Eiscue", "Eiscue-Noice"):
        state = battle(cloyster, mon(species, "Ice Face"))
        after = JIT_STEP(state, jnp.array([0, 0], jnp.int32))
        dealt[species] = _hp_lost(state, after, 1)
    # The first of five hits is absorbed; the other four land on Noice's 70
    # base Defense (on Eiscue's 110 they would come to barely half).
    assert dealt["Eiscue"] > 0.7 * dealt["Eiscue-Noice"]


def test_seed_sower_terrain_powers_the_rest_of_a_multi_hit_move():
    cinccino = mon("Cinccino", "Skill Link", ["bulletseed"])
    dealt = {}
    for ability, terrain in (("Harvest", ""), ("Seed Sower", ""),
                             ("Harvest", "grassyterrain")):
        state = battle(cinccino, mon("Arboliva", ability), terrain=terrain)
        after = JIT_STEP(state, jnp.array([0, 0], jnp.int32))
        # Less the terrain's healing at the end of the turn.
        healed = int(after.maxhp[1, 0]) // 16 if int(after.terrain) == C.GRASSY_TERRAIN else 0
        dealt[ability, terrain] = _hp_lost(state, after, 1) + healed
    assert dealt["Harvest", ""] < dealt["Seed Sower", ""] < dealt["Harvest", "grassyterrain"]


def test_endeavor_fails_without_contact_unless_the_target_has_more_hp():
    for user_hp, lands in ((1.0, False), (0.1, True)):
        state = battle(mon("Luvdisc", "Hydration", ["endeavor"], hpPercent=user_hp),
                       mon("Goodra", "Gooey", hpPercent=0.5))
        after = JIT_STEP(state, jnp.array([0, 0], jnp.int32))
        assert (_hp_lost(state, after, 1) > 0) == lands
        assert int(after.boosts[0, C.B_SPE]) == (-1 if lands else 0), "Gooey needs contact"


def test_transform_lets_go_of_a_choice_lock():
    ditto = mon("Ditto", "Limber", ["transform"], item="Choice Scarf")
    state = battle(ditto, mon("Blastoise", "Torrent",
                              ["shellsmash", "hydropump", "icebeam", "earthquake"]))
    state = JIT_STEP(state, jnp.array([0, 0], jnp.int32))
    assert bool(state.transformed[0])
    assert moves_offered(state, 0) == [0, 1, 2, 3]


def test_a_berry_unnerve_held_back_is_eaten_as_soon_as_it_leaves():
    """Before the newcomer's own Unnerve starts, too."""
    pyroar = mon("Pyroar", "Unnerve")
    state = battle(mon("Snorlax", "Thick Fat", item="Chesto Berry", status="slp",
                       statusTurns=3), pyroar, p2team=[pyroar, pyroar])
    after = JIT_STEP(state, jnp.array([0, C.ACTION_SWITCH_BASE + 1], jnp.int32))
    assert int(after.item[0, 0]) == 0
    assert int(after.status[0, 0]) == C.STATUS_NONE


def test_a_berry_the_weather_triggers_is_eaten_before_harvest():
    """Sandstorm's chip is followed by an Update; the berry goes down then, and
    Harvest, later in the residuals, can grow it back the same turn."""
    regrown = 0
    for seed in range(8):
        state = battle(mon("Exeggutor-Alola", "Harvest", item="Sitrus Berry", hpPercent=0.52),
                       BENCH, weather="sandstorm")
        state = state._replace(key=jax.random.PRNGKey(seed))
        after = JIT_STEP(state, jnp.array([0, 0], jnp.int32))
        sitrus = N.item_id("Sitrus Berry")
        assert int(after.hp[0, 0]) > int(state.hp[0, 0]), "the Sitrus Berry was eaten"
        if int(after.item[0, 0]) == sitrus:
            assert int(after.last_item[0, 0]) == 0
            regrown += 1
        else:
            assert int(after.last_item[0, 0]) == sitrus
    assert regrown > 0


def test_no_secondary_drop_lands_on_a_knocked_out_target():
    """So Mirror Armor has nothing to bounce."""
    for hp, bounced in ((1.0, True), (0.01, False)):
        state = battle(mon("Blissey", "Natural Cure", ["mysticalfire"]),
                       mon("Corviknight", "Mirror Armor", hpPercent=hp),
                       p2team=[mon("Corviknight", "Mirror Armor", hpPercent=hp), BENCH])
        after = JIT_STEP(state, jnp.array([0, 0], jnp.int32))
        assert int(after.boosts[0, C.B_SPA]) == (-1 if bounced else 0)
