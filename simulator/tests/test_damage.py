"""Differential test: our damage formula against Showdown's own `getDamage`.

`tools/showdown_damage.js` runs each case inside a real Showdown battle and
records the damage for all 16 rolls. Any mismatch here is a genuine divergence
from the reference implementation, not a rounding preference.
"""
from __future__ import annotations

import json
import pathlib

import jax.numpy as jnp
import pytest

from psjax import consts as C
from psjax.callbacks import base_power_modify, fixed_damage
from psjax.damage import calc_damage, current_types, type_effectiveness
from psjax.data import load_data, names
from helpers import make_attacker, make_cb_ctx, make_defender, make_move

TRUTH = pathlib.Path(__file__).resolve().parent.parent / "data" / "damage_truth.json"
CASES = json.load(open(TRUTH)) if TRUTH.exists() else []

WEATHER = {"sunnyday": C.SUN, "raindance": C.RAIN, "sandstorm": C.SAND,
           "snowscape": C.SNOW}
TERRAIN = {"electricterrain": C.ELECTRIC_TERRAIN, "grassyterrain": C.GRASSY_TERRAIN,
           "mistyterrain": C.MISTY_TERRAIN, "psychicterrain": C.PSYCHIC_TERRAIN}
SCREENS = {"reflect": C.SC_REFLECT, "lightscreen": C.SC_LIGHTSCREEN,
           "auroraveil": C.SC_AURORAVEIL}


def _run_case(case):
    data = load_data()
    atk = make_attacker(case["attacker"])
    dfn = make_defender(case["defender"])

    # Abilities set during case setup can create weather or terrain, so trust
    # what the reference battle reported over what the case requested.
    weather = jnp.int8(WEATHER.get(case.get("actualWeather") or case.get("weather"),
                                   C.WEATHER_NONE))
    terrain = jnp.int8(TERRAIN.get(case.get("actualTerrain") or case.get("terrain"),
                                   C.TERRAIN_NONE))
    # Air Lock and Cloud Nine switch the weather off for everyone. The engine
    # does this in `mechanics.effective_weather`; the harness has to match, or a
    # weather-boosted move looks boosted when it should not be.
    from psjax.hooks import A as _A
    suppressed = jnp.any(jnp.stack([atk.ability, dfn.ability]) == _A.AIRLOCK) | \
        jnp.any(jnp.stack([atk.ability, dfn.ability]) == _A.CLOUDNINE)
    weather = jnp.where(suppressed, jnp.int8(C.WEATHER_NONE), weather)

    cb_ctx = make_cb_ctx(case, atk, dfn, weather, terrain)
    mv = make_move(case["move"], cb_ctx)
    side = jnp.zeros(C.NUM_SIDE_CONDITIONS, jnp.int8)
    for s in case.get("screens", []):
        side = side.at[SCREENS[s]].set(5)

    def_types = current_types(dfn.types, dfn.terastallized, dfn.tera_type)
    exp, immune = type_effectiveness(
        data, mv.type, def_types, mv, data["move_ignore_immunity"][mv.id],
        dfn.ability, jnp.bool_(False))
    # Collision Course and friends key off the effectiveness, so the callback
    # context only becomes complete once it is known -- as in `execute_move`.
    cb_ctx = cb_ctx._replace(type_exp=exp)

    if bool(immune):
        return ["immune"] * 16

    # Moves with a `damageCallback` (Seismic Toss, Super Fang, Endeavor, ...)
    # bypass the damage formula entirely, exactly as `execute_move` does.
    fixed = fixed_damage(data["move_dmg_cb"][mv.id], cb_ctx)
    if int(fixed) >= 0:
        return [int(fixed)] * 16

    return [int(calc_damage(
        data, atk, dfn, mv, is_crit=jnp.bool_(case.get("crit", False)),
        damage_roll=jnp.int32(roll), weather=weather, terrain=terrain,
        side_conditions=side, type_exp=exp,
        bp_cb_mod=base_power_modify(data["move_bp_modify"][mv.id], cb_ctx),
        grounded_user=cb_ctx.grounded_user,
        grounded_target=cb_ctx.grounded_target,
        # Analytic keys off moving last, which is what a bare getDamage call
        # looks like to Showdown (there is no move queue to consult).
        analytic_ok=jnp.logical_not(cb_ctx.moves_first),
        technician_power=mv.base_power)) for roll in range(16)]


@pytest.mark.skipif(not CASES, reason="run tools/showdown_damage.js to build truth data")
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_damage_matches_showdown(case):
    got = _run_case(case)
    want = case["damages"]
    assert got == want, f"\n  showdown: {want}\n  psjax:    {got}"


@pytest.mark.skipif(not CASES, reason="no truth data")
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_stats_match_showdown(case):
    """Our stat formula must agree before the damage numbers can."""
    for role in ("attacker", "defender"):
        spec = case[role]
        p = make_attacker(spec) if role == "attacker" else make_defender(spec)
        want = case[f"{role}Stats"]
        got = p.stats
        assert int(got[C.ATK]) == want["atk"], f"{role} {spec['species']} atk"
        assert int(got[C.DEF]) == want["def"], f"{role} {spec['species']} def"
        assert int(got[C.SPA]) == want["spa"], f"{role} {spec['species']} spa"
        assert int(got[C.SPD]) == want["spd"], f"{role} {spec['species']} spd"
        assert int(got[C.SPE]) == want["spe"], f"{role} {spec['species']} spe"
        assert int(got[C.HP]) == case[f"{role}MaxHP"], f"{role} {spec['species']} hp"


# --- whole-movepool sweep ----------------------------------------------------

SWEEP_TRUTH = (pathlib.Path(__file__).resolve().parent.parent / "data"
               / "move_sweep_truth.json")
SWEEP = json.load(open(SWEEP_TRUTH)) if SWEEP_TRUTH.exists() else []

#: Moves this engine knowingly does not model, and why. Skipped rather than
#: deleted so the gap stays visible in the test report; `psjax.coverage` lists
#: them too. Anything not named here must match Showdown exactly.
KNOWN_UNMODELLED = {
    "beatup": "Beat Up's power and hit count come from the whole party's base "
              "Attack, which would need party data in the per-hit damage path. "
              "It appears in 1 of 4336 Random Battle sets (Fezandipiti only), "
              "so the hot-path cost is not worth it.",
}


@pytest.mark.skipif(not SWEEP, reason="run tools/move_sweep.py to build sweep data")
@pytest.mark.parametrize("case", SWEEP, ids=[c["name"] for c in SWEEP])
def test_every_randbats_move_matches_showdown(case):
    """Every damaging move in the Random Battle pool, against a live Showdown.

    The hand-written cases above probe particular mechanics; this checks that
    each compiled move row -- base power, type, category, base-power callback --
    is right for all 269 damaging moves the format can actually roll.
    """
    want = case["damages"]
    if all(d == 0 for d in want):
        pytest.skip("needs battle context a bare getDamage call cannot supply "
                    "(Counter/Mirror Coat read a prior hit; Triple Axel reads the "
                    "hit number) -- covered by the hand-written cases instead")
    if case["move"] in KNOWN_UNMODELLED:
        pytest.skip(KNOWN_UNMODELLED[case["move"]])
    got = _run_case(case)
    assert got == want, f"\n  showdown: {want}\n  psjax:    {got}"


# --- ability sweep -----------------------------------------------------------

ABILITY_TRUTH = (pathlib.Path(__file__).resolve().parent.parent / "data"
                 / "ability_sweep_truth.json")
ABILITIES = json.load(open(ABILITY_TRUTH)) if ABILITY_TRUTH.exists() else []


@pytest.mark.skipif(not ABILITIES, reason="run tools/ability_sweep.py to build data")
@pytest.mark.parametrize("case", ABILITIES, ids=[c["name"] for c in ABILITIES])
def test_every_wired_ability_matches_showdown(case):
    """Every ability the engine wires up, on both sides, against a live Showdown.

    Catches an ability applying when it should not, or on the wrong hook -- which
    is how Thick Fat, Heatproof, Water Bubble and Purifying Salt were found to be
    reducing final damage rather than the attacking stat.
    """
    want = case["damages"]
    got = _run_case(case)
    assert got == want, f"\n  showdown: {want}\n  psjax:    {got}"
