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
from psjax.callbacks import base_power_modify
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
    cb_ctx = make_cb_ctx(case, atk, dfn, weather, terrain)
    mv = make_move(case["move"], cb_ctx)
    side = jnp.zeros(C.NUM_SIDE_CONDITIONS, jnp.int8)
    for s in case.get("screens", []):
        side = side.at[SCREENS[s]].set(5)

    def_types = current_types(dfn.types, dfn.terastallized, dfn.tera_type)
    exp, immune = type_effectiveness(
        data, mv.type, def_types, mv, data["move_ignore_immunity"][mv.id],
        dfn.ability, jnp.bool_(False))

    if bool(immune):
        return ["immune"] * 16
    return [int(calc_damage(
        data, atk, dfn, mv, is_crit=jnp.bool_(case.get("crit", False)),
        damage_roll=jnp.int32(roll), weather=weather, terrain=terrain,
        side_conditions=side, type_exp=exp,
        bp_cb_mod=base_power_modify(data["move_bp_modify"][mv.id], cb_ctx),
        grounded_user=cb_ctx.grounded_user,
        grounded_target=cb_ctx.grounded_target)) for roll in range(16)]


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
