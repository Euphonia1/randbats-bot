"""Build a psjax battle from an effect-case spec, and snapshot it for comparison.

The spec format is the one `tools/effect_cases.py` emits and
`tools/showdown_effects.js` consumes, so the same scenario drives both engines.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from psjax import consts as C
from psjax.data import load_data, names
from psjax.state import empty_state
from psjax.stats import compute_all_stats

DATA = load_data()
N = names()

STATUS_TO_PS = {C.STATUS_NONE: "", C.BRN: "brn", C.PAR: "par", C.SLP: "slp",
                C.FRZ: "frz", C.PSN: "psn", C.TOX: "tox"}
PS_TO_STATUS = {v: k for k, v in STATUS_TO_PS.items()}

WEATHER_TO_PS = {C.WEATHER_NONE: "", C.SUN: "sunnyday", C.RAIN: "raindance",
                 C.SAND: "sandstorm", C.SNOW: "snowscape",
                 C.HARSH_SUN: "desolateland", C.HEAVY_RAIN: "primordialsea",
                 C.STRONG_WINDS: "deltastream"}
PS_TO_WEATHER = {v: k for k, v in WEATHER_TO_PS.items()}

TERRAIN_TO_PS = {C.TERRAIN_NONE: "", C.ELECTRIC_TERRAIN: "electricterrain",
                 C.GRASSY_TERRAIN: "grassyterrain", C.MISTY_TERRAIN: "mistyterrain",
                 C.PSYCHIC_TERRAIN: "psychicterrain"}
PS_TO_TERRAIN = {v: k for k, v in TERRAIN_TO_PS.items()}

SIDE_TO_PS = {v: k for k, v in C.SIDE_CONDITION_IDX.items()}
VOL_TO_PS = {v: k for k, v in C.VOLATILE_IDX.items()}
BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]

#: Names psjax models at all. Showdown tracks many more volatiles and side
#: conditions than we do, so comparisons are restricted to this vocabulary --
#: anything outside it is a known gap, not a mismatch.
KNOWN_VOLATILES = frozenset(C.VOLATILE_IDX)
KNOWN_SIDE_CONDITIONS = frozenset(C.SIDE_CONDITION_IDX)


def build(case, key):
    """A `BattleState` matching the scenario spec, with both leads sent out."""
    state = empty_state(key)
    teams = ((0, case.get("p1team") or [case["p1"]]),
             (1, case.get("p2team") or [case["p2"]]))
    for side, team in teams:
      for slot, spec in enumerate(team):
        sid = N.species_id(spec["species"])
        base = DATA["species_base_stats"][sid]
        stats = compute_all_stats(base[None], jnp.array([100], jnp.int8))[0]
        moves = [N.move_id(m) for m in spec.get("moves") or ["splash"]]
        moves = (moves + [-1] * 4)[:4]
        maxhp = stats[C.HP]
        hp = maxhp
        if spec.get("hpPercent"):
            hp = jnp.int16(max(1, int(maxhp * spec["hpPercent"])))

        state = state._replace(
            species=state.species.at[side, slot].set(sid),
            level=state.level.at[side, slot].set(100),
            hp=state.hp.at[side, slot].set(hp),
            maxhp=state.maxhp.at[side, slot].set(maxhp),
            stats=state.stats.at[side, slot].set(stats),
            types=state.types.at[side, slot].set(DATA["species_types"][sid]),
            moves=state.moves.at[side, slot].set(jnp.asarray(moves, jnp.int16)),
            pp=state.pp.at[side, slot].set(jnp.full(4, 16, jnp.int8)),
            maxpp=state.maxpp.at[side, slot].set(jnp.full(4, 16, jnp.int8)),
            ability=state.ability.at[side, slot].set(N.ability_id(spec.get("ability", ""))),
            item=state.item.at[side, slot].set(N.item_id(spec.get("item", ""))),
            status=state.status.at[side, slot].set(
                PS_TO_STATUS.get(spec.get("status") or "", 0)),
            status_turns=state.status_turns.at[side, slot].set(
                1 if spec.get("status") == "tox" else 0),
        )
        if slot == 0:
            for name, amount in (spec.get("boosts") or {}).items():
                state = state._replace(
                    boosts=state.boosts.at[side, BOOST_KEYS.index(name)].set(amount))
            for sc in spec.get("sideConditions") or []:
                idx = C.SIDE_CONDITION_IDX[sc]
                layers = C.SIDE_CONDITION_MAX.get(idx)
                state = state._replace(
                    side_conditions=state.side_conditions.at[side, idx].set(
                        1 if layers else 5))

    if case.get("weather"):
        state = state._replace(weather=jnp.int8(PS_TO_WEATHER[case["weather"]]),
                               weather_turns=jnp.int8(8))
    if case.get("terrain"):
        state = state._replace(terrain=jnp.int8(PS_TO_TERRAIN[case["terrain"]]),
                               terrain_turns=jnp.int8(8))
    if case.get("seedP2"):
        state = state._replace(
            volatiles=state.volatiles.at[1, C.V_LEECHSEED].set(jnp.int8(1)))
    return state


def snapshot(state):
    """The same shape `tools/showdown_effects.js` dumps, for direct comparison."""
    def side(p):
        active = int(state.active[p])
        side_conditions = sorted(
            SIDE_TO_PS[i] for i in range(C.NUM_SIDE_CONDITIONS)
            if int(state.side_conditions[p, i]) > 0 and i in SIDE_TO_PS)
        volatiles = sorted(
            VOL_TO_PS[i] for i in range(C.NUM_VOLATILES)
            if int(state.volatiles[p, i]) > 0 and i in VOL_TO_PS)
        return {
            "active": active,
            "hp": [int(x) for x in state.hp[p]],
            "maxhp": [int(x) for x in state.maxhp[p]],
            "status": [STATUS_TO_PS[int(s)] for s in state.status[p]],
            "item": [N.items_by_id.get(int(i), "") if int(i) else ""
                     for i in state.item[p]],
            "boosts": {k: int(state.boosts[p, j]) for j, k in enumerate(BOOST_KEYS)},
            "sideConditions": side_conditions,
            "spikeLayers": int(state.side_conditions[p, C.SC_SPIKES]),
            "toxicSpikeLayers": int(state.side_conditions[p, C.SC_TOXICSPIKES]),
            "volatiles": volatiles,
            "subHP": int(state.sub_hp[p]),
        }

    return {
        "turn": int(state.turn),
        "weather": WEATHER_TO_PS[int(state.weather)],
        "terrain": TERRAIN_TO_PS[int(state.terrain)],
        "p1": side(0),
        "p2": side(1),
    }


def normalise(snap, field):
    """Pull one dotted field out of a snapshot, restricted to what psjax models."""
    parts = field.split(".")
    value = snap
    for part in parts:
        value = value[part]
    if parts[-1] == "volatiles":
        return sorted(set(value) & KNOWN_VOLATILES)
    if parts[-1] == "sideConditions":
        return sorted(set(value) & KNOWN_SIDE_CONDITIONS)
    if parts[-1] == "item":
        # Showdown reports display names; psjax reports ids.
        return [N.to_id(v) for v in value]
    return value
