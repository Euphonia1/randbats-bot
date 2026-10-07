"""Generate damage cases for every ability the engine actually wires up.

`move_sweep.py` checks each move's compiled row; this does the same for
abilities. Each one is put on the attacker and on the defender, across contexts
chosen to trigger as many as possible in few cases: the attacker is burned and
at a third of its HP (Guts, Overgrow, Flare Boost, Marvel Scale, Defeatist),
and both sun and rain are represented.

An ability that needs some other trigger will simply not fire -- but the case
still catches it firing when it should not, which is how the Thick Fat family of
bugs showed up.

    python tools/ability_sweep.py > tools/ability_sweep_cases.json
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from psjax.coverage import wired_abilities            # noqa: E402

raw = json.load(open("data/gen9_raw.json"))
display = {aid: a["name"] for aid, a in raw["abilities"].items()}

# Neutral bodies; the harness forces the ability, so typing is all the species
# contributes. Close Combat is super effective on Normal-type Blissey, which
# exercises Filter, Solid Rock, Tinted Lens and Expert Belt.
ATK = {"species": "Clefable"}
DEF_NORMAL = {"species": "Blissey"}
DEF_WATER = {"species": "Slowbro"}

CONTEXTS = [
    ("phys burned pinch sun", "closecombat",
     {"status": "brn", "hpPercent": 0.3}, {"weather": "sunnyday"}),
    ("spec rain", "surf", {}, {"weather": "raindance"}),
    # A burned special attacker in a pinch using a Fire move with a secondary:
    # Blaze, Flare Boost and Sheer Force, none of which the first two reach.
    ("spec burned pinch fire", "flamethrower", {"status": "brn", "hpPercent": 0.3}, {}),
    # A poisoned physical attacker: Toxic Boost.
    ("phys poisoned", "closecombat", {"status": "psn"}, {}),
]

cases = []
for ability in sorted(wired_abilities()):
    if ability not in display or ability == "none":
        continue
    name = display[ability]
    for label, move, a_extra, field in CONTEXTS:
        cases.append({
            "name": f"ability atk {ability} [{label}]",
            "attacker": {**ATK, "ability": name, **a_extra},
            "defender": {**DEF_NORMAL, "ability": "Natural Cure"},
            "move": move, "movesFirst": False, **field,
        })
        cases.append({
            "name": f"ability def {ability} [{label}]",
            "attacker": {**ATK, "ability": "Unaware", **a_extra},
            "defender": {**DEF_WATER, "ability": name},
            "move": move, "movesFirst": False, **field,
        })

print(json.dumps(cases, indent=1))
