"""Generate a damage case for every damaging move in the Random Battle pool.

The hand-written cases in `damage_cases.json` probe specific mechanics. This
sweeps the whole movepool instead, so every compiled row -- base power, type,
category, and any base-power callback -- is checked against Showdown at least
once. Status moves are excluded: they deal no damage, and their effects are
covered by the effects harness.

    python tools/move_sweep.py > tools/move_sweep_cases.json
"""
import collections
import json

raw = json.load(open("data/gen9_raw.json"))
moves, randbats = raw["moves"], raw["randbats"]


def to_id(s):
    return "".join(c for c in s.lower() if c.isalnum())


used = collections.Counter()
for entry in randbats.values():
    for s in entry["sets"]:
        for mv in s["movepool"]:
            used[to_id(mv)] += 1

# Clefable is Fairy (few odd interactions) and Unaware keeps boosts out of it.
# The defender is chosen per move so the hit is never a type immunity -- an
# immune result would tell us nothing about the move's compiled row.
ATTACKER = {"species": "Clefable", "ability": "Unaware"}
DEFENDERS = [
    {"species": "Blissey", "ability": "Natural Cure"},      # Normal
    {"species": "Clefable", "ability": "Unaware"},          # Fairy
    {"species": "Skarmory", "ability": "Sturdy"},           # Steel / Flying
]

typechart = raw["typechart"]


def effectiveness(move_type, defender_species):
    """Type multiplier, so the generator can avoid immunities."""
    types = raw["species"][to_id(defender_species)]["types"]
    mult = 1.0
    for t in types:
        taken = typechart[to_id(t)]["damageTaken"].get(move_type, 0)
        mult *= {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}.get(taken, 1.0)
    return mult


def pick_defender(move):
    for d in DEFENDERS:
        if effectiveness(move["type"], d["species"]) > 0:
            return d
    return DEFENDERS[0]

cases = []
for mid in sorted(used):
    m = moves.get(mid)
    if not m or m["category"] == "Status":
        continue
    # Charge moves and multi-turn moves do not resolve damage on the tested turn
    # in a single getDamage call; the harness measures the hit itself, which is
    # what we want, so they are kept.
    cases.append({
        "name": f"sweep {mid}",
        "attacker": dict(ATTACKER),
        "defender": dict(pick_defender(m)),
        "move": mid,
        # A bare getDamage call has no move queue, so Showdown treats the user
        # as moving second -- match that for the order-sensitive callbacks.
        "movesFirst": False,
    })

print(json.dumps(cases, indent=1))
