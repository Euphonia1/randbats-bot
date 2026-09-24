"""Print a Pokemon's exact stats at a given level, with 31 IVs and 85 EVs.

    python model/tools/stats_at_level.py <species> [level ...]

    python model/tools/stats_at_level.py garchomp 78
    python model/tools/stats_at_level.py "Iron Valiant" 50 79 100

With no level, uses the species' Random Battle level. Stats come from the
engine's own `compute_all_stats` (neutral nature, Showdown's integer
rounding), so they match what a battle instantiates.
"""
from __future__ import annotations

import difflib
import sys

import jax.numpy as jnp
import numpy as np

from psjax import consts as C
from psjax.data import load_data, names
from psjax.stats import compute_all_stats


def stats_at(species: int, level: int) -> list[int]:
    """`[hp, atk, def, spa, spd, spe]` at `level`, 31 IVs and 85 EVs."""
    base = load_data()["species_base_stats"][species]
    return [int(s) for s in compute_all_stats(base, jnp.asarray(level))]


def randbats_level(species: int) -> int | None:
    data = load_data()
    hits = np.nonzero(np.asarray(data["rb_species"]) == species)[0]
    return int(data["rb_level"][hits[0]]) if len(hits) else None


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if argv else 1

    n = names()
    sid = n.to_id(argv[0])
    if sid not in n.species:
        close = difflib.get_close_matches(sid, n.species, n=5)
        print(f"unknown species {argv[0]!r}"
              + (f"; did you mean: {', '.join(close)}?" if close else ""),
              file=sys.stderr)
        return 1
    species = n.species[sid]

    if len(argv) > 1:
        levels = [int(a) for a in argv[1:]]
    else:
        lvl = randbats_level(species)
        if lvl is None:
            print(f"{sid} has no Random Battle set; pass a level", file=sys.stderr)
            return 1
        levels = [lvl]
    for lvl in levels:
        if not 1 <= lvl <= 100:
            print(f"level must be 1-100, got {lvl}", file=sys.stderr)
            return 1

    base = [int(b) for b in load_data()["species_base_stats"][species]]
    header = "".join(f"{s:>6}" for s in C.STAT_NAMES)
    print(f"{sid}  (31 IVs, 85 EVs, neutral nature)")
    print(f"{'':>6}{header}")
    print(f"{'base':>6}" + "".join(f"{b:>6}" for b in base))
    for lvl in levels:
        print(f"{'L' + str(lvl):>6}" + "".join(f"{s:>6}" for s in stats_at(species, lvl)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
