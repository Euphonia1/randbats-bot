"""Compile Showdown-generated teams into `data/team_pool.npz`.

Reads the newline-delimited JSON that `tools/dump_teams.js` writes and turns it
into the arrays the simulator instantiates a battle from. Names are resolved
through the same index the engine uses, and stats, types and PP are derived with
the engine's own helpers, so there is exactly one definition of each.

    node tools/dump_teams.js 20000 > data/team_pool.jsonl
    python tools/build_team_pool.py data/team_pool.jsonl

Reports anything the engine cannot represent rather than silently zeroing it --
an unmodelled item would otherwise turn into "no item" and quietly change the
teams the simulator plays with.
"""
from __future__ import annotations

import collections
import json
import os
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psjax import consts as C            # noqa: E402
from psjax.data import DATA_DIR, load_data, names  # noqa: E402
from psjax.stats import compute_all_stats  # noqa: E402

STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")


def main(path: str) -> None:
    nm, data = names(), load_data()
    teams = [json.loads(line) for line in open(path) if line.strip()]
    n = len(teams)
    if not n:
        raise SystemExit(f"{path} is empty")

    species = np.zeros((n, C.TEAM_SIZE), np.int16)
    level = np.zeros((n, C.TEAM_SIZE), np.int8)
    item = np.zeros((n, C.TEAM_SIZE), np.int16)
    ability = np.zeros((n, C.TEAM_SIZE), np.int16)
    moves = np.full((n, C.TEAM_SIZE, C.MOVES_PER_POKEMON), -1, np.int16)
    tera = np.zeros((n, C.TEAM_SIZE), np.int8)
    ivs = np.full((n, C.TEAM_SIZE, 6), 31, np.int32)
    evs = np.full((n, C.TEAM_SIZE, 6), 85, np.int32)

    missing = collections.Counter()
    cosmetic = collections.Counter()
    for t, team in enumerate(teams):
        if len(team) != C.TEAM_SIZE:
            raise SystemExit(f"team {t} has {len(team)} members, expected {C.TEAM_SIZE}")
        for s, mon in enumerate(team):
            name = mon["species"]
            if nm.to_id(name) not in nm.species:
                # A cosmetic forme the engine does not index separately. The
                # base shares stats, types and abilities, so this changes
                # nothing mechanically -- but count it rather than hide it.
                cosmetic[f"{name} -> {mon['base']}"] += 1
                name = mon["base"]
            species[t, s] = nm.species_id(name)
            level[t, s] = mon["level"]
            tera[t, s] = nm.type_id(mon["tera"])

            key = nm.to_id(mon["item"])
            if mon["item"] and key not in nm.items:
                missing[f"item:{mon['item']}"] += 1
            item[t, s] = nm.item_id(mon["item"])

            key = nm.to_id(mon["ability"])
            if key not in nm.abilities:
                missing[f"ability:{mon['ability']}"] += 1
            ability[t, s] = nm.ability_id(mon["ability"])

            for m, mv in enumerate(mon["moves"][:C.MOVES_PER_POKEMON]):
                if nm.to_id(mv) not in nm.moves:
                    missing[f"move:{mv}"] += 1
                    continue
                moves[t, s, m] = nm.move_id(mv)

            for i, sk in enumerate(STAT_KEYS):
                ivs[t, s, i] = mon["ivs"][sk]
                evs[t, s, i] = mon["evs"][sk]

    # Derived with the engine's own code, not a second implementation of it.
    base = np.asarray(data["species_base_stats"])[species]
    stats = np.asarray(compute_all_stats(
        jnp.asarray(base), jnp.asarray(level.astype(np.int32)),
        jnp.asarray(ivs), jnp.asarray(evs))).astype(np.int16)
    types = np.asarray(data["species_types"])[species].astype(np.int8)
    move_pp = np.asarray(data["move_pp"])
    pp = np.where(moves >= 0,
                  (move_pp[np.maximum(moves, 0)].astype(np.int32) * 8) // 5,
                  0).astype(np.int8)

    out = DATA_DIR / "team_pool.npz"
    np.savez_compressed(
        out,
        pool_species=species, pool_level=level, pool_stats=stats,
        pool_types=types, pool_moves=moves, pool_pp=pp,
        pool_item=item, pool_ability=ability, pool_tera_type=tera)

    print(f"{n} teams -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"distinct species {len(set(species.ravel().tolist()))}, "
          f"items {len(set(item.ravel().tolist()))}, "
          f"moves {len(set(moves.ravel().tolist()) - {-1})}")
    empty = int((moves < 0).sum())
    if empty:
        print(f"note: {empty} empty move slots "
              f"({empty / (n * C.TEAM_SIZE * 4) * 100:.2f}% of all slots)")
    if cosmetic:
        total = sum(cosmetic.values())
        print(f"\ncosmetic formes folded into their base ({total} slots): "
              f"{', '.join(sorted(cosmetic))}")
    if missing:
        print("\nNOT REPRESENTABLE -- these were dropped:")
        for name, count in missing.most_common(20):
            print(f"  {count:7d}  {name}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "team_pool.jsonl"))
