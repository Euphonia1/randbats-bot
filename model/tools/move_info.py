"""Print a move's type, category, base power and flags.

    python model/tools/move_info.py [--one-hot] <move> [move ...]

    python model/tools/move_info.py earthquake
    python model/tools/move_info.py "Close Combat" uturn "Swords Dance"
    python model/tools/move_info.py --one-hot earthquake

`--one-hot` prints the flags as a 0/1 vector with one entry per flag in
`FLAG_NAMES` order instead of as names.

Values come from the engine's compiled tables (`gen9.npz`), so they match what
a battle actually uses.
"""
from __future__ import annotations

import difflib
import sys

import numpy as np

from psjax import consts as C
from psjax.data import load_data, names

CATEGORY_NAMES = {v: k for k, v in C.CATEGORIES.items()}
# Flag names in bit order; index i of `flags_one_hot` is `FLAG_NAMES[i]`.
FLAG_NAMES = sorted(C.FLAG_BITS, key=C.FLAG_BITS.get)


def flags_one_hot(move: int) -> np.ndarray:
    """0/1 vector over `FLAG_NAMES`: 1 where the move has that flag."""
    mask = int(load_data()["move_flags"][move])
    return np.array([mask >> C.FLAG_BITS[f] & 1 for f in FLAG_NAMES], dtype=np.int8)


def move_info(move: int, one_hot: bool = False) -> dict:
    """`{type, category, base_power, flags}` for a move index.

    `flags` is a list of names, or with `one_hot` the `flags_one_hot` vector.
    """
    data = load_data()
    n = names()
    mask = int(data["move_flags"][move])
    return {
        "type": n.types_by_id[int(data["move_type"][move])],
        "category": CATEGORY_NAMES[int(data["move_category"][move])],
        "base_power": int(data["move_base_power"][move]),
        "flags": (flags_one_hot(move) if one_hot
                  else [f for f in FLAG_NAMES if mask >> C.FLAG_BITS[f] & 1]),
    }


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip())
        return 0 if argv else 1

    one_hot = "--one-hot" in argv
    argv = [a for a in argv if a != "--one-hot"]
    if not argv:
        print("no move given", file=sys.stderr)
        return 1

    n = names()
    status = 0
    for arg in argv:
        mid = n.to_id(arg)
        if mid not in n.moves:
            close = difflib.get_close_matches(mid, n.moves, n=5)
            print(f"unknown move {arg!r}"
                  + (f"; did you mean: {', '.join(close)}?" if close else ""),
                  file=sys.stderr)
            status = 1
            continue
        info = move_info(n.moves[mid], one_hot)
        print(mid)
        print(f"  type:       {info['type']}")
        print(f"  category:   {info['category']}")
        print(f"  base power: {info['base_power']}")
        if one_hot:
            print(f"  flags:      {''.join(map(str, info['flags']))}")
        else:
            print(f"  flags:      {', '.join(info['flags']) or '(none)'}")
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
