"""Load the compiled tables into device arrays.

`GameData` is a pytree of `jnp` arrays, so it can be closed over by jitted
functions and passed through `vmap` without being re-uploaded per battle. It is
immutable: nothing in the engine ever writes to it.
"""
from __future__ import annotations

import functools
import json
import pathlib
from typing import Any, Dict

import jax.numpy as jnp
import numpy as np

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
NPZ = DATA_DIR / "gen9.npz"
INDEX = DATA_DIR / "gen9_index.json"


class GameData(dict):
    """Dict of compiled arrays, registered as a JAX pytree.

    Subclassing `dict` keeps `data["move_base_power"]` working while letting JAX
    flatten it; the key order is sorted so the treedef is deterministic.
    """

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


import jax.tree_util as tu

tu.register_pytree_node(
    GameData,
    lambda d: (tuple(d[k] for k in sorted(d)), tuple(sorted(d))),
    lambda keys, vals: GameData(zip(keys, vals)),
)


@functools.lru_cache(maxsize=1)
def load_index() -> Dict[str, Any]:
    if not INDEX.exists():
        raise FileNotFoundError(
            f"{INDEX} not found -- run `node tools/dump_data.js && python -m psjax.build`")
    return json.load(open(INDEX))


@functools.lru_cache(maxsize=1)
def load_data() -> GameData:
    """Compiled tables as device arrays (cached: one upload per process)."""
    if not NPZ.exists():
        raise FileNotFoundError(
            f"{NPZ} not found -- run `node tools/dump_data.js && python -m psjax.build`")
    with np.load(NPZ) as f:
        return GameData({k: jnp.asarray(f[k]) for k in f.files})


# --- name lookup (host side only; never inside a traced function) -------------

class Names:
    """String <-> index helpers for tests, team building and rendering."""

    def __init__(self):
        idx = load_index()
        self.species = idx["species"]
        self.moves = idx["moves"]
        self.abilities = idx["abilities"]
        self.items = idx["items"]
        self.types = idx["types"]
        self.gen = idx["gen"]
        self.showdown_version = idx["showdown_version"]
        self.species_by_id = {v: k for k, v in self.species.items()}
        self.moves_by_id = {v: k for k, v in self.moves.items()}
        self.abilities_by_id = {v: k for k, v in self.abilities.items()}
        self.items_by_id = {v: k for k, v in self.items.items()}
        self.types_by_id = {v: k for k, v in self.types.items()}

    @staticmethod
    def to_id(s: str) -> str:
        return "".join(c for c in s.lower() if c.isalnum())

    def species_id(self, name: str) -> int:
        return self.species[self.to_id(name)]

    def move_id(self, name: str) -> int:
        return self.moves[self.to_id(name)]

    def ability_id(self, name: str) -> int:
        """0 (no effect) for an ability the engine does not model."""
        return self.abilities.get(self.to_id(name), 0)

    def item_id(self, name: str) -> int:
        return self.items.get(self.to_id(name), 0)

    def type_id(self, name: str) -> int:
        return self.types[self.to_id(name)]


@functools.lru_cache(maxsize=1)
def names() -> Names:
    return Names()
