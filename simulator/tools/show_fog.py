"""Print one battle's observation with and without fog of war, field by field.

    python tools/show_fog.py [seed]

The observation is a flat 526-float vector, so this rebuilds the segment layout
`BattleEnv._observe` writes and decodes both views through it. The layout is
asserted against `obs_dim`, which catches it drifting out of sync with the
encoder.
"""
from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psjax import consts as C              # noqa: E402
from psjax.data import names               # noqa: E402
from psjax.env import BattleEnv            # noqa: E402
from psjax.fog import FogOfWarEnv          # noqa: E402


def layout():
    """(name, width) for every segment, in the order `_observe` concatenates."""
    rows = []
    for side in ("mine", "theirs"):
        rows += [
            (f"{side}.hp_frac", C.TEAM_SIZE),
            (f"{side}.alive", C.TEAM_SIZE),
            (f"{side}.status", C.TEAM_SIZE * C.NUM_STATUS),
            (f"{side}.boosts", 7),
            (f"{side}.active_types", 2 * (C.NUM_TYPES + 1)),
            (f"{side}.volatiles", C.NUM_VOLATILES),
            (f"{side}.side_conditions", C.NUM_SIDE_CONDITIONS),
            (f"{side}.speed", 1),
            (f"{side}.terastallized", C.TEAM_SIZE),
            (f"{side}.move_type", 4 * C.NUM_TYPES),
            (f"{side}.move_category", 4 * 3),
            (f"{side}.move_power", 4),
            (f"{side}.move_pp", 4),
            (f"{side}.move_valid", 4),
        ]
    rows += [("field.weather", C.NUM_WEATHER), ("field.terrain", C.NUM_TERRAIN),
             ("field.trickroom_gravity", 2), ("field.turn", 1)]
    return rows


def segments(obs, rows):
    out, i = {}, 0
    for name, width in rows:
        out[name] = obs[i:i + width]
        i += width
    assert i == obs.shape[-1], f"layout covers {i} of {obs.shape[-1]}"
    return out


def fmt(v, prec=2):
    return "[" + " ".join(f"{float(x):.{prec}f}".rstrip("0").rstrip(".") or "0"
                          for x in v) + "]"


def moves_of(state, side, nm):
    i = int(state.active[side])
    return [nm.moves_by_id[int(m)] for m in state.moves[side, i] if m >= 0]


def report(env, fog_env, fs, rows, title):
    plain = env.observe(fs.battle)[0]      # player 0, full information
    fogged = fog_env.observe(fs)[0]        # player 0, fog of war
    a, b = segments(plain, rows), segments(fogged, rows)

    print(f"\n{title}")
    print(f"{'segment':26s} {'full information':34s} {'with fog'}")
    print("-" * 96)
    for name, _ in rows:
        if not name.startswith("theirs"):
            continue
        same = bool(jnp.allclose(a[name], b[name]))
        if same and "move" not in name and "hp" not in name:
            continue
        mark = "  " if same else "->"
        x, y = fmt(a[name]), fmt(b[name])
        if len(x) > 33:                     # one-hot blocks: show which index is set
            x = f"argmax/4 {[int(v) for v in jnp.argmax(a[name].reshape(4, -1), -1)]}"
            y = f"argmax/4 {[int(v) for v in jnp.argmax(b[name].reshape(4, -1), -1)]}"
        print(f"{mark} {name:24s} {x:34s} {y}")


def main(seed: int) -> None:
    nm = names()
    env = BattleEnv()
    fog_env = FogOfWarEnv(env)
    rows = layout()

    fs = fog_env.reset(jax.random.PRNGKey(seed))
    assert sum(w for _, w in rows) == env.obs_dim, "layout is out of sync"

    st = fs.battle
    p0, p1 = int(st.active[0]), int(st.active[1])
    print(f"player 0 leads {nm.species_by_id[int(st.species[0, p0])]}")
    print(f"player 1 leads {nm.species_by_id[int(st.species[1, p1])]}  "
          f"moves {moves_of(st, 1, nm)}")

    report(env, fog_env, fs, rows,
           "TURN 1 -- player 0 looking at player 1, nothing disclosed yet")

    # Both sides use their first move, which discloses exactly that move.
    fs, _, _, _ = fog_env.step(fs, jnp.array([0, 0], jnp.int32))
    used = moves_of(fs.battle, 1, nm)
    seen = [m for m, flag in zip(used, fs.revealed_moves[1, int(fs.battle.active[1])])
            if bool(flag)]
    print(f"\nplayer 1 has now used: {seen}")
    report(env, fog_env, fs, rows,
           "AFTER ONE TURN -- one of player 1's four moves is now disclosed")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
