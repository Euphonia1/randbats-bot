"""Report what the engine does and does not model.

Run as `python -m psjax.coverage`. The point is to make the gaps visible: an
ability outside `hooks.ABILITY_NAMES` compiles to id 0 and simply has no effect,
which is invisible at runtime. Numbers are weighted by how often each thing
appears in Gen 9 Random Battle sets, since that is the format this engine targets.
"""
from __future__ import annotations

import collections
import json
import pathlib

from . import consts as C
from . import effects as E
from . import hooks as H
from .build import RAW, to_id

ROOT = pathlib.Path(__file__).resolve().parent.parent


def randbats_usage(raw):
    """How many Random Battle sets reference each move and ability."""
    moves, abilities = collections.Counter(), collections.Counter()
    for entry in raw["randbats"].values():
        for s in entry["sets"]:
            for mv in s["movepool"]:
                moves[to_id(mv)] += 1
            for ab in s["abilities"]:
                abilities[to_id(ab)] += 1
    return moves, abilities


def report():
    raw = json.load(open(RAW))
    move_usage, ability_usage = randbats_usage(raw)
    lines = []
    add = lines.append

    add("=" * 74)
    add(f"psjax coverage  (Showdown v{raw['showdownVersion']}, gen {raw['gen']})")
    add("=" * 74)

    # --- moves ---
    from .moves import MISSING_EFFECTS
    handled = set(E.BP_REPLACE_INDEX) | set(E.BP_MODIFY_INDEX) | \
        set(E.DMG_INDEX) | set(E.TYPE_INDEX) | set(E.EFFECT_INDEX)
    missing_effect_moves = {mid for h in MISSING_EFFECTS
                            for mid in E.EFFECT_MOVES.get(h, [])}

    unmodelled = []
    for mid, m in raw["moves"].items():
        if not m["callbacks"]:
            continue                      # purely declarative: fully modelled
        if mid in handled and mid not in missing_effect_moves:
            continue
        unmodelled.append(mid)

    total_sets = sum(move_usage.values())
    rb_moves = set(move_usage)
    rb_unmodelled = sorted(rb_moves & set(unmodelled),
                           key=lambda m: -move_usage[m])
    covered_sets = total_sets - sum(move_usage[m] for m in rb_unmodelled)

    add("")
    add(f"MOVES  {len(raw['moves'])} total, {len(unmodelled)} with unmodelled behaviour")
    add(f"  Random Battle movepool: {len(rb_moves) - len(rb_unmodelled)}/{len(rb_moves)} "
        f"moves modelled")
    add(f"  weighted by set appearances: {100 * covered_sets / total_sets:.1f}% covered")
    if rb_unmodelled:
        add("  most-used unmodelled moves:")
        for mid in rb_unmodelled[:20]:
            add(f"    {move_usage[mid]:5d}x  {mid:22s} {','.join(raw['moves'][mid]['callbacks'])}")

    # --- abilities ---
    known = set(H.ABILITY_NAMES)
    rb_abilities = set(ability_usage)
    missing_ab = sorted(rb_abilities - known, key=lambda a: -ability_usage[a])
    total_ab = sum(ability_usage.values())
    covered_ab = total_ab - sum(ability_usage[a] for a in missing_ab)
    add("")
    add(f"ABILITIES  {len(H.ABILITY_NAMES) - 1} in the registry")
    add(f"  Random Battle abilities: {len(rb_abilities) - len(missing_ab)}/{len(rb_abilities)} "
        f"have an id")
    add(f"  weighted by set appearances: {100 * covered_ab / total_ab:.1f}%")
    add("  NOTE: an id is necessary but not sufficient -- only abilities wired into")
    add("        damage.py / mechanics.py / engine.py actually do anything.")
    if missing_ab:
        add("  most-used abilities with no id (no effect at all):")
        for a in missing_ab[:20]:
            add(f"    {ability_usage[a]:5d}x  {a}")

    # --- special effect handlers ---
    add("")
    add(f"SPECIAL EFFECT HANDLERS  {len(E.EFFECT_HANDLERS) - 1} declared")
    if MISSING_EFFECTS:
        add(f"  {len(MISSING_EFFECTS)} not implemented (the move still runs, but its")
        add("  special behaviour is skipped):")
        for h in MISSING_EFFECTS:
            movelist = ", ".join(E.EFFECT_MOVES.get(h, []))
            add(f"    {h:18s} {movelist}")

    # --- items ---
    add("")
    add(f"ITEMS  {len(H.ITEM_NAMES) - 1} in the registry "
        f"(all 33 used by the Gen 9 Random Battle generator are present)")

    add("")
    add("STRUCTURAL LIMITATIONS")
    for note in (
        "singles only -- no doubles targeting, spread damage or ally effects",
        "self-switch and forced switches resolve at end of turn, not immediately",
        "held items are assigned by set role, not by Showdown's generator logic",
        "no Dynamax, Z-moves or Mega Evolution (none appear in Gen 9 singles)",
        "happiness is fixed at 255 (Return 102 BP, Frustration 1 BP)",
    ):
        add(f"  - {note}")
    add("")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report())
