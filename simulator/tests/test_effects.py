"""Differential test for move and ability *effects*, against real Showdown turns.

`test_damage.py` pins the damage numbers; this pins everything else -- status,
boosts, hazards, residual chip and healing, item swaps, switch-in abilities.
`tools/showdown_effects.js` runs each scenario for one turn inside a real
Showdown battle with the RNG pinned so every check succeeds, and records the
resulting state. Here we replay the same scenario and compare.

Showdown's RNG is pinned; psjax's is not, so a move with less than perfect
accuracy will sometimes miss. Each scenario therefore runs under several keys
and the *most common* psjax outcome must be Showdown's. For the many scenarios
that cannot miss, that is simply every run agreeing.
"""
from __future__ import annotations

import collections
import json
import pathlib

import jax
import jax.numpy as jnp
import pytest

from psjax.engine import step
from scenario import build, normalise, snapshot

TRUTH = pathlib.Path(__file__).resolve().parent.parent / "data" / "effect_truth.json"
CASES = json.load(open(TRUTH)) if TRUTH.exists() else []
SEEDS = 9

_JIT_STEP = jax.jit(step)


def _action(choice):
    """Translate a Showdown choice string into a psjax action id.

    'move 2' -> move slot 1; 'switch 2' -> the switch action for team slot 1.
    """
    from psjax import consts as C
    if isinstance(choice, str) and choice.startswith("switch"):
        return C.ACTION_SWITCH_BASE + int(choice.split()[1]) - 1
    if isinstance(choice, str) and choice.startswith("move"):
        return int(choice.split()[1]) - 1
    return 0


def _run(case, seed):
    from psjax import consts as C
    state = build(case, jax.random.PRNGKey(seed))
    actions = jnp.array([_action(case.get("p1move")),
                         _action(case.get("p2move"))], jnp.int32)
    state = _JIT_STEP(state, actions)
    # A self-switch suspends the turn to ask for a replacement; answer it so the
    # rest of the turn runs, exactly as the Showdown harness does.
    if case.get("p1switchAfter") and int(state.phase) == C.PHASE_SWITCH:
        reply = jnp.array([C.ACTION_SWITCH_BASE + case["p1switchAfter"] - 1, 0],
                          jnp.int32)
        state = _JIT_STEP(state, reply)
    return snapshot(state)


def _truncate(value, width):
    """psjax always carries six team slots; Showdown's scenario teams are shorter."""
    return value[:width] if isinstance(value, list) else value


@pytest.mark.skipif(not CASES, reason="run tools/showdown_effects.js to build truth data")
@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_effect_matches_showdown(case):
    assert not case.get("error"), f"Showdown scenario itself failed: {case.get('error')}"
    want_snap = case["after"]
    team_width = {"p1": len(case["after"]["p1"]["hp"]),
                  "p2": len(case["after"]["p2"]["hp"])}

    outcomes = collections.Counter()
    for seed in range(SEEDS):
        got_snap = _run(case, seed)
        got = tuple(
            json.dumps(_truncate(normalise(got_snap, f), team_width.get(f.split(".")[0], 6)),
                       sort_keys=True)
            for f in case["check"])
        outcomes[got] += 1

    want = tuple(
        json.dumps(_truncate(normalise(want_snap, f), team_width.get(f.split(".")[0], 6)),
                   sort_keys=True)
        for f in case["check"])

    modal, count = outcomes.most_common(1)[0]
    if modal == want:
        return
    detail = "\n".join(
        f"    {n}/{SEEDS}x  " + "  ".join(f"{f}={v}" for f, v in zip(case["check"], o))
        for o, n in outcomes.most_common())
    pytest.fail(
        f"\n  showdown: " + "  ".join(f"{f}={v}" for f, v in zip(case["check"], want))
        + f"\n  psjax outcomes over {SEEDS} seeds:\n{detail}")
