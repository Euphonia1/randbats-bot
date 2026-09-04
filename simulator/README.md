# psjax — Pokémon Showdown's battle simulator in JAX

A reimplementation of [Pokémon Showdown](https://github.com/smogon/pokemon-showdown)'s
Generation 9 singles battle engine as pure JAX. Everything is a function from a
`BattleState` pytree of fixed-shape arrays to a new one, so battles can be
jitted, and rolled out with `lax.scan` without leaving the device.

The target format is **Gen 9 Random Battles**, and teams are generated from
Showdown's own `sets.json`.

## Why this is not a rewrite from memory

Showdown expresses most game logic as arbitrary JavaScript callbacks, which
cannot cross into JAX. Rather than reimplement mechanics from a wiki, this repo:

1. **Extracts the real data.** `tools/dump_data.js` loads the `pokemon-showdown`
   npm package and dumps every declarative field for all 848 moves, 911 species,
   abilities, items and the Random Battle sets — plus the *names* of the JS
   callbacks each entry defines, so the builder knows exactly what behaviour is
   unaccounted for.
2. **Compiles it to flat arrays.** `psjax/build.py` turns that JSON into a 51 KB
   `.npz` of integer columns. At runtime there are no dicts, strings or Python
   branches on game data — only array indexing.
3. **Ports the callbacks explicitly.** Each JS callback we support is transcribed
   from Showdown's source into a `lax.switch` branch in `psjax/callbacks.py`,
   including its integer rounding.
4. **Verifies against Showdown itself.** `tools/showdown_damage.js` runs each
   test case inside a real Showdown battle and records the damage for all 16
   damage rolls. `tests/test_damage.py` asserts our numbers match exactly.

```
$ pytest -q
187 passed
```

166 of those are damage checks: 83 scenarios × 16 rolls, matching Showdown's
`getDamage` exactly — crits, weather, screens, Tera, Adaptability, Unaware, Low
Kick's weight steps, Wring Out's fixed-point rounding, and so on. The remaining
21 cover the turn engine: battles terminate with a consistent winner, HP, PP and
boosts stay in range, rewards are zero-sum and paid once, priority beats Speed,
Trick Room inverts it, Choice locks hold, and Stealth Rock scales with the Rock
matchup.

Four real bugs were caught this way and fixed: Thick Fat, Heatproof, Water Bubble
and Purifying Salt were on the wrong hook (Showdown reduces the *attacking stat*,
not final damage, and the rounding differs); Ice Scales was the reverse; Low Kick
used kilograms where Showdown uses hectograms; and Adaptability was missing
entirely.

## Layout

| File | What it does |
| --- | --- |
| `tools/dump_data.js` | Showdown → `data/gen9_raw.json` |
| `tools/showdown_damage.js` | ground-truth damage from a real Showdown battle |
| `psjax/build.py` | `gen9_raw.json` → `data/gen9.npz` (flat arrays) |
| `psjax/consts.py` | enums shared by the builder and the engine |
| `psjax/effects.py` | registry mapping JS callbacks → handler ids |
| `psjax/hooks.py` | abilities and items with stable ids |
| `psjax/callbacks.py` | the callback implementations |
| `psjax/stats.py` | stat formulas and Showdown's fixed-point modifier math |
| `psjax/damage.py` | type effectiveness and the damage pipeline |
| `psjax/mechanics.py` | speed, grounding, boosts, status, HP |
| `psjax/moves.py` | one move: checks, accuracy, hits, secondaries |
| `psjax/engine.py` | turn order, switching, hazards, residuals, `step` |
| `psjax/teams.py` | Random Battle team generation |
| `psjax/env.py` | batched RL environment wrapper |
| `psjax/coverage.py` | what is and is not modelled |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e 'simulator[dev]'

cd simulator/tools && npm install pokemon-showdown    # data source
cd .. && node tools/dump_data.js && python -m psjax.build
node tools/showdown_damage.js tools/damage_cases.json data/damage_truth.json
pytest -q
```

## Use

```python
import jax
from psjax.env import BattleEnv

env = BattleEnv()
state = env.reset(jax.random.PRNGKey(0))

while not state.phase == 2:                       # PHASE_END
    mask = env.legal_actions(state)               # [2, 14] bool
    actions = env.sample_actions(state, key)      # your policy goes here
    state, obs, rewards, done = env.step(state, actions)
```

Actions are a single integer per player:

| Value | Meaning |
| --- | --- |
| `0..3` | use move in that slot |
| `4..7` | terastallize, then use move `n - 4` |
| `8..13` | switch to team slot `n - 8` |

`env.observe` returns `[2, 524]` floats — each row is one player's view.

## Performance

On one CPU core (Apple Silicon), single battle:

| | |
| --- | --- |
| `jax.jit(step)` compile | ~8 s, once per process |
| per decision point | 0.57 ms (~1,700 steps/s) |
| full random battle | ~520 ms (mean 42 turns) |

**Batching caveat.** `jax.vmap` works on `reset`, `observe` and the individual
mechanics helpers, but `jax.vmap(step)` currently takes many minutes to compile
even at a batch of 8. This is a compile-time blow-up inside
`moves.execute_move`, not a runtime cost — `residuals`, `switch_to` and the
31-branch effect `lax.switch` each vmap in under a second, so the large switches
are not the cause and it has not yet been narrowed further. Until it is,
`env.step_batch` and `env.rollout_batch` use `lax.map`, which compiles the
unbatched body once and still runs entirely on device, but sequentially rather
than in parallel. Getting `vmap(step)` to compile is the single highest-value
next piece of work: it is what would turn this into a fast self-play
environment.

## State of the implementation

Run `python -m psjax.coverage` for the current numbers. As of Showdown v0.11.11:

- **Moves.** 299/349 of the Random Battle movepool is fully modelled; weighted by
  how often moves appear in sets, **94.8%** of usage is covered. The rest run as
  ordinary moves with their special behaviour skipped.
- **Abilities.** 164/203 Random Battle abilities have an id (**89.4%** by usage).
  An id is necessary but not sufficient: only abilities wired into
  `damage.py` / `mechanics.py` / `engine.py` actually do anything. An ability
  with no id has *no effect at all* — it does not error, it is simply inert.
- **Items.** All 33 items the Gen 9 Random Battle generator can assign are
  present, plus ~55 more.

### Known deviations

These are deliberate and documented rather than hidden:

- **Singles only.** No doubles targeting, spread damage or ally effects.
- **Self-switch and forced switches resolve at end of turn**, not immediately.
  After a U-turn the opponent attacks the Pokémon that used it, rather than the
  replacement. This is the most behaviourally significant gap.
- **Held items are assigned by set role**, not by Showdown's generator logic,
  which is full of species- and move-specific special cases. Teams are drawn
  from the real species/move/ability/Tera pools; only the item differs.
- **Happiness is fixed at 255**, pinning Return at 102 BP and Frustration at 1.
- No Dynamax, Z-moves or Mega Evolution — none exist in Gen 9 singles.

