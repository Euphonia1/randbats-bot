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
4. **Verifies against Showdown itself.** Two differential harnesses run
   scenarios inside a real Showdown battle and record the result, which the
   Python tests then have to reproduce:
   - `tools/showdown_damage.js` records damage for all 16 damage rolls.
   - `tools/showdown_effects.js` runs a full turn and records the resulting
     state — status, boosts, hazards, residual chip and healing, item swaps,
     switch-in abilities.

```
$ pytest -q
570 passed, 3 skipped
```

**442 damage checks.** 88 hand-written scenarios × 16 rolls probe specific
mechanics — crits, weather, screens, Tera, Adaptability, Unaware, Low Kick's
weight steps, Wring Out's fixed-point rounding. On top of that, a sweep runs
**every one of the 269 damaging moves in the Random Battle pool** against a live
Showdown, so each compiled row — base power, type, category, base-power callback
— is verified. All of them match exactly except Beat Up, which is skipped with
its reason attached (see below).

**101 effect checks** — one real Showdown turn each, comparing the fields the
scenario is about. Because Showdown's RNG is pinned and psjax's is not, each
scenario runs under several keys and the most common psjax outcome must be
Showdown's; for the many that cannot miss, that is every run agreeing.

**27 engine and batching checks** — battles terminate with a consistent winner,
HP/PP/boosts stay in range, rewards are zero-sum and paid once, priority beats
Speed, Trick Room inverts it, Choice locks hold, and a batched step matches a
sequential one field for field.

### Bugs this caught

The damage harness caught four: Thick Fat, Heatproof, Water Bubble and Purifying
Salt were on the wrong hook (Showdown reduces the *attacking stat*, not final
damage, and the rounding differs); Ice Scales was the reverse; Low Kick used
kilograms where Showdown uses hectograms; and Adaptability was missing entirely.

The effects harness caught eight more, several of them serious:

- **Every move healed its user 1 HP and dealt 1 recoil.** An absent fraction is
  compiled as `0/1`, and the guards tested the *denominator*, so they were always
  true; `fraction_of_max` then floors at 1. This quietly skewed every battle.
- **Enum index 0 collided with "absent".** `move_side_condition` and
  `move_volatile` defaulted to `0`, which *is* Stealth Rock and Confusion. Every
  move set Stealth Rock on its own side, and no move could ever confuse, because
  the guard `> 0` excluded the real index 0.
- **Status moves ignored type immunity**, so Thunder Wave paralysed Ground types.
  Showdown defaults `ignoreImmunity` to true for status moves but individual
  moves override it; the compiled mask already had this right.
- **Knock Off never removed the item** — the handler existed but no move was
  mapped to it, so only the base-power bonus applied.
- **Rapid Spin boosted Speed twice**, once declaratively and once in its handler.
- **Leech Seed could seed Grass types**: the declarative volatile was applied
  before the handler's immunity check.
- **Rough Skin and Iron Barbs used 1/6** instead of 1/8 (Rocky Helmet's fraction).
- **Defiant and Competitive only answered Intimidate**, not stat drops from moves.
- Weather-scaled recovery (Synthesis, Shore Up) used a plain fraction where
  Showdown quantises the factor to 4096ths, so it was a point off.

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
| `tests/test_batching.py` | batched execution == sequential, field by field |
| `tools/showdown_effects.js` | ground-truth turn results from a real Showdown battle |
| `tools/effect_cases.py` | the effect scenarios, as readable Python |
| `tools/move_sweep.py` | generates a damage case for every damaging move |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e 'simulator[dev]'

cd simulator/tools && npm install pokemon-showdown    # data source
cd .. && node tools/dump_data.js && python -m psjax.build
node tools/showdown_damage.js tools/damage_cases.json data/damage_truth.json
python tools/effect_cases.py > tools/effect_cases.json
node tools/showdown_effects.js tools/effect_cases.json data/effect_truth.json
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

## Batching

Battles run in parallel through `jax.vmap`, and batched results are bit-identical
to stepping each battle on its own (`tests/test_batching.py` asserts this field
by field).

```python
env = BattleEnv()
states = env.reset_batch(jax.random.split(key, 1024))     # 1024 battles
states, obs, rewards, done = env.step_batch(states, actions)

final, rewards = rollout_batch(env, states, keys)         # play them all out
```

Measured on one CPU (Apple Silicon):

| batch | compile | per step | throughput |
| --- | --- | --- | --- |
| 1 | 25 s | 0.66 ms | 1,500 steps/s |
| 64 | 26 s | 4.0 ms | 16,000 steps/s |
| 256 | 27 s | 9.7 ms | 26,500 steps/s |
| 1024 | 28 s | 27.9 ms | 36,700 steps/s |

Compile time is a flat one-off and barely moves with batch size; throughput
scales roughly 24x from a single battle to 1024.

### Why the XLA flag

`psjax/__init__.py` sets `--xla_backend_optimization_level=0` before JAX starts.
This is load-bearing, and worth explaining because the symptom was baffling:
`vmap(step)` at a batch of **one** compiled in 6 seconds, and at a batch of
**two** did not finish in 150.

A battle step is thousands of tiny operations on six- and seven-element arrays.
Unbatched they fold away; batched, each becomes a real vectorised op, and XLA's
CPU backend then runs LLVM optimisation passes whose cost grows superlinearly in
that op count. Dropping the backend optimisation level skips those passes:
compilation becomes ~26 s at any batch size. It costs roughly 35% on
*unbatched* per-step time, which the batching repays many times over. Set
`PSJAX_NO_XLA_TUNING=1` to opt out, or set `xla_backend_optimization_level`
yourself and psjax will leave it alone.

Getting there also needed the engine itself to stop generating pathological
batched code. Each of these was found by bisecting compile time and is worth
knowing about if you extend the engine:

- **Integer division by a traced divisor.** XLA's lowering is enormous; a handful
  inside one `lax.switch` was minutes on its own. `stats.idiv` does an exact
  float divide with a two-step correction instead.
- **Indexing with a traced index.** `state.hp[side, slot]` is a gather over the
  team axis, and there are dozens per move. `mechanics.slot_get` / `slot_set`
  select over the six slots elementwise instead. Same for the stat index in
  `damage._pick_stat`, which the move's category makes traced.
- **`lax.cond` returning a `BattleState`.** Under `vmap` it computes both
  branches and selects over the *entire* pytree, so gating a two-field update
  costs as much as gating a forty-field one. Every such site now takes a `when`
  mask and no-ops internally; `run_effect` additionally returns only the 15
  fields any handler writes rather than all 47.
- **Repeated calls that could be folded.** Ten `apply_boosts` calls keyed on a
  single ability value became one accumulated vector.

## Performance (single battle)

| | |
| --- | --- |
| `jax.jit(step)` compile | ~11 s, once per process |
| per decision point | 0.90 ms |
| full random battle | ~600 ms (mean 42 turns) |

## State of the implementation

Run `python -m psjax.coverage` for the current numbers. As of Showdown v0.11.11:

- **Moves.** 324/349 of the Random Battle movepool is fully modelled; weighted by
  how often moves appear in sets, **98.7%** of usage is covered. Every damaging
  move is verified against Showdown by the sweep. The rest run as ordinary moves
  with their special behaviour skipped.
- **Abilities.** All 203 Random Battle abilities have an id, and **138 of them
  (81.9% by set usage)** are wired to actual behaviour. The rest are inert: they
  do not error, they simply have no effect. `python -m psjax.coverage` lists
  them. Some are legitimately passive — Multitype, the single most common, only
  fixes a forme's type, which the species data already encodes.
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
- **Beat Up is not modelled.** Its power and hit count come from the whole
  party's base Attack, which would need party data threaded into the per-hit
  damage path. It appears in 1 of 4336 Random Battle sets.
- No Dynamax, Z-moves or Mega Evolution — none exist in Gen 9 singles.

