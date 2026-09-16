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

## Leaving the field

Three things can take a Pokémon off the field, and they behave differently:

| | when the replacement arrives | who picks |
| --- | --- | --- |
| Fainting | after the turn finishes, residuals and all | the player |
| Self-switch (U-turn, Parting Shot) | mid-turn, before the opponent's move | the player |
| Phazing (Whirlwind, Dragon Tail) | mid-turn, immediately | random |

The self-switch case is why `step` is a state machine rather than a plain
turn function: the user picks a replacement *during* the turn, so the turn
suspends with the opponent's already-locked move held in
`pending_side`/`pending_action`, and that move then runs against whoever comes
in. Phazing needs no decision point, so it is settled inline.

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

| batch | compile | throughput |
| --- | --- | --- |
| 1 | 15 s | 2,000 steps/s |
| 64 | 18 s | 45,000 steps/s |
| 256 | 15 s | 83,000 steps/s |
| 1024 | 15 s | 106,000 steps/s |

Compile time is a flat one-off and barely moves with batch size.

### What keeps it fast

The compiled program is one enormous basic block, and XLA's cost is superlinear
in *basic-block size*, not in operation count. For a long time this package had
to force `--xla_backend_optimization_level=0` because `vmap(step)` would not
finish compiling at all: a batch of two ran past 150 seconds where a batch of one
took six. Two fixes removed that, and the engine is about five times faster for
it. Both are easy to undo by accident:

**Copies of the big pieces.** `run_action` (the whole move engine) and
`switch_to` (hazards, entry abilities) are large. Running them from a
`lax.fori_loop` over the two players puts *one* copy in the compiled graph; a
Python loop puts one per player. Collapsing the two action slots took the program
from 99,901 HLO operations to 58,469, and compile from 28 s to 12 s. Getting this
wrong the other way, with `switch_to`, once cost 4x throughput. The loops in
`run_turn`, `apply_forced_switches` and `resolve_phazing` are deliberate.

**Rank.** Mapping the damage calculation over a move's hits only needs the damage
roll to vary, and `vmap` batches just the values that depend on it. But
Technician tests `base_power <= 60`, and with a per-hit scaled power that test
pulled the entire forty-step modifier chain up a rank along with it. That single
comparison was the difference between compiling at optimisation level 1 and not
compiling at all — same operation count either way. `calc_damage` now takes the
unscaled power for that test (`technician_power`), which is exactly equivalent:
the only moves that scale power per hit are Triple Kick and Triple Axel, and all
of their per-hit powers are under the threshold regardless.

Diagnosing this was mostly a matter of measuring HLO size with
`jax.jit(f).lower(...).as_text()`, which is cheap, rather than compile time,
which is not.

### Still on the table

Throughput is implementation-bound, not hardware-bound: this uses about 2.5 of 8
cores, and per-battle cost was still falling at a batch of 2048. The remaining
gain is in parallelism rather than codegen.

Tried and rejected for that: `--xla_force_host_platform_device_count=<cores>`
with a `pmap` over the devices. Forcing N host devices splits the CPU thread pool
N ways, and measured end to end it came out ~10x slower than a plain `vmap` on
one device.

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
- **Held items are assigned by set role**, not by Showdown's generator logic,
  which is full of species- and move-specific special cases. Teams are drawn
  from the real species/move/ability/Tera pools; only the item differs.
- **Happiness is fixed at 255**, pinning Return at 102 BP and Frustration at 1.
- **Beat Up is not modelled.** Its power and hit count come from the whole
  party's base Attack, which would need party data threaded into the per-hit
  damage path. It appears in 1 of 4336 Random Battle sets.
- No Dynamax, Z-moves or Mega Evolution — none exist in Gen 9 singles.

