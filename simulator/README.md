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
1232 passed, 3 skipped
```

**1,094 damage checks**, all against a live Showdown:

- 88 hand-written scenarios × 16 rolls probe specific mechanics — crits, weather,
  screens, Tera, Adaptability, Unaware, Low Kick's weight steps, Wring Out's
  fixed-point rounding.
- A **move sweep** runs every one of the 269 damaging moves in the Random Battle
  pool, so each compiled row (base power, type, category, base-power callback) is
  verified. All match exactly except Beat Up, skipped with its reason attached.
- An **ability sweep** puts each of the 138 wired abilities on the attacker and
  on the defender across two contexts — burned and in a pinch under sun, and a
  special move in rain — so an ability that fires when it should not, or on the
  wrong hook, shows up.

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

The ability sweep then caught three more, all of them abilities that were
half-wired — the switch-in half was there and the damage half was not:

- **Orichalcum Pulse** set the sun but did not boost Attack in it, and **Hadron
  Engine** set Electric Terrain but did not boost Sp. Atk on it. Both are
  5461/4096, not the 1.3 they are usually quoted as.
- **Slow Start** halved Speed but not Attack.

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
| `tools/ability_sweep.py` | generates damage cases for every wired ability |

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e 'simulator[dev]'

cd simulator/tools && npm install pokemon-showdown    # data source
cd .. && node tools/dump_data.js && python -m psjax.build
node tools/showdown_damage.js tools/damage_cases.json data/damage_truth.json
python tools/move_sweep.py > tools/move_sweep_cases.json
node tools/showdown_damage.js tools/move_sweep_cases.json data/move_sweep_truth.json
python tools/ability_sweep.py > tools/ability_sweep_cases.json
node tools/showdown_damage.js tools/ability_sweep_cases.json data/ability_sweep_truth.json
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

### Batches desynchronise, and that is fine

A replacement prompt opens in some battles and not others, so a batch does not
stay in lockstep. Within a single `step_batch` call the same action integer can
mean a replacement slot in one battle and a move slot in the next; each battle
decodes it against its own `phase`. `lax.cond` on a batched predicate evaluates
both readings and selects per element, so every battle follows its own path, and
the result is identical to stepping each one alone.

Two consequences for anything driving the env:

* **Ask for the mask per battle.** `legal_actions` offers switches only to the
  battles that owe a replacement. Sampling from a mask taken for the wrong battle
  will produce an illegal action.
* **Turn counters drift apart.** Answering a prompt is a decision point, not a
  turn, so a battle that paused sits a turn behind its neighbours. Episode
  boundaries are per battle, which is what an RL loop wants anyway.

`tests/test_batching.py` pins all of this, including a finished battle sitting in
a batch of running ones without disturbing them.

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

Measured on one CPU device (8-core Apple M1) by `tools/psjax_bench.py`, which
plays complete battles to their end under uniformly random legal actions:

| batch | compile | battles/s | turns/s | env steps/s |
| --- | --- | --- | --- | --- |
| 1 | 17.3 s | 5.1 | 905 | 1,063 |
| 64 | 18.5 s | 47.8 | 3,578 | 48,180 |
| 256 | 16.5 s | 78.9 | 5,598 | 79,519 |
| 1024 | 16.0 s | 110.7 | 7,329 | 111,596 |
| 4096 | 16.0 s | 140.4 | 9,159 | 141,536 |
| 8192 | 16.7 s | 191.3 | 12,174 | 192,797 |

Compile time is a flat one-off and barely moves with batch size. It is also the
only part of this that is slow enough to notice, which makes one mistake
expensive: `rollout_batch` must stay under `jax.jit`. As a bare `vmap` of the
scan it was an eager XLA call that recompiled the whole rollout on *every*
invocation -- about 20 s a call, against 0.03 s once the compile is cached.

### Against Showdown itself

`tools/showdown_bench.js` is the same benchmark against Showdown's own engine:
complete `gen9randombattle` games, uniformly random legal choices, the same
reported fields. It drives `Battle` directly rather than through `BattleStream`,
so Showdown is measured on engine work rather than protocol plumbing. Same
machine, Showdown v0.11.11 on Node:

| | Showdown | @256 | @1024 | @4096 | @8192 |
| --- | --- | --- | --- | --- | --- |
| battles/s | 37.1 | 78.9 | 110.7 | 140.4 | 191.3 |
| turns/s | 2,176 | 5,598 | 7,329 | 9,159 | 12,174 |
| mean turns | 58.6 | 71.0 | 66.2 | 65.2 | 63.6 |
| compile | none | 16.5 s | 16.0 s | 16.0 s | 16.7 s |

The `@n` columns are psjax batch sizes.

About **5.2x on battles and 5.6x on turns** at a batch of 8192 (repeat runs vary
by a few percent), and the gap is still widening there while Showdown's rate
stays flat. Sharding that batch across threads rather than handing it to one
`vmap` takes it to **9.6x and 10.4x** -- see below. Compilation is a fixed cost Showdown never pays, so the crossover is
around **800 battles**: below that, shelling out to Showdown finishes sooner.

Four things the ratio does not say:

- **Showdown is doing more work per battle.** It emits a full protocol log,
  implements every move and ability rather than 98.7% and 81.9% of usage, and
  supports every format and generation. Part of its cost buys what this package
  does not have.
- **A batch finishes at the pace of its slowest battle**, so `battles/s` --
  whole batch over wall time -- is dragged down by the tail. `env steps/s`
  (192,797 at 8192) is the rate without that effect.
- **The turn counts are not the same**, 63.6 against 58.6, because psjax has no
  Endless Battle Clause and its stalemates never resolve; 43 of 8192 battles had
  not ended after 1000 steps. `turns/s` is the metric this does not distort.
- **Showdown's random chooser fell back** to `default` on 37 of roughly 11,700
  decisions (0.3%), where a uniformly picked option turned out to be illegal.

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

### Filling the machine

One `vmap` over a large batch does not saturate the CPU: at 8192 it keeps only
about 3.7 of 8 cores busy (`cores_used` in the benchmark output). XLA:CPU
parallelises inside a kernel only so far, and the rest of the machine idles.
Splitting the same work across Python threads fixes it, because JAX releases the
GIL while XLA runs -- the threads genuinely overlap:

| 8192 battles | battles/s | turns/s | cores | compile | total wall |
| --- | --- | --- | --- | --- | --- |
| one `vmap` of 8192 | 191.3 | 12,174 | 3.68 | 16.7 s | 61.6 s |
| 2 threads x 4096 | 246.4 | 15,451 | 5.30 | 16.7 s | 52.0 s |
| 4 threads x 2048 | 298.3 | 18,813 | 6.67 | 16.8 s | 46.3 s |
| 8 threads x 1024 | **355.9** | **22,699** | 7.49 | 16.8 s | **42.0 s** |
| 8 processes x 1024 | 281.3 | 18,623 | ~7.8 | 56-62 s | 98.8 s |

Threads are the win: 1.9x the single-batch throughput, and
`tools/psjax_bench.py --shards 8 8192` is the whole change.

Throughput tracks the core count almost exactly -- 52.0, 46.5, 44.7 and 47.5
battles per second per busy core at one, two, four and eight shards, which is
flat within run-to-run noise. The threads are not buying concurrency so much as
collecting cores the single `vmap` never asked for, and the shrinking gain per
doubling (1.29x, 1.21x, 1.19x) is the core count running out rather than
coordination overhead setting in. Four to eight still pays despite the second
four being the M1's efficiency cores.

Processes pin the cores marginally harder and are still a quarter slower, which
is the interesting part. Each one carries its own copy of the data tables and its
own XLA runtime, and the shared executable and shared tables of the threaded run
are worth more than that last 0.3 of a core. They also pay the compile eight
times: eight concurrent LLVM runs contend badly, 56-62 s each against 16.8 s
alone, which is most of why the end-to-end wall is more than twice the threaded
one.

Also tried and rejected: `--xla_force_host_platform_device_count=<cores>` with a
`pmap` over the devices. Forcing N host devices splits the CPU thread pool N ways,
and measured end to end it came out ~10x slower than a plain `vmap` on one
device. Threads get the parallelism that approach was reaching for without
partitioning the pool.

## Teams

Teams come from Showdown's own generator, not from a reimplementation of it.
`Teams.generate('gen9randombattle')` is a few thousand lines of sequential
logic -- per-species move enforcement, team-level state that accumulates as the
team is built, item rules keyed on the set's role and its final move list -- and
none of it is expressible as a fixed-shape traced computation. It also does not
need to be: team generation happens once, before the battle starts, so it can be
ordinary host-side code.

```
node tools/dump_teams.js 50000 > data/team_pool.jsonl
python tools/build_team_pool.py data/team_pool.jsonl     # -> data/team_pool.npz
```

`reset` then samples a team from the compiled pool, which makes items and
movesets exact rather than plausible. The pool is loaded separately from
`GameData` on purpose: the engine closes over `data` as a compile-time constant,
so folding fifty thousand teams into it would embed the lot in every compiled
module, `step` included. Only `reset` carries it.

`tools/build_team_pool.py` reports anything the engine cannot represent rather
than quietly zeroing it. Closing the gap it found added 40 items -- the seventeen
type plates, their pre-plate equivalents, and the forme items (Rusted Sword,
the Ogerpon masks, the creation orbs), which need to exist even though the forme
they force is already baked into the species the generator picked, because
holding one is not the same as holding nothing. Coverage is now complete: no
item, move or ability the generator produces is dropped.

Without a pool built, `new_battle` falls back to the procedural generator that
preceded this; `new_battle(key, data, pool=False)` forces it.

## Partial observability

`BattleEnv.observe` is full-information by design, which suits self-play but
trains a policy that could never be deployed against a real opponent.
`psjax.fog.FogOfWarEnv` wraps it and shows each player only what Showdown would:

```python
from psjax.fog import FogOfWarEnv

env = FogOfWarEnv()
fs = env.reset(key)                        # a FogState, not a BattleState
fs, obs, rewards, done = env.step(fs, actions)
```

`FogState` carries the battle plus `revealed[2,6]` and `revealed_moves[2,6,4]`,
and disclosure is read off public state rather than instrumented into the engine:
a Pokemon is revealed when it takes the field, and a move is revealed when its PP
falls. The observation layout is unchanged, so a policy trains under fog and
loads against the plain environment.

Less needed hiding than it first appears, and the reasoning is worth recording
because it is easy to hide too much:

- **The active Pokemon's moveset** is the leak worth closing. The encoder writes
  all four moves' type, category, power and PP from the moment it appears; a real
  opponent learns them one at a time.
- **Exact HP** becomes a whole percent over a denominator of 100, which also
  conceals the total -- 50% of 300 and 50% of 180 censor to the same reading.
- **The per-slot bench figures are not a leak.** A Pokemon that has never been
  active in singles has never been hit and never been statused, so its HP is full
  and its status clear by construction. The encoder is reporting what the opponent
  could already infer, and masking it would be theatre. The alive bits are public
  too -- Showdown shows the fainted count -- and bench species and movesets never
  enter the observation at all.

One approximation is left in place and noted rather than fixed: the opponent's
effective speed is exposed exactly, where a real player infers it from turn order.

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
- **Cosmetic formes are folded into their base.** Florges-Blue, the Sawsbuck
  seasons and Alcremie's creams are indexed as Florges, Sawsbuck and Alcremie.
  They share every mechanical property with the base, so this affects nothing
  but the name (47 slots in 12,000).
- **Happiness is fixed at 255**, pinning Return at 102 BP and Frustration at 1.
- **No Endless Battle Clause**, so a stalemate never resolves itself. Showdown
  ends games that neither side can win; here they run until the caller's step
  cap. Under uniformly random play this catches 43 battles in 8192 (0.5%) at a
  cap of 1000 steps, and it lengthens the mean game from Showdown's 58.6 turns
  to 63.6. The shape of it is always the same: `tests/test_engine.py` seed 7
  ends up with Toxapex (Recover, and Poison-typed so Toxic cannot touch it)
  against Umbreon (Wish, Protect), both at full HP and still there five thousand
  steps later. Real Showdown sets make this likelier than the procedural
  generator did, because they assemble coherent stall cores. Anything driving
  this engine needs a step cap, and an RL loop needs one that terminates the
  episode rather than leaving it hanging. Rollouts need a cap for this reason, which `rollout_batch` takes as
  `max_steps`.
- **Beat Up is not modelled.** Its power and hit count come from the whole
  party's base Attack, which would need party data threaded into the per-hit
  damage path. It appears in 1 of 4336 Random Battle sets.
- No Dynamax, Z-moves or Mega Evolution — none exist in Gen 9 singles.

