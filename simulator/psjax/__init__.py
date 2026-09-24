"""psjax -- Pokemon Showdown's Gen 9 battle simulator in JAX.

    from psjax.env import BattleEnv
    env = BattleEnv()
    state = env.reset(jax.random.PRNGKey(0))
    state, obs, rewards, done = env.step(state, actions)

Game data is compiled from the `pokemon-showdown` npm package; see
`tools/dump_data.js` and `psjax/build.py`. Run `python -m psjax.coverage` for an
honest account of what is and is not modelled.
"""
# --- a note on XLA optimisation levels ---------------------------------------
# This package used to force `--xla_backend_optimization_level=0`, because
# `vmap(step)` would not finish compiling otherwise: at a batch of two it ran
# past 150 seconds, while a batch of one took six. That is no longer needed, and
# the engine is roughly five times faster for it, but the reason is worth
# recording because it is easy to reintroduce.
#
# The cost is superlinear in *basic-block size*, not in operation count, and two
# things drove it:
#
#   * Copies of the big pieces. `run_action` (the whole move engine) and
#     `switch_to` (hazards, entry abilities) are large. Running them from a
#     `lax.fori_loop` over the two players puts one copy in the graph; a Python
#     loop puts one per player. Collapsing those took the program from 99,901
#     HLO operations to 58,469.
#   * Rank. Mapping the damage calculation over a move's hits only needs the
#     roll to vary, and `vmap` batches just what depends on it -- but Technician
#     reading a per-hit scaled `base_power` pulled the entire forty-step modifier
#     chain up a rank along with it. See `damage.calc_damage`'s `technician_power`.
#
# If a change here ever makes compilation hang again, those are the two things to
# look at first, in that order.
#
# --- and on XLA:GPU ------------------------------------------------------------
# On GPU the time goes mostly to XLA's priority-fusion pass, whose cost grows
# with how many consumers the busiest values have, and then to LLVM on each fused
# kernel. Fusion decisions depend on the batch size: at 1024 they once produced
# kernels of 2,409 operations, mostly Threefry rounds and division corrections,
# and a compile that ran past fifteen minutes on an RTX 4070. It is now about
# eight seconds at every batch size tried. Three things keep it there:
#
#   * Randomness drawn in bulk (`mechanics.random_words`). Each `jax.random` call
#     is a Threefry hash of ~100 operations; one per decision put ~56 of them in
#     the move loop, 40% of it.
#   * No `lax.switch` over the effect handlers (`moves.run_effect`). Batched, a
#     switch evaluates every branch at the batch shape and selects every output
#     across all 43 of them; that was a quarter of the traced program.
#   * Shifts and plain truncating division rather than `//` on signed integers
#     (`stats.chain_modify`, `stats.floordiv`), which adds a sign correction to
#     each of a few hundred divisions.
#
# Two more rules keep the step itself fast on GPU, where it was half as fast
# before them (see "Running on GPU" in the README):
#
#   * No scatters: state is written with `state.set_at`, never `.at[].set`. Each
#     scatter is a kernel of its own behind a copy of its operand.
#   * No `lax.cond` or `lax.switch` on a batched value: `state.select_state`.
#     Batched, they broadcast every table their branches read to the whole batch.

from .data import load_data, names
from .state import BattleState

__all__ = ["load_data", "names", "BattleState", "__version__"]
__version__ = "0.1.0"
