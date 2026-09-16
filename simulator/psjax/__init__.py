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

from .data import load_data, names
from .state import BattleState

__all__ = ["load_data", "names", "BattleState", "__version__"]
__version__ = "0.1.0"
