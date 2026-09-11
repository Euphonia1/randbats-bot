"""psjax -- Pokemon Showdown's Gen 9 battle simulator in JAX.

    from psjax.env import BattleEnv
    env = BattleEnv()
    state = env.reset(jax.random.PRNGKey(0))
    state, obs, rewards, done = env.step(state, actions)

Game data is compiled from the `pokemon-showdown` npm package; see
`tools/dump_data.js` and `psjax/build.py`. Run `python -m psjax.coverage` for an
honest account of what is and is not modelled.
"""
import os as _os

# --- XLA tuning for batched compilation --------------------------------------
# A battle step is thousands of tiny operations on six- and seven-element arrays.
# Unbatched that is fine, but `vmap` turns every one of them into a vectorised
# op, and XLA's CPU backend runs LLVM optimisation passes whose cost grows
# superlinearly in that op count: `vmap(step)` at a batch of two did not finish
# compiling in 150 seconds, while a batch of one took 6. Dropping the backend
# optimisation level removes those passes, and compilation becomes ~26 seconds
# for any batch size, with results bit-identical to the unbatched engine.
#
# It has to be set before JAX initialises its backend, which is why it lives at
# the top of the package. Set PSJAX_NO_XLA_TUNING=1 to opt out.
def _tune_xla() -> None:
    if _os.environ.get("PSJAX_NO_XLA_TUNING"):
        return
    flags = _os.environ.get("XLA_FLAGS", "")
    if "xla_backend_optimization_level" in flags:
        return          # respect an explicit setting
    _os.environ["XLA_FLAGS"] = (flags + " --xla_backend_optimization_level=0").strip()


_tune_xla()

from .data import load_data, names          # noqa: E402
from .state import BattleState              # noqa: E402

__all__ = ["load_data", "names", "BattleState", "__version__"]
__version__ = "0.1.0"
