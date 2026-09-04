"""psjax -- Pokemon Showdown's Gen 9 battle simulator in JAX.

    from psjax.env import BattleEnv
    env = BattleEnv()
    state = env.reset(jax.random.PRNGKey(0))
    state, obs, rewards, done = env.step(state, actions)

Game data is compiled from the `pokemon-showdown` npm package; see
`tools/dump_data.js` and `psjax/build.py`. Run `python -m psjax.coverage` for an
honest account of what is and is not modelled.
"""
from .data import load_data, names
from .state import BattleState

__all__ = ["load_data", "names", "BattleState", "__version__"]
__version__ = "0.1.0"
