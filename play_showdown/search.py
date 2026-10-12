"""pUCT search for the bot's decisions.

    policy = SearchPolicy(checkpoint)                # play.Policy's interface
    decision = policy.choose(battle, battle.options())

play.Policy plays the network's policy as it stands: one look at the battle
and one action drawn. SearchPolicy plays the turns ahead out in psjax, with
the network as its guide, and picks the action that held up best. It is
AlphaZero's search, PUCT, adapted to Random Battles in three ways:

- **The opponent's hidden information is sampled.** psjax needs the whole
  battle, and the bot sees only part of it. Each search draws
  `determinizations` battles that agree with everything the bot has seen
  (`determinize`): the opponent's sets from the team pool Showdown's own
  generator made, keeping the moves, Tera type, item and ability each Pokemon
  has shown, and its HP from the percentage; unseen Pokemon are drawn whole.
  Each battle is searched in a tree of its own, and their visit counts for the
  bot's options are summed at the end.
- **Moves are simultaneous.** Both players choose at every node, each by PUCT
  on statistics of its own actions alone (decoupled PUCT), with priors from
  the network's policy for that player's view of the battle. The opponent
  plays as well as the network can make it.
- **Chance is rolled again on every pass.** A tree node stands for the joint
  actions that led to it rather than for one state (an open-loop tree), so
  damage rolls, misses and crits are drawn afresh each time an action is
  tried, and its value averages over them. Outcomes that change what can be
  chosen next -- a faint, a forced switch -- lead to different nodes.

A leaf is valued by the network: the bot's chance of winning, averaged with
one minus the opponent's. Simulations run `batch` at a time, shared out
evenly between the trees and, within each, sent down different paths by
virtual loss, so the engine and the network see batches rather than single
states.

The battle searched is the one ShowdownBattle rebuilds, so what that does
not follow, the search plays as if it were not there: a Wish or Future Sight
on its way, an Illusion. How much HP an opponent's Substitute has left, and
how long its confusion lasts, are guesses.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import math
import sys
import threading
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
import torch

from psjax import consts as C
from psjax.data import load_team_pool
from psjax.fog import FogOfWarEnv, FogState, seen_order
from psjax.state import BattleState
from psjax.teams import legal_action_mask

from play import DATA, N, SPECIES_ABILITIES, Decision, Policy, ShowdownBattle
from game_inputs import _view  # noqa: E402  (play puts model/ on the path)
from train import policy as network  # noqa: E402

SPECIES_NUM = np.asarray(DATA["species_num"])
SPECIES_SET = np.asarray(DATA["species_set"])  # the species a set names, for a forme reached in battle
SPECIES_TYPES = np.asarray(DATA["species_types"])
BASE_STATS = np.asarray(DATA["species_base_stats"]).astype(np.int64)
MOVE_PP = np.asarray(DATA["move_pp"]).astype(np.int64)
IS_CHOICE = np.asarray(DATA["item_is_choice"])
STRUGGLE = N.move_id("struggle")


# --- the opponent's hidden information ---------------------------------------------

def stat_lines(species, level, zeroed=None) -> np.ndarray:
    """`[..., 6]` stats of `species` at `level` with Random Battles' 85 EVs and
    31 IVs, or 0 of each where `zeroed`, as psjax.stats works them out."""
    base = BASE_STATS[species]
    level = np.asarray(level, np.int64)[..., None]
    zeroed = np.zeros(base.shape, bool) if zeroed is None else zeroed
    iv, ev = np.where(zeroed, 0, 31), np.where(zeroed, 0, 85)
    core = (2 * base + iv + ev // 4) * level // 100
    stats = core + 5
    stats[..., C.HP] = np.where(base[..., C.HP] == 1, 1, core[..., C.HP] + level[..., 0] + 10)
    return stats


class SetSampler:
    """Sets for the opponent's Pokemon from the team pool that Showdown's own
    generator made (simulator/data/team_pool.npz), so each is one a Random
    Battle really deals: moves that go together, the item the generator gives
    them, the stats."""

    def __init__(self):
        pool = load_team_pool()
        if pool is None:
            raise FileNotFoundError("the search draws the opponent's sets from "
                                    "simulator/data/team_pool.npz, which is missing")

        def rows(name, *shape):
            return np.asarray(pool[name]).reshape(-1, *shape)

        self.species, self.level = rows("pool_species"), rows("pool_level")
        self.stats = rows("pool_stats", C.NUM_STATS)
        self.moves = rows("pool_moves", C.MOVES_PER_POKEMON)
        self.item, self.ability, self.tera = rows("pool_item"), rows("pool_ability"), rows("pool_tera_type")
        # The stats the generator built with 0 IVs and EVs: Attack on special
        # sets, Speed on Trick Room ones
        self.zeroed = self.stats < stat_lines(self.species, self.level)
        order = np.argsort(self.species, kind="stable")
        species, starts = np.unique(self.species[order], return_index=True)
        self.of_species = dict(zip(species.tolist(), np.split(order, starts[1:])))

    def fitting(self, species: int, moves: np.ndarray, tera: int, item: int, ability: int,
                choice: bool = True) -> np.ndarray:
        """The pool rows of `species` that best fit what it has shown: its
        `moves` (-1 for each not seen), and its Tera type, item and ability (-1
        until seen); with `choice=False`, sets without a Choice item. When none
        fits all of it (the pool lacks the set), those that fit the most. Empty
        if the pool has no Pokemon of that species."""
        rows = self.of_species.get(species)
        if rows is None:
            rows = self.of_species.get(int(SPECIES_SET[species]))
        if rows is None:
            return np.zeros(0, np.int64)
        score = 8 * np.isin(self.moves[rows], moves[moves >= 0]).sum(-1)
        if tera >= 0:
            score = score + 4 * (self.tera[rows] == tera)
        if item > 0:
            score = score + 2 * (self.item[rows] == item)
        if ability > 0:
            score = score + 2 * (self.ability[rows] == ability)
        if not choice:
            score = score + ~IS_CHOICE[self.item[rows]]
        return rows[score == score.max()]

    def unseen(self, rng: np.random.Generator, taken: set[int]) -> int:
        """A pool row for a Pokemon not yet seen: any in the pool, of a species
        (by Pokedex number, as Species Clause counts) the team lacks so far."""
        while True:
            row = int(rng.integers(len(self.species)))
            if int(SPECIES_NUM[self.species[row]]) not in taken:
                return row


def used_since_entry(history, side: int, slot: int) -> set[int]:
    """The moves the Pokemon in `slot` has used since it last came in, as far
    back as the history goes."""
    used = set()
    for event in np.asarray(history)[::-1]:
        if event[C.EV_SIDE] != side or event[C.EV_SLOT] != slot:
            continue
        if event[C.EV_MOVE] < 0:  # its switch-in
            break
        used.add(int(event[C.EV_MOVE]))
    used.discard(STRUGGLE)
    return used


@dataclasses.dataclass
class Facts:
    """What the bot knows of the battle besides `ShowdownBattle.fog_state()`.

    items, abilities: [6] each of the opponent's Pokemon's item and ability
        once the log has shown them, -1 until then
    choice: whether the opponent's active Pokemon may hold a Choice item -- not
        once it has used two different moves since it came in
    mid_turn: the bot is sending a Pokemon in partway through a turn (after its
        own U-turn, say), so the rest of the turn is still to come
    their_move_to_come: and the opponent has not acted yet this turn, so its
        move will land on whoever comes in
    """
    items: np.ndarray
    abilities: np.ndarray
    choice: bool = True
    mid_turn: bool = False
    their_move_to_come: bool = False

    @classmethod
    def of(cls, battle: ShowdownBattle) -> Facts:
        st, request = battle.st, battle.request or {}
        mine, theirs = int(st["active"][0]), int(st["active"][1])
        # A switch the turn's residuals have not come after is not a faint's
        # replacement, which waits until the end of the turn
        mid_turn = bool(request.get("forceSwitch")) and st["hp"][0, mine] > 0 \
            and not battle.after_upkeep
        return cls(items=battle.items[1].copy(), abilities=battle.abilities[1].copy(),
                   choice=len(used_since_entry(battle.history, 1, theirs)) < 2,
                   mid_turn=mid_turn,
                   their_move_to_come=mid_turn and not battle.acted[1] and st["hp"][1, theirs] > 0)


def determinize(observed: dict, seen: np.ndarray, facts: Facts, sampler: SetSampler,
                rng: np.random.Generator) -> dict:
    """One battle that the bot's view could be of.

    `observed` holds the BattleState fields the bot has rebuilt, side 0 its
    own (`ShowdownBattle.fog_state`), and `seen` which of the opponent's
    Pokemon it has seen. What it cannot see is drawn at random: each of the
    opponent's sets, from those of the pool that fit what that Pokemon has
    shown, and a whole Pokemon for each unseen one; then the turns each sleeper
    has left, which neither player is told. Returns the fields of the battle,
    as numpy arrays.
    """
    st = {k: np.array(v) for k, v in observed.items()}
    them = 1
    taken = {int(SPECIES_NUM[s]) for s in st["species"][them][seen]}
    for slot in range(C.TEAM_SIZE):
        if seen[slot]:
            species = int(st["species"][them, slot])
            tera = int(st["tera_type"][them, slot]) if st["terastallized"][them, slot] else -1
            active = slot == int(st["active"][them])
            rows = sampler.fitting(species, st["moves"][them, slot], tera, int(facts.items[slot]),
                                   int(facts.abilities[slot]), choice=facts.choice or not active)
            row = int(rng.choice(rows)) if len(rows) else None
        else:
            row = sampler.unseen(rng, taken)
            species = int(sampler.species[row])
            taken.add(int(SPECIES_NUM[species]))
            st["species"][them, slot] = species
            st["types"][them, slot] = SPECIES_TYPES[species]
            st["level"][them, slot] = sampler.level[row]
            st["hp"][them, slot] = 100  # a percentage, as the rest of their side's
        _give_set(st, slot, row, facts, sampler)

    # Their active Pokemon: held to its last move by a Choice item, and behind
    # a Substitute of a quarter of its own HP, not of the 100 the bot counts
    i = int(st["active"][them])
    if IS_CHOICE[st["item"][them, i]] and st["moves_since_switch"][them] > 0:
        found = np.flatnonzero(st["moves"][them, i] == st["last_move"][them])
        st["choice_slot"][them] = found[0] if len(found) else -1
    if st["volatiles"][them, C.V_SUBSTITUTE] > 0:
        st["sub_hp"][them] = st["maxhp"][them, i] // 4

    # Sleep lasts 1-3 turns (status_turns counts 2-4, see psjax.mechanics.
    # set_status), Rest's exactly 2, and a sleeper has slept `tried` of them
    for side, slot in zip(*np.nonzero(st["status"] == C.SLP)):
        tried = int(st["sleep_attempts"][side, slot])
        if st["rest_sleep"][side, slot]:
            left = 3 - tried
        else:
            left = int(rng.integers(min(max(2, tried + 1), 4), 5)) - tried
        st["status_turns"][side, slot] = max(left, 1)

    if facts.mid_turn:
        # The rest of the turn follows the switch: the opponent's move, if it
        # has yet to come (`Search` has it choose one), then the residuals.
        # A Pokemon of theirs that fainted meanwhile is replaced at the end.
        st["force_switch"][them] = False
        st["pending_side"][...] = 1 if facts.their_move_to_come else 0
        st["pending_action"][...] = -1
    return st


def _give_set(st: dict, slot: int, row: int | None, facts: Facts, sampler: SetSampler):
    """Give the opponent's Pokemon in `slot` the set in pool row `row`, keeping
    what it has shown; for None (no such species in the pool), only its stats."""
    them = 1
    species, level = int(st["species"][them, slot]), int(st["level"][them, slot])
    zeroed = sampler.zeroed[row] if row is not None else np.zeros(C.NUM_STATS, bool)
    if row is not None and sampler.species[row] == species and sampler.level[row] == level:
        stats = sampler.stats[row]
    else:  # a forme, or a level that differs: the set's spread on what it shows
        stats = stat_lines(species, level, zeroed)
    st["stats"][them, slot] = stats
    st["spread_zero"][them, slot] = np.sum(zeroed << np.arange(C.NUM_STATS))
    # Showdown shows HP as a percentage rounded up, and 99 for anything short of full
    percent, maxhp = int(st["hp"][them, slot]), int(stats[C.HP])
    st["maxhp"][them, slot] = maxhp
    st["hp"][them, slot] = (maxhp if percent >= 100 else 0 if percent <= 0 else
                            min(max(round((percent - 0.5) * maxhp / 100), 1), maxhp - 1))
    if row is None:
        st["ability"][them, slot] = SPECIES_ABILITIES[species, 0]
        return
    # The moves it has used stay in their slots; the set's others fill the rest
    moves = st["moves"][them, slot]
    unused = [m for m in sampler.moves[row] if m >= 0 and m not in moves]
    for i, move in zip(np.flatnonzero(moves < 0), unused):
        moves[i] = move
        st["maxpp"][them, slot, i] = st["pp"][them, slot, i] = MOVE_PP[move] * 8 // 5
    if st["item"][them, slot] != 0:  # 0 once it is gone
        st["item"][them, slot] = facts.items[slot] if facts.items[slot] > 0 else sampler.item[row]
    st["ability"][them, slot] = (facts.abilities[slot] if facts.abilities[slot] > 0
                                 else sampler.ability[row])
    if not st["terastallized"][them, slot]:
        st["tera_type"][them, slot] = sampler.tera[row]


def batch_of(battles: list[dict], fs: FogState) -> FogState:
    """`battles` (as `determinize` gives them) as one batch of FogStates, each
    with what `fs` records as revealed."""
    n = len(battles)
    battle = BattleState(**{k: jnp.asarray(np.stack([b[k] for b in battles]))
                            for k in BattleState._fields})
    revealed_moves = np.asarray(fs.revealed_moves)

    def tile(x):
        x = np.asarray(x)
        return jnp.asarray(np.broadcast_to(x, (n,) + x.shape))

    return FogState(battle=battle, revealed=tile(fs.revealed), revealed_moves=tile(revealed_moves),
                    history=tile(fs.history), move_order=tile(seen_order(revealed_moves)),
                    clay_screens=tile(np.zeros((C.NUM_PLAYERS, 3), bool)))


# --- the engine, batched ---------------------------------------------------------------

def _signature(env: FogOfWarEnv, battle: BattleState) -> tuple[jnp.ndarray, jnp.ndarray]:
    """`[batch, 7]`, what the tree tells decision points apart by: the phase,
    the winner, both active slots, each player's legal actions as a bitmask (0
    while it waits on the other), and the side owed the rest of a turn. Also
    the legal actions, `[batch, 2, NUM_ACTIONS]`, waiting or not."""
    legal = jax.vmap(lambda b: legal_action_mask(env.data, b))(battle)
    phase = battle.phase.astype(jnp.int32)
    acting = (phase[:, None] == C.PHASE_MOVE) | \
        ((phase[:, None] == C.PHASE_SWITCH) & battle.force_switch)
    bits = jnp.where(acting, jnp.sum(jnp.where(legal, 1 << jnp.arange(C.NUM_ACTIONS), 0), -1), 0)
    return jnp.stack([phase, battle.winner.astype(jnp.int32), battle.active[:, 0].astype(jnp.int32),
                      battle.active[:, 1].astype(jnp.int32), bits[:, 0], bits[:, 1],
                      battle.pending_side.astype(jnp.int32)], axis=-1), legal


@functools.partial(jax.jit, static_argnums=0)
def _inspect(env: FogOfWarEnv, fs: FogState):
    return _signature(env, fs.battle)


@functools.partial(jax.jit, static_argnums=0)
def _restart(env: FogOfWarEnv, fs: FogState, key) -> FogState:
    """`fs` with a random key of its own in each row, for a new round."""
    return fs._replace(battle=fs.battle._replace(key=jax.random.split(key, fs.battle.turn.shape[0])))


@functools.partial(jax.jit, static_argnums=0)
def _advance(env: FogOfWarEnv, fs: FogState, actions, moving, pending):
    """Step the rows that are `moving` with `actions` (`[batch, 2]`), the
    others staying as they are; first, where `pending`, the opponent's action
    becomes the move it has held over (see `Facts.their_move_to_come`).
    Returns the batch and its `_signature`."""
    battle = fs.battle
    held = jnp.where(pending, actions[:, 1], battle.pending_action).astype(battle.pending_action.dtype)
    stepped = env.step_batch(fs._replace(battle=battle._replace(pending_action=held)), actions)[0]
    fs = jax.tree.map(lambda new, old: jnp.where(moving.reshape((-1,) + (1,) * (old.ndim - 1)),
                                                 new, old), stepped, fs)
    return fs, _signature(env, fs.battle)[0]


@functools.partial(jax.jit, static_argnums=0)
def _views(env: FogOfWarEnv, fs: FogState) -> dict:
    """Both players' GameNetwork inputs, player 0's rows first."""
    return jax.tree.map(lambda a, b: jnp.concatenate([a, b]), _view(env, fs, 0), _view(env, fs, 1))


# --- the tree --------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def actions_of(bits: int) -> np.ndarray:
    """The actions in a `_signature` bitmask; a player with none, waiting on
    the other, has action 0, which the engine ignores."""
    if not bits:
        return np.zeros(1, np.int64)
    return np.flatnonzero((bits >> np.arange(C.NUM_ACTIONS)) & 1)


class Node:
    """A decision point in one determinization's tree: the joint actions that
    led to it from the root, and what came of them (`_signature`).

    Each player chooses by PUCT on statistics of its own actions (decoupled
    PUCT), so `visits[p]` and `values[p]` are by player p's actions, each
    value from that player's side: a sum of its chances of winning."""
    __slots__ = ("actions", "prior", "visits", "values", "count", "value", "children", "ready",
                 "terminal")

    def __init__(self, actions: tuple[np.ndarray, np.ndarray], value: float = 0.5,
                 terminal: bool = False):
        self.actions = actions  # (the bot's, the opponent's): psjax actions
        self.prior = None
        self.visits = [np.zeros(len(a)) for a in actions]
        self.values = [np.zeros(len(a)) for a in actions]
        self.count = 0
        self.value = value  # the bot's chance of winning from here, once evaluated
        self.children: dict[tuple, Node] = {}
        self.ready = self.terminal = terminal

    @classmethod
    def reached(cls, signature: list[int]) -> Node:
        """A new node for a decision point of this `_signature`."""
        phase, winner, _, _, ours, theirs, _ = signature
        if phase == C.PHASE_END:
            value = 1.0 if winner == 0 else 0.0 if winner == 1 else 0.5
            return cls((actions_of(0), actions_of(0)), value, terminal=True)
        return cls((actions_of(ours), actions_of(theirs)))

    def expand(self, probs: np.ndarray, value: float):
        """Take the network's evaluation: `probs`, each player's probability
        of each action (`[2, NUM_ACTIONS]`), and the bot's chance of winning."""
        self.prior = []
        for actions, p in zip(self.actions, probs):
            prior = p[actions]
            total = prior.sum()
            self.prior.append(prior / total if total > 0 else np.full(len(actions), 1 / len(actions)))
        self.value, self.ready = float(value), True

    def select(self, c_puct: float) -> tuple[int, int]:
        """Each player's choice, as indices into its actions. Each is counted
        as visited at once, and valued only once the simulation is back; until
        then it looks like a loss, which steers the rest of the batch elsewhere
        (virtual loss)."""
        scale = c_puct * math.sqrt(max(self.count, 1))
        picks = []
        for p in (0, 1):
            visits, values = self.visits[p], self.values[p]
            if len(visits) == 1:
                picks.append(0)
                continue
            # An action not tried yet counts as worth what the node is
            untried = self.value if p == 0 else 1 - self.value
            q = np.where(visits > 0, values / np.maximum(visits, 1), untried)
            picks.append(int(np.argmax(q + scale * self.prior[p] / (1 + visits))))
        self.count += 1
        self.visits[0][picks[0]] += 1
        self.visits[1][picks[1]] += 1
        return picks[0], picks[1]


@dataclasses.dataclass
class Result:
    """A search's verdict on the bot's options."""
    actions: np.ndarray  # the options, as psjax actions
    visits: np.ndarray  # each one's visits, over all the trees
    wins: np.ndarray  # the bot's mean chance of winning after each
    prior: np.ndarray  # the network's probability of each
    simulations: int


class Search:
    """PUCT over psjax with a GameNetwork; see the module docstring.

    net, device: the network and the device it is on
    bf16: run the network under bfloat16 autocast, as train.py does
    determinizations: battles the opponent's hidden information is drawn into,
        each searched in its own tree
    batch: simulations at once, a multiple of `determinizations`; the engine
        compiles for it
    c_puct: how far PUCT trusts the prior over the values found so far
    max_depth: the most decisions one simulation looks ahead
    """

    def __init__(self, net: torch.nn.Module, device: torch.device, bf16: bool = True,
                 env: FogOfWarEnv | None = None, sampler: SetSampler | None = None,
                 determinizations: int = 8, batch: int = 64, c_puct: float = 1.0,
                 max_depth: int = 10, seed: int | None = None):
        if determinizations < 1 or batch % determinizations:
            raise ValueError(f"the batch ({batch}) must be a multiple of the "
                             f"determinizations ({determinizations})")
        self.net, self.device, self.bf16 = net, torch.device(device), bf16
        self.env = env if env is not None else FogOfWarEnv()
        self.sampler = sampler if sampler is not None else SetSampler()
        self.trees, self.batch, self.c_puct, self.max_depth = determinizations, batch, c_puct, max_depth
        self.rng = np.random.default_rng(seed)

    @torch.inference_mode()
    def run(self, fs: FogState, options, facts: Facts, seconds: float = 2.0,
            simulations: int = 0) -> Result:
        """Search the decision `fs` holds, as the bot has rebuilt it (side 0 the
        bot's), over `options`, its legal actions. Runs until `seconds` have
        passed or `simulations` have been run, whichever comes first (0 for no
        limit), and always one batch."""
        if not seconds and not simulations:
            raise ValueError("a search needs a time or a number of simulations to stop at")
        deadline = time.monotonic() + seconds if seconds else math.inf
        observed = jax.device_get(fs.battle._asdict())
        seen = np.asarray(fs.revealed)[1]
        battles = [determinize(observed, seen, facts, self.sampler, self.rng)
                   for _ in range(self.trees)]
        start = batch_of([battles[r % self.trees] for r in range(self.batch)], fs)
        roots = self._roots(start, np.array(sorted(options)), facts)
        done = 0
        while not done or ((not simulations or done < simulations) and time.monotonic() < deadline):
            self._simulate(start, roots, facts.their_move_to_come)
            done += self.batch
        visits = np.sum([root.visits[0] for root in roots], axis=0)
        values = np.sum([root.values[0] for root in roots], axis=0)
        return Result(actions=roots[0].actions[0], visits=visits,
                      wins=values / np.maximum(visits, 1),
                      prior=np.mean([root.prior[0] for root in roots], axis=0), simulations=done)

    def _roots(self, start: FogState, ours: np.ndarray, facts: Facts) -> list[Node]:
        """A root for each determinization (row t of `start` for tree t),
        evaluated: the bot chooses from `ours`, the opponent from what the
        engine allows it."""
        trees, batch = self.trees, self.batch
        signature, legal = (np.asarray(x) for x in _inspect(self.env, start))
        views = _views(self.env, start)
        if facts.their_move_to_come:
            # It chose its move as the turn began, from the battle as it was then
            b = start.battle
            before = start._replace(battle=b._replace(
                phase=jnp.full_like(b.phase, C.PHASE_MOVE), force_switch=jnp.zeros_like(b.force_switch),
                pending_side=jnp.full_like(b.pending_side, -1)))
            views = jax.tree.map(lambda now, then: jnp.concatenate([now[:batch], then[batch:]]),
                                 views, _views(self.env, before))
        roots = []
        for t in range(trees):
            if facts.their_move_to_come:  # a move: a switch or Tera would have come first
                theirs = np.flatnonzero(legal[t, 1, :C.ACTION_TERA_BASE])
                theirs = theirs if len(theirs) else actions_of(0)
            else:
                theirs = actions_of(int(signature[t, 5]))
            roots.append(Node((ours, theirs)))
        self._evaluate(views, np.arange(trees), roots)
        return roots

    def _simulate(self, start: FogState, roots: list[Node], their_move_to_come: bool):
        """One round: `batch` simulations, row r in tree r % determinizations,
        each from its root down to a decision point the tree has not evaluated
        yet (or the end of the battle, or `max_depth`), and back up."""
        batch, trees = self.batch, len(roots)
        fs = _restart(self.env, start, jax.random.PRNGKey(int(self.rng.integers(2 ** 31))))
        nodes = [roots[r % trees] for r in range(batch)]
        paths: list[list[tuple[Node, int, int]]] = [[] for _ in range(batch)]
        leaves: list[Node | None] = [None] * batch
        moving = np.ones(batch, bool)
        for depth in range(self.max_depth):
            actions = np.zeros((batch, 2), np.int32)
            for r in np.flatnonzero(moving):
                node = nodes[r]
                i, j = node.select(self.c_puct)
                paths[r].append((node, i, j))
                actions[r] = node.actions[0][i], node.actions[1][j]
            pending = moving & (depth == 0 and their_move_to_come)
            fs, signature = _advance(self.env, fs, actions, moving, pending)
            signature = np.asarray(signature).tolist()
            last = depth == self.max_depth - 1
            for r in np.flatnonzero(moving):
                node, i, j = paths[r][-1]
                key = (i, j, *signature[r])
                child = node.children.get(key)
                if child is None:
                    child = node.children[key] = Node.reached(signature[r])
                if child.ready and not child.terminal and not last:
                    nodes[r] = child
                else:
                    leaves[r], moving[r] = child, False
            if not moving.any():
                break
        # Evaluate each new leaf once, at the first row that reached it
        first = {}
        for r, leaf in enumerate(leaves):
            if not leaf.ready:
                first.setdefault(id(leaf), r)
        if first:
            rows = np.array(sorted(first.values()))
            self._evaluate(_views(self.env, fs), rows, [leaves[r] for r in rows])
        for path, leaf in zip(paths, leaves):
            for node, i, j in path:
                node.values[0][i] += leaf.value
                node.values[1][j] += 1 - leaf.value

    def _evaluate(self, views: dict, rows: np.ndarray, nodes: list[Node]):
        """Expand `nodes` with the network's evaluation of rows `rows` of
        `views` (both players' views, as `_views` gives them)."""
        n = len(rows)
        index = np.concatenate([rows, rows + self.batch])
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.asarray(x)[index]).to(self.device),
                              jax.device_get(views))
        legal = np.zeros((2, n, C.NUM_ACTIONS), bool)
        for k, node in enumerate(nodes):
            legal[0, k, node.actions[0]] = legal[1, k, node.actions[1]] = True
        log_probs, win_logit = network(self.net, inputs, self.bf16,
                                       torch.from_numpy(legal.reshape(2 * n, -1)).to(self.device))
        probs = log_probs.exp().view(2, n, -1).transpose(0, 1).cpu().numpy()
        win = torch.sigmoid(win_logit).view(2, n).cpu().numpy()
        # Two estimates of one chance: the bot's own, and one less the opponent's
        for node, p, ours, theirs in zip(nodes, probs, win[0], win[1]):
            node.expand(p, (ours + 1 - theirs) / 2)


# --- the policy ----------------------------------------------------------------------

class SearchPolicy(Policy):
    """`play.Policy`, deciding by `Search` rather than by the network alone.

    Searches `seconds` per decision, or `simulations` if that comes first (0
    for no limit), then plays the action searched the most -- or, unless
    `greedy`, one drawn by its share of the visits from those within three
    quarters of the most, so that it does not always answer a position the
    same way when it rates two answers near enough alike. One search runs at
    a time, using the whole CPU (or `device`, for the network). If one fails,
    the network decides alone.
    """

    def __init__(self, checkpoint, device: str = "cpu", greedy: bool = False,
                 seconds: float = 2.0, simulations: int = 0, determinizations: int = 8,
                 c_puct: float = 1.0):
        if seconds <= 0 and simulations <= 0:
            raise ValueError("a search needs a time or a number of simulations to stop at")
        self.seconds, self.simulations = seconds, simulations
        self.search: Search | None = None  # until it is ready, Policy's own choices
        self.lock = threading.Lock()
        super().__init__(checkpoint, device, greedy)
        self.rng = np.random.default_rng()
        self.search = self._warm_up(Search(self.net, self.device, env=self.env,
                                           determinizations=determinizations, c_puct=c_puct))

    def _warm_up(self, search: Search) -> Search:
        """Compile the search on a battle psjax makes up, rather than at the
        first decision, and run the network in bfloat16 if that is faster here,
        as it is on a GPU and on CPUs that do bfloat16 natively."""
        fs = self.env.reset(jax.random.PRNGKey(0))
        legal = np.asarray(self.env.legal_actions(fs))[0]
        options = {int(a): "" for a in np.flatnonzero(legal)}
        facts = Facts(items=np.full(C.TEAM_SIZE, -1), abilities=np.full(C.TEAM_SIZE, -1))
        search.run(fs, options, facts, seconds=0, simulations=search.batch)
        inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self.device),
                              jax.device_get(_views(self.env, batch_of(
                                  [jax.device_get(fs.battle._asdict())] * search.batch, fs))))
        seconds = {}
        with torch.inference_mode():
            for bf16 in (False, True, False, True):
                start = time.perf_counter()
                network(self.net, inputs, bf16)
                seconds[bf16] = min(seconds.get(bf16, math.inf), time.perf_counter() - start)
        search.bf16 = seconds[True] < seconds[False]
        return search

    def choose(self, battle: ShowdownBattle, options: dict[int, str]) -> Decision:
        request = battle.request or {}
        reviving = any(mon.get("reviving") for mon in request.get("side", {}).get("pokemon", []))
        if self.search is None or len(options) < 2 or reviving:
            return super().choose(battle, options)
        with self.lock:
            try:
                self.search.net = self.net  # the latest, should it have been reloaded
                result = self.search.run(battle.fog_state(), options, Facts.of(battle),
                                         self.seconds, self.simulations)
            except Exception:
                traceback.print_exc()
                print("the search failed, so the network chooses alone", file=sys.stderr, flush=True)
                return super().choose(battle, options)
        visits = result.visits
        if self.greedy:
            pick = int(np.argmax(visits))
        else:
            near = np.where(visits >= 0.75 * visits.max(), visits, 0)
            pick = int(self.rng.choice(len(visits), p=near / near.sum()))
        return Decision(int(result.actions[pick]), float(visits[pick] / visits.sum()),
                        float(result.wins[pick]),
                        f"; {result.simulations} simulations, the network alone gave it "
                        f"p={result.prior[pick]:.2f}")


# --- the command line ----------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser, default: bool):
    """The search's settings, on by default or not."""
    group = parser.add_argument_group(
        "search", "pUCT search for each decision, from play_showdown/search.py")
    if default:
        group.add_argument("--no-search", dest="search", action="store_false",
                           help="play the network's own choices, without searching")
    else:
        group.add_argument("--search", action="store_true",
                           help="search each decision rather than play the network's own choice")
    group.add_argument("--search-time", type=float, default=2.0, metavar="SECONDS",
                       help="how long to search each decision (default: %(default)s)")
    group.add_argument("--simulations", type=int, default=0,
                       help="stop each search after this many simulations instead, if sooner; "
                            "0 for no limit (default: %(default)s)")
    group.add_argument("--determinizations", type=int, default=8,
                       help="battles the opponent's hidden information is drawn into, each "
                            "searched in a tree of its own; a divisor of 64 (default: %(default)s)")
    group.add_argument("--c-puct", type=float, default=1.0,
                       help="how far the search trusts the network's policy over what it has "
                            "found (default: %(default)s)")


def make_policy(args: argparse.Namespace, checkpoint, device: str) -> Policy:
    """The policy `add_arguments`' settings ask for."""
    if not args.search:
        return Policy(checkpoint, device, args.greedy)
    return SearchPolicy(checkpoint, device, args.greedy, args.search_time, args.simulations,
                        args.determinizations, args.c_puct)
