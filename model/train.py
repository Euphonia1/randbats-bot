"""Self-play PPO for GameNetwork.

    python model/train.py --run-dir runs/first
    python model/train.py --run-dir runs/first --resume     # carry on from its checkpoint
    python model/train.py --help                            # every setting

One network plays both sides of `num_envs` battles at once, each side from its
own fog-of-war view (`game_inputs`), and learns from both. Each iteration plays
`rollout_steps` decisions in every battle, starting a new battle wherever one
ends, then makes `epochs` passes of PPO over what it collected, with advantages
from generalized advantage estimation (GAE). JAX steps the battles on whatever
backend it has; PyTorch runs the network on `--device`, under bfloat16
autocast unless `--precision fp32`.

In an `opponent_pool` share of the battles the other side is not the network
but a snapshot of it from an earlier evaluation, drawn uniformly from all of
them each iteration, and the network learns from its own side only. Against
nothing but its current self, a game of simultaneous moves can send it round
in circles: each version learns to beat the last, forgetting what beat the one
before. The pool keeps it playing against its whole past (fictitious self-play).

The reward is undiscounted: 1 for a win and 0 for a loss, paid when the battle
ends, and nothing before. A tie pays 0.5 to each side, and so does a battle
still going at `max_turns`. A state's value is then the probability of winning
from it, which is what the win head's sigmoid already means, so the value loss
is the cross-entropy between that sigmoid and the GAE returns, which lie in
[0, 1].

The policy loss carries two regularizers, each averaged over the decisions with
more than one legal action and measured against U, the uniform policy over the
n legal actions:

- **Entropy**, H(pi) = -sum pi log pi, a bonus for spreading probability out.
  It is the reverse KL to uniform, KL(pi || U) = log n - H(pi), which hardly
  notices an action that is dying out: its gradient on that action's logit,
  -pi(a) (log pi(a) + H), vanishes along with pi(a).
- **Zero-avoiding**, KL(U || pi) = -log n - (1/n) sum_a log pi(a), the forward
  KL from uniform, a penalty. It goes to infinity as any legal action's
  probability goes to zero, and its gradient on each logit is pi(a) - 1/n, so
  however unlikely an action has become, it is pushed back up at a steady
  1/n. In a game of hidden information and simultaneous moves, a policy that
  never plays an action can no longer find out it was wrong to.

Every `eval_every` iterations the network plays `num_envs` new battles against
uniform random play, as many against itself as it was at the previous
evaluation, and as many against the newest snapshot at least `eval_lookback`
iterations old. It takes player 0 in half of each set and player 1 in the
rest, and samples its actions as it does in training. Then it is saved as a
snapshot, in the run directory's `snapshots/`, for later evaluations and the
opponent pool.
"""
from __future__ import annotations

import os

# By default JAX takes 75% of GPU memory the moment it starts, leaving PyTorch
# little room; this has to be set before JAX is first imported.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import copy
import dataclasses
import functools
import json
import math
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from psjax import consts as C
from psjax.fog import FogOfWarEnv, FogState

from architechture import GameNetwork
from game_inputs import _view


@dataclasses.dataclass
class TrainConfig:
    # Self-play
    num_envs: int = 2048  # battles in flight; each gives a sample per side per decision
    rollout_steps: int = 64  # decisions every battle plays between updates
    max_turns: int = 500  # a battle still going at this turn ends as a tie
    opponent_pool: float = 0.25  # share of battles against a past snapshot, not the current network
    # PPO
    lr: float = 3e-4
    epochs: int = 4  # passes over each rollout
    minibatch_size: int = 8192  # rows; a rollout has (2 - opponent_pool) * num_envs * rollout_steps
    clip: float = 0.2  # how far the probability ratio may move before PPO stops pushing
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    zero_avoid_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.0  # stop an update's epochs once one goes past this KL; 0 never stops
    # The run
    iterations: int = 5000
    eval_every: int = 20
    eval_lookback: int = 100  # also evaluate against the newest snapshot this many iterations old; 0 skips it
    save_every: int = 10
    seed: int = 0
    device: str = "auto"  # PyTorch's: cuda if it has a GPU, else cpu
    precision: str = "bf16"  # the network's matrix multiplies: bf16 (autocast) or fp32


# --- battles -----------------------------------------------------------------
#
# Every batch has a row per battle and side: row i is player 0 in battle i, and
# row num_envs + i is player 1 in the same battle.

def _views(env: FogOfWarEnv, fs: FogState) -> dict:
    """Both players' GameNetwork inputs, as JAX arrays, player 0's rows first."""
    return jax.tree.map(lambda a, b: jnp.concatenate([a, b]),
                        _view(env, fs, 0), _view(env, fs, 1))


@functools.partial(jax.jit, static_argnums=(0, 2))
def start(env: FogOfWarEnv, key, num_envs: int) -> tuple[FogState, dict]:
    """`num_envs` new battles, and both players' inputs for them."""
    fs = env.reset_batch(jax.random.split(key, num_envs))
    return fs, _views(env, fs)


@functools.partial(jax.jit, static_argnums=0)
def advance(env: FogOfWarEnv, fs: FogState, actions, key, max_turns):
    """Play one decision in every battle, then start a new one wherever a
    battle ended.

    actions: (2 * num_envs,) int32, one per row
    max_turns: a battle still going at this turn ends as a tie

    Returns (fs, rewards, done, inputs, key), each of the middle three by row:
        rewards: 1 to the winner and 0 to the loser of a battle that ended
            with this decision, 0.5 to each side for a tie, and 0 while it
            goes on
        done: whether the row's battle ended with this decision, in which
            case its inputs are the next battle's first
        inputs: both players' GameNetwork inputs for the next decision
    """
    n = fs.battle.turn.shape[0]
    key, reset_key = jax.random.split(key)
    fs, _, _, _ = env.step_batch(fs, actions.reshape(C.NUM_PLAYERS, n).T)
    battle = fs.battle
    over = battle.phase == C.PHASE_END
    done = over | (battle.turn >= max_turns)
    winner = jnp.where(over, battle.winner, 2).astype(jnp.int32)[:, None]  # 2 is a tie
    rewards = jnp.where(winner == 2, 0.5, (winner == jnp.arange(C.NUM_PLAYERS)).astype(jnp.float32))
    rewards = jnp.where(done[:, None], rewards, 0.0)

    fresh = env.reset_batch(jax.random.split(reset_key, n))
    fs = jax.tree.map(lambda new, old: jnp.where(done.reshape((n,) + (1,) * (old.ndim - 1)),
                                                 new, old), fresh, fs)
    return fs, rewards.T.reshape(-1), jnp.tile(done, C.NUM_PLAYERS), _views(env, fs), key


# --- the policy ----------------------------------------------------------------

def to_torch(tree, device: torch.device) -> dict:
    """JAX or NumPy arrays as torch tensors on `device`, each keeping its own
    compact dtype, for storing."""
    return jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(device),
                        jax.device_get(tree))


def model_inputs(tree: dict) -> dict:
    """Stored inputs in the dtypes GameNetwork takes, as `game_inputs` gives
    them: IDs and counts long, flags bool, the rest float32."""
    def cast(t: torch.Tensor) -> torch.Tensor:
        if t.dtype == torch.bool:
            return t
        return t.float() if t.is_floating_point() else t.long()
    return jax.tree.map(cast, tree)


def rows(tree: dict, index) -> dict:
    return jax.tree.map(lambda t: t[index], tree)


def policy(net: GameNetwork, inputs: dict,
           bf16: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """(log_probs, win_logit) for a batch of stored inputs: log_probs is
    (batch, NUM_ACTIONS) float32, with illegal actions at the lowest float.

    With bf16 the network runs under bfloat16 autocast. Its forward and
    backward are memory-bound, moving large activations more than they
    multiply, so halving those bytes makes an update about a third faster.
    The outputs are float32 either way."""
    device_type = next(net.parameters()).device.type
    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=bf16):
        logits, win_logit = net(**model_inputs(inputs))
    return torch.log_softmax(logits.float(), dim=-1), win_logit.float()


def sample(log_probs: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
    """An action per row drawn from `log_probs` (by Gumbel-max), never an
    illegal one. Zeros for `log_probs` draw uniformly from the legal actions."""
    gumbel = -torch.empty_like(log_probs).exponential_().log()
    return (log_probs + gumbel).masked_fill(~legal, -math.inf).argmax(-1)


def entropy_and_zero_avoid(log_probs: torch.Tensor,
                           legal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Each row's entropy H(pi) and KL(U || pi), U being uniform over its legal
    actions; see the module docstring. Both are 0 with a single legal action."""
    legal_log_probs = log_probs.masked_fill(~legal, 0.0)
    n = legal.sum(-1)
    entropy = -(log_probs.exp() * legal_log_probs).sum(-1)
    zero_avoid = -legal_log_probs.sum(-1) / n - torch.log(n.float())
    return entropy, zero_avoid


# --- PPO -----------------------------------------------------------------------

def gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
        last_values: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimates, undiscounted.

    rewards, values, dones: (T, B), a row per decision; dones[t] marks the
        battles that ended with decision t, whose values[t + 1] belong to the
        next battle and so are not bootstrapped from
    last_values: (B,) the values of the states after the last decision

    Returns (advantages, returns), each (T, B); the returns, advantages plus
    values, are the value head's targets.
    """
    advantages = torch.zeros_like(rewards)
    next_value, next_advantage = last_values, torch.zeros_like(last_values)
    for t in reversed(range(rewards.shape[0])):
        live = 1.0 - dones[t].float()
        delta = rewards[t] + live * next_value - values[t]
        next_advantage = delta + lam * live * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
    return advantages, advantages + values


def ppo_loss(log_probs: torch.Tensor, win_logit: torch.Tensor, legal: torch.Tensor,
             actions: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor,
             returns: torch.Tensor, cfg: TrainConfig) -> tuple[torch.Tensor, dict]:
    """PPO's clipped surrogate, the value loss and the two regularizers, for
    one minibatch; returns the loss and its parts as floats.

    The policy terms average over the decisions with a choice to make. The
    others -- the side that waits while the opponent replaces a fainted
    Pokemon, a Pokemon locked into Outrage -- give the policy nothing to learn,
    but still train the value.
    """
    choice = legal.sum(-1) > 1
    mean = lambda x: (x * choice).sum() / choice.sum().clamp(min=1)

    log_ratio = log_probs.gather(-1, actions[:, None]).squeeze(-1) - old_log_probs
    ratio = log_ratio.exp()
    surrogate = torch.minimum(ratio * advantages,
                              ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantages)
    policy_loss = -mean(surrogate)
    value_loss = F.binary_cross_entropy_with_logits(win_logit, returns)
    entropy, zero_avoid = (mean(x) for x in entropy_and_zero_avoid(log_probs, legal))
    loss = (policy_loss + cfg.value_coef * value_loss
            - cfg.entropy_coef * entropy + cfg.zero_avoid_coef * zero_avoid)

    with torch.no_grad():
        parts = dict(policy_loss=policy_loss, value_loss=value_loss, entropy=entropy,
                     zero_avoid=zero_avoid, approx_kl=mean(ratio - 1 - log_ratio),
                     clip_frac=mean(((ratio - 1).abs() > cfg.clip).float()))
    return loss, {k: v.item() for k, v in parts.items()}


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    """1 - Var(returns - values) / Var(returns): 1 for a value head that is
    always right, 0 for one no better than the mean."""
    var = returns.var()
    return float(1 - (returns - values).var() / var) if var > 0 else math.nan


def save_atomically(obj, path: pathlib.Path):
    """torch.save, written so a crash never leaves half of the file."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    # Windows will not replace a file another process has open, as
    # play_showdown/play.py does for a moment whenever it loads a new checkpoint.
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25)
    os.replace(tmp, path)


class Trainer:
    """The network, its optimizer, and the battles it is in the middle of.

    `snapshots` is the directory of the networks kept at each evaluation, for
    the opponent pool and the lookback evaluation; with None neither happens.
    """

    def __init__(self, cfg: TrainConfig, env: FogOfWarEnv | None = None,
                 snapshots: pathlib.Path | None = None):
        if not 0 <= cfg.opponent_pool <= 1:
            raise ValueError(f"opponent_pool is a share, 0 to 1, not {cfg.opponent_pool}")
        self.cfg = cfg
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                                   if cfg.device == "auto" else cfg.device)
        torch.manual_seed(cfg.seed)
        # TF32 matrix multiplies on the tensor cores of Ampere and later GPUs
        # (an A100, a 4070): several times plain fp32's speed, at 10 bits of
        # mantissa. Without it an A100 is slower than a 4070.
        torch.set_float32_matmul_precision("high")
        if cfg.precision not in ("bf16", "fp32"):
            raise ValueError(f"precision is bf16 or fp32, not {cfg.precision!r}")
        self.bf16 = cfg.precision == "bf16"
        self.env = env if env is not None else FogOfWarEnv()
        self.net = GameNetwork().to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr, eps=1e-5)
        # The opponent for evaluation: the network as it was at the previous one
        self.past = copy.deepcopy(self.net).requires_grad_(False)
        self.past_iteration = 0
        self.iteration = 0
        self.samples = 0

        self.snapshots = snapshots
        # Whichever snapshot the opponent pool's battles, or an evaluation, last loaded
        self.opponent = copy.deepcopy(self.net).requires_grad_(False)
        self.opponent_iteration = None
        self.rng = np.random.default_rng(cfg.seed)
        # The pool's battles are the last k. The opponent is player 1 in the
        # first half of them and player 0 in the rest, so the network learns
        # from both seats.
        n = cfg.num_envs
        k = round(cfg.opponent_pool * n) if snapshots is not None else 0
        pool = torch.arange(n - k, n)
        self.opponent_rows = torch.cat([n + pool[:k // 2], pool[k // 2:]]).to(self.device)
        self.learner_rows = torch.ones(2 * n, dtype=torch.bool, device=self.device)
        self.learner_rows[self.opponent_rows] = False

        self.key, key = jax.random.split(jax.random.PRNGKey(cfg.seed))
        self.fs, inputs = start(self.env, key, cfg.num_envs)
        self.inputs = to_torch(inputs, self.device)
        self.decisions = np.zeros(cfg.num_envs, np.int64)  # so far, in each battle

    def _advance(self, fs: FogState, actions: torch.Tensor, key):
        return advance(self.env, fs, jnp.asarray(actions.cpu().numpy(), jnp.int32),
                       key, self.cfg.max_turns)

    @torch.no_grad()
    def collect(self) -> tuple[dict, dict]:
        """Play `rollout_steps` decisions in every battle, against a snapshot
        drawn afresh in the opponent pool's.

        Returns the batch, a row per decision the network made itself, with its
        advantages and returns; and statistics on the battles that ended.
        """
        cfg, n = self.cfg, self.cfg.num_envs
        pool = len(self.opponent_rows) > 0
        if pool:
            # A pool battle still going from the last rollout carries on against
            # the new draw; the network's side of it is on-policy all the same.
            self.load_opponent(int(self.rng.choice(self.snapshot_iterations())))
        steps, lengths, ties = [], [], 0
        for _ in range(cfg.rollout_steps):
            log_probs, win_logit = policy(self.net, self.inputs, self.bf16)
            played = log_probs
            if pool:
                theirs = policy(self.opponent, rows(self.inputs, self.opponent_rows), self.bf16)[0]
                played = log_probs.index_copy(0, self.opponent_rows, theirs)
            actions = sample(played, self.inputs["legal_actions"])
            self.fs, rewards, done, inputs, self.key = self._advance(self.fs, actions, self.key)
            rewards, done, inputs = jax.device_get((rewards, done, inputs))
            steps.append(dict(
                inputs=self.inputs, action=actions,
                log_prob=log_probs.gather(-1, actions[:, None]).squeeze(-1),
                value=torch.sigmoid(win_logit),
                reward=torch.from_numpy(np.array(rewards)).to(self.device),
                done=torch.from_numpy(np.array(done)).to(self.device)))
            self.inputs = to_torch(inputs, self.device)

            self.decisions += 1
            ended = done[:n]
            lengths.extend(self.decisions[ended])
            ties += int(np.sum(rewards[:n][ended] == 0.5))
            self.decisions[ended] = 0

        _, last_logit = policy(self.net, self.inputs, self.bf16)
        batch = jax.tree.map(lambda *xs: torch.stack(xs), *steps)
        batch["advantage"], batch["return"] = gae(
            batch["reward"], batch["value"], batch["done"], torch.sigmoid(last_logit),
            cfg.gae_lambda)
        # The opponent's rows go: another policy chose their actions
        batch = jax.tree.map(lambda x: x[:, self.learner_rows].flatten(0, 1), batch)
        stats = dict(games=len(lengths), ties=ties,
                     battle_length=float(np.mean(lengths)) if lengths else math.nan,
                     explained_variance=explained_variance(batch["value"], batch["return"]))
        if pool:
            stats["opponent_iteration"] = self.opponent_iteration
        return batch, stats

    def update(self, batch: dict) -> dict:
        """`epochs` passes of PPO over `batch` in shuffled minibatches; returns
        the mean of each minibatch's loss parts and gradient norm."""
        cfg = self.cfg
        for group in self.opt.param_groups:
            group["lr"] = cfg.lr
        legal = batch["inputs"]["legal_actions"]
        # Normalized over the decisions with a choice, the only ones the policy reads
        advantages, choice = batch["advantage"], legal.sum(-1) > 1
        if choice.sum() > 1:
            chosen = advantages[choice]
            advantages = (advantages - chosen.mean()) / (chosen.std() + 1e-8)

        parts = []
        for _ in range(cfg.epochs):
            epoch = []
            for idx in torch.randperm(len(advantages), device=self.device).split(cfg.minibatch_size):
                log_probs, win_logit = policy(self.net, rows(batch["inputs"], idx), self.bf16)
                loss, info = ppo_loss(log_probs, win_logit, legal[idx], batch["action"][idx],
                                      batch["log_prob"][idx], advantages[idx],
                                      batch["return"][idx], cfg)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                info["grad_norm"] = nn.utils.clip_grad_norm_(
                    self.net.parameters(), cfg.max_grad_norm).item()
                self.opt.step()
                epoch.append(info)
            parts += epoch
            if cfg.target_kl and np.mean([p["approx_kl"] for p in epoch]) > cfg.target_kl:
                break
        return {k: float(np.mean([p[k] for p in parts])) for k in parts[0]}

    @torch.no_grad()
    def evaluate(self, opponent: GameNetwork | None) -> float:
        """The network's score over `num_envs` new battles against `opponent`,
        or uniform random play if None: 1 a win, 0.5 a tie and 0 a loss. The
        network plays player 0 in the first half of the battles and player 1 in
        the rest, and both sides sample their actions."""
        n = self.cfg.num_envs
        battle = np.arange(n)
        second = battle >= n // 2
        mine_np = battle + n * second  # the network's row in each battle
        mine, theirs = (torch.from_numpy(r).to(self.device)
                        for r in (mine_np, battle + n * ~second))
        self.key, key = jax.random.split(self.key)
        fs, inputs = start(self.env, key, n)
        score = np.full(n, np.nan)
        # Finished battles start again like any other, but only the first counts
        while np.isnan(score).any():
            inputs = to_torch(inputs, self.device)
            mine_lp = policy(self.net, rows(inputs, mine), self.bf16)[0]
            theirs_lp = (policy(opponent, rows(inputs, theirs), self.bf16)[0]
                         if opponent is not None else torch.zeros_like(mine_lp))
            log_probs = mine_lp.new_empty(2 * n, mine_lp.shape[-1])
            log_probs[mine], log_probs[theirs] = mine_lp, theirs_lp
            actions = sample(log_probs, inputs["legal_actions"])
            fs, rewards, done, inputs, key = self._advance(fs, actions, key)
            rewards, done = jax.device_get((rewards, done))
            rewards, done = np.asarray(rewards)[mine_np], np.asarray(done)[:n]
            first = done & np.isnan(score)
            score[first] = rewards[first]
        return float(score.mean())

    def evaluate_all(self) -> dict:
        """Scores against random play, against the previous evaluation's
        network, which this one then replaces, and against the newest snapshot
        at least `eval_lookback` iterations old, if there is one yet; then keeps
        the network as a snapshot."""
        stats = dict(vs_random=self.evaluate(None), vs_past=self.evaluate(self.past),
                     past_iteration=self.past_iteration)
        if self.snapshots is not None and self.cfg.eval_lookback:
            old = [i for i in self.snapshot_iterations()
                   if i <= self.iteration - self.cfg.eval_lookback]
            if old:
                stats |= dict(vs_lookback=self.evaluate(self.load_opponent(old[-1])),
                              lookback_iteration=old[-1])
        self.past.load_state_dict(self.net.state_dict())
        self.past_iteration = self.iteration
        self.keep(self.net, self.iteration)
        return stats

    def snapshot_iterations(self) -> list[int]:
        """The iterations there are snapshots of, oldest first."""
        return sorted(int(p.stem) for p in self.snapshots.glob("*.pt"))

    def keep(self, net: GameNetwork, iteration: int):
        """Save `net` as the snapshot of `iteration`, in a checkpoint's format,
        so that play_showdown/play.py can play it too."""
        if self.snapshots is not None:
            self.snapshots.mkdir(parents=True, exist_ok=True)
            save_atomically(dict(net=net.state_dict(), iteration=iteration),
                            self.snapshots / f"{iteration:05d}.pt")

    def load_opponent(self, iteration: int) -> GameNetwork:
        """`self.opponent`, as the snapshot of `iteration`."""
        ckpt = torch.load(self.snapshots / f"{iteration:05d}.pt", map_location=self.device)
        self.opponent.load_state_dict(ckpt["net"])
        self.opponent_iteration = iteration
        return self.opponent

    def save(self, path: pathlib.Path):
        """A checkpoint of the training, written so a crash never leaves half of one."""
        save_atomically(dict(net=self.net.state_dict(), opt=self.opt.state_dict(),
                             past=self.past.state_dict(), past_iteration=self.past_iteration,
                             iteration=self.iteration, samples=self.samples,
                             config=dataclasses.asdict(self.cfg)), path)

    def load(self, path: pathlib.Path):
        """Pick up from a checkpoint. The battles in flight start afresh."""
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.opt.load_state_dict(ckpt["opt"])
        self.past.load_state_dict(ckpt["past"])
        self.past_iteration = ckpt["past_iteration"]
        self.iteration = ckpt["iteration"]
        self.samples = ckpt["samples"]


# --- the run ---------------------------------------------------------------------

def parse_args(argv=None) -> tuple[TrainConfig, pathlib.Path, bool]:
    """The config, from TrainConfig's defaults (or the run's own config, when
    resuming) with any settings given on the command line on top."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=pathlib.Path, default=pathlib.Path("runs/selfplay"),
                        help="where the checkpoint, config, log and snapshots go "
                             "(default: %(default)s)")
    parser.add_argument("--resume", action="store_true",
                        help="carry on from the run directory's checkpoint, with its config")
    fields = dataclasses.fields(TrainConfig)
    for f in fields:
        parser.add_argument("--" + f.name.replace("_", "-"), type=type(f.default),
                            default=None, metavar=type(f.default).__name__.upper(),
                            help=f"default: {f.default}")
    args = parser.parse_args(argv)

    cfg = TrainConfig()
    if args.resume:
        cfg = TrainConfig(**json.loads((args.run_dir / "config.json").read_text()))
    given = {f.name: getattr(args, f.name) for f in fields if getattr(args, f.name) is not None}
    return dataclasses.replace(cfg, **given), args.run_dir, args.resume


def summary(s: dict) -> str:
    """One iteration's statistics as a line of the console log."""
    line = (f"iter {s['iteration']:5d}  {s['samples'] / 1e6:7.2f}M samples  "
            f"{s['games']:4d} games (length {s['battle_length']:5.1f}, {s['ties']} ties)  "
            f"policy {s['policy_loss']:+.4f}  value {s['value_loss']:.4f} "
            f"(ev {s['explained_variance']:+.3f})  entropy {s['entropy']:.3f}  "
            f"zero-avoid {s['zero_avoid']:.3f}  kl {s['approx_kl']:.4f}  "
            f"clip {s['clip_frac']:.3f}  [{s['collect_seconds']:.1f}s + {s['update_seconds']:.1f}s]")
    if "vs_random" in s:
        line += (f"\n           eval: {s['vs_random']:.3f} against random play, "
                 f"{s['vs_past']:.3f} against iteration {s['past_iteration']}")
        if "vs_lookback" in s:
            line += f", {s['vs_lookback']:.3f} against iteration {s['lookback_iteration']}"
    return line


def main(argv=None):
    cfg, run_dir, resume = parse_args(argv)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint, snapshots = run_dir / "checkpoint.pt", run_dir / "snapshots"
    if not resume and any(snapshots.glob("*.pt")):
        # They would join the new run's opponent pool
        raise SystemExit(f"{snapshots} holds another run's snapshots: pass --resume to "
                         "carry that run on, or a new --run-dir to start afresh")
    trainer = Trainer(cfg, snapshots=snapshots)
    if resume:
        trainer.load(checkpoint)
    # The pool's first opponent: the network at the last evaluation, or at the start
    trainer.keep(trainer.past, trainer.past_iteration)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2) + "\n")
    print(f"{sum(p.numel() for p in trainer.net.parameters()):,} parameters on "
          f"{trainer.device}, battles on JAX's {jax.default_backend()}; "
          f"{len(trainer.opponent_rows)} of {cfg.num_envs} battles against past snapshots; "
          f"{int(trainer.learner_rows.sum()) * cfg.rollout_steps:,} samples an iteration",
          flush=True)

    with open(run_dir / "log.jsonl", "a") as log:
        try:
            while trainer.iteration < cfg.iterations:
                t0 = time.perf_counter()
                batch, stats = trainer.collect()
                t1 = time.perf_counter()
                stats |= trainer.update(batch)
                trainer.iteration += 1
                trainer.samples += len(batch["action"])
                stats |= dict(iteration=trainer.iteration, samples=trainer.samples,
                              collect_seconds=t1 - t0, update_seconds=time.perf_counter() - t1)
                if trainer.iteration % cfg.eval_every == 0:
                    stats |= trainer.evaluate_all()
                if trainer.iteration % cfg.save_every == 0:
                    trainer.save(checkpoint)
                print(summary(stats), flush=True)
                log.write(json.dumps(stats) + "\n")
                log.flush()
        except KeyboardInterrupt:
            print("interrupted; saving")
    trainer.save(checkpoint)


if __name__ == "__main__":
    main()
