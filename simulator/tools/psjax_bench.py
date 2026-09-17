"""Benchmark this engine on complete Gen 9 Random Battles.

The counterpart to `tools/showdown_bench.js`: both play uniformly random legal
choices until every battle has ended, and both report the same fields. See the
"Benchmarks" section of the README for what the two engines are and aren't
doing equally, which the raw numbers do not show.

Compilation is reported apart from throughput because it is a fixed startup
cost -- roughly twenty seconds, paid once per shape, whether the batch holds
one battle or four thousand -- and folding it into a rate would say more about
the batch size chosen than about the engine.

    python tools/psjax_bench.py [batch sizes...]
"""
import functools
import json
import os
import sys
import threading
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from psjax import consts as C          # noqa: E402
from psjax.env import BattleEnv        # noqa: E402

CAP = 1000          # a randbats game is ~60 turns; this is a safety stop
CHECK_EVERY = 16    # steps between "is everything finished?" host syncs


@functools.partial(jax.jit, static_argnums=0)
def _step(env, state, key):
    """One decision point for the whole batch, actions sampled on device."""
    keys = jax.random.split(key, state.turn.shape[0])
    actions = jax.vmap(env.sample_actions)(state, keys)
    return jax.vmap(env.step)(state, actions)[0]


def bench(env, n, seed=0):
    t0 = time.time()
    states = env.reset_batch(jax.random.split(jax.random.PRNGKey(seed), n))
    jax.block_until_ready(states.hp)
    gen = time.time() - t0

    # Compile against a throwaway copy so the timed run starts warm.
    t0 = time.time()
    jax.block_until_ready(_step(env, states, jax.random.PRNGKey(0)).hp)
    compile_s = time.time() - t0

    key = jax.random.PRNGKey(seed + 1)
    t0, c0, steps, st = time.time(), time.process_time(), 0, states
    while steps < CAP:
        for _ in range(CHECK_EVERY):
            key, k = jax.random.split(key)
            st = _step(env, st, k)
            steps += 1
        if bool(jnp.all(st.phase == C.PHASE_END)):
            break
    jax.block_until_ready(st.hp)
    dt, cpu = time.time() - t0, time.process_time() - c0

    turns = int(jnp.sum(st.turn))
    return {
        "engine": "psjax",
        "battles": n,
        "finished": int(jnp.sum(st.phase == C.PHASE_END)),
        "compile_s": round(compile_s, 1),
        "team_generation_s": round(gen, 2),
        "battle_time_s": round(dt, 2),
        "battles_per_s": round(n / dt, 1),
        "mean_turns": round(turns / n, 1),
        "turns_per_s": round(turns / dt),
        "env_steps_per_s": round(n * steps / dt),
        # CPU seconds per wall second: how many of the machine's cores the
        # run actually kept busy, which for XLA:CPU here is close to one.
        "cores_used": round(cpu / dt, 2),
    }


def _play(env, st, key, out, idx):
    """Run one shard to completion. Called on its own Python thread."""
    steps = 0
    while steps < CAP:
        for _ in range(CHECK_EVERY):
            key, k = jax.random.split(key)
            st = _step(env, st, k)
            steps += 1
        if bool(jnp.all(st.phase == C.PHASE_END)):
            break
    jax.block_until_ready(st.hp)
    out[idx] = (st, steps)


def bench_threaded(env, n, shards, seed=0):
    """Split `n` battles across `shards` Python threads instead of one vmap.

    JAX drops the GIL while XLA runs, so the threads really do overlap. Whether
    that buys anything is the question: they share one process-wide XLA thread
    pool, so this is splitting the same cores rather than adding any.
    """
    per = n // shards
    t0 = time.time()
    states = [env.reset_batch(jax.random.split(jax.random.PRNGKey(seed + i), per))
              for i in range(shards)]
    jax.block_until_ready(states[0].hp)
    gen = time.time() - t0

    t0 = time.time()
    jax.block_until_ready(_step(env, states[0], jax.random.PRNGKey(0)).hp)
    compile_s = time.time() - t0

    out = [None] * shards
    t0, c0 = time.time(), time.process_time()
    threads = [threading.Thread(target=_play, args=(
        env, states[i], jax.random.PRNGKey(seed + 100 + i), out, i))
        for i in range(shards)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    dt, cpu = time.time() - t0, time.process_time() - c0

    turns = sum(int(jnp.sum(st.turn)) for st, _ in out)
    steps = sum(s for _, s in out)
    return {
        "engine": "psjax",
        "mode": f"{shards} threads x {per}",
        "battles": n,
        "finished": sum(int(jnp.sum(st.phase == C.PHASE_END)) for st, _ in out),
        "compile_s": round(compile_s, 1),
        "team_generation_s": round(gen, 2),
        "battle_time_s": round(dt, 2),
        "battles_per_s": round(n / dt, 1),
        "mean_turns": round(turns / n, 1),
        "turns_per_s": round(turns / dt),
        "env_steps_per_s": round(per * steps / dt),
        "cores_used": round(cpu / dt, 2),
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    shards = 0
    if "--shards" in args:
        i = args.index("--shards")
        shards = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    sizes = [int(a) for a in args] or [1024]
    env = BattleEnv()
    for n in sizes:
        print(json.dumps(bench_threaded(env, n, shards) if shards
                         else bench(env, n)), flush=True)
