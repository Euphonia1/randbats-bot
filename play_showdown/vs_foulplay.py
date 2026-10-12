"""Score a snapshot of the network against Foul Play
(https://github.com/pmariglia/foul-play), an open-source Showdown bot that
searches each decision with its own engine, poke-engine.

    python play_showdown/vs_foulplay.py runs/selfplay/snapshots/01080.pt
    python play_showdown/vs_foulplay.py runs/selfplay/snapshots/01080.pt --battles 400

This starts a private Showdown server, logs the network in to it as play.py
does, and runs `--processes` copies of Foul Play, which challenge it to
`--battles` Random Battles between them. The network runs on the CPU and
samples its actions, as in training's evaluations. Its score, 1 a win, 0.5 a
tie and 0 a loss, goes to stdout as a line of JSON, and is appended to `--out`
if given: model/train.py runs this in the background at every evaluation when
given --foulplay-battles.

Foul Play's strength is its search time per decision, `--search-time-ms`; keep
it the same for scores to compare. A score over n battles is good to about
0.5 / sqrt(n) either way: 0.035 over 200.

With `--search`, the network searches its decisions as play_online.py does
(play_showdown/search.py), for `--search-time` seconds or `--simulations`
each. Its searches run one at a time, so the battles in flight wait on each
other: a smaller budget than online keeps the run to a sensible length.

    python play_showdown/vs_foulplay.py runs/selfplay/snapshots/01080.pt --search --simulations 512

Foul Play needs Python 3.11+, and Rust to build its engine, so it gets a conda
env of its own, apart from training's, which also holds the Node.js that the
Showdown server runs on. Once:

    conda create -n foulplay --override-channels -c conda-forge python=3.12 nodejs=22 rust git
    cd simulator/tools
    PATH=~/miniconda3/envs/foulplay/bin:$PATH npm install    # the Showdown server
    git clone https://github.com/pmariglia/foul-play.git
    ~/miniconda3/envs/foulplay/bin/python -m pip install -r foul-play/requirements.txt

Foul Play logs in as a guest through play.pokemonshowdown.com, so this needs
an internet connection, though the battles themselves stay on this machine.
"""
from __future__ import annotations

import os

# One decision at a time is quicker on the CPU, and leaves the GPU to
# training. This has to be set before JAX is first imported.
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import asyncio
import json
import pathlib
import signal
import socket
import subprocess
import sys
import time

import play
import search

FOUL_PLAY = play.ROOT / "simulator" / "tools" / "foul-play"


def default_python() -> pathlib.Path:
    """$FOULPLAY_PYTHON, or the Python of a conda env named foulplay beside this one."""
    if "FOULPLAY_PYTHON" in os.environ:
        return pathlib.Path(os.environ["FOULPLAY_PYTHON"])
    return pathlib.Path(sys.executable).parents[2] / "foulplay" / "bin" / "python"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def evaluate(args: argparse.Namespace) -> dict:
    """Play the battles; the score and its parts."""
    start = time.monotonic()
    url = f"ws://localhost:{args.port}/showdown/websocket"
    policy = search.make_policy(args, args.snapshot, "cpu")
    bot = play.Bot(url, args.name, policy, verbose=False)
    task = asyncio.create_task(bot.run())
    await asyncio.wait([task, asyncio.create_task(bot.ready.wait())],
                       return_when=asyncio.FIRST_COMPLETED)
    if task.done():
        task.result()  # it failed to log in: raise why

    log = open(args.log, "a") if args.log else subprocess.DEVNULL
    launched = []

    def launch(name: str, battles: int) -> subprocess.Popen:
        # In a process group of its own, with the workers it searches in
        launched.append(subprocess.Popen(
            [str(args.foulplay_python), "run.py", "--websocket-uri", url,
             "--ps-username", name, "--bot-mode", "challenge_user",
             "--user-to-challenge", args.name, "--pokemon-format", play.FORMAT,
             "--run-count", str(battles), "--search-time-ms", str(args.search_time_ms),
             "--log-level", "WARNING"],
            cwd=args.foulplay_dir, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True))
        return launched[-1]

    def stop(p: subprocess.Popen):
        """Kill a copy of Foul Play and any search workers it left."""
        if os.name == "nt":
            p.kill()
        else:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        p.wait()

    def played(name: str) -> int:
        return sum(o is not None and play.to_id(o) == play.to_id(name) for o in bot.opponents)

    # Each copy of Foul Play plays its share of the battles one after another.
    # Now and then one crashes on a position it cannot handle, or waits forever
    # on a challenge that went astray; a fresh copy, under a new name, then
    # plays what was left of its share. The battle it was in never finishes,
    # so does not count.
    shares = [args.battles // args.processes + (i < args.battles % args.processes)
              for i in range(args.processes)]
    running = {f"FoulPlayEval{i + 1}": n for i, n in enumerate(shares) if n}
    processes = {name: launch(name, n) for name, n in running.items()}
    progress = {name: (0, time.monotonic()) for name in running}  # battles done, and when
    restarts = 0
    try:
        deadline = time.monotonic() + args.timeout
        while processes and not task.done() and time.monotonic() < deadline:
            await asyncio.sleep(1)
            now = time.monotonic()
            for name, p in list(processes.items()):
                done = played(name)
                if done != progress[name][0]:
                    progress[name] = (done, now)
                stalled = now - progress[name][1] > args.stall_seconds
                if p.poll() is None and not stalled:
                    continue
                if p.poll() is None:
                    stop(p)
                del processes[name]
                left = running.pop(name) - done
                if (stalled or p.returncode != 0) and left > 0 and restarts < 2 * args.processes:
                    restarts += 1
                    new = f"{name.split('x')[0]}x{restarts}"
                    running[new], processes[new] = left, launch(new, left)
                    progress[new] = (0, now)
        if task.done():
            task.result()  # the connection failed: raise why
        for _ in range(50):  # the bot hears the last battle end a moment after Foul Play
            if len(bot.results) >= args.battles:
                break
            await asyncio.sleep(0.1)
    finally:
        for p in launched:
            stop(p)
        task.cancel()
        if log is not subprocess.DEVNULL:
            log.close()

    n = len(bot.results)
    if n == 0:
        sys.exit(f"no battles finished against Foul Play"
                 + (f"; see {args.log}" if args.log else "; pass --log to see why"))
    if n < args.battles:
        print(f"only {n} of {args.battles} battles finished", file=sys.stderr)
    wins = sum(w is not None and play.to_id(w) == bot.userid for w in bot.results)
    ties = sum(w is None for w in bot.results)
    result = dict(iteration=policy.iteration, battles=n, wins=wins, losses=n - wins - ties,
                  ties=ties, score=(wins + 0.5 * ties) / n, search_time_ms=args.search_time_ms,
                  foulplay_restarts=restarts, seconds=round(time.monotonic() - start, 1),
                  snapshot=str(args.snapshot))
    if args.search:  # the network's own search, apart from Foul Play's search_time_ms
        result |= dict(network_search_time=args.search_time, simulations=args.simulations,
                       determinizations=args.determinizations, c_puct=args.c_puct)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("snapshot", type=pathlib.Path,
                        help="a snapshot or checkpoint that train.py wrote")
    parser.add_argument("--battles", type=int, default=200, help="default: %(default)s")
    parser.add_argument("--processes", type=int, default=8,
                        help="copies of Foul Play playing at once (default: %(default)s)")
    parser.add_argument("--search-time-ms", type=int, default=100,
                        help="Foul Play's search time per decision; its own default "
                             "(default: %(default)s)")
    parser.add_argument("--greedy", action="store_true",
                        help="always take the likeliest action (with --search, the one searched "
                             "the most), instead of sampling as in training")
    search.add_arguments(parser, default=False)
    parser.add_argument("--out", type=pathlib.Path, help="a JSON-lines file to append the score to")
    parser.add_argument("--log", type=pathlib.Path, help="a file to append Foul Play's output to")
    parser.add_argument("--foulplay-dir", type=pathlib.Path, default=FOUL_PLAY,
                        help="the Foul Play checkout (default: %(default)s)")
    parser.add_argument("--foulplay-python", type=pathlib.Path, default=default_python(),
                        help="the Python that Foul Play is installed in; its directory must "
                             "also hold node (default: $FOULPLAY_PYTHON, else %(default)s)")
    parser.add_argument("--port", type=int, default=0,
                        help="the private server's port; 0 picks a free one (default: %(default)s)")
    parser.add_argument("--name", default="RandbatsEval",
                        help="the network's name on the server (default: %(default)s)")
    parser.add_argument("--timeout", type=float, default=1800,
                        help="seconds to wait for the battles to finish (default: %(default)s)")
    parser.add_argument("--stall-seconds", type=float, default=300,
                        help="seconds a copy of Foul Play may go without finishing a battle "
                             "before a fresh one replaces it (default: %(default)s)")
    args = parser.parse_args(argv)
    for path, what in ((args.snapshot, "snapshot"), (args.foulplay_dir / "run.py", "Foul Play"),
                       (args.foulplay_python, "Python for Foul Play")):
        if not path.exists():
            sys.exit(f"no {what} at {path}; see the setup in --help")

    args.port = args.port or free_port()
    # The Showdown server runs on the Node.js in Foul Play's environment
    os.environ["PATH"] = f"{args.foulplay_python.parent}{os.pathsep}{os.environ['PATH']}"
    server = play.start_server(args.port)
    try:
        result = asyncio.run(evaluate(args))
    finally:
        play.stop_server(server)
    line = json.dumps(result)
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


if __name__ == "__main__":
    main()
