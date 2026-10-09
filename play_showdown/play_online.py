"""Play the trained network on the real Pokemon Showdown server.

    python play_showdown/play_online.py --name MyBotName
    python play_showdown/play_online.py --name MyBotName --accept-from MyOwnName  # only yours
    python play_showdown/play_online.py --name MyBotName --checkpoint runs/first/checkpoint.pt

This logs the network in to play.pokemonshowdown.com (or the server --server
names) and waits there for challenges, as play.py does on a local server: it
accepts challenges to [Gen 9] Random Battle, turns down the rest, plays any
number of battles at once up to --max-battles, and at the start of each battle
loads the checkpoint again if train.py has written a newer one since. To play
it, open https://play.pokemonshowdown.com, press "Find a user", look up the
bot's name and challenge it to [Gen 9] Random Battle.

The name can be a registered account or a free one. For a registered name, put
its password in SHOWDOWN_PASSWORD, or type it when asked; for a free one, leave
the password empty. SHOWDOWN_USERNAME stands in for --name. In PowerShell:

    $env:SHOWDOWN_USERNAME = "MyBotName"; $env:SHOWDOWN_PASSWORD = "..."
    python play_showdown/play_online.py

If the connection drops, the bot logs in again and rejoins its battles, which
Showdown keeps open meanwhile; it also picks up any battles it left unfinished
when it last stopped. A free name can only be taken back once the server
notices the old connection is gone, so until then it keeps trying.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import pathlib
import sys
import time
import traceback
import urllib.parse
import urllib.request

import websockets

from play import FORMAT, ROOT, Bot, Policy, to_id

SERVER = "wss://sim3.psim.us/showdown/websocket"
LOGIN_SERVER = "https://play.pokemonshowdown.com/action.php"
#: Showdown takes one message per 0.6 s from a user it does not trust, queues
#: what comes faster, and drops what queues up past five.
SEND_INTERVAL = 0.6


class LoginError(Exception):
    pass


def get_assertion(username: str, password: str, challstr: str) -> str:
    """The login server's signature on this connection taking `username`: as
    its account, with the password, or as a free name, without one."""
    if password:
        fields = {"act": "login", "name": username, "pass": password, "challstr": challstr}
    else:
        fields = {"act": "getassertion", "userid": to_id(username), "challstr": challstr}
    request = urllib.request.Request(LOGIN_SERVER, data=urllib.parse.urlencode(fields).encode(),
                                     headers={"User-Agent": "randbats-bot"})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read().decode()
    assertion = body
    if password:  # "]" and then JSON
        try:
            assertion = json.loads(body.removeprefix("]")).get("assertion") or ""
        except ValueError:
            raise LoginError(f"the login server answered {body[:200]!r}") from None
    if assertion == ";;@gmail":
        raise LoginError(f"{username} signs in with Google, which the bot cannot do: "
                         "give it an account with a password")
    if assertion.startswith(";;"):
        raise LoginError(f"could not log in as {username}: {assertion[2:]}")
    if assertion.startswith(";"):
        raise LoginError(f"{username} is a registered name: set SHOWDOWN_PASSWORD to its password")
    if not assertion:
        raise LoginError(f"could not log in as {username}: the login server answered {body[:200]!r}")
    return assertion


class OnlineBot(Bot):
    """`Bot` on a public server: it proves its name to the login server, keeps
    under the server's message limit, turns down challenges past `max_battles`
    (and, if `accept_from` names anyone, from everyone else), and comes back
    when the connection drops."""

    def __init__(self, url: str, username: str, password: str, policy: Policy,
                 max_battles: int = 5, accept_from=(), verbose: bool = True):
        super().__init__(url, username, policy, verbose)
        self.password, self.max_battles = password, max_battles
        self.accept_from = {to_id(name) for name in accept_from}
        self.sending, self.next_send = asyncio.Lock(), 0.0
        self.lost: set[str] = set()  # the battles it was in when the connection dropped
        self.joined: set[str] = set()  # the battles it has asked to rejoin on this connection
        self.logged_in = self.ever_logged_in = False
        self.check_games = False  # whether the next |updatesearch| lists only older battles

    async def run(self):
        delay = 0
        while True:
            self.logged_in, self.check_games, self.joined = False, False, set()
            try:
                async with websockets.connect(self.url, max_size=None) as self.ws:
                    async for message in self.ws:
                        await self.handle(message)
                reason = "the server closed the connection"
            except LoginError as error:
                if not self.ever_logged_in:
                    raise
                # It has logged in before, so this should pass: most likely the
                # server has not yet noticed the old connection is gone
                reason = str(error)
            except websockets.InvalidURI:
                raise
            except (OSError, websockets.WebSocketException) as error:
                reason = f"{type(error).__name__}: {error}"
            self.lost |= set(self.battles)
            self.battles.clear()
            # Straight back after a connection that worked, then slower and slower
            delay = 5 if self.logged_in else min(max(2 * delay, 5), 300)
            self.say("", f"disconnected ({reason}); trying again in {delay} s")
            await asyncio.sleep(delay)

    async def send(self, room: str, text: str):
        async with self.sending:
            await asyncio.sleep(self.next_send - time.monotonic())
            await self.ws.send(f"{room}|{text}")
            self.next_send = time.monotonic() + SEND_INTERVAL

    async def login(self, challstr: str):
        assertion = await asyncio.to_thread(get_assertion, self.username, self.password, challstr)
        await self.send("", f"/trn {self.username},0,{assertion}")

    async def handle(self, message: str):
        if not message.startswith(">"):
            for line in message.split("\n"):
                if line.startswith("|nametaken|"):
                    raise LoginError(f"could not log in as {self.username}: "
                                     + line.split("|", 3)[-1])
                if (line.startswith("|updateuser|") and not self.logged_in
                        and to_id(line.split("|")[2]) == self.userid):
                    self.logged_in = self.ever_logged_in = self.check_games = True
                    for room in sorted(self.lost):
                        await self.rejoin(room)
                    self.lost.clear()
                elif line.startswith("|updatesearch|") and self.logged_in and self.check_games:
                    # Logging in to a name lists the battles it has going
                    self.check_games = False
                    for room in json.loads(line[len("|updatesearch|"):]).get("games") or {}:
                        await self.rejoin(room)
        await super().handle(message)

    async def rejoin(self, room: str):
        if room.startswith("battle-") and room not in self.battles and room not in self.joined:
            self.joined.add(room)
            self.say(room, "rejoining")
            await self.send("", f"/join {room}")

    async def on_pm(self, line: str):
        _, _, sender, receiver, text = line.split("|", 4)
        if (text.startswith("/challenge ") and to_id(receiver) == self.userid
                and to_id(sender) != self.userid):
            challenger, format_id = to_id(sender), text.split(" ", 1)[1].split("|")[0]
            if self.accept_from and challenger not in self.accept_from:
                self.say("", f"turned down {sender[1:]}, who is not on --accept-from")
                await self.send("", f"/reject {challenger}")
                return
            if format_id == FORMAT and len(self.battles) >= self.max_battles:
                self.say("", f"turned down {sender[1:]}: already in {len(self.battles)} battles")
                await self.send("", f"/reject {challenger}")
                await self.send("", f"/pm {challenger}, I'm playing {len(self.battles)} "
                                    "battles already; challenge me again in a few minutes.")
                return
            self.say("", f"{sender[1:]} challenged it to {format_id}"
                         + ("" if format_id == FORMAT else ", which it turns down"))
            if format_id == FORMAT:
                self.check_games = False  # the next |updatesearch| will list this battle
        await super().on_pm(line)

    async def on_battle(self, room: str, lines: list[str]):
        try:
            await super().on_battle(room, lines)
            return
        except websockets.ConnectionClosed:
            raise
        except Exception:
            traceback.print_exc()
        # A message it cannot follow should cost it a move, not every battle it is in
        if any(line.startswith("|win|") or line == "|tie" for line in lines):
            self.battles.pop(room, None)
            await self.send("", f"/leave {room}")
            return
        for line in lines:
            if line.startswith("|request|") and line[len("|request|"):]:
                request = json.loads(line[len("|request|"):])
                if not request.get("wait"):
                    self.say(room, "lets Showdown choose this one")
                    await self.send(room, f"/choose default|{request['rqid']}")


async def run(args: argparse.Namespace, password: str):
    policy = Policy(args.checkpoint, args.device, args.greedy)
    bot = OnlineBot(args.server, args.name, password, policy, args.max_battles,
                    args.accept_from, verbose=not args.quiet)
    task = asyncio.create_task(bot.run())
    await asyncio.wait([task, asyncio.create_task(bot.ready.wait())],
                       return_when=asyncio.FIRST_COMPLETED)
    if task.done():
        task.result()  # it failed to log in: raise why
    print(f"{args.name} is online at {args.server}, playing iteration {policy.iteration} of "
          f"{args.checkpoint}. Challenge it to [Gen 9] Random Battle from "
          "https://play.pokemonshowdown.com. Ctrl+C stops it.", flush=True)
    await task


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default=os.environ.get("SHOWDOWN_USERNAME"),
                        help="the name to play under (default: $SHOWDOWN_USERNAME)")
    parser.add_argument("--checkpoint", type=pathlib.Path,
                        default=ROOT / "runs" / "selfplay" / "checkpoint.pt",
                        help="a checkpoint train.py wrote (default: %(default)s)")
    parser.add_argument("--server", default=SERVER,
                        help="the server's websocket (default: %(default)s)")
    parser.add_argument("--accept-from", nargs="+", default=[], metavar="USER",
                        help="take challenges from these users only")
    parser.add_argument("--max-battles", type=int, default=5,
                        help="turn down challenges while in this many battles; the main server "
                             "allows no more than 5 or so (default: %(default)s)")
    parser.add_argument("--greedy", action="store_true",
                        help="always take the likeliest action, instead of sampling as in training")
    parser.add_argument("--device", default="cpu", help="PyTorch's (default: %(default)s)")
    parser.add_argument("--quiet", action="store_true", help="do not print every decision")
    args = parser.parse_args(argv)
    if not args.name:
        parser.error("give the bot a name with --name or SHOWDOWN_USERNAME")
    if not args.checkpoint.exists():
        sys.exit(f"no checkpoint at {args.checkpoint}; train one with model/train.py, "
                 "or pass --checkpoint")
    password = os.environ.get("SHOWDOWN_PASSWORD")
    if password is None:
        password = getpass.getpass(f"Password for {args.name} (empty if it is not registered): ") \
            if sys.stdin.isatty() else ""

    try:
        asyncio.run(run(args, password))
    except KeyboardInterrupt:
        pass
    except LoginError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
