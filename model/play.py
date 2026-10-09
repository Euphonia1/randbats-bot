"""Play the trained network on a local Pokemon Showdown server.

    python model/play.py                                  # plays runs/selfplay/checkpoint.pt
    python model/play.py --checkpoint runs/first/checkpoint.pt
    python model/play.py --vs-random 20                   # check it against random play first

This starts Showdown's own server -- the `pokemon-showdown` package that
simulator/tools installs (`cd simulator/tools && npm install`) -- and logs the
network in to it as     . To play it, open http://localhost:8000, which
hands you on to the official client pointed at this server; choose any name,
press "Find a user", look up RandbatsBot and challenge it to [Gen 9] Random
Battle. It plays any number of battles at once, and at the start of each one
loads the checkpoint again if train.py has written a newer one since, so it
can play while it trains.

The bot sees what any player sees: its own team, in full, from Showdown's
requests, and the opponent through the battle log. `ShowdownBattle` follows the
battle and rebuilds from them the psjax state that `game_inputs` reads, with
placeholders for what the bot cannot know about the opponent (unseen Pokemon,
unused moves, held items), which `game_inputs` withholds in any case, as it
does in training. The legal actions come from the request itself.

The server runs with `--no-security`, so anyone can log in under any name; it
only listens on this machine (127.0.0.1).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import pathlib
import random
import shutil
import signal
import socket
import subprocess
import sys
import time

import jax
import numpy as np
import torch
import websockets

from psjax import consts as C
from psjax.data import load_data, names
from psjax.fog import HISTORY_LEN, FogOfWarEnv, FogState, empty_history
from psjax.state import BattleState, empty_state

from architechture import GameNetwork
from game_inputs import _view
from train import policy, sample, to_torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHOWDOWN = ROOT / "simulator" / "tools" / "node_modules" / "pokemon-showdown"
FORMAT = "gen9randombattle"

DATA = load_data()
N = names()
to_id = N.to_id

STATUSES = {"brn": C.BRN, "par": C.PAR, "slp": C.SLP, "frz": C.FRZ, "psn": C.PSN, "tox": C.TOX}
BOOSTS = {name: i for i, name in enumerate(C.BOOST_NAMES)}
STATS = ("atk", "def", "spa", "spd", "spe")  # stats[1:], as in C.STAT_NAMES
#: The opponent's item while it still has one: game_inputs shows only whether
#: it is gone, so any item will do.
HELD = N.item_id("leftovers")
CHOICE_ITEMS = {N.item_id(i) for i in ("choiceband", "choicescarf", "choicespecs")}
LIGHT_CLAY = N.item_id("lightclay")
PRESSURE = N.ability_id("pressure")
RAMPAGES = {N.move_id(m) for m in ("outrage", "petaldance", "thrash", "ragingfury")}
STRUGGLE = N.move_id("struggle")
#: How psjax starts the timed volatiles' counters; like psjax, `ShowdownBattle`
#: takes one off each at the end of every turn. The rest only need to be there.
VOLATILE_TURNS = {C.V_TAUNT: 3, C.V_ENCORE: 3, C.V_DISABLE: 4, C.V_YAWN: 2,
                  C.V_MAGNETRISE: 5, C.V_THROATCHOP: 2, C.V_SYRUPBOMB: 4, C.V_HEALBLOCK: 2,
                  C.V_SLOWSTART: 5, C.V_TORMENT: 3, C.V_PARTIALLYTRAPPED: 5, C.V_CONFUSION: 3}
TIMED_VOLATILES = [C.V_TAUNT, C.V_ENCORE, C.V_DISABLE, C.V_YAWN, C.V_MAGNETRISE,
                   C.V_THROATCHOP, C.V_TORMENT, C.V_PARTIALLYTRAPPED, C.V_SYRUPBOMB,
                   C.V_HEALBLOCK, C.V_SLOWSTART]
SINGLE_TURN_VOLATILES = [C.V_PROTECT, C.V_FLINCH, C.V_ROOST, C.V_HELPINGHAND, C.V_ENDURE,
                         C.V_FOCUSPUNCH, C.V_BEAKBLAST, C.V_MATBLOCK]
#: Showdown keeps each of these as a volatile of its own; psjax as Protect.
PROTECTS = {"protect", "detect", "spikyshield", "kingsshield", "banefulbunker", "obstruct",
            "silktrap", "burningbulwark", "maxguard"}
BINDING = {"bind", "wrap", "firespin", "whirlpool", "sandtomb", "magmastorm", "infestation",
           "snaptrap", "thundercage", "clamp"}


# --- reading the protocol ----------------------------------------------------------

def species_id(name: str) -> int:
    """A species as Showdown writes it ("Urshifu-Rapid-Strike"), or the nearest
    forme psjax has, so a cosmetic one ("Gastrodon-East") is its base species."""
    parts = name.split("-")
    while parts:
        sid = N.species.get(to_id("-".join(parts)))
        if sid is not None:
            return sid
        parts.pop()
    print(f"warning: unknown species {name!r}", file=sys.stderr)
    return 0


def move_id(name: str) -> int:
    """A move's psjax ID, -1 if psjax does not have it. Requests write Return
    as "return102" and Hidden Power with its type."""
    move = to_id(name)
    for candidate in (move, move.rstrip("0123456789"), "hiddenpower"):
        if candidate in N.moves:
            return N.moves[candidate]
        if not move.startswith("hiddenpower"):
            break
    return -1


def type_line(species: int) -> np.ndarray:
    return np.asarray(DATA["species_types"][species])


def parse_ident(ident: str) -> tuple[str, str]:
    """("p1", "Greedent") from "p1a: Greedent" or "p1: Greedent"."""
    side, _, name = ident.partition(": ")
    return side[:2], name


def parse_details(details: str) -> tuple[int, int, int, int | None]:
    """(species, level, gender, Tera type or None) from "Greedent, L86, F, tera:Ghost"."""
    species, *rest = details.split(", ")
    level, gender, tera = 100, C.GENDER_NONE, None
    for part in rest:
        if part[:1] == "L" and part[1:].isdigit():
            level = int(part[1:])
        elif part in ("M", "F"):
            gender = C.GENDER_M if part == "M" else C.GENDER_F
        elif part.startswith("tera:"):
            tera = N.type_id(part[5:])
    return species_id(species), level, gender, tera


def parse_condition(condition: str) -> tuple[int, int | None, int]:
    """(hp, maxhp, status) from "301/347 brn", "72/100" or "0 fnt"; maxhp is
    None for a fainted Pokemon, which does not say."""
    hp, _, status = condition.partition(" ")
    if status == "fnt" or hp == "0":
        return 0, None, C.STATUS_NONE
    hp, _, maxhp = hp.partition("/")
    return int(hp), int(maxhp) if maxhp else None, STATUSES.get(status, C.STATUS_NONE)


def effect_id(effect: str) -> str:
    """"move: Leech Seed" -> "leechseed"."""
    for prefix in ("move: ", "ability: ", "item: "):
        if effect.startswith(prefix):
            return to_id(effect[len(prefix):])
    return to_id(effect)


def split_args(args: list[str]) -> tuple[list[str], dict[str, str]]:
    """A protocol line's arguments, and its [tags]: "[from] item: Life Orb"
    gives tags["from"] = "item: Life Orb"."""
    positional, tags = [], {}
    for arg in args:
        if arg.startswith("["):
            key, _, value = arg[1:].partition("]")
            tags[key] = value.strip()
        else:
            positional.append(arg)
    return positional, tags


def legal_choices(request: dict) -> list[tuple[str, str, int, bool]]:
    """Every choice a request allows, as (choice, kind, index, tera): kind is
    "move", with the move's index in the request, or "switch", with the
    Pokemon's position in it."""
    team = request["side"]["pokemon"]
    choices = []

    def switches(revive=False):
        for pos, mon in enumerate(team):
            if not mon["active"] and mon["condition"].endswith("fnt") == revive:
                choices.append((f"switch {pos + 1}", "switch", pos, False))

    if request.get("forceSwitch"):
        # Revival Blessing asks for a fainted Pokemon instead
        switches(revive=any(mon.get("reviving") for mon in team))
        return choices
    active = request["active"][0]
    for i, move in enumerate(active["moves"]):
        if move.get("disabled") or move.get("pp", 1) <= 0:
            continue
        choices.append((f"move {i + 1}", "move", i, False))
        if active.get("canTerastallize"):
            choices.append((f"move {i + 1} terastallize", "move", i, True))
    if not active.get("trapped"):
        switches()
    return choices


# --- following a battle ------------------------------------------------------------

class ShowdownBattle:
    """One battle as one player follows it, kept as the psjax state that
    `game_inputs` reads. psjax side 0 is always the bot, side 1 its opponent.

    Pass every line of the battle room to `receive`, call `end_update` after
    each message the room sends (each is one decision point, after which psjax
    stamps its history with the turn), and `update` with every request. The
    bot's Pokemon take team slots in the order of its first request; the
    opponent's, in the order they appear.
    """

    def __init__(self, username: str):
        self.userid = to_id(username)
        self.me: str | None = None  # "p1" or "p2"
        self.st = {k: np.array(v) for k, v in
                   jax.device_get(empty_state(jax.random.PRNGKey(0)))._asdict().items()}
        # The opponent's unseen Pokemon are at full health and hold an item
        self.st["hp"][:] = 100
        self.st["maxhp"][:] = 100
        self.st["item"][1] = HELD
        self.revealed = np.zeros((C.NUM_PLAYERS, C.TEAM_SIZE), bool)
        self.revealed_moves = np.zeros((C.NUM_PLAYERS, C.TEAM_SIZE, C.MOVES_PER_POKEMON), bool)
        self.history = np.array(empty_history())
        self.events: list[tuple[int, int, int, int]] = []  # since the last update
        self.names: tuple[list[str], list[str]] = ([], [])  # each side's Pokemon by slot
        self.base_species = np.full((C.NUM_PLAYERS, C.TEAM_SIZE), -1)  # under a temporary forme
        self.out = [False, False]  # whether the side has sent anyone out yet
        self.first_seen = [None, None]  # the slot the latest switch-in revealed, if it did
        self.passing: list[str | None] = [None, None]  # Baton Pass or Shed Tail, until the switch
        self.turn, self.after_upkeep = 0, False
        self.request: dict | None = None
        self.ended, self.winner = False, None

    # --- who is who

    def _side(self, showdown_side: str) -> int:
        return 0 if showdown_side == self.me else 1

    def _slot(self, side: int, name: str) -> int:
        team = self.names[side]
        if name not in team:
            if len(team) == C.TEAM_SIZE:
                print(f"warning: a seventh Pokemon, {name}", file=sys.stderr)
                return C.TEAM_SIZE - 1
            team.append(name)
        return team.index(name)

    def _who(self, ident: str) -> tuple[int, int]:
        side, name = parse_ident(ident)
        return self._side(side), self._slot(self._side(side), name)

    def _set_species(self, side: int, slot: int, species: int):
        self.st["species"][side, slot] = species
        self.st["types"][side, slot] = type_line(species)

    # --- the log

    def receive(self, line: str):
        """Take one line of the battle log."""
        if not line.startswith("|"):
            return
        kind, *args = line[1:].split("|")
        handler = getattr(self, "on_" + kind.replace("-", "_"), None)
        if handler is not None:
            handler(*split_args(args))

    def end_update(self):
        """The end of one message from the battle room: stamp the events it
        held with the turn, as psjax does after each step -- the turns played
        to the end, which takes in this one once its residuals are done."""
        if not self.events:
            return
        turn = max(self.completed_turns(), 0)
        rows = np.array([event + (turn,) for event in self.events], self.history.dtype)
        self.history = np.concatenate([self.history, rows])[-HISTORY_LEN:]
        self.events = []

    def completed_turns(self) -> int:
        return self.turn if self.after_upkeep else self.turn - 1

    def on_player(self, args, tags):
        if len(args) >= 2 and to_id(args[1]) == self.userid:
            self.me = args[0]

    def on_turn(self, args, tags):
        self.turn, self.after_upkeep = int(args[0]), False

    def on_upkeep(self, args, tags):
        """The end of the turn's residuals, where psjax counts the timers down."""
        st = self.st
        self.after_upkeep = True
        for field in ("trick_room", "gravity"):
            st[field][...] = np.where(st[field] > 1, st[field] - 1, st[field])
        sc = st["side_conditions"]
        timed = np.arange(C.NUM_SIDE_CONDITIONS) >= C.SC_REFLECT
        sc[:, timed] = np.where(sc[:, timed] > 1, sc[:, timed] - 1, sc[:, timed])
        vol = st["volatiles"]
        vol[:, TIMED_VOLATILES] = np.where(vol[:, TIMED_VOLATILES] > 1,
                                           vol[:, TIMED_VOLATILES] - 1, vol[:, TIMED_VOLATILES])
        vol[:, SINGLE_TURN_VOLATILES] = 0

    def on_win(self, args, tags):
        self.ended, self.winner = True, args[0]

    def on_tie(self, args, tags):
        self.ended = True

    def on_switch(self, args, tags):
        side, name = parse_ident(args[0])
        side = self._side(side)
        new = name not in self.names[side]
        slot = self._slot(side, name)
        self._enter(side, slot, args[1], args[2] if len(args) > 2 else None)
        self.first_seen[side] = slot if new else None

    on_drag = on_switch

    def _enter(self, side: int, slot: int, details: str, condition: str | None):
        st = self.st
        if self.out[side]:
            self._leave(side, int(st["active"][side]))
        self.out[side] = True
        kept = {k: st[k][side].copy() for k in ("boosts", "volatiles", "sub_hp")}
        for field, value in (("boosts", 0), ("volatiles", 0), ("sub_hp", 0),
                             ("disabled_slot", -1), ("encore_slot", -1), ("locked_slot", -1),
                             ("choice_slot", -1), ("last_move", -1), ("boosted_stat", -1),
                             ("moves_since_switch", 0), ("transformed", False),
                             ("paradox_booster", False), ("illusion", -1)):
            st[field][side] = value
        passing, self.passing[side] = self.passing[side], None
        if passing == "batonpass":
            st["boosts"][side] = kept["boosts"]
            st["volatiles"][side, list(C.BATON_PASS_VOLATILES)] = \
                kept["volatiles"][list(C.BATON_PASS_VOLATILES)]
            st["sub_hp"][side] = kept["sub_hp"]
        elif passing == "shedtail":
            st["volatiles"][side, C.V_SUBSTITUTE] = kept["volatiles"][C.V_SUBSTITUTE]
            st["sub_hp"][side] = kept["sub_hp"]
        st["active"][side] = slot
        self._details(side, slot, details)
        if condition:
            self._condition(side, slot, condition)
        if st["status"][side, slot] == C.TOX:
            st["status_turns"][side, slot] = 1  # Toxic's count starts again
        self.revealed[side, slot] = True
        self.events.append((side, slot, int(st["species"][side, slot]), -1))

    def _leave(self, side: int, slot: int):
        """Undo what lasts only while on the field: a forme, Transform, a new type."""
        base = self.base_species[side, slot]
        species = base if base >= 0 else self.st["species"][side, slot]
        self._set_species(side, slot, species)
        self.base_species[side, slot] = -1

    def _details(self, side: int, slot: int, details: str):
        st = self.st
        species, level, gender, tera = parse_details(details)
        self._set_species(side, slot, species)
        st["level"][side, slot] = level
        st["gender"][side, slot] = gender
        if tera is not None:
            st["terastallized"][side, slot] = True
            st["tera_type"][side, slot] = tera
            st["tera_used"][side] = True

    def _condition(self, side: int, slot: int, condition: str):
        st = self.st
        hp, maxhp, status = parse_condition(condition)
        st["hp"][side, slot] = hp
        if maxhp is not None:
            st["maxhp"][side, slot] = maxhp
        self._status(side, slot, status)

    def _status(self, side: int, slot: int, status: int, rest: bool = False):
        st = self.st
        before = st["status"][side, slot]
        st["status"][side, slot] = status
        if status == C.SLP and (before != C.SLP or rest):
            st["sleep_attempts"][side, slot] = 0
            st["rest_sleep"][side, slot] = rest
        if status == C.TOX and before != C.TOX:
            st["status_turns"][side, slot] = 1

    def on_detailschange(self, args, tags):
        """A forme for good: Palafin-Hero, Terapagos-Terastal."""
        side, slot = self._who(args[0])
        self._details(side, slot, args[1])
        self.base_species[side, slot] = -1

    def on__formechange(self, args, tags):
        """A forme for as long as it stays in: Minior's core, Eiscue's Noice Face."""
        side, slot = self._who(args[0])
        if self.base_species[side, slot] < 0:
            self.base_species[side, slot] = self.st["species"][side, slot]
        self._set_species(side, slot, species_id(args[1]))

    def on__transform(self, args, tags):
        side, slot = self._who(args[0])
        target_side, target = self._who(args[1])
        if self.base_species[side, slot] < 0:
            self.base_species[side, slot] = self.st["species"][side, slot]
        self._set_species(side, slot, self.st["species"][target_side, target])
        self.st["transformed"][side] = True

    def on_replace(self, args, tags):
        """Illusion broken: the Pokemon out is really this one."""
        st = self.st
        showdown_side, name = parse_ident(args[0])
        side = self._side(showdown_side)
        shown = int(st["active"][side])
        if name not in self.names[side] and self.first_seen[side] == shown:
            # The disguise was never really seen, so its slot is this one's
            self.names[side][shown] = name
            self.revealed_moves[side, shown] = False
            st["moves"][side, shown] = -1
        slot = self._slot(side, name)
        if slot != shown:
            st["hp"][side, slot] = st["hp"][side, shown]
            st["status"][side, slot] = st["status"][side, shown]
            st["active"][side] = slot
        self._details(side, slot, args[1])
        self.revealed[side, slot] = True

    def on_move(self, args, tags):
        st = self.st
        side, slot = self._who(args[0])
        move = move_id(args[1])
        vol = st["volatiles"][side]
        if vol[C.V_TWOTURN]:  # the charged move going off
            vol[C.V_TWOTURN] = 0
            st["locked_slot"][side] = -1
        vol[C.V_DESTINYBOND] = 0
        source = tags.get("from", "")
        # A move another move or an ability called (Sleep Talk, Dancer, Magic
        # Bounce) is not the one the Pokemon chose, which is all psjax logs
        if source and source != "lockedmove":
            return
        st["last_move"][side] = move
        st["moves_since_switch"][side] = min(st["moves_since_switch"][side] + 1, 127)
        self.events.append((side, slot, int(st["species"][side, slot]), move))
        if move < 0 or move == STRUGGLE:
            return
        index = self._reveal(side, slot, move)
        if index is None:
            return
        if side == 1 and source != "lockedmove":
            # Pressure takes an extra PP from a move aimed at its holder
            mine = int(st["active"][0])
            aimed = len(args) > 2 and args[2] and self._side(args[2][:2]) == 0
            spent = 2 if aimed and st["ability"][0, mine] == PRESSURE else 1
            st["pp"][side, slot, index] = max(st["pp"][side, slot, index] - spent, 0)
        if move in RAMPAGES:
            vol[C.V_LOCKEDMOVE] = 1
            st["locked_slot"][side] = index
        elif vol[C.V_LOCKEDMOVE]:
            vol[C.V_LOCKEDMOVE] = 0
            st["locked_slot"][side] = -1
        if move == N.move_id("batonpass"):
            self.passing[side] = "batonpass"
        elif move == N.move_id("shedtail"):
            self.passing[side] = "shedtail"
        elif move == N.move_id("minimize"):
            vol[C.V_MINIMIZE] = 1
        elif move == N.move_id("defensecurl"):
            vol[C.V_DEFENSECURL] = 1

    def _reveal(self, side: int, slot: int, move: int) -> int | None:
        """The move's index in that Pokemon's moveset; the opponent's moves go
        in as they appear."""
        st = self.st
        moves = st["moves"][side, slot]
        found = np.nonzero(moves == move)[0]
        if len(found):
            index = int(found[0])
        elif side == 1 and np.any(moves < 0):
            index = int(np.nonzero(moves < 0)[0][0])
            moves[index] = move
            st["maxpp"][side, slot, index] = int(DATA["move_pp"][move]) * 8 // 5
            st["pp"][side, slot, index] = st["maxpp"][side, slot, index]
        else:
            return None
        self.revealed_moves[side, slot, index] = True
        return index

    def on_cant(self, args, tags):
        side, slot = self._who(args[0])
        if args[1] == "slp":
            self.st["sleep_attempts"][side, slot] += 1
        elif args[1] == "recharge":
            self.st["volatiles"][side, C.V_RECHARGE] = 0

    def on_faint(self, args, tags):
        side, slot = self._who(args[0])
        self.st["hp"][side, slot] = 0
        self.st["status"][side, slot] = C.STATUS_NONE
        self.st["last_faint"][...] = side

    def on__damage(self, args, tags):
        side, slot = self._who(args[0])
        self._condition(side, slot, args[1])
        if tags.get("from") in ("psn", "tox") and self.st["status"][side, slot] == C.TOX:
            self.st["status_turns"][side, slot] = min(self.st["status_turns"][side, slot] + 1, 16)

    on__heal = on__sethp = on__damage

    def on__status(self, args, tags):
        side, slot = self._who(args[0])
        self._status(side, slot, STATUSES.get(args[1], C.STATUS_NONE),
                     rest=tags.get("from") == "move: Rest")

    def on__curestatus(self, args, tags):
        side, slot = self._who(args[0])
        self.st["status"][side, slot] = C.STATUS_NONE

    def on__cureteam(self, args, tags):
        side, _ = self._who(args[0])
        self.st["status"][side] = C.STATUS_NONE

    def on__boost(self, args, tags, sign=1):
        side, _ = self._who(args[0])
        boosts = self.st["boosts"][side]
        stat = BOOSTS[args[1]]
        boosts[stat] = np.clip(boosts[stat] + sign * int(args[2]), -6, 6)

    def on__unboost(self, args, tags):
        self.on__boost(args, tags, sign=-1)

    def on__setboost(self, args, tags):
        side, _ = self._who(args[0])
        self.st["boosts"][side, BOOSTS[args[1]]] = int(args[2])

    def on__clearboost(self, args, tags):
        side, _ = self._who(args[0])
        self.st["boosts"][side] = 0

    def on__clearallboost(self, args, tags):
        self.st["boosts"][:] = 0

    def on__clearnegativeboost(self, args, tags):
        side, _ = self._who(args[0])
        self.st["boosts"][side] = np.maximum(self.st["boosts"][side], 0)

    def on__clearpositiveboost(self, args, tags):
        side, _ = self._who(args[0])
        self.st["boosts"][side] = np.minimum(self.st["boosts"][side], 0)

    def on__invertboost(self, args, tags):
        side, _ = self._who(args[0])
        self.st["boosts"][side] = -self.st["boosts"][side]

    def on__copyboost(self, args, tags):
        """The first Pokemon copies the second's stat stages (Psych Up)."""
        side, _ = self._who(args[0])
        source, _ = self._who(args[1])
        self.st["boosts"][side] = self.st["boosts"][source]

    def on__swapboost(self, args, tags):
        a, _ = self._who(args[0])
        b, _ = self._who(args[1])
        stats = [BOOSTS[s.strip()] for s in args[2].split(",")] if len(args) > 2 \
            else list(range(C.NUM_BOOSTS))
        boosts = self.st["boosts"]
        boosts[a, stats], boosts[b, stats] = boosts[b, stats].copy(), boosts[a, stats].copy()

    def on__weather(self, args, tags):
        self.st["weather"][...] = C.WEATHER_IDX.get(to_id(args[0]), C.WEATHER_NONE)
        if "upkeep" not in tags:
            self.st["weather_turns"][...] = 5

    def on__fieldstart(self, args, tags, on=True):
        effect, st = effect_id(args[0]), self.st
        if effect in C.TERRAIN_IDX:
            st["terrain"][...] = C.TERRAIN_IDX[effect] if on else C.TERRAIN_NONE
            st["terrain_turns"][...] = 5 if on else 0
        elif effect in ("trickroom", "gravity"):
            st[effect.replace("trickroom", "trick_room")][...] = 5 if on else 0

    def on__fieldend(self, args, tags):
        self.on__fieldstart(args, tags, on=False)

    def on__sidestart(self, args, tags):
        side = self._side(args[0][:2])
        condition = C.SIDE_CONDITION_IDX.get(effect_id(args[1]))
        if condition is None:
            return
        sc, st = self.st["side_conditions"][side], self.st
        if condition in C.SIDE_CONDITION_MAX:  # a hazard: one more layer
            sc[condition] = min(sc[condition] + 1, C.SIDE_CONDITION_MAX[condition])
        elif condition == C.SC_TAILWIND:
            sc[condition] = 4
        else:
            # The bot knows when its screens have Light Clay behind them; the
            # opponent's look like 5 turns, all a player can tell
            screen = condition in (C.SC_REFLECT, C.SC_LIGHTSCREEN, C.SC_AURORAVEIL)
            clay = side == 0 and st["item"][0, int(st["active"][0])] == LIGHT_CLAY
            sc[condition] = 8 if screen and clay else 5

    def on__sideend(self, args, tags):
        condition = C.SIDE_CONDITION_IDX.get(effect_id(args[1]))
        if condition is not None:
            self.st["side_conditions"][self._side(args[0][:2]), condition] = 0

    def on__swapsideconditions(self, args, tags):
        self.st["side_conditions"][:] = self.st["side_conditions"][::-1].copy()

    def on__start(self, args, tags):
        st = self.st
        side, slot = self._who(args[0])
        effect, vol = effect_id(args[1]), st["volatiles"][side]
        if effect == "typechange":
            types = [N.type_id(t) for t in args[2].split("/") if to_id(t) in N.types][:2]
            st["types"][side, slot] = (types + [C.TYPE_NONE, C.TYPE_NONE])[:2]
            return
        if effect.startswith("perish"):  # Showdown counts it down out loud
            vol[C.V_PERISHSONG] = int(effect[6:] or 3)
            return
        for paradox, volatile in (("protosynthesis", C.V_PROTOSYNTHESIS),
                                  ("quarkdrive", C.V_QUARKDRIVE)):
            if effect.startswith(paradox) and effect != paradox:
                vol[volatile] = 1
                st["boosted_stat"][side] = 1 + STATS.index(effect[len(paradox):])
                return
        moves = st["moves"][side, slot]
        if effect == "disable" and len(args) > 2:
            found = np.nonzero(moves == move_id(args[2]))[0]
            st["disabled_slot"][side] = found[0] if len(found) else -1
        elif effect == "encore":
            found = np.nonzero(moves == st["last_move"][side])[0]
            st["encore_slot"][side] = found[0] if len(found) else -1
        elif effect == "substitute":
            st["sub_hp"][side] = st["maxhp"][side, slot] // 4
        elif effect == "confusion" and "fatigue" in tags:
            vol[C.V_LOCKEDMOVE] = 0
            st["locked_slot"][side] = -1
        if effect in C.VOLATILE_IDX:
            volatile = C.VOLATILE_IDX[effect]
            vol[volatile] = VOLATILE_TURNS.get(volatile, 1)

    def on__end(self, args, tags):
        st = self.st
        side, _ = self._who(args[0])
        effect, vol = effect_id(args[1]), st["volatiles"][side]
        if "partiallytrapped" in tags or effect in BINDING:
            vol[C.V_PARTIALLYTRAPPED] = 0
        elif effect in ("protosynthesis", "quarkdrive"):
            st["boosted_stat"][side] = -1
            st["paradox_booster"][side] = False
        elif effect == "disable":
            st["disabled_slot"][side] = -1
        elif effect == "encore":
            st["encore_slot"][side] = -1
        elif effect == "substitute":
            st["sub_hp"][side] = 0
        if effect in C.VOLATILE_IDX:
            vol[C.VOLATILE_IDX[effect]] = 0

    def on__activate(self, args, tags):
        side, _ = self._who(args[0])
        effect = effect_id(args[1])
        if effect in BINDING:
            self.st["volatiles"][side, C.V_PARTIALLYTRAPPED] = VOLATILE_TURNS[C.V_PARTIALLYTRAPPED]
        elif effect == "trapped":
            self.st["volatiles"][side, C.V_TRAPPED] = 1
        elif effect in ("protosynthesis", "quarkdrive") and "fromitem" in tags:
            self.st["paradox_booster"][side] = True

    def on__singleturn(self, args, tags):
        side, _ = self._who(args[0])
        effect = effect_id(args[1])
        volatile = C.V_PROTECT if effect in PROTECTS else C.VOLATILE_IDX.get(effect)
        if volatile is not None:
            self.st["volatiles"][side, volatile] = 1

    def on__singlemove(self, args, tags):
        side, _ = self._who(args[0])
        volatile = C.VOLATILE_IDX.get(effect_id(args[1]))
        if volatile is not None:
            self.st["volatiles"][side, volatile] = 1

    def on__prepare(self, args, tags):
        """The first turn of a two-turn move: it is locked into the second."""
        side, slot = self._who(args[0])
        self.st["volatiles"][side, C.V_TWOTURN] = 1
        found = np.nonzero(self.st["moves"][side, slot] == move_id(args[1]))[0]
        self.st["locked_slot"][side] = found[0] if len(found) else -1

    def on__mustrecharge(self, args, tags):
        side, _ = self._who(args[0])
        self.st["volatiles"][side, C.V_RECHARGE] = 1

    def on__terastallize(self, args, tags):
        side, slot = self._who(args[0])
        self.st["terastallized"][side, slot] = True
        self.st["tera_type"][side, slot] = N.type_id(args[1])
        self.st["tera_used"][side] = True

    def on__item(self, args, tags):
        side, slot = self._who(args[0])
        self.st["item"][side, slot] = N.item_id(args[1]) if side == 0 else HELD

    def on__enditem(self, args, tags):
        side, slot = self._who(args[0])
        self.st["item"][side, slot] = 0

    # --- the request

    def update(self, request: dict):
        """Take what a request says about the bot's own side, which is everything."""
        st = self.st
        self.request = request
        self.me = self.me or request["side"]["id"]
        active_moves = (request.get("active") or [{}])[0].get("moves", [])
        for mon in request["side"]["pokemon"]:
            slot = self._slot(0, parse_ident(mon["ident"])[1])
            species, level, gender, _ = parse_details(mon["details"])
            if self.base_species[0, slot] < 0:  # not in a temporary forme
                self._set_species(0, slot, species)
            st["level"][0, slot], st["gender"][0, slot] = level, gender
            hp, maxhp, status = parse_condition(mon["condition"])
            st["hp"][0, slot] = hp
            if maxhp is not None:
                st["maxhp"][0, slot] = st["stats"][0, slot, C.HP] = maxhp
            self._status(0, slot, status)
            st["stats"][0, slot, 1:] = [mon["stats"][s] for s in STATS]
            moves = [move_id(m) for m in mon["moves"]][:C.MOVES_PER_POKEMON]
            moves += [-1] * (C.MOVES_PER_POKEMON - len(moves))
            for i, move in enumerate(moves):
                if st["moves"][0, slot, i] != move:  # new to us: full PP until we hear otherwise
                    st["moves"][0, slot, i] = move
                    st["maxpp"][0, slot, i] = st["pp"][0, slot, i] = \
                        int(DATA["move_pp"][move]) * 8 // 5 if move >= 0 else 0
            st["item"][0, slot] = N.item_id(mon["item"]) if mon["item"] else 0
            st["ability"][0, slot] = N.ability_id(mon["ability"])
            st["tera_type"][0, slot] = N.type_id(mon["teraType"]) if mon.get("teraType") \
                else C.TYPE_NONE
            st["terastallized"][0, slot] = bool(mon.get("terastallized"))
            if mon["active"]:
                st["active"][0] = slot
                self.revealed[0, slot] = True
                ids = [move_id(m["id"]) for m in active_moves]
                if ids == moves[:len(ids)] and len(ids) == len(mon["moves"]):
                    for i, m in enumerate(active_moves):
                        if "pp" in m:
                            st["pp"][0, slot, i], st["maxpp"][0, slot, i] = m["pp"], m["maxpp"]
                elif len(ids) == 1 and ids[0] in moves:  # locked into a move
                    st["locked_slot"][0] = moves.index(ids[0])
        st["tera_used"][0] |= st["terastallized"][0].any()

    def options(self) -> dict[int, str]:
        """Each psjax action the request allows, and its Showdown choice."""
        request = self.request
        team = request["side"]["pokemon"]
        active_moves = (request.get("active") or [{}])[0].get("moves", [])
        own = list(self.st["moves"][0, int(self.st["active"][0])])
        options = {}
        for choice, kind, index, tera in legal_choices(request):
            if kind == "switch":
                action = C.ACTION_SWITCH_BASE + self._slot(0, parse_ident(team[index]["ident"])[1])
            else:
                # Struggle and Recharge are slot 0 to psjax, as is a move it lacks
                move = move_id(active_moves[index]["id"])
                slot = own.index(move) if move in own else 0
                action = (C.ACTION_TERA_BASE if tera else C.ACTION_MOVE_BASE) + slot
            options.setdefault(action, choice)
        return options

    def fog_state(self) -> FogState:
        """The battle as psjax would hold it at this decision, and what each
        side has revealed."""
        st = self.st
        switching = bool(self.request and self.request.get("forceSwitch"))
        st["turn"][...] = max(self.completed_turns(), 0)
        st["phase"][...] = C.PHASE_SWITCH if switching else C.PHASE_MOVE
        theirs = int(st["active"][1])
        st["force_switch"][:] = [switching, switching and st["hp"][1, theirs] <= 0]
        st["fainted_count"][:] = (st["hp"] <= 0).sum(1)
        # A Choice item holds the bot to the move it last used, since it came in
        mine = int(st["active"][0])
        last = np.nonzero(st["moves"][0, mine] == st["last_move"][0])[0]
        locked = st["item"][0, mine] in CHOICE_ITEMS and st["moves_since_switch"][0] > 0
        st["choice_slot"][0] = last[0] if locked and len(last) else -1
        battle = BattleState(**{k: jax.numpy.asarray(v) for k, v in st.items()})
        return FogState(battle=battle, revealed=jax.numpy.asarray(self.revealed),
                        revealed_moves=jax.numpy.asarray(self.revealed_moves),
                        history=jax.numpy.asarray(self.history))


# --- the network -------------------------------------------------------------------

class Policy:
    """The network from a checkpoint, loaded again whenever the file changes."""

    def __init__(self, checkpoint: pathlib.Path, device: str = "cpu", greedy: bool = False):
        self.checkpoint, self.device, self.greedy = checkpoint, torch.device(device), greedy
        self.env = FogOfWarEnv()
        self.net = GameNetwork().to(self.device).requires_grad_(False)
        self.loaded_at, self.iteration = None, None
        self.reload()
        # Compile the input pipeline now rather than on the first move
        self.choose(ShowdownBattle(""), {0: "default"})

    def reload(self) -> bool:
        """Load the checkpoint if it has changed; whether it had."""
        mtime = self.checkpoint.stat().st_mtime
        if mtime == self.loaded_at:
            return False
        # Read in one go: train.py cannot replace the file while it is open here
        ckpt = torch.load(io.BytesIO(self.checkpoint.read_bytes()), map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.loaded_at, self.iteration = mtime, ckpt.get("iteration")
        return True

    @torch.no_grad()
    def choose(self, battle: ShowdownBattle, options: dict[int, str]) -> tuple[int, float, float]:
        """(action, its probability, the network's chance of winning) from `options`."""
        fs = jax.tree.map(lambda x: x[None], battle.fog_state())
        inputs = to_torch(_view(self.env, fs, 0), self.device)
        legal = torch.zeros(1, C.NUM_ACTIONS, dtype=torch.bool, device=self.device)
        legal[0, list(options)] = True
        inputs["legal_actions"] = legal
        log_probs, win_logit = policy(self.net, inputs)
        action = log_probs.argmax(-1) if self.greedy else sample(log_probs, legal)
        return (int(action), float(log_probs[0, action].exp()),
                float(torch.sigmoid(win_logit[0])))


# --- the bot -----------------------------------------------------------------------

class Bot:
    """The network as a Showdown user: it accepts challenges to Random Battles
    and plays every battle it is in."""

    def __init__(self, url: str, username: str, policy: Policy, verbose: bool = True):
        self.url, self.username, self.userid = url, username, to_id(username)
        self.policy, self.verbose = policy, verbose
        self.battles: dict[str, ShowdownBattle] = {}
        self.results: list[str | None] = []  # each finished battle's winner; None, a tie
        self.ready = asyncio.Event()
        self.ws = None

    async def run(self):
        async with websockets.connect(self.url, max_size=None) as self.ws:
            async for message in self.ws:
                await self.handle(message)

    async def send(self, room: str, text: str):
        await self.ws.send(f"{room}|{text}")

    def say(self, room: str, text: str):
        if self.verbose:
            print(f"[{room.removeprefix('battle-')}] {text}" if room else text, flush=True)

    async def handle(self, message: str):
        lines = message.split("\n")
        room = lines.pop(0)[1:] if lines[0].startswith(">") else ""
        if room.startswith("battle-"):
            await self.on_battle(room, lines)
            return
        for line in lines:
            if line.startswith("|challstr|"):
                await self.send("", f"/trn {self.username},0,")
            elif line.startswith("|updateuser|"):
                if to_id(line.split("|")[2]) == self.userid and not self.ready.is_set():
                    self.ready.set()
            elif line.startswith("|nametaken|"):
                raise RuntimeError(f"could not log in as {self.username}: {line}")
            elif line.startswith("|pm|"):
                await self.on_pm(line)
            elif line.startswith("|popup|"):
                self.say("", line[len("|popup|"):].replace("||", "\n"))

    async def on_pm(self, line: str):
        _, _, sender, receiver, text = line.split("|", 4)
        if to_id(receiver) != self.userid or to_id(sender) == self.userid:
            return
        if text.startswith("/challenge "):
            challenger, format_id = to_id(sender), text.split(" ", 1)[1].split("|")[0]
            if format_id == FORMAT:
                await self.send("", "/utm null")
                await self.send("", f"/accept {challenger}")
            else:
                await self.send("", f"/reject {challenger}")
                await self.send("", f"/pm {challenger}, I only play [Gen 9] Random Battle.")

    async def on_battle(self, room: str, lines: list[str]):
        battle = self.battles.get(room)
        if battle is None:
            if not any(line.startswith("|init|battle") for line in lines):
                return
            battle = self.battles[room] = ShowdownBattle(self.username)
            if self.policy.reload():
                self.say(room, f"loaded {self.policy.checkpoint} "
                               f"(iteration {self.policy.iteration})")
            self.say(room, f"started; the network is at iteration {self.policy.iteration}")
        request = None
        for line in lines:
            if line.startswith("|request|"):
                request = json.loads(line[len("|request|"):]) if line[len("|request|"):] else None
            elif line.startswith("|error|"):
                self.say(room, line)
                if "[Invalid choice]" in line and "nothing to choose" not in line:
                    await self.send(room, "/choose default")
                # An [Unavailable choice] (a trapping ability it could not see) is
                # followed by a new request
            else:
                battle.receive(line)
        battle.end_update()
        if request is not None:
            battle.update(request)
            if not request.get("wait") and not request.get("teamPreview"):
                await self.decide(room, battle, request)
        if battle.ended:
            self.results.append(battle.winner)
            won = battle.winner is not None and to_id(battle.winner) == self.userid
            self.say(room, "won" if won else "tie" if battle.winner is None else "lost")
            await self.send(room, "gg")
            await self.send("", f"/leave {room}")
            del self.battles[room]

    async def decide(self, room: str, battle: ShowdownBattle, request: dict):
        options = battle.options()
        if not options:
            await self.send(room, f"/choose default|{request['rqid']}")
            return
        action, prob, win = self.policy.choose(battle, options)
        choice = options[action]
        await self.send(room, f"/choose {choice}|{request['rqid']}")
        self.say(room, f"turn {battle.turn}: {self.describe(request, choice)} "
                       f"(p={prob:.2f}, thinks it wins {win:.0%})")

    @staticmethod
    def describe(request: dict, choice: str) -> str:
        kind, index, *tera = choice.split()
        if kind == "switch":
            return "switch to " + parse_ident(request["side"]["pokemon"][int(index) - 1]["ident"])[1]
        move = request["active"][0]["moves"][int(index) - 1]["move"]
        return move + (" with Tera" if tera else "")


async def random_opponent(url: str, name: str, bot: str, battles: int) -> None:
    """Challenge `bot` to `battles` Random Battles one after another, and play
    each with uniformly random legal choices."""
    played, logged_in = 0, False
    async with websockets.connect(url, max_size=None) as ws:
        async def send(room, text):
            await ws.send(f"{room}|{text}")

        async def challenge():
            await send("", "/utm null")
            await send("", f"/challenge {to_id(bot)}, {FORMAT}")

        async for message in ws:
            lines = message.split("\n")
            room = lines.pop(0)[1:] if lines[0].startswith(">") else ""
            for line in lines:
                if line.startswith("|challstr|"):
                    await send("", f"/trn {name},0,")
                elif (line.startswith("|updateuser|") and not logged_in
                      and to_id(line.split("|")[2]) == to_id(name)):
                    logged_in = True
                    await challenge()
                elif line.startswith("|request|") and line[len("|request|"):]:
                    request = json.loads(line[len("|request|"):])
                    if not request.get("wait"):
                        choices = legal_choices(request)
                        choice = random.choice(choices)[0] if choices else "default"
                        await send(room, f"/choose {choice}|{request['rqid']}")
                elif line.startswith("|error|[Invalid choice]"):
                    await send(room, "/choose default")
                elif line.startswith("|win|") or line == "|tie":
                    played += 1
                    await send("", f"/leave {room}")
                    if played == battles:
                        return
                    await challenge()


# --- the server ----------------------------------------------------------------------

def listening(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def start_server(port: int) -> subprocess.Popen:
    """Start Showdown's server on `port` and wait until it takes connections."""
    if not (SHOWDOWN / "pokemon-showdown").exists():
        sys.exit(f"Pokemon Showdown is not installed at {SHOWDOWN}; "
                 "run `npm install` in simulator/tools")
    if shutil.which("node") is None:
        sys.exit("Node.js is not on the PATH")
    if listening(port):
        sys.exit(f"something is already listening on port {port}: pass --no-server to "
                 "play on it, if it is a Showdown server, or --port to choose another")
    # What `node build` would set up, which --skip-build leaves out. The config
    # only overrides config-example.js, which Showdown loads first.
    config = SHOWDOWN / "config" / "config.js"
    if not config.exists():
        config.write_text("// Written by model/play.py: only this machine may connect.\n"
                          "exports.bindaddress = '127.0.0.1';\n")
    for folder in (SHOWDOWN / "logs" / "repl", SHOWDOWN / "config" / "chat-plugins"):
        folder.mkdir(parents=True, exist_ok=True)

    log = SHOWDOWN / "logs" / "play-server.log"
    group = (dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP) if os.name == "nt"
             else dict(start_new_session=True))
    server = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--skip-build", "--no-security", str(port)],
        cwd=SHOWDOWN, stdout=log.open("w"), stderr=subprocess.STDOUT, **group)
    deadline = time.monotonic() + 120
    while not listening(port):
        if server.poll() is not None or time.monotonic() > deadline:
            stop_server(server)
            sys.exit(f"the Showdown server did not start; see {log}")
        time.sleep(0.5)
    return server


def stop_server(server: subprocess.Popen):
    """Stop the server and the processes it started (it runs battles in its own)."""
    if server.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)], capture_output=True)
    else:
        os.killpg(server.pid, signal.SIGTERM)
    server.wait(timeout=10)


async def run(args: argparse.Namespace):
    url = f"ws://{args.host}:{args.port}/showdown/websocket"
    policy = Policy(args.checkpoint, args.device, args.greedy)
    bot = Bot(url, args.name, policy, verbose=not args.quiet)
    task = asyncio.create_task(bot.run())
    await asyncio.wait([task, asyncio.create_task(bot.ready.wait())],
                       return_when=asyncio.FIRST_COMPLETED)
    if task.done():
        task.result()  # it failed to log in: raise why
    print(f"{args.name} is online, playing iteration {policy.iteration} of {args.checkpoint}.")

    if not args.vs_random:
        print(f"Open http://localhost:{args.port}, choose a name, then press "
              f'"Find a user" and challenge {args.name} to [Gen 9] Random Battle. '
              "Ctrl+C stops it.", flush=True)
        await task
        return
    await random_opponent(url, "RandomPlayer", args.name, args.vs_random)
    for _ in range(100):  # the bot hears the last battle end a moment later
        if len(bot.results) >= args.vs_random:
            break
        await asyncio.sleep(0.1)
    task.cancel()
    wins = sum(w is not None and to_id(w) == bot.userid for w in bot.results)
    ties = sum(w is None for w in bot.results)
    print(f"\n{args.name} won {wins} of {len(bot.results)} battles against random play"
          + (f" ({ties} ties)" if ties else ""))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=pathlib.Path,
                        default=ROOT / "runs" / "selfplay" / "checkpoint.pt",
                        help="a checkpoint train.py wrote (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8000, help="default: %(default)s")
    parser.add_argument("--host", default="localhost",
                        help="the server to connect to, with --no-server (default: %(default)s)")
    parser.add_argument("--no-server", action="store_true",
                        help="play on a Showdown server that is already running")
    parser.add_argument("--name", default="RandbatsBot", help="the bot's name (default: %(default)s)")
    parser.add_argument("--greedy", action="store_true",
                        help="always take the likeliest action, instead of sampling as in training")
    parser.add_argument("--device", default="cpu", help="PyTorch's (default: %(default)s)")
    parser.add_argument("--vs-random", type=int, default=0, metavar="N",
                        help="instead of waiting for challenges, play N battles against a "
                             "client that plays at random, and report the score")
    parser.add_argument("--quiet", action="store_true", help="do not print every decision")
    args = parser.parse_args(argv)
    if not args.checkpoint.exists():
        sys.exit(f"no checkpoint at {args.checkpoint}; train one with model/train.py, "
                 "or pass --checkpoint")

    server = None if args.no_server else start_server(args.port)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            stop_server(server)


if __name__ == "__main__":
    main()
