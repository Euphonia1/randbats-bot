"""ShowdownBattle must rebuild, from what a player is told, the inputs the
network trained on.

    python -m pytest model/test_play.py

The battle lines are real Showdown output, from a Random Battle on a local server.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

from psjax import consts as C
from psjax.fog import FogOfWarEnv

from architechture import UNKNOWN
from game_inputs import _view
from play import N, ShowdownBattle, legal_choices

ENV = FogOfWarEnv()

#: The bot's team: (name, details, moves, item, ability, Tera type)
TEAM = [
    ("Greedent", "Greedent, L86, F", ["knockoff", "swordsdance", "earthquake", "doubleedge"],
     "sitrusberry", "cheekpouch", "Ghost"),
    ("Wyrdeer", "Wyrdeer, L87, M", ["psychicnoise", "bodyslam", "earthquake", "thunderwave"],
     "leftovers", "intimidate", "Ground"),
    ("Cobalion", "Cobalion, L80", ["aurasphere", "vacuumwave", "flashcannon", "calmmind"],
     "leftovers", "justified", "Water"),
    ("Coalossal", "Coalossal, L89, M", ["flamethrower", "rapidspin", "willowisp", "stoneedge"],
     "heavydutyboots", "flamebody", "Water"),
    ("Vespiquen", "Vespiquen, L99, F", ["roost", "airslash", "toxic", "uturn"],
     "heavydutyboots", "pressure", "Steel"),
    ("Entei", "Entei, L78", ["stoneedge", "flareblitz", "extremespeed", "sacredfire"],
     "choiceband", "innerfocus", "Fire"),
]
MAXHP = [347, 321, 277, 341, 299, 307]


def request(conditions=None, order=range(6), active_moves=None, trapped=False, tera=True,
            **extra) -> dict:
    """A request for the bot, as Showdown words one: `order` is the team as
    the request lists it, active first; `conditions` each Pokemon's HP and
    status by team index."""
    conditions = conditions or {}
    pokemon = []
    for position, i in enumerate(order):
        name, details, moves, item, ability, tera_type = TEAM[i]
        pokemon.append(dict(
            ident=f"p1: {name}", details=details,
            condition=conditions.get(i, f"{MAXHP[i]}/{MAXHP[i]}"), active=position == 0,
            stats=dict(atk=200, **{"def": 200}, spa=200, spd=200, spe=200 + i), moves=moves,
            baseAbility=ability, item=item, ability=ability, teraType=tera_type, terastallized=""))
    if active_moves is None:
        active_moves = [dict(move=m, id=m, pp=16, maxpp=16, disabled=False)
                        for m in TEAM[order[0]][2]]
    out = dict(side=dict(name="AlphaTest", id="p1", pokemon=pokemon), rqid=1, **extra)
    if not extra.get("forceSwitch"):
        active = dict(moves=active_moves)
        if tera:
            active["canTerastallize"] = TEAM[order[0]][5]
        if trapped:
            active["trapped"] = True
        out["active"] = [active]
    return out


START = """|player|p1|AlphaTest|169|
|player|p2|BetaTest|266|
|gen|9
|teamsize|p1|6
|teamsize|p2|6
|start
|switch|p1a: Greedent|Greedent, L86, F|347/347
|switch|p2a: Talonflame|Talonflame, L83, F|100/100
|turn|1"""

TURNS = ["""|
|move|p2a: Talonflame|Tera Blast|p1a: Greedent
|-damage|p1a: Greedent|301/347
|move|p1a: Greedent|Swords Dance|p1a: Greedent
|-boost|p1a: Greedent|atk|2
|
|upkeep
|turn|2""", """|
|move|p2a: Talonflame|Flare Blitz|p1a: Greedent
|-damage|p1a: Greedent|205/347
|-damage|p2a: Talonflame|88/100|[from] Recoil
|move|p1a: Greedent|Double-Edge|p2a: Talonflame
|-damage|p2a: Talonflame|0 fnt
|-status|p1a: Greedent|brn|[from] ability: Flame Body|[of] p2a: Talonflame
|faint|p2a: Talonflame
|-damage|p1a: Greedent|128/347 brn|[from] Recoil
|-enditem|p1a: Greedent|Sitrus Berry|[eat]
|-heal|p1a: Greedent|214/347 brn|[from] item: Sitrus Berry
|-heal|p1a: Greedent|329/347 brn|[from] ability: Cheek Pouch
|
|-damage|p1a: Greedent|308/347 brn|[from] brn
|upkeep""", """|
|switch|p2a: Swanna|Swanna, L89, F|100/100
|turn|3""", """|
|move|p2a: Swanna|Hydro Pump|p1a: Greedent
|-damage|p1a: Greedent|185/347 brn
|move|p1a: Greedent|Knock Off|p2a: Swanna
|-damage|p2a: Swanna|72/100
|-enditem|p2a: Swanna|Heavy-Duty Boots|[from] move: Knock Off|[of] p1a: Greedent
|
|-damage|p1a: Greedent|164/347 brn|[from] brn
|upkeep
|turn|4"""]


def play(battle: ShowdownBattle, *messages: str):
    """Each message as the battle room sends one."""
    for message in messages:
        for line in message.split("\n"):
            battle.receive(line)
        battle.end_update()


def view(battle: ShowdownBattle) -> dict:
    fs = jax.tree.map(lambda x: x[None], battle.fog_state())
    return jax.tree.map(lambda x: np.asarray(x)[0], jax.device_get(_view(ENV, fs, 0)))


@pytest.fixture(scope="module")
def battle():
    """The battle four turns in: the opponent's Talonflame has fainted, and
    Swanna's boots have been Knocked Off."""
    battle = ShowdownBattle("AlphaTest")
    play(battle, START)
    battle.update(request())
    play(battle, *TURNS)
    battle.update(request({0: "164/347 brn"}))
    return battle


def test_the_bot_is_side_zero_with_its_team_in_request_order(battle):
    assert battle.me == "p1"
    assert battle.names[0] == [name for name, *_ in TEAM]
    assert battle.names[1] == ["Talonflame", "Swanna"]


def test_the_log_and_requests_give_the_state(battle):
    st = battle.st
    assert st["hp"][0, 0] == 164 and st["maxhp"][0, 0] == 347 and st["status"][0, 0] == C.BRN
    assert st["boosts"][0, C.B_ATK] == 2
    assert st["item"][0, 0] == N.item_id("sitrusberry"), "the request is the word on its own side"
    assert st["hp"][1].tolist()[:2] == [0, 72] and st["maxhp"][1, 1] == 100
    assert int(st["active"][1]) == 1 and battle.turn == 4 and battle.completed_turns() == 3


def test_the_opponent_shows_only_what_it_revealed(battle):
    them = view(battle)["opponent"]["team"]
    talonflame, swanna = (N.species_id(s) for s in ("talonflame", "swanna"))
    assert them["species_ids"].tolist() == [talonflame, swanna] + [UNKNOWN] * 4
    assert them["level"].tolist() == [83, 89, 0, 0, 0, 0]
    assert them["item_ids"].tolist() == [UNKNOWN, 0] + [UNKNOWN] * 4, "Knocked Off"
    moves = [N.move_id(m) for m in ("terablast", "flareblitz")]
    assert them["move_ids"][0].tolist() == moves + [UNKNOWN] * 2
    assert them["move_ids"][1].tolist() == [N.move_id("hydropump")] + [UNKNOWN] * 3
    assert them["hp"].tolist() == [0, 72, 100, 100, 100, 100] and np.all(them["maxhp"] == 100)


def test_the_history_is_stamped_as_psjax_stamps_it(battle):
    h = view(battle)["history"]
    n = int(h["present"].sum())
    assert n == 9, "two leads, four moves, a switch-in and two more moves"
    # Each step's events take the turns completed after it: the leads 0, then
    # the moves of turns 1-3, with Swanna's replacement at the end of turn 2.
    assert h["turns_ago"][-n:].tolist() == [3, 3, 2, 2, 1, 1, 1, 0, 0]
    assert h["mine"][-n:].tolist() == [True, False, False, True, False, True, False, False, True]
    assert h["move_ids"][-1] == N.move_id("knockoff") and h["move_ids"][-3] == -1


def test_actions_map_to_showdown_choices(battle):
    options = battle.options()
    assert options[0] == "move 1" and options[3] == "move 4"
    assert options[C.ACTION_TERA_BASE + 2] == "move 3 terastallize"
    assert options[C.ACTION_SWITCH_BASE + 5] == "switch 6"
    assert C.ACTION_SWITCH_BASE not in options, "Greedent is already out"
    legal = view(battle)["legal_actions"]
    assert legal.shape == (C.NUM_ACTIONS,)


def test_a_switch_is_to_a_pokemon_wherever_the_request_lists_it():
    battle = ShowdownBattle("AlphaTest")
    play(battle, START)
    battle.update(request())
    # Entei came in, so Showdown now lists it first and Greedent where it was
    battle.update(request(order=[5, 1, 2, 3, 4, 0]))
    options = battle.options()
    assert int(battle.st["active"][0]) == 5
    assert options[C.ACTION_SWITCH_BASE + 0] == "switch 6"
    assert options[C.ACTION_SWITCH_BASE + 1] == "switch 2"


def test_locks_forced_switches_and_struggle():
    battle = ShowdownBattle("AlphaTest")
    play(battle, START)
    outrage = [dict(move="Earthquake", id="earthquake")]  # locked into its third move
    battle.update(request(active_moves=outrage, trapped=True, tera=False))
    assert battle.options() == {2: "move 1"}
    assert int(battle.st["locked_slot"][0]) == 2

    banded = [dict(move=m, id=m, pp=8, maxpp=8, disabled=m != "flareblitz") for m in TEAM[5][2]]
    battle.update(request(order=[5, 1, 2, 3, 4, 0], active_moves=banded))
    assert {a for a in battle.options() if a < C.ACTION_SWITCH_BASE} == {1, 5}

    struggling = [dict(move="Struggle", id="struggle", disabled=False)]
    battle.update(request(active_moves=struggling, trapped=True, tera=False))
    assert battle.options() == {0: "move 1"}

    fainted = request({0: "0 fnt", 3: "0 fnt"}, forceSwitch=[True])
    assert [c for c, *_ in legal_choices(fainted)] == ["switch 2", "switch 3", "switch 5", "switch 6"]
    battle.update(fainted)
    assert battle.fog_state().battle.phase == C.PHASE_SWITCH
    assert sorted(battle.options()) == [C.ACTION_SWITCH_BASE + s for s in (1, 2, 4, 5)]


def test_a_broken_illusion_was_never_the_disguise():
    battle = ShowdownBattle("AlphaTest")
    play(battle, START.replace("Talonflame|Talonflame, L83, F", "Corviknight|Corviknight, L84, M"),
         """|
|move|p1a: Greedent|Double-Edge|p2a: Corviknight
|-damage|p2a: Corviknight|60/100
|replace|p2a: Zoroark|Zoroark-Hisui, L84, M|60/100
|-end|p2a: Zoroark|Illusion
|upkeep
|turn|2""")
    battle.update(request())
    assert battle.names[1] == ["Zoroark"]
    them = view(battle)["opponent"]["team"]
    assert them["species_ids"][0] == N.species_id("zoroarkhisui")
    assert them["hp"][0] == 60 and np.all(them["species_ids"][1:] == UNKNOWN)
