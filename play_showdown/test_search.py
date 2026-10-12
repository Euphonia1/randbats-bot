"""The search must play out battles that agree with what the bot has seen, and
choose among the options Showdown gave it.

    python -m pytest play_showdown/test_search.py

The battles are test_play's, from real Showdown output; the network is an
untrained one, as these check the search's workings, not its judgement.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

from psjax import consts as C
from psjax.fog import FogOfWarEnv

from test_play import START, TURNS, play, request  # noqa: E402
from play import N, ShowdownBattle  # noqa: E402
import search as S  # noqa: E402
from architechture import GameNetwork  # noqa: E402
from game_inputs import _view  # noqa: E402

ENV = FogOfWarEnv()
SAMPLER = S.SetSampler()


@pytest.fixture(scope="module")
def battle():
    """Four turns in: the opponent's Talonflame (Tera Blast, Flare Blitz, Flame
    Body) has fainted, and Swanna, at 72%, has had its boots Knocked Off."""
    battle = ShowdownBattle("AlphaTest")
    play(battle, START)
    battle.update(request())
    play(battle, *TURNS)
    battle.update(request({0: "164/347 brn"}))
    return battle


@pytest.fixture(scope="module")
def search():
    return S.Search(GameNetwork().requires_grad_(False), "cpu", bf16=False, env=ENV,
                    sampler=SAMPLER, determinizations=2, batch=4, seed=0)


def determinized(battle, facts=None, n=20, seed=0):
    fs = battle.fog_state()
    observed = jax.device_get(fs.battle._asdict())
    rng = np.random.default_rng(seed)
    facts = facts or S.Facts.of(battle)
    return fs, [S.determinize(observed, np.asarray(fs.revealed)[1], facts, SAMPLER, rng)
                for _ in range(n)]


def test_the_log_shows_items_abilities_and_who_has_acted(battle):
    assert battle.items[1, 1] == N.item_id("heavydutyboots"), "Knocked Off"
    assert battle.abilities[1, 0] == N.ability_id("flamebody"), "[of] the burn's source"
    assert battle.items[0, 0] == N.item_id("sitrusberry")
    assert battle.abilities[0, 0] == N.ability_id("cheekpouch")
    assert battle.acted == [False, False], "a new turn"


def test_determinized_battles_keep_what_was_seen(battle):
    _, battles = determinized(battle)
    talonflame, swanna = N.species_id("talonflame"), N.species_id("swanna")
    for st in battles:
        assert st["species"][1, 0] == talonflame and st["hp"][1, 0] == 0
        assert st["moves"][1, 0, :2].tolist() == [N.move_id("terablast"), N.move_id("flareblitz")]
        assert st["ability"][1, 0] == N.ability_id("flamebody")
        assert st["species"][1, 1] == swanna and st["item"][1, 1] == 0
        assert st["moves"][1, 1, 0] == N.move_id("hydropump")
        assert np.ceil(100 * st["hp"][1, 1] / st["maxhp"][1, 1]) == 72
        assert np.all(st["moves"][1, :2] >= 0) and np.all(st["pp"][1, :2] > 0)
        # The four unseen ones: whole, healthy, and no species twice
        assert np.all(st["hp"][1, 2:] == st["maxhp"][1, 2:]) and np.all(st["maxhp"][1, 2:] > 1)
        assert len(set(S.SPECIES_NUM[st["species"][1]].tolist())) == C.TEAM_SIZE
        assert np.all(st["item"][1, 2:] > 0) and np.all(st["stats"][1] > 1)
        for k in ("species", "hp", "maxhp", "stats", "moves", "item", "ability"):
            assert np.array_equal(st[k][0], battle.st[k][0]), f"the bot's own {k}"
    assert len({tuple(st["species"][1]) for st in battles}) > 1, "drawn afresh each time"


def test_the_bot_sees_a_determinized_battle_as_it_sees_the_battle(battle):
    fs, battles = determinized(battle, n=4)
    one = lambda x: x[None]
    seen = jax.device_get(_view(ENV, jax.tree.map(one, fs), 0))
    drawn = jax.device_get(_view(ENV, S.batch_of(battles, fs), 0))
    for path, x in jax.tree_util.tree_leaves_with_path(seen):
        y = drawn
        for key in path:
            y = y[key.key]
        assert np.allclose(np.broadcast_to(x, y.shape), y), jax.tree_util.keystr(path)


def test_sets_fit_the_moves_seen():
    moves = np.array([N.move_id("terablast"), N.move_id("flareblitz"), -1, -1])
    rows = SAMPLER.fitting(N.species_id("talonflame"), moves, -1, -1, -1)
    assert len(rows) and np.all(np.isin(moves[:2], SAMPLER.moves[rows]).all())
    for row in rows:
        assert set(moves[:2]) <= set(SAMPLER.moves[row].tolist())


def test_sleep_lasts_what_is_left_of_it(battle):
    st = {k: v.copy() for k, v in battle.st.items()}
    st["status"][0, 1], st["sleep_attempts"][0, 1] = C.SLP, 2  # slept two turns of 1-3
    st["status"][0, 2], st["sleep_attempts"][0, 2], st["rest_sleep"][0, 2] = C.SLP, 1, True
    fs = battle.fog_state()
    facts = S.Facts.of(battle)
    rng = np.random.default_rng(0)
    left = [S.determinize(st, np.asarray(fs.revealed)[1], facts, SAMPLER, rng)["status_turns"][0, 1:3]
            for _ in range(50)]
    assert {int(x[0]) for x in left} == {1, 2}, "slept two: one turn more, or two"
    assert {int(x[1]) for x in left} == {2}, "Rest's two turns, one slept"


def test_protect_gets_likelier_to_fail_as_psjax_counts_it():
    battle = ShowdownBattle("AlphaTest")
    play(battle, START)
    battle.update(request())
    protect = ("|\n|move|p1a: Greedent|Protect|p1a: Greedent\n{}"
               "|move|p2a: Talonflame|Brave Bird|p1a: Greedent\n|\n|upkeep\n|turn|{}")
    streaks = []
    for turn, outcome in enumerate(["|-singleturn|p1a: Greedent|Protect\n", "|-fail|p1a: Greedent\n",
                                    "|-singleturn|p1a: Greedent|Protect\n",
                                    "|-singleturn|p1a: Greedent|move: Endure\n"], start=2):
        play(battle, protect.format(outcome, turn))
        streaks.append(int(battle.st["protect_streak"][0]))
    play(battle, "|\n|move|p1a: Greedent|Knock Off|p2a: Talonflame\n|\n|upkeep\n|turn|6")
    assert streaks + [int(battle.st["protect_streak"][0])] == [1, 0, 1, 2, 0]


VESPIQUEN_LEAD = START.replace("|switch|p1a: Greedent|Greedent, L86, F|347/347",
                               "|switch|p1a: Vespiquen|Vespiquen, L99, F|299/299")
VESPIQUEN_FIRST = [4, 0, 1, 2, 3, 5]


def u_turned(before: str = "") -> ShowdownBattle:
    """Vespiquen has U-turned on turn 1, after `before`, and is to be replaced."""
    battle = ShowdownBattle("AlphaTest")
    play(battle, VESPIQUEN_LEAD)
    battle.update(request(order=VESPIQUEN_FIRST))
    play(battle, "|\n" + before + "|move|p1a: Vespiquen|U-turn|p2a: Talonflame\n"
                 "|-damage|p2a: Talonflame|80/100")
    battle.update(request(order=VESPIQUEN_FIRST, forceSwitch=[True]))
    return battle


def test_a_u_turn_before_their_move_leaves_it_to_land_on_the_replacement(search):
    battle = u_turned()
    facts = S.Facts.of(battle)
    assert facts.mid_turn and facts.their_move_to_come
    _, battles = determinized(battle, n=2)
    assert all(st["pending_side"] == 1 for st in battles)
    roots = search._roots(S.batch_of(battles * 2, battle.fog_state()),
                          np.array(sorted(battle.options())), facts)
    for root in roots:
        assert np.all(root.actions[0] >= C.ACTION_SWITCH_BASE), "the bot's switches"
        assert len(root.actions[1]) and np.all(root.actions[1] < C.ACTION_TERA_BASE), "their moves"
    result = search.run(battle.fog_state(), battle.options(), facts, seconds=0, simulations=8)
    assert set(result.actions.tolist()) == set(battle.options())


def test_a_u_turn_after_their_move_only_leaves_the_residuals():
    battle = u_turned("|move|p2a: Talonflame|Brave Bird|p1a: Vespiquen\n"
                      "|-damage|p1a: Vespiquen|200/299\n")
    facts = S.Facts.of(battle)
    assert facts.mid_turn and not facts.their_move_to_come
    _, battles = determinized(battle, n=1)
    assert battles[0]["pending_side"] == 0 and battles[0]["pending_action"] == -1


def test_a_replacement_for_a_faint_comes_at_the_end_of_the_turn():
    battle = ShowdownBattle("AlphaTest")
    play(battle, START)
    battle.update(request())
    play(battle, *TURNS)
    battle.update(request({0: "164/347 brn"}))
    assert not S.Facts.of(battle).mid_turn
    play(battle, "|\n|move|p2a: Swanna|Hydro Pump|p1a: Greedent\n|-damage|p1a: Greedent|0 fnt\n"
                 "|faint|p1a: Greedent\n|\n|upkeep")
    battle.update(request({0: "0 fnt"}, forceSwitch=[True]))
    facts = S.Facts.of(battle)
    assert not facts.mid_turn and not facts.their_move_to_come
    _, battles = determinized(battle, n=1)
    assert battles[0]["pending_side"] == -1


def test_the_search_chooses_among_the_options_without_compiling_again(battle, search):
    # Compiled on a battle psjax made, as SearchPolicy warms up...
    fs = ENV.reset(jax.random.PRNGKey(1))
    options = {int(a): "" for a in np.flatnonzero(np.asarray(ENV.legal_actions(fs))[0])}
    facts = S.Facts(items=np.full(6, -1), abilities=np.full(6, -1))
    search.run(fs, options, facts, seconds=0, simulations=4)
    compiled = [f._cache_size() for f in (S._advance, S._views, S._inspect, S._restart)]
    # ...then searching what the bot rebuilt from Showdown compiles nothing new
    options = battle.options()
    result = search.run(battle.fog_state(), options, S.Facts.of(battle), seconds=0, simulations=12)
    assert [f._cache_size() for f in (S._advance, S._views, S._inspect, S._restart)] == compiled
    assert result.actions.tolist() == sorted(options)
    assert result.simulations == 12 and result.visits.sum() == 12
    assert np.isclose(result.prior.sum(), 1) and np.all((0 <= result.wins) & (result.wins <= 1))
