"""GameNetwork inputs must show the opponent's public side and nothing more.

    python -m pytest model/test_game_inputs.py
"""
from __future__ import annotations

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from psjax import consts as C
from psjax.engine import step
from psjax.fog import FogOfWarEnv, FogState
from psjax.mechanics import species_stats

from architechture import GameNetwork
from game_inputs import game_inputs

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "simulator" / "tests"))
from scenario import build  # noqa: E402  (the simulator tests' battle builder)

ENV = FogOfWarEnv()
BATCH = 16


@pytest.fixture(scope="module")
def battles():
    """A batch of battles a few random turns in, so some moves, Pokemon and
    Tera types have been revealed and some have not."""
    key = jax.random.PRNGKey(0)
    fs = ENV.reset_batch(jax.random.split(key, BATCH))
    for i in range(6):
        keys = jax.random.split(jax.random.fold_in(key, i), BATCH)
        fs, _, _, _ = ENV.step_batch(fs, jax.vmap(ENV.sample_actions)(fs, keys))
    return fs


def _t(x) -> torch.Tensor:
    return torch.from_numpy(np.array(x))


def _same(a: dict, b: dict) -> bool:
    flat_a, flat_b = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    return all(torch.equal(x, y) for x, y in zip(flat_a, flat_b))


@pytest.mark.parametrize("player", [0, 1])
def test_the_network_runs_on_both_players_views(battles, player):
    inputs = game_inputs(ENV, battles, player)
    logits, win = GameNetwork()(**inputs)
    assert logits.shape == (BATCH, C.NUM_ACTIONS) and win.shape == (BATCH,)
    legal = _t(jax.vmap(ENV.legal_actions)(battles))[:, player]
    assert torch.equal(inputs["legal_actions"], legal)
    assert torch.all(logits.softmax(-1)[~legal] == 0)


def test_your_own_side_is_shown_in_full(battles):
    me = game_inputs(ENV, battles, 0)["me"]
    b = battles.battle
    for name, field in [("species_ids", b.species), ("item_ids", b.item),
                        ("ability_ids", b.ability), ("move_ids", b.moves),
                        ("tera_type_ids", b.tera_type), ("hp", b.hp),
                        ("level", b.level), ("stats", b.stats[..., 1:])]:
        assert torch.equal(me["team"][name], _t(field[:, 0]).long())
    assert torch.equal(me["choice_slot"], _t(b.choice_slot[:, 0]).long())


def test_the_opponent_shows_only_what_has_been_revealed(battles):
    them = game_inputs(ENV, battles, 0)["opponent"]["team"]
    b, revealed = battles.battle, _t(battles.revealed[:, 1])
    assert torch.all((them["species_ids"] >= 0) == revealed)
    # An ability only where the species has no other
    abilities = _t(ENV.data["species_abilities"][jnp.asarray(
        them["species_ids"].clamp(min=0).numpy())]).long()
    only = revealed & (abilities[..., 1] == 0) & (abilities[..., 2] == 0)
    assert torch.equal(them["ability_ids"], torch.where(only, abilities[..., 0], -1))
    assert torch.all((them["item_ids"] == 0) | (them["item_ids"] == -1))
    tera = _t(b.terastallized[:, 1])
    assert torch.all((them["tera_type_ids"] >= 0) == tera)
    seen = _t(battles.revealed_moves[:, 1])
    assert torch.all((them["move_ids"] >= 0) == (seen & (them["move_ids"] >= 0)))
    assert torch.all(them["maxhp"] == 100), "the opponent's HP is a percent"

    # Level as shown, stats as a player would work them out from it.
    assert torch.equal(them["level"], torch.where(revealed, _t(b.level[:, 1]).long(), 0))
    for i, j in torch.nonzero(revealed).tolist():
        expected = species_stats(ENV.data, b.species[i, 1, j], b.level[i, 1, j], jnp.int8(0))
        assert torch.equal(them["stats"][i, j], _t(expected[1:]).long())
    assert torch.all(them["stats"][~revealed] == 0)


def test_the_opponents_secrets_do_not_change_the_inputs(battles):
    """Rewrite everything about the opponent that a real player cannot see."""
    b = battles.battle
    unseen = ~battles.revealed[:, 1]
    unused = ~battles.revealed_moves[:, 1]
    key = jax.random.PRNGKey(1)
    secret = b._replace(
        species=b.species.at[:, 1].set(jnp.where(unseen, 1, b.species[:, 1])),
        ability=b.ability.at[:, 1].set(b.ability[:, 1] % 7 + 1),
        item=b.item.at[:, 1].set(jnp.where(b.item[:, 1] > 0, b.item[:, 1] % 5 + 1, 0)),
        tera_type=b.tera_type.at[:, 1].set(
            jnp.where(b.terastallized[:, 1], b.tera_type[:, 1], C.FAIRY)),
        moves=b.moves.at[:, 1].set(jnp.where(unused, 5, b.moves[:, 1])),
        # Sleep turns left are secret; a toxic counter is not.
        status_turns=b.status_turns.at[:, 1].set(jnp.where(
            b.status[:, 1] == C.TOX, b.status_turns[:, 1],
            jax.random.randint(key, b.status_turns[:, 1].shape, 1, 4).astype(jnp.int8))),
        level=b.level.at[:, 1].set(jnp.where(unseen, 50, b.level[:, 1])),
        stats=b.stats.at[:, 1].add(7),  # the true spread, not the standard one
        choice_slot=b.choice_slot.at[:, 1].set(2),
    )
    before = game_inputs(ENV, battles, 0)
    after = game_inputs(ENV, battles._replace(battle=secret), 0)
    assert _same(before["opponent"], after["opponent"])
    assert _same(before["field"], after["field"])
    assert _same(before["matchups"], after["matchups"])

    # A control: the same rewrite of a Pokemon they have seen does show.
    seen = b._replace(species=b.species.at[:, 1].set(jnp.where(unseen, b.species[:, 1], 1)))
    assert not _same(before["opponent"],
                     game_inputs(ENV, battles._replace(battle=seen), 0)["opponent"])


def test_illusion_shows_the_disguise(battles):
    b = battles.battle
    disguised = battles._replace(battle=b._replace(illusion=b.illusion.at[:, 1].set(5)))
    them = game_inputs(ENV, disguised, 0)["opponent"]["team"]
    active = _t(b.active[:, 1]).long()
    shown = them["species_ids"][torch.arange(BATCH), active]
    assert torch.equal(shown, _t(b.species[:, 1, 5]).long())


def test_counters_and_phase_are_seen_from_your_side(battles):
    b = battles.battle
    for player in (0, 1):
        inputs = game_inputs(ENV, battles, player)
        them = inputs["opponent"]
        assert torch.equal(inputs["field"]["force_switch"],
                           _t(b.force_switch[:, [player, 1 - player]]))
        assert torch.equal(inputs["field"]["phase"], _t(b.phase).long())
        assert torch.equal(them["last_move"], _t(b.last_move[:, 1 - player]).long())
        assert torch.all(them["choice_slot"] == -1), "a Choice lock gives the item away"
        toxic = _t(b.status[:, 1 - player] == C.TOX)
        assert torch.all(them["team"]["toxic_counter"][~toxic] == 0)


# --- damage calcs ----------------------------------------------------------------

CHOMP = {"species": "Garchomp", "ability": "Rough Skin",
         "moves": ["earthquake", "dragonclaw", "swordsdance", "bulletseed"]}


def _view(state, seen_slots=(0,)):
    """Player 0's inputs, with player 1's `seen_slots` revealed and no moves."""
    seen = jnp.zeros((2, 6), bool).at[0, 0].set(True).at[1, jnp.array(seen_slots)].set(True)
    fs = FogState(battle=state, revealed=seen, revealed_moves=jnp.zeros((2, 6, 4), bool))
    return game_inputs(ENV, jax.tree_util.tree_map(lambda x: x[None], fs), 0)


def test_damage_calcs_bracket_the_damage_the_engine_deals():
    """Every hit lands between the min and max roll, except crits above it."""
    state = build({"p1": CHOMP, "p2": {"species": "Hippowdon", "ability": "Sand Stream"}},
                  jax.random.PRNGKey(0))
    calcs = _view(state)["matchups"]["moves"][0]
    maxhp = int(state.maxhp[1, 0])
    run = jax.jit(step)
    for slot in (0, 1, 3):  # Earthquake, Dragon Claw, Bullet Seed (2-5 hits)
        low, high = float(calcs[slot, 0]), float(calcs[slot, 1])
        for i in range(24):
            after = run(state._replace(key=jax.random.PRNGKey(i)), jnp.array([slot, 0], jnp.int32))
            dealt = (maxhp - int(after.hp[1, 0])) / maxhp
            assert dealt >= low - 1e-6, (slot, dealt, low)
            assert dealt <= high + 1e-6 or dealt >= 1.4 * low, (slot, dealt, high)
    assert torch.all(calcs[2] == 0), "Swords Dance deals no damage"


def test_moves_are_calculated_against_the_active_and_three_revealed():
    team = [{"species": s, "ability": a} for s, a in [
        ("Blissey", "Natural Cure"), ("Skarmory", "Sturdy"), ("Rotom-Wash", "Levitate"),
        ("Hippowdon", "Sand Stream"), ("Heatran", "Flash Fire"), ("Gengar", "Cursed Body")]]
    state = build({"p1": CHOMP, "p2": team[0], "p2team": team}, jax.random.PRNGKey(0))
    state = state._replace(hp=state.hp.at[1, 4].set(0))  # Heatran has fainted
    quake = _view(state, seen_slots=(0, 2, 3, 4, 5))["matchups"]["moves"][0, 0].view(4, 2)
    # Blissey (active), then Rotom-Wash, Hippowdon and Gengar: Skarmory is
    # unseen and Heatran fainted.
    assert torch.all(quake[1] == 0), "Rotom-Wash's only ability is Levitate"
    assert torch.all(quake[3] > 1), "Ground is super effective on Gengar"
    assert torch.all(quake[[0, 2]] > 0)

    unseen = _view(state)["matchups"]
    assert torch.all(unseen["moves"][0, :, 2:] == -1), "no other Pokemon revealed"
    assert torch.all(unseen["switches"][0, :, 2:] == -1), "no opposing move revealed"
