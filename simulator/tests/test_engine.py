"""Engine-level tests: invariants over random battles, plus targeted mechanics.

The differential damage tests in `test_damage.py` pin the numbers; these pin the
*state machine* -- that battles terminate, that HP and boosts stay in range, and
that a handful of mechanics behave as the games do.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from psjax import consts as C
from psjax.data import load_data, names
from psjax.engine import step, turn_order
from psjax.env import BattleEnv
from psjax.mechanics import effective_speed
from psjax.state import empty_state
from psjax.teams import legal_action_mask, new_battle

DATA = load_data()
N = names()

# Compiled once for the whole module: `step` is a large program and re-jitting it
# per test would dominate the run time.
JIT_STEP = jax.jit(step)


def play(key, max_steps=400, trace=None):
    """Play a battle out with random legal actions, checking invariants each step.

    `trace`, if given, collects total HP per side each step so a caller can tell
    a stalemate from a battle that is stuck.
    """
    state = new_battle(key, DATA)
    jstep = JIT_STEP
    for _ in range(max_steps):
        if trace is not None:
            trace.append(tuple(int(jnp.sum(state.hp[p])) for p in range(C.NUM_PLAYERS)))
        if int(state.phase) == C.PHASE_END:
            break
        mask = legal_action_mask(DATA, state)
        assert bool(jnp.all(jnp.any(mask, axis=1))), "a player had no legal action"
        key, k = jax.random.split(key)
        actions = jnp.stack([
            jax.random.categorical(jax.random.fold_in(k, p),
                                   jnp.where(mask[p], 0.0, -1e9))
            for p in range(C.NUM_PLAYERS)])
        state = jstep(state, actions, DATA)
        check_invariants(state)
    return state


def check_invariants(state):
    assert bool(jnp.all(state.hp >= 0)), "negative HP"
    assert bool(jnp.all(state.hp <= state.maxhp)), "HP above maximum"
    assert bool(jnp.all(state.pp >= 0)), "negative PP"
    assert bool(jnp.all(jnp.abs(state.boosts) <= 6)), "boost outside -6..+6"
    assert bool(jnp.all((state.status >= 0) & (state.status < C.NUM_STATUS)))
    assert bool(jnp.all((state.active >= 0) & (state.active < C.TEAM_SIZE)))
    assert bool(jnp.all(state.side_conditions >= 0))
    assert bool(jnp.all(state.volatiles >= 0))


@pytest.mark.parametrize("seed", range(8))
def test_random_battle_finishes_or_provably_stalls(seed):
    """A battle either ends properly or is a genuine stalemate.

    Termination is not something this engine can promise, because it implements
    no Endless Battle Clause: two Pokemon that out-recover each other's damage
    will trade moves forever, and under uniformly random play that happens. Seed
    7 is Toxapex (Recover, and Poison-typed so Toxic cannot touch it) against
    Umbreon (Wish, Protect); both sit at full HP indefinitely.

    So the assertion is the falsifiable one: a battle that has not finished must
    have made no progress at all over its final stretch. That still catches a
    battle stuck in a phase loop, or one that hangs while HP is moving -- which
    is what this test was really guarding against.
    """
    trace = []
    state = play(jax.random.PRNGKey(seed), trace=trace)
    if int(state.phase) != C.PHASE_END:
        tail = trace[-100:]
        assert len(set(tail)) == 1, (
            f"battle did not finish by turn {int(state.turn)} and was still "
            f"changing: total HP over the last {len(tail)} steps was {sorted(set(tail))}")
        return

    assert int(state.winner) in (0, 1, 2)
    alive = [int(jnp.sum(state.hp[p] > 0)) for p in range(2)]
    if int(state.winner) in (0, 1):
        assert alive[1 - int(state.winner)] == 0, "declared a winner with the loser alive"
        assert alive[int(state.winner)] > 0


def test_most_random_battles_still_finish():
    """Stalemates must stay the exception, not become the rule."""
    finished = sum(
        int(play(jax.random.PRNGKey(seed)).phase) == C.PHASE_END
        for seed in range(8))
    assert finished >= 6, f"only {finished}/8 battles finished"


def test_a_finished_battle_absorbs_further_steps():
    env = BattleEnv(DATA)
    state = play(jax.random.PRNGKey(0))
    assert int(state.phase) == C.PHASE_END
    frozen, _, rewards, done = env.step(state, jnp.array([0, 0]))
    assert bool(done)
    assert bool(jnp.all(rewards == 0)), "a finished battle paid out twice"
    assert bool(jnp.all(frozen.hp == state.hp)), "a finished battle kept mutating"


def test_reward_is_zero_sum_and_paid_once():
    env = BattleEnv(DATA)
    state = env.reset(jax.random.PRNGKey(5))
    key = jax.random.PRNGKey(6)
    total = jnp.zeros(2)
    for _ in range(400):
        key, k = jax.random.split(key)
        state, _, rewards, done = env.step(state, env.sample_actions(state, k))
        total = total + rewards
        if bool(done):
            break
    assert float(jnp.sum(total)) == 0.0, "rewards were not zero-sum"
    assert float(jnp.max(jnp.abs(total))) in (0.0, 1.0), "reward paid more than once"


def test_observation_is_finite_and_symmetric_in_shape():
    env = BattleEnv(DATA)
    state = env.reset(jax.random.PRNGKey(1))
    obs = env.observe(state)
    assert obs.shape == (2, env.obs_dim)
    assert bool(jnp.all(jnp.isfinite(obs))), "observation contained NaN or inf"


def test_batched_reset_and_step_shapes():
    env = BattleEnv(DATA)
    states = env.reset_batch(jax.random.split(jax.random.PRNGKey(0), 4))
    assert states.hp.shape == (4, 2, C.TEAM_SIZE)
    actions = jnp.zeros((4, 2), jnp.int32)
    states, obs, rewards, done = env.step_batch(states, actions)
    assert obs.shape == (4, 2, env.obs_dim)
    assert rewards.shape == (4, 2)
    assert done.shape == (4,)


# --- targeted mechanics ------------------------------------------------------

def _one_v_one(species_a, moves_a, species_b, moves_b, **kw):
    """A minimal two-Pokemon battle for testing a single interaction."""
    state = empty_state(jax.random.PRNGKey(0))
    for side, (sp, mvs) in enumerate(((species_a, moves_a), (species_b, moves_b))):
        sid = N.species_id(sp)
        base = DATA["species_base_stats"][sid]
        from psjax.stats import compute_all_stats
        stats = compute_all_stats(base[None], jnp.array([100], jnp.int8))[0]
        move_ids = jnp.array([N.move_id(m) for m in mvs] + [-1] * (4 - len(mvs)),
                             jnp.int16)
        state = state._replace(
            species=state.species.at[side, 0].set(sid),
            level=state.level.at[side, 0].set(100),
            hp=state.hp.at[side, 0].set(stats[C.HP]),
            maxhp=state.maxhp.at[side, 0].set(stats[C.HP]),
            stats=state.stats.at[side, 0].set(stats),
            types=state.types.at[side, 0].set(DATA["species_types"][sid]),
            moves=state.moves.at[side, 0].set(move_ids),
            pp=state.pp.at[side, 0].set(jnp.full(4, 16, jnp.int8)),
            maxpp=state.maxpp.at[side, 0].set(jnp.full(4, 16, jnp.int8)),
        )
    return state._replace(**kw)


def test_switch_resets_boosts_but_keeps_hp():
    state = new_battle(jax.random.PRNGKey(2), DATA)
    state = state._replace(boosts=state.boosts.at[0].set(jnp.full(7, 3, jnp.int8)))
    hp_before = state.hp[0, 1]
    from psjax.engine import switch_to
    state = switch_to(DATA, state, 0, jnp.int32(1))
    assert int(state.active[0]) == 1
    assert bool(jnp.all(state.boosts[0] == 0)), "boosts survived a switch"
    # Hazards can shave HP on entry, but the Pokemon must not be at full reset.
    assert int(state.hp[0, 1]) <= int(hp_before)


def test_stealth_rock_scales_with_the_rock_matchup():
    """Stealth Rock does maxhp/8, scaled by the Rock type matchup.

    Talonflame (Fire/Flying) takes 4x, so half its HP; Skarmory (Steel/Flying) is
    neutral at 1/8 because Steel's resistance cancels Flying's weakness; Lucario
    (Fighting/Steel) resists twice for 1/32.
    """
    from psjax.engine import apply_entry_hazards
    for species, expected_fraction in (("Talonflame", 0.5), ("Skarmory", 0.125),
                                       ("Lucario", 0.03125)):
        state = _one_v_one(species, ["tackle"], "Blissey", ["tackle"])
        state = state._replace(
            side_conditions=state.side_conditions.at[0, C.SC_STEALTHROCK].set(1))
        maxhp = int(state.maxhp[0, 0])
        after = apply_entry_hazards(DATA, state, 0)
        lost = maxhp - int(after.hp[0, 0])
        assert abs(lost / maxhp - expected_fraction) < 0.01, \
            f"{species}: lost {lost}/{maxhp}"


def test_heavy_duty_boots_ignore_hazards():
    from psjax.engine import apply_entry_hazards
    state = _one_v_one("Talonflame", ["tackle"], "Blissey", ["tackle"])
    state = state._replace(
        side_conditions=state.side_conditions.at[0, C.SC_STEALTHROCK].set(1),
        item=state.item.at[0, 0].set(N.item_id("Heavy-Duty Boots")))
    after = apply_entry_hazards(DATA, state, 0)
    assert int(after.hp[0, 0]) == int(state.hp[0, 0])


def test_priority_beats_speed():
    """Slow Extreme Speed still moves before fast Tackle."""
    state = _one_v_one("Snorlax", ["extremespeed"], "Dragapult", ["tackle"])
    assert int(effective_speed(state, 1)) > int(effective_speed(state, 0))
    first = turn_order(DATA, state, jnp.array([0, 0]), jax.random.PRNGKey(0))
    assert int(first) == 0, "priority did not override Speed"


def test_trick_room_inverts_speed_order():
    state = _one_v_one("Snorlax", ["tackle"], "Dragapult", ["tackle"])
    normal = turn_order(DATA, state, jnp.array([0, 0]), jax.random.PRNGKey(0))
    assert int(normal) == 1, "the faster Pokemon did not move first"
    tricked = turn_order(DATA, state._replace(trick_room=jnp.int8(5)),
                         jnp.array([0, 0]), jax.random.PRNGKey(0))
    assert int(tricked) == 0, "Trick Room did not invert the Speed order"


def test_switching_outranks_any_move():
    state = _one_v_one("Snorlax", ["tackle"], "Dragapult", ["extremespeed"])
    switch = jnp.array([C.ACTION_SWITCH_BASE + 1, 0])
    first = turn_order(DATA, state, switch, jax.random.PRNGKey(0))
    assert int(first) == 0, "a switch did not resolve before a priority move"


def test_paralysis_halves_speed():
    state = _one_v_one("Dragapult", ["tackle"], "Blissey", ["tackle"])
    fast = int(effective_speed(state, 0))
    slowed = state._replace(status=state.status.at[0, 0].set(jnp.int8(C.PAR)))
    assert int(effective_speed(slowed, 0)) == fast // 2


def test_choice_item_locks_the_move():
    state = new_battle(jax.random.PRNGKey(7), DATA)
    lead = int(state.active[0])
    state = state._replace(choice_slot=state.choice_slot.at[0].set(jnp.int8(2)),
                           item=state.item.at[0, lead].set(N.item_id("Choice Scarf")))
    mask = legal_action_mask(DATA, state)
    usable = [i for i in range(4) if bool(mask[0, i])]
    assert usable == [2], f"Choice lock allowed {usable}"
    # Switching out is still allowed while Choice-locked.
    assert bool(jnp.any(mask[0, C.ACTION_SWITCH_BASE:]))
    # The lock goes with the item: once it is knocked off, every move is back.
    state = state._replace(item=state.item.at[0, lead].set(0))
    mask = legal_action_mask(DATA, state)
    assert [i for i in range(4) if bool(mask[0, i])] == [0, 1, 2, 3]


def test_forced_switch_phase_only_allows_switches():
    state = new_battle(jax.random.PRNGKey(8), DATA)
    state = state._replace(force_switch=state.force_switch.at[0].set(True))
    mask = legal_action_mask(DATA, state)
    assert not bool(jnp.any(mask[0, :C.ACTION_SWITCH_BASE])), "moves offered while forced to switch"
    assert bool(jnp.any(mask[0, C.ACTION_SWITCH_BASE:]))


def test_no_replacement_requested_with_an_empty_bench():
    """A self-switch with nothing left to bring in must not ask for a switch.

    Regression: `has_bench` counted the active Pokemon, so a side whose active
    survived a U-turn with a wiped-out bench was still put into PHASE_SWITCH.
    The only "legal" action then decoded to team slot -8, which JAX silently
    clamped, corrupting `active`.
    """
    state = new_battle(jax.random.PRNGKey(1), DATA)
    # Wipe player 1's bench, leaving only the active alive, and ask to switch.
    hp = state.hp.at[1].set(jnp.zeros(C.TEAM_SIZE, jnp.int16))
    hp = hp.at[1, int(state.active[1])].set(state.maxhp[1, int(state.active[1])])
    state = state._replace(hp=hp, force_switch=state.force_switch.at[1].set(True))

    mask = legal_action_mask(DATA, state)
    assert bool(jnp.any(mask[1])), "player 1 was left with no legal action"
    for action in jnp.nonzero(mask[1])[0]:
        assert int(action) >= C.ACTION_SWITCH_BASE, \
            "a forced switch offered a non-switch action"

    # Whatever the engine does next, the active slot must stay in range.
    stepped = step(state, jnp.array([0, int(jnp.argmax(mask[1]))]), DATA)
    assert bool(jnp.all((stepped.active >= 0) & (stepped.active < C.TEAM_SIZE))), \
        f"active slot out of range: {stepped.active}"


def test_effect_handlers_only_write_declared_fields():
    """`run_effect` returns a projection, so a handler writing elsewhere is lost."""
    from psjax.moves import EFFECT_FNS, EFFECT_WRITES
    state = new_battle(jax.random.PRNGKey(0), DATA)
    stray = []
    for name, fn in EFFECT_FNS.items():
        out = fn(DATA, state, 0, 1, jnp.uint32(0))
        for field in state._fields:
            if field in EFFECT_WRITES or field == "key":
                continue
            before, after = getattr(state, field), getattr(out, field)
            if before.shape != after.shape or not bool(jnp.all(before == after)):
                stray.append((name, field))
    assert not stray, f"handlers wrote fields outside EFFECT_WRITES: {stray}"


# --- switching out mid-turn --------------------------------------------------

def _two_v_one(p1_species, p1_moves, bench_species, p2_species, p2_moves):
    """A 2-Pokemon team against a 1-Pokemon team, for switch tests."""
    from psjax.stats import compute_all_stats
    state = empty_state(jax.random.PRNGKey(0))
    team = [(0, 0, p1_species, p1_moves), (0, 1, bench_species, ["splash"]),
            (1, 0, p2_species, p2_moves)]
    for side, slot, species, moves in team:
        sid = N.species_id(species)
        stats = compute_all_stats(DATA["species_base_stats"][sid][None],
                                  jnp.array([100], jnp.int8))[0]
        ids = [N.move_id(m) for m in moves] + [-1] * (4 - len(moves))
        state = state._replace(
            species=state.species.at[side, slot].set(sid),
            level=state.level.at[side, slot].set(100),
            hp=state.hp.at[side, slot].set(stats[C.HP]),
            maxhp=state.maxhp.at[side, slot].set(stats[C.HP]),
            stats=state.stats.at[side, slot].set(stats),
            types=state.types.at[side, slot].set(DATA["species_types"][sid]),
            moves=state.moves.at[side, slot].set(jnp.asarray(ids, jnp.int16)),
            pp=state.pp.at[side, slot].set(jnp.full(4, 16, jnp.int8)),
            maxpp=state.maxpp.at[side, slot].set(jnp.full(4, 16, jnp.int8)),
        )
    return state


def test_self_switch_resolves_before_the_opponent_moves():
    """U-turn's replacement arrives first, so the opponent's move hits it.

    Regression: self-switches used to be deferred to the end of the turn, which
    meant the opponent attacked the Pokemon that had just left.
    """
    # Weavile outspeeds Snorlax, so the U-turn resolves first.
    state = _two_v_one("Weavile", ["uturn"], "Blissey", "Snorlax", ["bodyslam"])
    weavile_hp = int(state.hp[0, 0])
    blissey_hp = int(state.hp[0, 1])

    after_uturn = JIT_STEP(state, jnp.array([0, 0]), DATA)
    assert int(after_uturn.phase) == C.PHASE_SWITCH, "U-turn did not ask for a switch"
    assert bool(after_uturn.force_switch[0])
    assert int(after_uturn.pending_side) == 1, "the opponent's move was not held over"

    resumed = JIT_STEP(after_uturn, jnp.array([C.ACTION_SWITCH_BASE + 1, 0]), DATA)
    assert int(resumed.active[0]) == 1, "the replacement did not come in"
    assert int(resumed.hp[0, 0]) == weavile_hp, \
        "Body Slam hit the Pokemon that used U-turn instead of its replacement"
    assert int(resumed.hp[0, 1]) < blissey_hp, "the replacement was never attacked"
    assert int(resumed.phase) == C.PHASE_MOVE, "the turn did not finish"


def test_self_switch_with_an_empty_bench_does_not_suspend():
    """U-turn with nothing to switch to just carries on."""
    state = _two_v_one("Weavile", ["uturn"], "Blissey", "Snorlax", ["bodyslam"])
    state = state._replace(hp=state.hp.at[0, 1].set(jnp.int16(0)))
    after = JIT_STEP(state, jnp.array([0, 0]), DATA)
    assert int(after.phase) == C.PHASE_MOVE, "suspended with no replacement available"
    assert int(after.pending_side) == -1
    assert int(after.hp[0, 0]) < int(state.hp[0, 0]), \
        "the opponent's move never landed"


def test_phazing_brings_in_a_random_replacement_immediately():
    """Whirlwind drags someone in during the turn, with no decision point."""
    from psjax.engine import apply_entry_hazards  # noqa: F401
    picked = set()
    for seed in range(8):
        state = _two_v_one("Snorlax", ["splash"], "Clefable",
                           "Corviknight", ["whirlwind"])
        # Hazards on the phazed side prove the arrival really switched in.
        state = state._replace(
            side_conditions=state.side_conditions.at[0, C.SC_STEALTHROCK].set(1),
            key=jax.random.PRNGKey(seed))
        after = JIT_STEP(state, jnp.array([0, 0]), DATA)
        assert int(after.phase) != C.PHASE_SWITCH, \
            "phazing asked for a choice; it should pick at random"
        picked.add(int(after.active[0]))
        assert int(after.active[0]) == 1, "nobody was dragged in"
        assert int(after.hp[0, 1]) < int(state.hp[0, 1]), \
            "the arrival did not take entry hazards"
    assert picked == {1}, f"only one bench slot was available, got {picked}"


def test_phazing_a_side_with_no_bench_does_nothing():
    state = _two_v_one("Snorlax", ["splash"], "Clefable", "Corviknight", ["whirlwind"])
    state = state._replace(hp=state.hp.at[0, 1].set(jnp.int16(0)))
    after = JIT_STEP(state, jnp.array([0, 0]), DATA)
    assert int(after.active[0]) == 0
    assert int(after.phase) == C.PHASE_MOVE
