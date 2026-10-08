"""Replay Showdown-recorded Random Battles in psjax and report where they part.

`tools/showdown_lockstep.js` plays complete Gen 9 Random Battles with every
random word pinned to a constant `w` and records the teams, each decision and
the state after it. This builds the same teams through psjax's own
`new_battle`, pins psjax's random words to the same `w`, feeds it the same
actions, and compares the two states after every decision. A battle that
matches to the end agrees with Showdown on every roll-free outcome under that
`w`. Every decision where they differ is reported with Showdown's log. When
only values differ (HP, boosts, a counter), psjax is resynced to Showdown's
state and the battle goes on, so one bug does not hide the next; a `FATAL`
difference -- a different Pokemon out, a faint in one engine only -- stops it.

    node tools/showdown_lockstep.js 50 lockstep.json
    python tools/lockstep.py lockstep.json [report.json]

Pinning psjax takes four patches, applied only in this process:

* `random_words` returns `w` everywhere.
* `below(w, n)` becomes Showdown's `floor(w * n / 2**32)`. psjax uses
  `w % n`, which has the same distribution but maps a given word elsewhere.
  (`uniform` already reads `w / 2**32`, and the recorder's words have no bits
  below the 24 it reads.)
* A Speed tie always goes to p1. Showdown queues p1's action first and sorts
  the queue an even number of times before the first move, each sort
  shuffling the tied pair with the same constant word -- so the swaps cancel
  and p1 leads whatever `w` is (checked directly against Showdown).
* Phazing drags in the eligible Pokemon at `floor(w * n / 2**32)` in party
  order, as Showdown's `sample` over `possibleSwitches` does.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

from psjax import consts as C                       # noqa: E402
from psjax import engine as PE                       # noqa: E402
from psjax import mechanics as PMC                   # noqa: E402
from psjax import moves as PM                        # noqa: E402
from psjax import teams as PT                        # noqa: E402
from psjax.data import load_data, names              # noqa: E402
from psjax.mechanics import active_types, effective_speed   # noqa: E402
from psjax.state import select_state                 # noqa: E402
from psjax.stats import compute_all_stats            # noqa: E402
from scenario import STATUS_TO_PS, TERRAIN_TO_PS, WEATHER_TO_PS   # noqa: E402

DATA = load_data()
N = names()
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
BOOST_KEYS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
VOL_NAMES = {v: k for k, v in C.VOLATILE_IDX.items()}
SC_NAMES = {v: k for k, v in C.SIDE_CONDITION_IDX.items()}
SLOT_CONDITIONS = {"futuremove", "wish", "healingwish", "lunardance"}
#: Differences that are tallied but do not end the comparison: state psjax
#: does not keep, whose consequences are compared wherever they surface.
SOFT = {"volatile:lockedmove"}
#: Volatiles psjax keeps that Showdown keeps somewhere else, so they are never
#: in its volatile list: Slow Start's count lives in the ability's state.
NOT_VOLATILE_IN_SHOWDOWN = {"slowstart"}
PSEUDO_WEATHER = {"trickroom", "gravity"}

# --- pinning psjax's randomness to Showdown's constant word -------------------

_W = [None]       # the traced word, set by whichever jitted function is tracing


def _const_words(key, shape):
    return jnp.broadcast_to(_W[0], shape).astype(jnp.uint32)


def _below(word, n):
    return (((word >> 8) * jnp.uint32(n)) >> 24).astype(jnp.int32)


def _turn_order(data, state, actions, key):
    p0 = PE.action_priority(data, state, 0, actions[0])
    p1 = PE.action_priority(data, state, 1, actions[1])
    s0, s1 = effective_speed(state, 0), effective_speed(state, 1)
    faster0 = jnp.where(state.trick_room > 0, s0 < s1, s0 > s1)
    first = jnp.where(p0 != p1, p0 > p1, jnp.where(s0 == s1, True, faster0))
    return jnp.where(first, 0, 1).astype(jnp.int32)


def _resolve_phazing(data, state, key):
    def one_side(side, st):
        slots = jnp.arange(C.TEAM_SIZE)
        eligible = (st.hp[side] > 0) & (slots != st.active[side].astype(jnp.int32))
        count = jnp.sum(eligible).astype(jnp.uint32)
        k = ((_W[0] >> 8) * count) >> 24
        pos = st.party_pos[side]
        rank = jnp.sum(eligible[None, :] & (pos[None, :] < pos[:, None]), axis=1)
        slot = jnp.argmax(eligible & (rank == k.astype(rank.dtype)))
        return select_state(st.phazed[side] & jnp.any(eligible),
                            PE.switch_to(data, st, side, slot), st)
    state = jax.lax.fori_loop(0, C.NUM_PLAYERS, one_side, state)
    return state._replace(phazed=jnp.zeros((C.NUM_PLAYERS,), bool))


PE.random_words = _const_words
PE.below = PM.below = PMC.below = _below
PE.turn_order = _turn_order
PE.resolve_phazing = _resolve_phazing


def _step(state, actions, w):
    _W[0] = w
    return PE.step(state, actions, DATA)


def _init(t0, t1, gender, w):
    _W[0] = w
    teams = iter([t0, t1])
    original = PT.random_team
    PT.random_team = lambda data, key, pool=None: next(teams)
    try:
        state = PT.new_battle(jax.random.PRNGKey(0), DATA, pool=False)
    finally:
        PT.random_team = original
    return state._replace(gender=gender)


def _views(state):
    """What the snapshot needs that is not a plain field."""
    return (jnp.stack([active_types(state, 0), active_types(state, 1)]),
            PT.legal_action_mask(DATA, state))


STEP = jax.jit(jax.vmap(_step))
INIT = jax.jit(jax.vmap(_init))
VIEWS = jax.jit(jax.vmap(_views))


# --- teams --------------------------------------------------------------------

def species_key(species, base):
    return species if species in N.species else base


def build_teams(records):
    """Team arrays for `new_battle`, `[B, 2, 6, ...]`, plus anything psjax lacks."""
    B = len(records)
    shape = (B, 2, C.TEAM_SIZE)
    species = np.zeros(shape, np.int32)
    level = np.zeros(shape, np.int32)
    item = np.zeros(shape, np.int32)
    ability = np.zeros(shape, np.int32)
    tera = np.zeros(shape, np.int32)
    gender = np.zeros(shape, np.int32)
    moves = np.full(shape + (4,), -1, np.int32)
    ivs = np.full(shape + (6,), 31, np.int32)
    evs = np.full(shape + (6,), 85, np.int32)
    sd_stats = np.zeros(shape + (6,), np.int32)
    sd_pp = np.zeros(shape + (4,), np.int32)
    missing = collections.Counter()
    for b, rec in enumerate(records):
        for p, team in enumerate(rec["teams"]):
            for s, mon in enumerate(team):
                species[b, p, s] = N.species_id(species_key(mon["species"], mon["speciesBase"]))
                level[b, p, s] = mon["level"]
                if mon["item"] and mon["item"] not in N.items:
                    missing[f"item:{mon['item']}"] += 1
                item[b, p, s] = N.item_id(mon["item"]) if mon["item"] in N.items else 0
                if mon["ability"] not in N.abilities:
                    missing[f"ability:{mon['ability']}"] += 1
                ability[b, p, s] = N.ability_id(mon["ability"]) if mon["ability"] in N.abilities else 0
                tera[b, p, s] = N.type_id(mon["tera"])
                gender[b, p, s] = {"M": C.GENDER_M, "F": C.GENDER_F}.get(mon["gender"], C.GENDER_NONE)
                for m, mv in enumerate(mon["moves"][:4]):
                    if mv in N.moves:
                        moves[b, p, s, m] = N.move_id(mv)
                    else:
                        missing[f"move:{mv}"] += 1
                    sd_pp[b, p, s, m] = mon["pp"][m]
                for i, k in enumerate(STAT_KEYS):
                    ivs[b, p, s, i] = mon["ivs"][k]
                    evs[b, p, s, i] = mon["evs"][k]
                    sd_stats[b, p, s, i] = mon["stats"][k]

    base = np.asarray(DATA["species_base_stats"])[species]
    stats = np.asarray(compute_all_stats(jnp.asarray(base), jnp.asarray(level),
                                         jnp.asarray(ivs), jnp.asarray(evs)))
    move_pp = np.asarray(DATA["move_pp"])
    pp = np.where(moves >= 0, (move_pp[np.maximum(moves, 0)].astype(np.int32) * 8) // 5, 0)
    types = np.asarray(DATA["species_types"])[species]

    # Derived the way psjax derives them; Showdown's numbers are the check.
    init_diffs = collections.defaultdict(list)
    for b, p, s in zip(*np.nonzero(np.any(stats != sd_stats, axis=-1))):
        mon = records[b]["teams"][p][s]
        # Showdown's recorded stats are a lead Minior's Meteor forme, which
        # Shields Down has already taken by the time teams are recorded; the
        # forme change is compared through `species` and the turns that follow.
        if mon["species"].startswith("minior") or mon["species"] == "ditto":
            # (A lead Ditto has likewise already transformed with Imposter.)
            continue
        init_diffs[b].append(("stats", f"p{p + 1}[{s}] {mon['species']}",
                              sd_stats[b, p, s].tolist(), stats[b, p, s].tolist()))
    for b, p, s in zip(*np.nonzero(np.any((pp != sd_pp) & (moves >= 0), axis=-1))):
        mon = records[b]["teams"][p][s]
        init_diffs[b].append(("maxpp", f"p{p + 1}[{s}] {mon['species']}",
                              sd_pp[b, p, s].tolist(), pp[b, p, s].tolist()))

    team = dict(species=species, level=level, stats=stats, types=types, moves=moves,
                pp=pp, item=item, ability=ability, tera_type=tera)
    side = lambda p: {k: jnp.asarray(v[:, p]) for k, v in team.items()}
    return side(0), side(1), jnp.asarray(gender, jnp.int8), missing, init_diffs


# --- snapshots ----------------------------------------------------------------

def ps_snapshot(st, b, types, sd_hint):
    """psjax state `b` of a batch, shaped like the recorder's Showdown snapshot."""
    def side(p):
        a = int(st.active[b, p])
        out = {
            "active": a,
            "hp": [int(x) for x in st.hp[b, p]],
            "status": [STATUS_TO_PS[int(x)] for x in st.status[b, p]],
            "sleepTurns": [int(t) if int(s) == C.SLP else 0
                           for s, t in zip(st.status[b, p], st.status_turns[b, p])],
            # psjax counts Toxic from 1 (the next tick's sixteenths); Showdown's
            # stage counts ticks taken, so between turns psjax = stage + 1.
            "toxicStage": [int(t) - 1 if int(s) == C.TOX else 0
                           for s, t in zip(st.status[b, p], st.status_turns[b, p])],
            "item": [N.items_by_id.get(int(i), "") if int(i) else "" for i in st.item[b, p]],
            "ability": [N.abilities_by_id.get(int(i), "") if int(i) else ""
                        for i in st.ability[b, p]],
            "species": [N.species_by_id[int(x)] for x in st.species[b, p]],
            "tera": [bool(x) for x in st.terastallized[b, p]],
            "pp": [[int(x) for x in row] for row in st.pp[b, p]],
            "sideConditions": sorted(SC_NAMES[i] for i in range(C.NUM_SIDE_CONDITIONS)
                                     if int(st.side_conditions[b, p, i]) > 0),
            "sideConditionCounts": {SC_NAMES[i]: int(st.side_conditions[b, p, i])
                                    for i in range(C.NUM_SIDE_CONDITIONS)
                                    if int(st.side_conditions[b, p, i]) > 0},
            "spikeLayers": int(st.side_conditions[b, p, C.SC_SPIKES]),
            "toxicSpikeLayers": int(st.side_conditions[b, p, C.SC_TOXICSPIKES]),
            "slotConditions": sorted(
                (["futuremove"] if int(st.future_turns[b, p]) > 0 else []) +
                (["wish"] if int(st.wish_turns[b, p]) > 0 else []) +
                ({1: ["healingwish"], 2: ["lunardance"]}.get(int(st.healing_wish[b, p]), []))),
            "types": [N.types_by_id[int(t)] for t in types[b, p] if int(t) != C.TYPE_NONE],
            "boosts": {k: int(st.boosts[b, p, j]) for j, k in enumerate(BOOST_KEYS)},
            # Protosynthesis and Quark Drive live in `boosted_stat`, not a volatile.
            "volatiles": sorted([VOL_NAMES[i] for i in range(C.NUM_VOLATILES)
                                 if int(st.volatiles[b, p, i]) > 0] +
                                [n for n in ("protosynthesis", "quarkdrive")
                                 if int(st.boosted_stat[b, p]) >= 0 and
                                 N.abilities_by_id.get(int(st.ability[b, p, a])) == n]),
            "confusion": int(st.volatiles[b, p, C.V_CONFUSION]),
            "subHP": int(st.sub_hp[b, p]),
            "transformed": bool(st.transformed[b, p]),
        }
        return out

    phase = int(st.phase[b])
    if phase == C.PHASE_END:
        request = ["wait", "wait"]
    elif phase == C.PHASE_SWITCH:
        request = ["switch" if bool(st.force_switch[b, p]) else "wait" for p in range(2)]
    else:
        request = ["move", "move"]
    return {
        "turn": int(st.turn[b]) + (1 if phase == C.PHASE_MOVE else 0),
        "ended": phase == C.PHASE_END,
        "winner": int(st.winner[b]),
        "request": request,
        "weather": WEATHER_TO_PS[int(st.weather[b])],
        "weatherTurns": int(st.weather_turns[b]),
        "terrain": TERRAIN_TO_PS[int(st.terrain[b])],
        "terrainTurns": int(st.terrain_turns[b]),
        "pseudoWeather": sorted((["trickroom"] if int(st.trick_room[b]) > 0 else []) +
                                (["gravity"] if int(st.gravity[b]) > 0 else [])),
        "p1": side(0),
        "p2": side(1),
    }


def compare(sd, ps, rec):
    """Fields where the two snapshots disagree, as (field, where, showdown, psjax)."""
    diffs = []

    def check(field, where, a, b):
        if a != b:
            diffs.append((field, where, a, b))

    for f in ("request", "winner", "turn"):
        check(f, "", sd[f], ps[f])
    # A battle decided partway through the residuals stops there in Showdown --
    # the Leftovers and timers after the deciding knockout never run -- where
    # psjax finishes them. With a winner on both sides, the rest is moot.
    if sd["winner"] != -1 and sd["winner"] == ps["winner"]:
        return diffs
    for f in ("weather", "terrain"):
        check(f, "", sd[f], ps[f])
    if sd["weather"] == ps["weather"] and sd["weather"]:
        check("weatherTurns", sd["weather"], sd["weatherTurns"], ps["weatherTurns"])
    if sd["terrain"] == ps["terrain"] and sd["terrain"]:
        check("terrainTurns", sd["terrain"], sd["terrainTurns"], ps["terrainTurns"])
    check("pseudoWeather", "", sorted(set(sd["pseudoWeather"]) & PSEUDO_WEATHER),
          ps["pseudoWeather"])
    for p, key in enumerate(("p1", "p2")):
        a, b = sd[key], ps[key]
        team = rec["teams"][p]
        check("active", key, a["active"], b["active"])
        for s in range(C.TEAM_SIZE):
            where = f"{key}[{s}] {team[s]['species']}"
            check("hp", where, a["hp"][s], b["hp"][s])
            if a["hp"][s] == 0:
                continue        # Showdown reports 'fnt' and drops Tera; nothing else matters
            check("status", where, a["status"][s], b["status"][s])
            if a["status"][s] == "slp" and b["status"][s] == "slp":
                check("sleepTurns", where, a["sleepTurns"][s], b["sleepTurns"][s])
            if a["status"][s] == "tox" and b["status"][s] == "tox":
                check("toxicStage", where, a["toxicStage"][s], b["toxicStage"][s])
            sd_item = a["item"][s] if a["item"][s] in N.items else ""
            # (A battle won by a residual knockout stops there in Showdown;
            # psjax finishes the residuals -- a Harvest -- see `ended` below.)
            if sd["winner"] == -1:
                check("item", where, sd_item, b["item"][s])
            sd_ab = a["ability"][s] if a["ability"][s] in N.abilities else ""
            check("ability", where, sd_ab, b["ability"][s])
            sd_species = species_key(a["species"][s], a["speciesBase"][s])
            ps_species = b["species"][s]
            # A benched Minior's forme label (Meteor or a Core colour) is cosmetic:
            # Shields Down sets it again on the way in.
            if s != a["active"] and sd_species.startswith("minior") and ps_species.startswith("minior"):
                ps_species = sd_species
            check("species", where, sd_species, ps_species)
            check("tera", where, a["tera"][s], b["tera"][s])
            if not (s == a["active"] and a.get("transformed")):
                known = [m in N.moves for m in team[s]["moves"]]
                check("pp", where, [x for x, k in zip(a["pp"][s], known) if k],
                      [x for x, k in zip(b["pp"][s][:len(known)], known) if k])
        check("sideConditions", key,
              sorted(set(a["sideConditions"]) & set(C.SIDE_CONDITION_IDX)), b["sideConditions"])
        timed = sorted(set(a["sideConditionCounts"]) & set(b["sideConditionCounts"]) &
                       {k for k, i in C.SIDE_CONDITION_IDX.items() if i >= C.SC_REFLECT})
        check("sideConditionTurns", key, {k: a["sideConditionCounts"][k] for k in timed},
              {k: b["sideConditionCounts"][k] for k in timed})
        check("spikeLayers", key, a["spikeLayers"], b["spikeLayers"])
        check("toxicSpikeLayers", key, a["toxicSpikeLayers"], b["toxicSpikeLayers"])
        check("slotConditions", key, sorted(set(a["slotConditions"]) & SLOT_CONDITIONS),
              b["slotConditions"])
        if a["active"] >= 0 and a["hp"][a["active"]] > 0:
            where = f"{key} active {team[a['active']]['species']}"
            # Burn Up and Double Shock leave Showdown a "???" where the lost type
            # was; psjax drops it, which hits the same. (Not once the battle is
            # over: Roost's lost Flying comes back in psjax's last residuals.)
            if sd["winner"] == -1:
                check("types", where, [t for t in a["types"] if t != "???"], b["types"])
            # A battle decided by a knockout stops there in Showdown: the win is
            # checked before AfterFaint (Moxie, Beast Boost), and before the end
            # of the move or the residuals (Scale Shot's boosts, Protect
            # wearing off). psjax finishes the move and the turn. Nothing reads
            # boosts or volatiles once there is a winner.
            ended = sd["winner"] != -1
            if not ended:
                check("boosts", where, a["boosts"], b["boosts"])
            # psjax flags Unburden on any item loss; only the ability reads it.
            unburden = a["ability"][a["active"]] == "unburden"
            for v in sorted(((set(a["volatiles"]) & set(C.VOLATILE_IDX)) ^ set(b["volatiles"]))
                            - NOT_VOLATILE_IN_SHOWDOWN if not ended else ()):
                if v != "unburden" or unburden:
                    check(f"volatile:{v}", where, v in a["volatiles"], v in b["volatiles"])
            check("confusion", where, a["confusion"], b["confusion"])
            check("subHP", where, a["subHP"], b["subHP"])
    return diffs




# --- the replay ---------------------------------------------------------------

#: A difference in any of these means the two battles are no longer in the same
#: place -- someone fainted in one and not the other, a different Pokemon is out
#: -- and the replay of that battle stops. Anything else is copied over from
#: Showdown and the replay continues, so one bug does not hide every later one.
FATAL = {"request", "turn", "winner", "active", "species"}
PS_STATUS = {v: k for k, v in STATUS_TO_PS.items()}
PS_WEATHER = {v: k for k, v in WEATHER_TO_PS.items()}
PS_TERRAIN = {v: k for k, v in TERRAIN_TO_PS.items()}
RESYNC_FIELDS = ("hp", "status", "status_turns", "item", "ability", "terastallized", "pp",
                 "side_conditions", "boosts", "volatiles", "sub_hp", "weather",
                 "weather_turns", "terrain", "terrain_turns", "trick_room", "gravity",
                 "healing_wish", "tera_used", "paradox_booster")


def resync(cols, b, sd, rec):
    """Overwrite psjax battle `b` with Showdown's values for every compared field."""
    for p, key in enumerate(("p1", "p2")):
        a = sd[key]
        team = rec["teams"][p]
        for s in range(C.TEAM_SIZE):
            cols["hp"][b, p, s] = a["hp"][s]
            if a["hp"][s] == 0:
                continue
            cols["status"][b, p, s] = PS_STATUS.get(a["status"][s], 0)
            if a["status"][s] == "slp":
                cols["status_turns"][b, p, s] = a["sleepTurns"][s]
            if a["status"][s] == "tox":
                cols["status_turns"][b, p, s] = a["toxicStage"][s] + 1
            cols["item"][b, p, s] = N.item_id(a["item"][s]) if a["item"][s] in N.items else 0
            if a["ability"][s] in N.abilities:
                cols["ability"][b, p, s] = N.ability_id(a["ability"][s])
            cols["terastallized"][b, p, s] = a["tera"][s]
            if not (s == a["active"] and a.get("transformed")):
                for m, mv in enumerate(team[s]["moves"][:4]):
                    if mv in N.moves:
                        cols["pp"][b, p, s, m] = a["pp"][s][m]
        cols["healing_wish"][b, p] = 2 if "lunardance" in a["slotConditions"] else \
            1 if "healingwish" in a["slotConditions"] else 0
        if any(a["tera"]):
            cols["tera_used"][b, p] = True
        # A Pokemon still holding its Booster Energy is not running on it.
        if a["active"] >= 0 and a["item"][a["active"]] == "boosterenergy":
            cols["paradox_booster"][b, p] = False
        sc = cols["side_conditions"][b, p]
        for name, i in C.SIDE_CONDITION_IDX.items():
            count = a["sideConditionCounts"].get(name)
            sc[i] = 0 if count is None else (count or 1)
        if a["active"] >= 0 and a["hp"][a["active"]] > 0:
            cols["boosts"][b, p] = [a["boosts"][k] for k in BOOST_KEYS]
            vols = cols["volatiles"][b, p]
            for name, i in C.VOLATILE_IDX.items():
                if name in NOT_VOLATILE_IN_SHOWDOWN:
                    continue
                if (name in a["volatiles"]) != (vols[i] > 0):
                    on = a["confusion"] if name == "confusion" else 1
                    vols[i] = on if name in a["volatiles"] else 0
            cols["sub_hp"][b, p] = a["subHP"]
    cols["weather"][b] = PS_WEATHER.get(sd["weather"], 0)
    cols["weather_turns"][b] = sd["weatherTurns"]
    cols["terrain"][b] = PS_TERRAIN.get(sd["terrain"], 0)
    cols["terrain_turns"][b] = sd["terrainTurns"]
    for name, field in (("trickroom", "trick_room"), ("gravity", "gravity")):
        if (name in sd["pseudoWeather"]) != (cols[field][b] > 0):
            cols[field][b] = 3 if name in sd["pseudoWeather"] else 0


def run(records):
    B = len(records)
    t0, t1, gender, missing, init_diffs = build_teams(records)
    words = jnp.asarray([r["w"] for r in records], jnp.uint32)

    tic = time.time()
    state = INIT(t0, t1, gender, words)
    jax.block_until_ready(state)
    print(f"built {B} battles in {time.time() - tic:.1f}s")

    findings = []                       # every decision that differed, in order
    stopped = {}                        # battle -> decision where the replay stopped
    mask_diffs = collections.Counter()
    mask_examples = {}
    alive = np.ones(B, bool)
    progress = np.zeros(B, np.int32)
    steps_max = max(len(r["steps"]) for r in records)

    def snap_all(state, stage):
        types, masks = jax.device_get(VIEWS(state))
        st = jax.device_get(state)
        cols = None
        for b in np.nonzero(alive)[0]:
            rec = records[b]
            sd = rec["initial"] if stage < 0 else rec["steps"][stage]["after"]
            diffs = compare(sd, ps_snapshot(st, b, types, sd), rec)
            if stage < 0:
                diffs = init_diffs.get(b, []) + diffs
            if diffs:
                findings.append(dict(b=int(b), step=stage, diffs=diffs))
                if any(d[0] in FATAL for d in diffs):
                    stopped[int(b)] = stage
                    alive[b] = False
                    continue
                if any(d[0] not in SOFT for d in diffs):
                    if cols is None:
                        cols = {f: np.array(getattr(st, f)) for f in RESYNC_FIELDS}
                    resync(cols, b, sd, rec)
            if stage + 1 >= len(rec["steps"]):
                alive[b] = False          # reached the end of the battle
        if cols is not None:
            state = state._replace(**{f: jnp.asarray(v) for f, v in cols.items()})
        return state, masks

    tic = time.time()
    state, masks = snap_all(state, -1)
    for t in range(steps_max):
        if not alive.any():
            break
        actions = np.zeros((B, 2), np.int32)
        for b in np.nonzero(alive)[0]:
            step = records[b]["steps"][t]
            actions[b] = step["actions"]
            for p in range(2):
                want = step["legal"][p]
                if want is None:
                    continue
                have = set(np.nonzero(masks[b, p])[0].tolist())
                for kind, acts in (("showdown-only", set(want) - have),
                                   ("psjax-only", have - set(want))):
                    if acts:
                        key = (kind, "switch" if min(acts) >= 8 else
                               "tera" if min(acts) >= 4 else "move")
                        mask_diffs[key] += 1
                        mask_examples.setdefault(key, []).append(
                            (records[b]["id"], t, p, sorted(want), sorted(have)))
        progress[alive] = t + 1
        state = STEP(state, jnp.asarray(actions), words)
        state, masks = snap_all(state, t)
    print(f"replayed {int(progress.sum())} decisions in {time.time() - tic:.1f}s")
    return findings, stopped, progress, missing, mask_diffs, mask_examples


def summarise(records, findings, stopped, progress, missing, mask_diffs, mask_examples,
              out=None):
    B = len(records)
    replayed = int(progress.sum())
    differing = {(f["b"], f["step"]) for f in findings if f["step"] >= 0}
    print(f"\n{B} battles: {replayed} decisions replayed, {replayed - len(differing)} "
          f"matched Showdown exactly; {B - len(stopped)} battles replayed to the end")
    battles, decisions = collections.defaultdict(set), collections.Counter()
    for f in findings:
        for field in {d[0] for d in f["diffs"]}:
            battles[field].add(f["b"])
            decisions[field] += 1
    print("\nfield                      battles  decisions")
    for field in sorted(battles, key=lambda k: -len(battles[k])):
        tag = "  (stops replay)" if field in FATAL else ""
        print(f"  {field:24s} {len(battles[field]):6d}  {decisions[field]:9d}{tag}")
    if missing:
        print("\nnot in psjax's index (treated as absent):",
              ", ".join(f"{k} x{v}" for k, v in missing.most_common()))
    if mask_diffs:
        print("\nlegal-action disagreements:")
        for key, n in mask_diffs.most_common():
            print(f"  {n:4d}  {key[0]} {key[1]}")
    if out:
        report = []
        for f in findings:
            rec = records[f["b"]]
            step = f["step"]
            here = rec["steps"][step] if step >= 0 else None
            report.append(dict(
                id=rec["id"], battle=rec["battle"], w=rec["w"], step=step,
                turn=(here["after"]["turn"] if here else 1),
                choices=here["choices"] if here else None,
                stopped=stopped.get(f["b"]) == step,
                diffs=[list(d) for d in f["diffs"]],
                log=(here["log"] if here else []),
            ))
        with open(out, "w") as fh:
            json.dump(dict(findings=report,
                           mask_examples={f"{k[0]} {k[1]}": v[:20]
                                          for k, v in mask_examples.items()}),
                      fh, indent=1, default=str)
        print(f"\ndetails -> {out}")


def main():
    path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    records = [r for r in json.load(open(path)) if not r.get("error")]
    summarise(records, *run(records), out=out)


if __name__ == "__main__":
    main()
