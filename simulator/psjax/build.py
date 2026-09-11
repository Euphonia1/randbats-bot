"""Compile `data/gen9_raw.json` into flat arrays the JAX engine can index.

The engine never sees a dict, a string, or a Python branch on game data: every
lookup is an array index. This module is the only place that knows about
Showdown's field names, so a Showdown data update is absorbed by re-running
`tools/dump_data.js` and then this script.

    python -m psjax.build
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

from . import consts as C
from . import effects as E
from . import hooks as H

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "gen9_raw.json"
OUT_NPZ = ROOT / "data" / "gen9.npz"
OUT_INDEX = ROOT / "data" / "gen9_index.json"


def to_id(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


class BuildError(Exception):
    pass


# --- type chart --------------------------------------------------------------

def build_typechart(raw) -> np.ndarray:
    """`eff[attacker, defender]` as a float multiplier."""
    tc = raw["typechart"]
    missing = [t for t in C.TYPES if t not in tc]
    if missing:
        raise BuildError(f"types missing from dex: {missing}")
    eff = np.ones((C.NUM_TYPES, C.NUM_TYPES), dtype=np.float32)
    # Showdown stores this defender-first: damageTaken[defender][Attacker].
    code = {0: 1.0, 1: 2.0, 2: 0.5, 3: 0.0}
    for d_name, d_idx in C.TYPE_IDX.items():
        taken = tc[d_name]["damageTaken"]
        for a_name, a_idx in C.TYPE_IDX.items():
            v = taken.get(a_name.capitalize())
            if v is None:
                # Showdown capitalises type keys; Stellar has no entries at all.
                v = taken.get(a_name, 0)
            if v not in code:
                v = 0
            eff[a_idx, d_idx] = code[v]
    return eff


# --- species -----------------------------------------------------------------

def build_species(raw):
    ids = sorted(raw["species"])
    idx = {sid: i for i, sid in enumerate(ids)}
    n = len(ids)
    out = {
        "species_num": np.zeros(n, np.int16),
        "species_types": np.full((n, 2), C.TYPE_NONE, np.int8),
        "species_base_stats": np.zeros((n, 6), np.int16),
        "species_weight": np.zeros(n, np.float32),
        "species_abilities": np.zeros((n, 3), np.int16),
        "species_nfe": np.zeros(n, np.bool_),
    }
    for sid, i in idx.items():
        s = raw["species"][sid]
        out["species_num"][i] = s["num"]
        for k, t in enumerate(s["types"][:2]):
            out["species_types"][i, k] = C.TYPE_IDX[to_id(t)]
        bs = s["baseStats"]
        out["species_base_stats"][i] = [bs["hp"], bs["atk"], bs["def"],
                                        bs["spa"], bs["spd"], bs["spe"]]
        out["species_weight"][i] = s["weightkg"]
        for k, a in enumerate(s["abilities"][:3]):
            out["species_abilities"][i, k] = H.ABILITY_IDX.get(to_id(a), 0)
        out["species_nfe"][i] = s["nfe"]
    return idx, out


# --- moves -------------------------------------------------------------------

BOOST_KEYS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]


def _boost_vec(b):
    v = np.zeros(7, np.int8)
    if b:
        for k, amount in b.items():
            v[BOOST_KEYS.index(k)] = amount
    return v


def _fraction(f, default=(0, 1)):
    return tuple(f) if f else default


def build_moves(raw):
    ids = sorted(raw["moves"])
    idx = {mid: i for i, mid in enumerate(ids)}
    n = len(ids)
    z16 = lambda: np.zeros(n, np.int16)
    z8 = lambda: np.zeros(n, np.int8)
    out = {
        "move_num": z16(), "move_type": z8(), "move_category": z8(),
        "move_base_power": z16(), "move_accuracy": z16(), "move_pp": z8(),
        "move_priority": z8(), "move_target": z8(),
        "move_flags": np.zeros(n, np.int64), "move_crit_ratio": z8(),
        "move_will_crit": np.zeros(n, np.bool_),
        "move_drain": np.zeros((n, 2), np.int8),
        "move_recoil": np.zeros((n, 2), np.int8),
        "move_heal": np.zeros((n, 2), np.int8),
        "move_multihit": np.ones((n, 2), np.int8),
        "move_ohko": np.zeros(n, np.bool_),
        "move_fixed_damage": z16(),
        "move_status": z8(), "move_volatile": z8(),
        "move_side_condition": z8(), "move_weather": z8(), "move_terrain": z8(),
        "move_boosts": np.zeros((n, 7), np.int8),
        "move_self_boosts": np.zeros((n, 7), np.int8),
        "move_self_volatile": z8(), "move_self_side_condition": z8(),
        # up to two secondaries (e.g. Fire Fang: burn chance + flinch chance)
        "move_sec_chance": np.zeros((n, 2), np.int8),
        "move_sec_status": np.zeros((n, 2), np.int8),
        "move_sec_volatile": np.zeros((n, 2), np.int8),
        "move_sec_boosts": np.zeros((n, 2, 7), np.int8),
        "move_sec_self_boosts": np.zeros((n, 2, 7), np.int8),
        "move_force_switch": np.zeros(n, np.bool_),
        "move_self_switch": z8(),
        "move_selfdestruct": z8(),
        "move_breaks_protect": np.zeros(n, np.bool_),
        "move_stalling": np.zeros(n, np.bool_),
        "move_thaws_target": np.zeros(n, np.bool_),
        "move_sleep_usable": np.zeros(n, np.bool_),
        "move_ignore_ability": np.zeros(n, np.bool_),
        "move_ignore_defensive": np.zeros(n, np.bool_),
        "move_ignore_evasion": np.zeros(n, np.bool_),
        "move_ignore_offensive": np.zeros(n, np.bool_),
        "move_ignore_immunity": np.zeros(n, np.int64),  # bitmask; -1 == all
        "move_override_off_stat": z8(), "move_override_def_stat": z8(),
        "move_override_off_pokemon": np.zeros(n, np.bool_),
        "move_defensive_category": z8(),
        "move_crash_damage": np.zeros(n, np.bool_),
        "move_no_variance": np.zeros(n, np.bool_),
        "move_is_charge": np.zeros(n, np.bool_),
        "move_duration": z8(),
        "move_bp_replace": z8(), "move_bp_modify": z8(), "move_dmg_cb": z8(),
        "move_acc_cb": z8(),
        "move_type_cb": z8(), "move_effect_cb": z8(),
        "move_implemented": np.zeros(n, np.bool_),
    }
    stat_name_to_idx = {"atk": C.ATK, "def": C.DEF, "spa": C.SPA,
                        "spd": C.SPD, "spe": C.SPE}

    for mid, i in idx.items():
        m = raw["moves"][mid]
        out["move_num"][i] = m["num"]
        out["move_type"][i] = C.TYPE_IDX[to_id(m["type"])]
        out["move_category"][i] = C.CATEGORIES[m["category"]]
        out["move_base_power"][i] = m["basePower"]
        out["move_accuracy"][i] = m["accuracy"]      # -1 means "never misses"
        out["move_pp"][i] = min(m["pp"], 127)
        out["move_priority"][i] = m["priority"]
        out["move_target"][i] = C.TARGET_IDX[m["target"]]
        out["move_crit_ratio"][i] = m["critRatio"]
        out["move_will_crit"][i] = bool(m["willCrit"])

        bits = 0
        for f, present in (m["flags"] or {}).items():
            if present and f in C.FLAG_BITS:
                bits |= 1 << C.FLAG_BITS[f]
        out["move_flags"][i] = bits
        out["move_is_charge"][i] = bool((m["flags"] or {}).get("charge"))

        out["move_drain"][i] = _fraction(m["drain"])
        out["move_recoil"][i] = _fraction(m["recoil"])
        out["move_heal"][i] = _fraction(m["heal"])

        mh = m["multihit"]
        if isinstance(mh, list):
            out["move_multihit"][i] = [mh[0], mh[1]]
        elif isinstance(mh, int):
            out["move_multihit"][i] = [mh, mh]

        out["move_ohko"][i] = bool(m["ohko"])
        if isinstance(m["damage"], int):
            out["move_fixed_damage"][i] = m["damage"]

        if m["status"]:
            out["move_status"][i] = C.STATUS_IDX.get(m["status"], 0)
        if m["volatileStatus"]:
            out["move_volatile"][i] = C.VOLATILE_IDX.get(m["volatileStatus"], -1)
        if m["sideCondition"]:
            out["move_side_condition"][i] = C.SIDE_CONDITION_IDX.get(
                to_id(m["sideCondition"]), -1)
        if m["weather"]:
            out["move_weather"][i] = C.WEATHER_IDX.get(to_id(m["weather"]), 0)
        if m["terrain"]:
            out["move_terrain"][i] = C.TERRAIN_IDX.get(to_id(m["terrain"]), 0)

        out["move_boosts"][i] = _boost_vec(m["boosts"])
        if m["self"]:
            out["move_self_boosts"][i] = _boost_vec(m["self"]["boosts"])
            if m["self"]["volatileStatus"]:
                out["move_self_volatile"][i] = C.VOLATILE_IDX.get(
                    m["self"]["volatileStatus"], -1)
            if m["self"]["sideCondition"]:
                out["move_self_side_condition"][i] = C.SIDE_CONDITION_IDX.get(
                    to_id(m["self"]["sideCondition"]), -1)

        for k, s in enumerate(m["secondaries"][:2]):
            out["move_sec_chance"][i, k] = min(s["chance"], 100)
            if s["status"]:
                out["move_sec_status"][i, k] = C.STATUS_IDX.get(s["status"], 0)
            if s["volatileStatus"]:
                out["move_sec_volatile"][i, k] = C.VOLATILE_IDX.get(
                    s["volatileStatus"], -1)
            out["move_sec_boosts"][i, k] = _boost_vec(s["boosts"])
            if s["self"]:
                out["move_sec_self_boosts"][i, k] = _boost_vec(s["self"]["boosts"])

        out["move_force_switch"][i] = m["forceSwitch"]
        ss = m["selfSwitch"]
        out["move_self_switch"][i] = 0 if not ss else (2 if ss == "copyvolatile" else 1)
        sd = m["selfdestruct"]
        out["move_selfdestruct"][i] = 0 if not sd else (2 if sd == "ifHit" else 1)
        out["move_breaks_protect"][i] = m["breaksProtect"]
        out["move_stalling"][i] = m["stallingMove"]
        out["move_thaws_target"][i] = m["thawsTarget"]
        out["move_sleep_usable"][i] = m["sleepUsable"]
        out["move_ignore_ability"][i] = m["ignoreAbility"]
        out["move_ignore_defensive"][i] = m["ignoreDefensive"]
        out["move_ignore_evasion"][i] = m["ignoreEvasion"]
        out["move_ignore_offensive"][i] = m["ignoreOffensive"]

        ii = m["ignoreImmunity"]
        if ii is True:
            out["move_ignore_immunity"][i] = -1
        elif isinstance(ii, dict):
            mask = 0
            for t, on in ii.items():
                if on and to_id(t) in C.TYPE_IDX:
                    mask |= 1 << C.TYPE_IDX[to_id(t)]
            out["move_ignore_immunity"][i] = mask

        if m["overrideOffensiveStat"]:
            out["move_override_off_stat"][i] = stat_name_to_idx[m["overrideOffensiveStat"]]
        if m["overrideDefensiveStat"]:
            out["move_override_def_stat"][i] = stat_name_to_idx[m["overrideDefensiveStat"]]
        out["move_override_off_pokemon"][i] = m["overrideOffensivePokemon"] == "target"
        if m["defensiveCategory"]:
            out["move_defensive_category"][i] = C.CATEGORIES[m["defensiveCategory"]]
        out["move_crash_damage"][i] = m["hasCrashDamage"]
        out["move_no_variance"][i] = m["noDamageVariance"]
        if m["condition"] and m["condition"]["duration"]:
            out["move_duration"][i] = min(m["condition"]["duration"], 127)

        out["move_bp_replace"][i] = E.BP_REPLACE_INDEX.get(mid, 0)
        out["move_bp_modify"][i] = E.BP_MODIFY_INDEX.get(mid, 0)
        out["move_acc_cb"][i] = E.ACC_INDEX.get(mid, 0)
        out["move_dmg_cb"][i] = E.DMG_INDEX.get(mid, 0)
        out["move_type_cb"][i] = E.TYPE_INDEX.get(mid, 0)
        out["move_effect_cb"][i] = E.EFFECT_INDEX.get(mid, 0)

        # A move is "modelled" when it has no JS behaviour left unaccounted for:
        # either it declares no callbacks, or we route it to a handler.
        has_handler = any(out[k][i] for k in
                          ("move_bp_replace", "move_bp_modify", "move_dmg_cb",
                           "move_type_cb", "move_effect_cb", "move_acc_cb"))
        out["move_implemented"][i] = (not m["callbacks"]) or has_handler

    return idx, out


# --- abilities & items -------------------------------------------------------

def build_abilities(raw):
    dupes = [n for n in set(H.ABILITY_NAMES) if H.ABILITY_NAMES.count(n) > 1]
    if dupes:
        raise BuildError(f"duplicate ABILITY_NAMES entries: {sorted(dupes)}")
    known = set(raw["abilities"])
    unknown = [a for a in H.ABILITY_NAMES[1:] if a not in known]
    if unknown:
        raise BuildError(f"ABILITY_NAMES entries not in the Gen 9 dex: {unknown}")
    n = len(H.ABILITY_NAMES)
    out = {
        "ability_mold_breaker": np.zeros(n, np.bool_),
        "ability_absorb_type": np.full(n, C.TYPE_NONE, np.int8),
        "ability_absorb_heal": np.zeros(n, np.int8),      # sixteenths of max HP
        "ability_absorb_boost_stat": np.full(n, -1, np.int8),
        "ability_absorb_boost_amt": np.zeros(n, np.int8),
        "ability_immune_type": np.full(n, C.TYPE_NONE, np.int8),
        "ability_weather": np.zeros(n, np.int8),
        "ability_terrain": np.zeros(n, np.int8),
        "ability_status_immune": np.zeros(n, np.int64),   # bitmask over statuses
        "ability_ate_type": np.full(n, C.TYPE_NONE, np.int8),
    }
    boost_name_to_idx = {"atk": C.B_ATK, "def": C.B_DEF, "spa": C.B_SPA,
                         "spd": C.B_SPD, "spe": C.B_SPE}
    for name in H.MOLD_BREAKER:
        if name in H.ABILITY_IDX:
            out["ability_mold_breaker"][H.ABILITY_IDX[name]] = True
    for name, (t, heal, stat, amt) in H.ABSORB.items():
        i = H.ABILITY_IDX.get(name)
        if i is None:
            continue
        out["ability_absorb_type"][i] = C.TYPE_IDX[t]
        out["ability_absorb_heal"][i] = heal
        if stat != -1:
            out["ability_absorb_boost_stat"][i] = boost_name_to_idx[stat]
            out["ability_absorb_boost_amt"][i] = amt
    for name, t in H.TYPE_IMMUNE.items():
        if name in H.ABILITY_IDX:
            out["ability_immune_type"][H.ABILITY_IDX[name]] = C.TYPE_IDX[t]
    for name, w in H.WEATHER_SETTER.items():
        if name in H.ABILITY_IDX:
            out["ability_weather"][H.ABILITY_IDX[name]] = C.WEATHER_IDX[w]
    for name, t in H.TERRAIN_SETTER.items():
        if name in H.ABILITY_IDX:
            out["ability_terrain"][H.ABILITY_IDX[name]] = C.TERRAIN_IDX[t]
    for name, t in H.ATE_ABILITIES.items():
        if name in H.ABILITY_IDX:
            out["ability_ate_type"][H.ABILITY_IDX[name]] = C.TYPE_IDX[t]
    for name, statuses in H.STATUS_IMMUNE.items():
        i = H.ABILITY_IDX.get(name)
        if i is None:
            continue
        mask = 0
        for s in statuses:
            mask |= 1 << C.STATUS_IDX[s]
        out["ability_status_immune"][i] = mask
    return out


def build_items(raw):
    dupes = [n for n in set(H.ITEM_NAMES) if H.ITEM_NAMES.count(n) > 1]
    if dupes:
        raise BuildError(f"duplicate ITEM_NAMES entries: {sorted(dupes)}")
    known = set(raw["items"])
    unknown = [it for it in H.ITEM_NAMES[1:] if it not in known]
    if unknown:
        raise BuildError(f"ITEM_NAMES entries not in the Gen 9 dex: {unknown}")
    n = len(H.ITEM_NAMES)
    out = {
        "item_is_berry": np.zeros(n, np.bool_),
        "item_is_choice": np.zeros(n, np.bool_),
        "item_fling_power": np.zeros(n, np.int16),
        "item_boost_type": np.full(n, C.TYPE_NONE, np.int8),
        "item_boost_mod": np.full(n, 4096, np.int16),   # 4096 == x1
        "item_resist_type": np.full(n, C.TYPE_NONE, np.int8),
    }
    for name, i in H.ITEM_IDX.items():
        if i == 0:
            continue
        it = raw["items"][name]
        out["item_is_berry"][i] = it["isBerry"]
        out["item_is_choice"][i] = it["isChoice"]
        if it["fling"]:
            out["item_fling_power"][i] = it["fling"].get("basePower", 0)
    for name, (t, mod) in H.TYPE_BOOST_ITEM.items():
        if name in H.ITEM_IDX:
            out["item_boost_type"][H.ITEM_IDX[name]] = C.TYPE_IDX[t]
            out["item_boost_mod"][H.ITEM_IDX[name]] = mod
    for name, t in H.RESIST_BERRY.items():
        if name in H.ITEM_IDX:
            out["item_resist_type"][H.ITEM_IDX[name]] = C.TYPE_IDX[t]
    return out


# --- randbats sets -----------------------------------------------------------

MAX_SETS = 8
MAX_MOVEPOOL = 16
MAX_TERA = 12


def build_randbats(raw, species_idx, move_idx):
    """Pack the Gen 9 random-battle sets into rectangular arrays.

    Team generation samples from these on-device, so ragged movepools are padded
    with -1 and paired with an explicit count.
    """
    entries = []
    for name, e in raw["randbats"].items():
        sid = to_id(name)
        if sid not in species_idx:
            continue
        entries.append((sid, e))
    entries.sort()
    n = len(entries)
    out = {
        "rb_species": np.zeros(n, np.int16),
        "rb_level": np.zeros(n, np.int8),
        "rb_num_sets": np.zeros(n, np.int8),
        "rb_role": np.zeros((n, MAX_SETS), np.int8),
        "rb_movepool": np.full((n, MAX_SETS, MAX_MOVEPOOL), -1, np.int16),
        "rb_movepool_len": np.zeros((n, MAX_SETS), np.int8),
        "rb_abilities": np.full((n, MAX_SETS, 4), 0, np.int16),
        "rb_abilities_len": np.zeros((n, MAX_SETS), np.int8),
        "rb_tera": np.full((n, MAX_SETS, MAX_TERA), C.TYPE_NONE, np.int8),
        "rb_tera_len": np.zeros((n, MAX_SETS), np.int8),
    }
    overflow = []
    for i, (sid, e) in enumerate(entries):
        out["rb_species"][i] = species_idx[sid]
        out["rb_level"][i] = e["level"]
        sets = e["sets"][:MAX_SETS]
        if len(e["sets"]) > MAX_SETS:
            overflow.append((sid, "sets", len(e["sets"])))
        out["rb_num_sets"][i] = len(sets)
        for k, s in enumerate(sets):
            out["rb_role"][i, k] = C.ROLE_IDX[s["role"]]
            mp = [move_idx[to_id(m)] for m in s["movepool"] if to_id(m) in move_idx]
            if len(mp) > MAX_MOVEPOOL:
                overflow.append((sid, "movepool", len(mp)))
            mp = mp[:MAX_MOVEPOOL]
            out["rb_movepool"][i, k, :len(mp)] = mp
            out["rb_movepool_len"][i, k] = len(mp)
            ab = [H.ABILITY_IDX.get(to_id(a), 0) for a in s["abilities"]][:4]
            out["rb_abilities"][i, k, :len(ab)] = ab
            out["rb_abilities_len"][i, k] = len(ab)
            tt = [C.TYPE_IDX[to_id(t)] for t in s.get("teraTypes", [])
                  if to_id(t) in C.TYPE_IDX][:MAX_TERA]
            out["rb_tera"][i, k, :len(tt)] = tt
            out["rb_tera_len"][i, k] = len(tt)
    return out, overflow


# --- entrypoint --------------------------------------------------------------

def main() -> int:
    if not RAW.exists():
        print(f"missing {RAW}; run `node tools/dump_data.js` first", file=sys.stderr)
        return 1
    raw = json.load(open(RAW))

    arrays = {"type_eff": build_typechart(raw)}
    species_idx, sp = build_species(raw)
    arrays.update(sp)
    move_idx, mv = build_moves(raw)
    arrays.update(mv)
    arrays.update(build_abilities(raw))
    arrays.update(build_items(raw))
    rb, overflow = build_randbats(raw, species_idx, move_idx)
    arrays.update(rb)

    # Cross-check: every move named in the effect registry must exist.
    unknown = sorted({m for reg in (E.BP_REPLACE_MOVES, E.BP_MODIFY_MOVES,
                                    E.DMG_MOVES, E.TYPE_MOVES, E.EFFECT_MOVES,
                                    E.ACC_MOVES)
                      for ms in reg.values() for m in ms if m not in move_idx})
    if unknown:
        raise BuildError(f"effects.py references unknown moves: {unknown}")

    np.savez_compressed(OUT_NPZ, **arrays)
    json.dump(
        {
            "gen": raw["gen"],
            "showdown_version": raw["showdownVersion"],
            "species": species_idx,
            "moves": move_idx,
            "abilities": H.ABILITY_IDX,
            "items": H.ITEM_IDX,
            "types": C.TYPE_IDX,
        },
        open(OUT_INDEX, "w"),
    )

    n_impl = int(arrays["move_implemented"].sum())
    print(f"wrote {OUT_NPZ.name} ({OUT_NPZ.stat().st_size / 1e6:.2f} MB) "
          f"and {OUT_INDEX.name}")
    print(f"  species={len(species_idx)} moves={len(move_idx)} "
          f"abilities={len(H.ABILITY_NAMES)} items={len(H.ITEM_NAMES)} "
          f"randbats={len(arrays['rb_species'])}")
    print(f"  moves fully modelled: {n_impl}/{len(move_idx)}")
    if overflow:
        print(f"  WARNING: {len(overflow)} randbats entries truncated: {overflow[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
