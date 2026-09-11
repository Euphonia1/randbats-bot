"""Build damage-calculation contexts from plain names, for tests."""
from __future__ import annotations

import jax.numpy as jnp

from psjax import consts as C
from psjax.callbacks import CbCtx
from psjax.damage import Attacker, Defender, MoveCtx, resolve_move_ctx
from psjax.stats import boost_multiply
from psjax.data import load_data, names
from psjax.stats import compute_all_stats

STATUS = {"brn": C.BRN, "par": C.PAR, "slp": C.SLP, "frz": C.FRZ,
          "psn": C.PSN, "tox": C.TOX}
BOOST_ORDER = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]


def _stats_and_types(spec):
    data, n = load_data(), names()
    sid = n.species_id(spec["species"])
    level = jnp.int8(spec.get("level", 100))
    base = data["species_base_stats"][sid]
    stats = compute_all_stats(base[None], jnp.asarray([level]))[0]
    types = data["species_types"][sid]
    return sid, level, stats, types


def _boosts(spec):
    b = jnp.zeros(7, jnp.int8)
    for name, val in (spec.get("boosts") or {}).items():
        b = b.at[BOOST_ORDER.index(name)].set(val)
    return b


def _boosted_stat(spec, stats):
    """Which stat Protosynthesis / Quark Drive raised, or -1 for none.

    The case can name it explicitly; otherwise it is the highest non-HP stat,
    the way the abilities pick on switch-in.
    """
    if not spec.get("boosted"):
        return -1
    named = spec["boosted"]
    if named is True:
        return int(jnp.argmax(stats[1:])) + 1
    return {"atk": C.ATK, "def": C.DEF, "spa": C.SPA,
            "spd": C.SPD, "spe": C.SPE}[named]


def make_attacker(spec) -> Attacker:
    data, n = load_data(), names()
    sid, level, stats, types = _stats_and_types(spec)
    maxhp = stats[C.HP]
    hp = jnp.int16(max(1, int(maxhp * spec["hpPercent"]))) if spec.get("hpPercent") else maxhp
    tera = spec.get("tera")
    tera_type = jnp.int8(n.type_id(tera)) if tera else jnp.int8(C.TYPE_NONE)
    cur = types
    if tera and n.to_id(tera) != "stellar":
        cur = jnp.array([n.type_id(tera), C.TYPE_NONE], jnp.int8)
    return Attacker(
        level=level, stats=stats, boosts=_boosts(spec), types=cur, base_types=types,
        ability=jnp.int16(n.ability_id(spec.get("ability", ""))),
        item=jnp.int16(n.item_id(spec.get("item", ""))),
        status=jnp.int8(STATUS.get(spec.get("status"), 0)),
        hp=hp, maxhp=maxhp,
        terastallized=jnp.bool_(bool(tera)), tera_type=tera_type,
        boosted_stat=jnp.int8(_boosted_stat(spec, stats)),
    )


def make_defender(spec) -> Defender:
    data, n = load_data(), names()
    sid, level, stats, types = _stats_and_types(spec)
    maxhp = stats[C.HP]
    hp = jnp.int16(max(1, int(maxhp * spec["hpPercent"]))) if spec.get("hpPercent") else maxhp
    tera = spec.get("tera")
    tera_type = jnp.int8(n.type_id(tera)) if tera else jnp.int8(C.TYPE_NONE)
    cur = types
    if tera and n.to_id(tera) != "stellar":
        cur = jnp.array([n.type_id(tera), C.TYPE_NONE], jnp.int8)
    return Defender(
        stats=stats, status=jnp.int8(STATUS.get(spec.get("status"), 0)),
        boosts=_boosts(spec), types=cur,
        ability=jnp.int16(n.ability_id(spec.get("ability", ""))),
        item=jnp.int16(n.item_id(spec.get("item", ""))),
        hp=hp, maxhp=maxhp,
        terastallized=jnp.bool_(bool(tera)), tera_type=tera_type,
        nfe=jnp.bool_(bool(data["species_nfe"][sid])),
        boosted_stat=jnp.int8(_boosted_stat(spec, stats)),
    )


def make_cb_ctx(case, atk: Attacker, dfn: Defender, weather, terrain) -> CbCtx:
    """Callback context for a one-off calculation (no turn history)."""
    data, n = load_data(), names()
    a_spec, d_spec = case["attacker"], case["defender"]
    a_sid, d_sid = n.species_id(a_spec["species"]), n.species_id(d_spec["species"])
    boosted = lambda p, i: boost_multiply(p.stats[i], p.boosts[i - 1])
    return CbCtx(
        base_power=jnp.int32(0), move_type=jnp.int8(0),
        hit_number=jnp.int32(case.get("hitNumber", 1)),
        atk_status=atk.status, dfn_status=dfn.status,
        atk_hp=atk.hp, atk_maxhp=atk.maxhp, dfn_hp=dfn.hp, dfn_maxhp=dfn.maxhp,
        atk_item=atk.item, dfn_item=dfn.item,
        atk_weight=data["species_weight"][a_sid],
        dfn_weight=data["species_weight"][d_sid],
        atk_speed=boosted(atk, C.SPE), dfn_speed=boosted(dfn, C.SPE),
        atk_boosts=atk.boosts, dfn_boosts=dfn.boosts,
        moves_first=jnp.bool_(case.get("movesFirst", True)),
        user_damaged=jnp.bool_(case.get("userDamaged", False)),
        target_damaged=jnp.bool_(case.get("targetDamaged", False)),
        weather=weather, terrain=terrain,
        grounded_user=jnp.bool_(case.get("groundedUser", True)),
        grounded_target=jnp.bool_(case.get("groundedTarget", True)),
        pp_left=jnp.int32(case.get("ppLeft", 8)),
        times_hit=jnp.int32(case.get("timesAttacked", 0)),
        fainted_count=jnp.int32(case.get("faintedCount", 0)),
        terastallized=atk.terastallized, tera_type=atk.tera_type,
        fury_multiplier=jnp.int32(case.get("furyMultiplier", 1)),
        level=atk.level, last_damage=jnp.int32(case.get("lastDamage", 0)),
        last_damage_category=jnp.int8(case.get("lastDamageCategory", -1)),
        off_atk=boosted(atk, C.ATK), off_spa=boosted(atk, C.SPA),
        user_ability=atk.ability,
        last_move_failed=jnp.bool_(case.get("lastMoveFailed", False)),
        stats_lowered=jnp.bool_(case.get("statsLowered", False)),
    )


def make_move(move_name, cb_ctx=None) -> MoveCtx:
    """Resolve a move by name; without a context the declared row is used."""
    data, n = load_data(), names()
    mid = n.move_id(move_name)
    if cb_ctx is None:
        return MoveCtx(
            id=jnp.int32(mid), type=data["move_type"][mid],
            category=data["move_category"][mid],
            base_power=data["move_base_power"][mid].astype(jnp.int32),
            flags=data["move_flags"][mid], crit_ratio=data["move_crit_ratio"][mid],
            effect_cb=data["move_effect_cb"][mid],
        )
    return resolve_move_ctx(data, mid, cb_ctx)
