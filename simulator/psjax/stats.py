"""Stat computation and Showdown's fixed-point modifier arithmetic.

Showdown does all damage math in integers with specific truncation points, and
those truncations are load-bearing: a float reimplementation drifts by 1 HP
often enough to flip KO ranges. Everything here mirrors `sim/battle.ts`.
"""
from __future__ import annotations

import jax.numpy as jnp

from . import consts as C

# Showdown's boost tables, kept as integer numerator/denominator pairs so the
# division can be a single floor.
#   stages >= 0: stat * (2 + b) / 2        stages < 0: stat * 2 / (2 - b)
#   accuracy/evasion use 3 instead of 2.


def idiv(a, b):
    """`floor(a / b)` for non-negative integers, computed through a float divide.

    XLA's lowering of integer division by a non-constant divisor is very large,
    and several of them inside one `lax.switch` makes `vmap` compilation blow up
    (8 branches was already minutes on CPU). A float divide plus a two-step
    correction is exact for the magnitudes here -- the largest numerator is
    `4096 * maxhp`, around 2.9e6, well inside float32's exact integer range of
    2^24 -- and compiles to a handful of ops.
    """
    a = jnp.asarray(a, jnp.int32)
    b = jnp.maximum(jnp.asarray(b, jnp.int32), 1)
    q = jnp.floor(a.astype(jnp.float32) / b.astype(jnp.float32)).astype(jnp.int32)
    q = q - (a < q * b).astype(jnp.int32)          # float rounded up
    q = q + ((q + 1) * b <= a).astype(jnp.int32)   # float rounded down
    return q


def boost_multiply(stat, stage, denom=2):
    """`floor(stat * boostTable[stage])`, matching Pokemon.getStat."""
    stage = jnp.clip(stage, -6, 6).astype(jnp.int32)
    stat = stat.astype(jnp.int32)
    num = jnp.where(stage >= 0, denom + stage, denom)
    den = jnp.where(stage >= 0, denom, denom - stage)
    return idiv(stat * num, den)


def modify(value, numerator, denominator=1):
    """Showdown's `Battle.modify`: `tr((tr(value*mod) + 2047) / 4096)`.

    `numerator/denominator` is the multiplier; it is first quantised to 4096ths,
    which is why e.g. chaining two 1.3x boosts is not the same as one 1.69x.

    int32 throughout: the largest intermediate is `damage * mod`, and damage stays
    in the low thousands, so there is roughly two orders of magnitude of headroom.
    """
    mod = (jnp.asarray(numerator, jnp.int32) * 4096) // jnp.asarray(denominator, jnp.int32)
    return chain_modify(value, mod)


def chain_modify(value, mod4096):
    """`modify` for a multiplier already expressed in 4096ths (4096 == x1)."""
    return (value.astype(jnp.int32) * jnp.asarray(mod4096, jnp.int32) + 2047) // 4096


def compute_hp(base, level, iv=31, ev=85):
    """`floor((2*base + iv + floor(ev/4)) * level / 100) + level + 10`.

    Shedinja (base HP 1) is the standard exception and is always 1.
    """
    base = base.astype(jnp.int32)
    level = level.astype(jnp.int32)
    val = ((2 * base + iv + ev // 4) * level) // 100 + level + 10
    return jnp.where(base == 1, 1, val).astype(jnp.int16)


def compute_stat(base, level, iv=31, ev=85, nature_num=1, nature_den=1):
    """`floor((floor((2*base + iv + floor(ev/4)) * level / 100) + 5) * nature)`."""
    base = base.astype(jnp.int32)
    level = level.astype(jnp.int32)
    val = ((2 * base + iv + ev // 4) * level) // 100 + 5
    return idiv(val * nature_num, nature_den).astype(jnp.int16)


def compute_all_stats(base_stats, level, ivs=None, evs=None):
    """Full `[..., 6]` stat line from `[..., 6]` base stats.

    Random Battles uses a flat 85 EVs / 31 IVs and a neutral nature, so those are
    the defaults; `ivs`/`evs` let the team builder zero Attack on special sets and
    Speed on Trick Room sets, exactly as Showdown's generator does.
    """
    if ivs is None:
        ivs = jnp.full(base_stats.shape, 31, jnp.int32)
    if evs is None:
        evs = jnp.full(base_stats.shape, 85, jnp.int32)
    hp = compute_hp(base_stats[..., C.HP], level, ivs[..., C.HP], evs[..., C.HP])
    others = compute_stat(base_stats[..., 1:], level[..., None], ivs[..., 1:], evs[..., 1:])
    return jnp.concatenate([hp[..., None], others], axis=-1)


def accuracy_stage_multiply(value, stage):
    """Accuracy/evasion stages use thirds rather than halves."""
    return boost_multiply(value, stage, denom=3)
