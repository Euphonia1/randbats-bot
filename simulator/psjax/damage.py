"""Type effectiveness and the damage formula.

Mirrors `BattleActions.getDamage` / `modifyDamage` from Showdown, including the
order of operations and every truncation point. The pipeline is:

    base = tr(tr(tr(tr(2*L/5 + 2) * power * atk) / def) / 50) + 2
    -> weather -> crit(x1.5) -> random(85..100%) -> STAB -> type -> burn
    -> final modifiers (Life Orb, Expert Belt, Multiscale, ...) -> min 1

Nothing here mutates state; `moves.py` applies the result.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from . import callbacks as cb
from . import consts as C
from .hooks import A, I
from .stats import boost_multiply, chain_modify, modify

# Multipliers are carried in 4096ths, as Showdown does, so that chained
# modifiers quantise identically.
M1 = 4096


def m(x: float) -> int:
    """A float multiplier as a 4096ths modifier (`m(1.5) == 6144`)."""
    return int(x * 4096)


class Attacker(NamedTuple):
    """Everything the damage formula needs about the attacking side."""
    level: jnp.ndarray
    stats: jnp.ndarray        # [6]
    boosts: jnp.ndarray       # [7]
    types: jnp.ndarray        # [2]
    base_types: jnp.ndarray   # [2] pre-Tera, for Stellar/Tera STAB rules
    ability: jnp.ndarray
    item: jnp.ndarray
    status: jnp.ndarray
    hp: jnp.ndarray
    maxhp: jnp.ndarray
    terastallized: jnp.ndarray
    tera_type: jnp.ndarray


class Defender(NamedTuple):
    stats: jnp.ndarray
    status: jnp.ndarray
    boosts: jnp.ndarray
    types: jnp.ndarray
    ability: jnp.ndarray
    item: jnp.ndarray
    hp: jnp.ndarray
    maxhp: jnp.ndarray
    terastallized: jnp.ndarray
    tera_type: jnp.ndarray
    nfe: jnp.ndarray          # eligible for Eviolite


class MoveCtx(NamedTuple):
    """A move after dynamic type/power/category resolution."""
    id: jnp.ndarray
    type: jnp.ndarray
    category: jnp.ndarray
    base_power: jnp.ndarray
    flags: jnp.ndarray
    crit_ratio: jnp.ndarray
    effect_cb: jnp.ndarray


def has_flag(flags, bit_name: str):
    return (flags & (1 << C.FLAG_BITS[bit_name])) != 0


# --- typing ------------------------------------------------------------------

def current_types(types, terastallized, tera_type):
    """Tera replaces the type line entirely (Stellar keeps the original types)."""
    tera_line = jnp.stack([tera_type, jnp.int8(C.TYPE_NONE)])
    use_tera = terastallized & (tera_type != C.STELLAR)
    return jnp.where(use_tera, tera_line, types)


def type_effectiveness(data, move_type, def_types, move_ctx, ignore_immunity_mask,
                       defender_ability, scrappy):
    """Return `(exponent, immune)`.

    The exponent counts doublings so the caller can apply Showdown's
    double-then-truncate-halve sequence rather than a float multiply.
    """
    eff = data["type_eff"][move_type, def_types]          # [2] float
    present = def_types != C.TYPE_NONE
    exp = jnp.where(eff == 2.0, 1, jnp.where(eff == 0.5, -1, 0))
    zero = (eff == 0.0) & present

    # Moves that pierce specific type immunities (Thousand Arrows vs Flying);
    # -1 in the mask means "all immunities".
    pierced = (ignore_immunity_mask == -1) | (
        (ignore_immunity_mask >> def_types.astype(jnp.int32)) & 1).astype(bool)
    # Scrappy / Foresight: Normal and Fighting hit Ghost.
    ghost_pierce = scrappy & (def_types == C.GHOST)
    zero = zero & ~pierced & ~ghost_pierce

    exp = jnp.where(present, exp, 0)

    # Freeze-Dry is super effective on Water regardless of the chart.
    from .effects import EFFECT_HANDLERS
    fd = EFFECT_HANDLERS.index("freezedry")
    is_fd = move_ctx.effect_cb == fd
    fd_exp = jnp.where(present & (def_types == C.WATER), 1, exp)
    exp = jnp.where(is_fd, fd_exp, exp)
    zero = jnp.where(is_fd & jnp.any(present & (def_types == C.WATER)), False, zero)

    # Flying Press counts as Flying on top of its own Fighting typing.
    fp = EFFECT_HANDLERS.index("flyingpress")
    is_fp = move_ctx.effect_cb == fp
    fly = data["type_eff"][C.FLYING, def_types]
    fly_exp = jnp.where(fly == 2.0, 1, jnp.where(fly == 0.5, -1, 0))
    exp = exp + jnp.where(is_fp & present, fly_exp, 0)
    zero = zero | (is_fp & present & (fly == 0.0))

    total_exp = jnp.sum(exp)
    immune = jnp.any(zero)

    # Levitate and friends: an ability-granted immunity to a whole type.
    ab_immune = data["ability_immune_type"][defender_ability]
    immune = immune | ((ab_immune == move_type) & (ab_immune != C.TYPE_NONE))

    return jnp.clip(total_exp, -6, 6), immune


# --- offensive / defensive stats ---------------------------------------------

def _attack_stat(data, atk: Attacker, dfn: Defender, mv: MoveCtx, is_crit,
                 defender_stats_source, weather):
    """The attacking stat after boosts and ability/item modifiers.

    A critical hit ignores the attacker's *negative* offensive boosts.
    """
    override = data["move_override_off_stat"][mv.id]
    use_target = data["move_override_off_pokemon"][mv.id]
    default_stat = jnp.where(mv.category == C.CAT_PHYSICAL, C.ATK, C.SPA)
    stat_idx = jnp.where(override != 0, override.astype(jnp.int32), default_stat)

    # Foul Play reads the target's Attack (and the target's boosts).
    stats = jnp.where(use_target, defender_stats_source, atk.stats)
    boosts = jnp.where(use_target, dfn.boosts, atk.boosts)

    raw = stats[stat_idx]
    stage = boosts[stat_idx - 1]                 # boost vector skips HP
    stage = jnp.where(is_crit, jnp.maximum(stage, 0), stage)
    # Unaware ignores the attacker's offensive boosts entirely.
    stage = jnp.where(dfn.ability == A.UNAWARE, 0, stage)
    stat = boost_multiply(raw, stage)

    ab, it = atk.ability, atk.item
    phys = mv.category == C.CAT_PHYSICAL
    statused = atk.status != C.STATUS_NONE
    mod = jnp.int32(M1)

    def apply(mod, cond, factor):
        return jnp.where(cond, chain_modify(mod, factor), mod)

    mod = apply(mod, ((ab == A.HUGEPOWER) | (ab == A.PUREPOWER)) & phys, m(2.0))
    mod = apply(mod, (ab == A.GUTS) & statused & phys, m(1.5))
    mod = apply(mod, (ab == A.HUSTLE) & phys, m(1.5))
    mod = apply(mod, (ab == A.TOXICBOOST) & phys &
                ((atk.status == C.PSN) | (atk.status == C.TOX)), m(1.5))
    mod = apply(mod, (ab == A.FLAREBOOST) & ~phys & (atk.status == C.BRN), m(1.5))
    sun = (weather == C.SUN) | (weather == C.HARSH_SUN)
    mod = apply(mod, (ab == A.SOLARPOWER) & ~phys & sun, m(1.5))
    mod = apply(mod, (ab == A.DEFEATIST) & (atk.hp * 2 <= atk.maxhp), m(0.5))
    mod = apply(mod, (ab == A.WATERBUBBLE) & (mv.type == C.WATER), m(2.0))
    # Type-specialist abilities.
    for ability, typ in ((A.TRANSISTOR, C.ELECTRIC), (A.DRAGONSMAW, C.DRAGON),
                         (A.ROCKYPAYLOAD, C.ROCK), (A.STEELWORKER, C.STEEL)):
        mod = apply(mod, (ab == ability) & (mv.type == typ), m(1.5))

    # Defender-side abilities hooked on onSourceModifyAtk/SpA.
    d_ab = dfn.ability
    mod = apply(mod, (d_ab == A.THICKFAT) &
                ((mv.type == C.FIRE) | (mv.type == C.ICE)), m(0.5))
    mod = apply(mod, (d_ab == A.HEATPROOF) & (mv.type == C.FIRE), m(0.5))
    mod = apply(mod, (d_ab == A.WATERBUBBLE) & (mv.type == C.FIRE), m(0.5))
    mod = apply(mod, (d_ab == A.PURIFYINGSALT) & (mv.type == C.GHOST), m(0.5))

    mod = apply(mod, (it == I.CHOICEBAND) & phys, m(1.5))
    mod = apply(mod, (it == I.CHOICESPECS) & ~phys, m(1.5))
    mod = apply(mod, (it == I.LIGHTBALL), m(2.0))  # Pikachu-only in practice
    mod = apply(mod, (it == I.THICKCLUB) & phys, m(2.0))

    return jnp.maximum(chain_modify(stat, mod), 1)


def _defense_stat(data, atk: Attacker, dfn: Defender, mv: MoveCtx, is_crit,
                  weather, terrain):
    """The defending stat after boosts and ability/item modifiers.

    A critical hit ignores the defender's *positive* defensive boosts.
    """
    override = data["move_override_def_stat"][mv.id]
    def_cat = data["move_defensive_category"][mv.id]
    cat = jnp.where(def_cat != 0, def_cat, mv.category)
    default_stat = jnp.where(cat == C.CAT_PHYSICAL, C.DEF, C.SPD)
    stat_idx = jnp.where(override != 0, override.astype(jnp.int32), default_stat)

    raw = dfn.stats[stat_idx]
    stage = dfn.boosts[stat_idx - 1]
    stage = jnp.where(is_crit, jnp.minimum(stage, 0), stage)
    stage = jnp.where(atk.ability == A.UNAWARE, 0, stage)
    stage = jnp.where(data["move_ignore_defensive"][mv.id], jnp.minimum(stage, 0), stage)
    stat = boost_multiply(raw, stage)

    ab, it = dfn.ability, dfn.item
    phys = cat == C.CAT_PHYSICAL
    mod = jnp.int32(M1)

    def apply(mod, cond, factor):
        return jnp.where(cond, chain_modify(mod, factor), mod)

    mod = apply(mod, (ab == A.FURCOAT) & phys, m(2.0))
    mod = apply(mod, (ab == A.GRASSPELT) & phys & (terrain == C.GRASSY_TERRAIN), m(1.5))
    mod = apply(mod, (ab == A.MARVELSCALE) & phys & (dfn.status != C.STATUS_NONE), m(1.5))
    mod = apply(mod, (it == I.ASSAULTVEST) & ~phys, m(1.5))
    mod = apply(mod, (it == I.EVIOLITE) & dfn.nfe, m(1.5))
    # Snow raises the Defense of Ice types; Sand raises Sp. Def of Rock types.
    is_ice = jnp.any(dfn.types == C.ICE)
    is_rock = jnp.any(dfn.types == C.ROCK)
    mod = apply(mod, (weather == C.SNOW) & is_ice & phys, m(1.5))
    mod = apply(mod, (weather == C.SAND) & is_rock & ~phys, m(1.5))

    return jnp.maximum(chain_modify(stat, mod), 1)


# --- base power modifiers ----------------------------------------------------

def _base_power_modifiers(data, atk: Attacker, dfn: Defender, mv: MoveCtx,
                          terrain, has_secondary, analytic_ok, fainted_count,
                          bp_cb_mod, grounded_user, grounded_target):
    """Ability/item multipliers applied to base power before the main formula."""
    ab, it = atk.ability, atk.item
    mod = jnp.int32(M1)

    def apply(mod, cond, factor):
        return jnp.where(cond, chain_modify(mod, factor), mod)

    mod = apply(mod, (ab == A.TECHNICIAN) & (mv.base_power <= 60), m(1.5))
    mod = apply(mod, (ab == A.TOUGHCLAWS) & has_flag(mv.flags, "contact"), m(1.3))
    mod = apply(mod, (ab == A.STRONGJAW) & has_flag(mv.flags, "bite"), m(1.5))
    mod = apply(mod, (ab == A.MEGALAUNCHER) & has_flag(mv.flags, "bullet"), m(1.5))
    mod = apply(mod, (ab == A.IRONFIST) & has_flag(mv.flags, "punch"), m(1.2))
    mod = apply(mod, (ab == A.SHARPNESS) & has_flag(mv.flags, "slicing"), m(1.5))
    mod = apply(mod, (ab == A.PUNKROCK) & has_flag(mv.flags, "sound"), m(1.3))
    mod = apply(mod, (ab == A.RECKLESS) & (data["move_recoil"][mv.id, 0] > 0), m(1.2))
    mod = apply(mod, (ab == A.SHEERFORCE) & has_secondary, m(1.3))
    # Analytic only applies when the user moves last.
    mod = apply(mod, (ab == A.ANALYTIC) & analytic_ok, m(1.3))
    # Supreme Overlord: +10% per fallen ally, as a single 4096ths modifier.
    mod = jnp.where(ab == A.SUPREMEOVERLORD,
                    chain_modify(mod, 4096 + 819 * jnp.clip(fainted_count, 0, 5)), mod)
    # Pinch abilities: Overgrow/Blaze/Torrent/Swarm at <= 1/3 HP.
    pinch = atk.hp * 3 <= atk.maxhp
    for ability, typ in ((A.OVERGROW, C.GRASS), (A.BLAZE, C.FIRE),
                         (A.TORRENT, C.WATER), (A.SWARM, C.BUG)):
        mod = apply(mod, (ab == ability) & pinch & (mv.type == typ), m(1.5))

    # Type-boosting held items (Magnet, Mystic Water, ...).
    boost_type = data["item_boost_type"][it]
    mod = jnp.where(boost_type == mv.type,
                    chain_modify(mod, data["item_boost_mod"][it]), mod)
    mod = apply(mod, (it == I.MUSCLEBAND) & (mv.category == C.CAT_PHYSICAL), m(1.1))
    mod = apply(mod, (it == I.WISEGLASSES) & (mv.category == C.CAT_SPECIAL), m(1.1))
    mod = apply(mod, (it == I.PUNCHINGGLOVE) & has_flag(mv.flags, "punch"), m(1.1))

    # Terrain boosts require the user to be grounded; Misty Terrain instead
    # weakens Dragon moves aimed at a grounded target.
    mod = apply(mod, grounded_user & (terrain == C.ELECTRIC_TERRAIN) &
                (mv.type == C.ELECTRIC), m(1.3))
    mod = apply(mod, grounded_user & (terrain == C.GRASSY_TERRAIN) &
                (mv.type == C.GRASS), m(1.3))
    mod = apply(mod, grounded_user & (terrain == C.PSYCHIC_TERRAIN) &
                (mv.type == C.PSYCHIC), m(1.3))
    mod = apply(mod, grounded_target & (terrain == C.MISTY_TERRAIN) &
                (mv.type == C.DRAGON), m(0.5))

    # The move's own onBasePower callback, resolved once by the caller.
    mod = chain_modify(mod, bp_cb_mod)
    return mod


# --- weather -----------------------------------------------------------------

def weather_modifier(weather, move_type, utility_umbrella):
    """Sun/rain scaling of Fire and Water moves; 0 means the move fizzles."""
    active = jnp.logical_not(utility_umbrella)
    sun = ((weather == C.SUN) | (weather == C.HARSH_SUN)) & active
    rain = ((weather == C.RAIN) | (weather == C.HEAVY_RAIN)) & active
    mod = jnp.int32(M1)
    mod = jnp.where(sun & (move_type == C.FIRE), m(1.5), mod)
    mod = jnp.where(sun & (move_type == C.WATER), m(0.5), mod)
    mod = jnp.where(rain & (move_type == C.WATER), m(1.5), mod)
    mod = jnp.where(rain & (move_type == C.FIRE), m(0.5), mod)
    # The primal weathers null out the opposing type completely.
    mod = jnp.where((weather == C.HARSH_SUN) & (move_type == C.WATER), 0, mod)
    mod = jnp.where((weather == C.HEAVY_RAIN) & (move_type == C.FIRE), 0, mod)
    return mod


# --- STAB --------------------------------------------------------------------

def stab_modifier(atk: Attacker, move_type):
    """1.5x normally, 2x for a Tera type matching an original type, 1.2x Stellar."""
    base_match = jnp.any(atk.base_types == move_type)
    tera_match = atk.terastallized & (atk.tera_type == move_type)
    is_stab = base_match | tera_match

    stellar = atk.terastallized & (atk.tera_type == C.STELLAR)
    normal = jnp.where(is_stab, m(1.5), M1)
    # Tera into a type you already had gives the full 2x.
    normal = jnp.where(tera_match & base_match, m(2.0), normal)
    # Adaptability raises 1.5x to 2x, and stacks with Tera to 2.25x.
    adaptable = (atk.ability == A.ADAPTABILITY) & is_stab
    normal = jnp.where(adaptable,
                       jnp.where(tera_match & base_match, m(2.25), m(2.0)), normal)
    # Stellar: 2x on your own types, a flat 1.2x on everything else.
    stellar_mod = jnp.where(base_match, m(2.0), 4915)
    return jnp.where(stellar, stellar_mod, normal).astype(jnp.int32)


# --- final modifiers ---------------------------------------------------------

def _final_modifiers(data, atk: Attacker, dfn: Defender, mv: MoveCtx, type_exp,
                     side_conditions, is_crit):
    ab, it = atk.ability, atk.item
    d_ab, d_it = dfn.ability, dfn.item
    mod = jnp.int32(M1)

    def apply(mod, cond, factor):
        return jnp.where(cond, chain_modify(mod, factor), mod)

    super_eff = type_exp > 0
    resisted = type_exp < 0

    # Screens: halve damage unless the hit is a crit.
    phys = mv.category == C.CAT_PHYSICAL
    reflect = side_conditions[C.SC_REFLECT] > 0
    lightscreen = side_conditions[C.SC_LIGHTSCREEN] > 0
    veil = side_conditions[C.SC_AURORAVEIL] > 0
    screened = ((reflect & phys) | (lightscreen & ~phys) | veil) & ~is_crit
    screened = screened & (ab != A.INFILTRATOR)
    mod = apply(mod, screened, m(0.5))

    # Attacker-side damage abilities.
    mod = apply(mod, (ab == A.TINTEDLENS) & resisted, m(2.0))
    mod = apply(mod, (ab == A.NEUROFORCE) & super_eff, m(1.25))
    mod = apply(mod, (ab == A.SNIPER) & is_crit, m(1.5))

    # Defender-side reduction abilities.
    mod = apply(mod, ((d_ab == A.SOLIDROCK) | (d_ab == A.FILTER) |
                      (d_ab == A.PRISMARMOR)) & super_eff, m(0.75))
    mod = apply(mod, ((d_ab == A.MULTISCALE) | (d_ab == A.SHADOWSHIELD)) &
                (dfn.hp == dfn.maxhp), m(0.5))
    mod = apply(mod, (d_ab == A.ICESCALES) & (mv.category == C.CAT_SPECIAL), m(0.5))
    mod = apply(mod, (d_ab == A.FLUFFY) & has_flag(mv.flags, "contact"), m(0.5))
    mod = apply(mod, (d_ab == A.FLUFFY) & (mv.type == C.FIRE), m(2.0))
    mod = apply(mod, (d_ab == A.PUNKROCK) & has_flag(mv.flags, "sound"), m(0.5))

    # Items.
    mod = apply(mod, (it == I.LIFEORB), m(1.3))
    mod = apply(mod, (it == I.EXPERTBELT) & super_eff, m(1.2))
    # A resist berry halves a super-effective hit of its type (Chilan: any Normal).
    berry_type = data["item_resist_type"][d_it]
    berry_ok = (berry_type == mv.type) & (berry_type != C.TYPE_NONE) & \
               (super_eff | (berry_type == C.NORMAL))
    mod = apply(mod, berry_ok, m(0.5))
    return mod


# --- the formula -------------------------------------------------------------

def calc_damage(data, atk: Attacker, dfn: Defender, mv: MoveCtx, *,
                is_crit, damage_roll, weather, terrain, side_conditions,
                type_exp, bp_cb_mod=4096, grounded_user=True, grounded_target=True,
                has_secondary=False, utility_umbrella=False,
                analytic_ok=False, fainted_count=0, defender_stats_source=None):
    """Damage for one hit. `damage_roll` is 0..15, matching Showdown's `random(16)`.

    `type_exp` comes from `type_effectiveness`; immunity is handled by the caller
    so that ability-based absorption can run its own side effects.
    """
    if defender_stats_source is None:
        defender_stats_source = dfn.stats

    bp_mod = _base_power_modifiers(data, atk, dfn, mv, terrain, has_secondary,
                                   analytic_ok, fainted_count, bp_cb_mod,
                                   grounded_user, grounded_target)
    power = jnp.maximum(chain_modify(mv.base_power, bp_mod), 1)

    attack = _attack_stat(data, atk, dfn, mv, is_crit, defender_stats_source, weather)
    defense = _defense_stat(data, atk, dfn, mv, is_crit, weather, terrain)

    level = atk.level.astype(jnp.int32)
    base = ((2 * level) // 5) + 2
    base = (base * power * attack) // defense
    base = base // 50 + 2

    base = chain_modify(base, weather_modifier(weather, mv.type, utility_umbrella))
    base = jnp.where(is_crit, (base * 3) // 2, base)
    # Showdown: tr(tr(damage * (100 - random(16))) / 100)
    base = (base * (100 - damage_roll)) // 100
    base = chain_modify(base, stab_modifier(atk, mv.type))

    # Type effectiveness: double up, then halve with truncation, one step at a time.
    def double(i, v):
        return jnp.where(i < type_exp, v * 2, v)

    def halve(i, v):
        return jnp.where(i < -type_exp, v // 2, v)

    base = jax.lax.fori_loop(0, 6, double, base)
    base = jax.lax.fori_loop(0, 6, halve, base)

    # Burn halves physical damage, unless the attacker has Guts.
    burned = (atk.status == C.BRN) & (mv.category == C.CAT_PHYSICAL) & \
             (atk.ability != A.GUTS)
    base = jnp.where(burned, chain_modify(base, m(0.5)), base)

    base = chain_modify(base, _final_modifiers(
        data, atk, dfn, mv, type_exp, side_conditions, is_crit))

    return jnp.maximum(base, 1).astype(jnp.int32)


# --- accuracy and crits ------------------------------------------------------

CRIT_RATES = jnp.array([24, 8, 2, 1, 1], jnp.int32)   # 1-in-N by crit stage


def crit_chance_stage(data, mv: MoveCtx, atk: Attacker):
    """Crit stage: base ratio plus Focus Energy, Scope Lens, Super Luck, ..."""
    stage = data["move_crit_ratio"][mv.id].astype(jnp.int32) - 1
    stage = stage + jnp.where(atk.item == I.SCOPELENS, 1, 0)
    stage = stage + jnp.where(atk.item == I.RAZORCLAW, 1, 0)
    stage = stage + jnp.where(atk.ability == A.SUPERLUCK, 1, 0)
    return jnp.clip(stage, 0, 4)


def accuracy_check(data, mv: MoveCtx, atk: Attacker, dfn: Defender,
                   acc_boost, eva_boost, roll, gravity):
    """True if the move connects. `roll` is uniform in [0, 100)."""
    base_acc = data["move_accuracy"][mv.id].astype(jnp.int32)
    always = base_acc < 0

    # Accuracy and evasion share one stage table; evasion counts against you.
    stage = jnp.clip(acc_boost - eva_boost, -6, 6)
    stage = jnp.where(data["move_ignore_evasion"][mv.id], jnp.maximum(stage, 0), stage)
    stage = jnp.where(dfn.ability == A.UNAWARE, acc_boost, stage)
    num = jnp.where(stage >= 0, 3 + stage, 3)
    den = jnp.where(stage >= 0, 3, 3 - stage)
    acc = (base_acc * num) // den

    mod = jnp.int32(M1)
    mod = jnp.where(atk.ability == A.COMPOUNDEYES, chain_modify(mod, m(1.3)), mod)
    mod = jnp.where(atk.item == I.WIDELENS, chain_modify(mod, m(1.1)), mod)
    mod = jnp.where(dfn.item == I.BRIGHTPOWDER, chain_modify(mod, m(0.9)), mod)
    mod = jnp.where(gravity > 0, chain_modify(mod, m(5.0 / 3.0)), mod)
    acc = chain_modify(acc, mod)

    return always | (roll < acc)


# --- dynamic move resolution -------------------------------------------------

def resolve_move_ctx(data, move_id, cb_ctx: "cb.CbCtx") -> MoveCtx:
    """Apply the move's type, base-power and category callbacks.

    This is what turns the static compiled row into the move as actually used
    this turn: Weather Ball's type, Low Kick's power, Tera Blast's category.
    """
    move_id = jnp.asarray(move_id, jnp.int32)
    declared_type = data["move_type"][move_id]
    declared_bp = data["move_base_power"][move_id].astype(jnp.int32)
    category = data["move_category"][move_id]

    ctx = cb_ctx._replace(base_power=declared_bp, move_type=declared_type)
    mv_type = cb.modify_type(data["move_type_cb"][move_id], ctx)
    ctx = ctx._replace(move_type=mv_type)
    base_power = cb.base_power_replace(data["move_bp_replace"][move_id], ctx)

    # Tera Blast and Photon Geyser become physical when the user's Attack is
    # higher than its Sp. Atk (boosts counted, ability/item modifiers not).
    from .effects import BP_REPLACE_HANDLERS, EFFECT_HANDLERS
    is_terablast = data["move_bp_replace"][move_id] == BP_REPLACE_HANDLERS.index("terablast")
    is_photon = data["move_effect_cb"][move_id] == EFFECT_HANDLERS.index("photongeyser")
    physical_switch = ((is_terablast & cb_ctx.terastallized) | is_photon) & \
                      (cb_ctx.off_atk > cb_ctx.off_spa)
    category = jnp.where(physical_switch, C.CAT_PHYSICAL, category)

    return MoveCtx(
        id=move_id, type=mv_type, category=category,
        base_power=jnp.maximum(base_power, 0), flags=data["move_flags"][move_id],
        crit_ratio=data["move_crit_ratio"][move_id],
        effect_cb=data["move_effect_cb"][move_id],
    )


def scale_power_for_hit(data, move_id, mv: MoveCtx, hit_number) -> MoveCtx:
    """Triple Kick and Triple Axel gain power on each successive hit.

    They are the only base-power callbacks that depend on the hit index, so the
    multi-hit loop applies this instead of re-running the whole dispatch.
    """
    from .effects import BP_REPLACE_HANDLERS
    scaling = data["move_bp_replace"][move_id] == \
        BP_REPLACE_HANDLERS.index("multihit_scaling")
    power = jnp.where(scaling, mv.base_power * jnp.maximum(hit_number, 1),
                      mv.base_power)
    return mv._replace(base_power=power)
