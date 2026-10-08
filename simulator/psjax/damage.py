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
import numpy as np

from . import callbacks as cb
from . import consts as C
from .data import move_index, species_index
from .effects import BP_REPLACE_HANDLERS, EFFECT_HANDLERS, TYPE_HANDLERS
from .hooks import A, I, SPECIES_BOOST_ITEMS
from .stats import boost_multiply, chain_modify, floordiv, idiv, modify

# Multipliers are carried in 4096ths, as Showdown does, so that chained
# modifiers quantise identically.
M1 = 4096


def m(x: float) -> int:
    """A float multiplier as a 4096ths modifier (`m(1.5) == 6144`)."""
    return int(x * 4096)


#: What most of Showdown's "1.3x" effects really are: `[5325, 4096]`, a point
#: above `m(1.3)`. (Life Orb is the exception, at 5324.)
M13 = 5325


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
    boosted_stat: jnp.ndarray   # Protosynthesis / Quark Drive; -1 when inactive
    slow_start: jnp.ndarray     # Slow Start still counting down
    charged: jnp.ndarray = False    # Charge: the next Electric move hits twice as hard
    #: National Dex number, for the items only one species can use (Soul Dew,
    #: Adamant Crystal, Ogerpon's masks); 0 when unknown.
    species_num: jnp.ndarray = 0
    flash_fire: jnp.ndarray = False   # Flash Fire has absorbed a Fire move


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
    boosted_stat: jnp.ndarray
    glaive_rush: jnp.ndarray = False   # used Glaive Rush: takes double damage


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
    # Shift, then test the low bit: `futuremove` is bit 31, which as a mask
    # would not fit in the int32 the flags are loaded as.
    return (jnp.right_shift(flags, C.FLAG_BITS[bit_name]) & 1) != 0


# --- typing ------------------------------------------------------------------

def current_types(types, terastallized, tera_type):
    """Tera replaces the type line entirely (Stellar keeps the original types)."""
    tera_line = jnp.stack([tera_type, jnp.int8(C.TYPE_NONE)])
    use_tera = terastallized & (tera_type != C.STELLAR)
    return jnp.where(use_tera, tera_line, types)


def type_effectiveness(data, move_type, def_types, move_ctx, ignore_immunity_mask,
                       defender_ability, scrappy, def_terastallized=False,
                       def_grounded=True, def_full_hp=False):
    """Return `(exponent, immune)`.

    The exponent counts doublings so the caller can apply Showdown's
    double-then-truncate-halve sequence rather than a float multiply.
    """
    eff = data["type_eff"][move_type, def_types]          # [2] float
    present = def_types != C.TYPE_NONE
    exp = jnp.where(eff == 2.0, 1, jnp.where(eff == 0.5, -1, 0))
    # A Ground move goes by whether its target is on the ground at all
    # (`def_grounded`, Showdown's `isGrounded`) rather than by the chart: an Air
    # Balloon or Magnet Rise lifts anything, and Gravity, Smack Down or Ingrain
    # bring a Flying type down to be hit neutrally.
    ground = move_type == C.GROUND
    zero = (eff == 0.0) & present & jnp.logical_not(ground)

    # Moves that ignore the immunities to their own type (Thousand Arrows:
    # Ground, so Flying types are hit); -1 in the mask means "all immunities".
    # The mask is keyed by the attacking type, as Showdown's `ignoreImmunity`.
    pierced = (ignore_immunity_mask == -1) | (
        (ignore_immunity_mask >> move_type.astype(jnp.int32)) & 1).astype(bool)
    # Scrappy / Foresight: Normal and Fighting hit Ghost.
    ghost_pierce = scrappy & (def_types == C.GHOST)
    zero = zero & ~pierced & ~ghost_pierce

    exp = jnp.where(present, exp, 0)

    # Freeze-Dry is super effective on Water regardless of the chart.
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

    # Thousand Arrows hits an airborne Flying type for neutral damage, whatever
    # its other type -- Showdown zeroes every type's contribution.
    ta = move_ctx.effect_cb == EFFECT_HANDLERS.index("thousandarrows")
    airborne_flyer = ta & jnp.any(def_types == C.FLYING) & jnp.logical_not(def_grounded)
    total_exp = jnp.where(airborne_flyer, 0, total_exp)

    # A Stellar move is super effective on anything Terastallized and neutral on
    # everything else.
    stellar = move_type == C.STELLAR
    total_exp = jnp.where(stellar, jnp.where(def_terastallized, 1, 0), total_exp)
    immune = immune & jnp.logical_not(stellar)

    # Levitate and friends: an ability-granted immunity to a whole type. A move
    # that ignores that type's immunities goes through it too -- Showdown's
    # `runImmunity` returns before Levitate is ever consulted. Levitate's own is
    # part of being off the ground, so Gravity takes it away like the rest.
    ab_immune = data["ability_immune_type"][defender_ability]
    immune = immune | ((ab_immune == move_type) & (ab_immune != C.TYPE_NONE) &
                       jnp.logical_not(pierced) & jnp.logical_not(ground))
    immune = immune | (ground & jnp.logical_not(def_grounded) & jnp.logical_not(pierced))

    # Tera Shell: at full HP every damaging hit is not very effective.
    shell = (defender_ability == A.TERASHELL) & def_full_hp & (total_exp >= 0) & \
        jnp.logical_not(immune) & (move_ctx.category != C.CAT_STATUS)
    total_exp = jnp.where(shell, -1, total_exp)

    return jnp.clip(total_exp, -6, 6), immune


# --- offensive / defensive stats ---------------------------------------------

def _pick_stat(stats, boosts, stat_idx):
    """Read stat `stat_idx` (1..5) and its boost stage without a dynamic gather.

    `stat_idx` depends on the move's category, which is traced, so indexing
    directly would lower to a gather. Selecting between five statically-read
    values instead is what lets `vmap` compile this in seconds rather than
    minutes.
    """
    value = stats[C.ATK]
    stage = boosts[C.B_ATK]
    for stat, boost in ((C.DEF, C.B_DEF), (C.SPA, C.B_SPA),
                        (C.SPD, C.B_SPD), (C.SPE, C.B_SPE)):
        hit = stat_idx == stat
        value = jnp.where(hit, stats[stat], value)
        stage = jnp.where(hit, boosts[boost], stage)
    return value, stage


def _attack_stat(data, atk: Attacker, dfn: Defender, mv: MoveCtx, is_crit,
                 defender_stats_source, weather, terrain, target_switched_in=False):
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

    raw, stage = _pick_stat(stats, boosts, stat_idx)
    stage = jnp.where(is_crit, jnp.maximum(stage, 0), stage)
    # Unaware ignores the attacker's offensive boosts entirely.
    stage = jnp.where(dfn.ability == A.UNAWARE, 0, stage)
    stat = boost_multiply(raw, stage)

    ab, it = atk.ability, atk.item
    # Showdown runs onModifyAtk / onModifySpA by the move's category, whatever
    # stat it attacks with: Body Press reads Defense but is still a physical
    # move, so Choice Band, Huge Power and Tablets of Ruin all apply to it.
    phys = mv.category == C.CAT_PHYSICAL
    special = mv.category == C.CAT_SPECIAL
    statused = atk.status != C.STATUS_NONE
    mod = jnp.int32(M1)

    def apply(mod, cond, factor):
        return jnp.where(cond, chain_modify(mod, factor), mod)

    mod = apply(mod, ((ab == A.HUGEPOWER) | (ab == A.PUREPOWER)) & phys, m(2.0))
    mod = apply(mod, (ab == A.GUTS) & statused & phys, m(1.5))
    mod = apply(mod, (ab == A.HUSTLE) & phys, m(1.5))
    # Pinch abilities: Overgrow/Blaze/Torrent/Swarm at <= 1/3 HP. Showdown hooks
    # these on the attacking stat, not on base power.
    pinch = atk.hp * 3 <= atk.maxhp
    for ability, typ in ((A.OVERGROW, C.GRASS), (A.BLAZE, C.FIRE),
                         (A.TORRENT, C.WATER), (A.SWARM, C.BUG)):
        mod = apply(mod, (ab == ability) & pinch & (mv.type == typ), m(1.5))
    sun = (weather == C.SUN) | (weather == C.HARSH_SUN)
    mod = apply(mod, (ab == A.SOLARPOWER) & special & sun, m(1.5))
    # Orichalcum Pulse and Hadron Engine are 5461/4096, not the 1.3 they are
    # usually quoted as.
    mod = apply(mod, (ab == A.ORICHALCUMPULSE) & phys & sun, 5461)
    mod = apply(mod, (ab == A.HADRONENGINE) & special &
                (terrain == C.ELECTRIC_TERRAIN), 5461)
    # Slow Start halves Attack as well as Speed while it is counting down.
    mod = apply(mod, atk.slow_start & phys, m(0.5))
    mod = apply(mod, (ab == A.DEFEATIST) & (atk.hp * 2 <= atk.maxhp), m(0.5))
    mod = apply(mod, (ab == A.WATERBUBBLE) & (mv.type == C.WATER), m(2.0))
    # Type-specialist abilities.
    for ability, typ in ((A.DRAGONSMAW, C.DRAGON),
                         (A.ROCKYPAYLOAD, C.ROCK), (A.STEELWORKER, C.STEEL)):
        mod = apply(mod, (ab == ability) & (mv.type == typ), m(1.5))
    # Transistor was cut to 5325/4096 in Gen 9.
    mod = apply(mod, (ab == A.TRANSISTOR) & (mv.type == C.ELECTRIC), M13)
    # Flash Fire, once it has absorbed a Fire move, powers up the holder's own.
    mod = apply(mod, atk.flash_fire & (ab == A.FLASHFIRE) & (mv.type == C.FIRE) &
                (phys | special), m(1.5))
    # Stakeout doubles the attacking stat against a foe that just switched in.
    mod = apply(mod, (ab == A.STAKEOUT) & target_switched_in & (phys | special), m(2.0))

    # Defender-side abilities hooked on onSourceModifyAtk/SpA.
    d_ab = dfn.ability
    mod = apply(mod, (d_ab == A.THICKFAT) &
                ((mv.type == C.FIRE) | (mv.type == C.ICE)), m(0.5))
    mod = apply(mod, (d_ab == A.HEATPROOF) & (mv.type == C.FIRE), m(0.5))
    mod = apply(mod, (d_ab == A.WATERBUBBLE) & (mv.type == C.FIRE), m(0.5))
    mod = apply(mod, (d_ab == A.PURIFYINGSALT) & (mv.type == C.GHOST), m(0.5))

    # Protosynthesis / Quark Drive: x1.3 on the stat picked when it activated --
    # hooked on onModifyAtk / onModifySpA too, so by category.
    mod = apply(mod, atk.boosted_stat == jnp.where(phys, C.ATK, C.SPA), M13)

    mod = apply(mod, (it == I.CHOICEBAND) & phys, m(1.5))
    mod = apply(mod, (it == I.CHOICESPECS) & special, m(1.5))
    mod = apply(mod, (it == I.LIGHTBALL), m(2.0))  # Pikachu-only in practice
    mod = apply(mod, (it == I.THICKCLUB) & phys, m(2.0))

    # The Ruin abilities weaken everyone else's stat while their holder is out:
    # Tablets of Ruin their Attack, Vessel of Ruin their Sp. Atk.
    mod = apply(mod, phys & (d_ab == A.TABLETSOFRUIN) & (ab != A.TABLETSOFRUIN), m(0.75))
    mod = apply(mod, special & (d_ab == A.VESSELOFRUIN) & (ab != A.VESSELOFRUIN), m(0.75))

    return jnp.maximum(chain_modify(stat, mod), 1)


def _defense_stat(data, atk: Attacker, dfn: Defender, mv: MoveCtx, is_crit,
                  weather, terrain, def_stage_delta=0, def_raw=None):
    """The defending stat after boosts and ability/item modifiers.

    A critical hit ignores the defender's *positive* defensive boosts.
    `def_stage_delta` moves the Defense stage for one hit of a multi-hit move
    (Weak Armor, Stamina), and `def_raw` replaces the unboosted Defense (Ice
    Face broken by an earlier hit); only those vary with them, so a vmap over
    the hits does not drag the modifier chain along.
    """
    override = data["move_override_def_stat"][mv.id]
    def_cat = data["move_defensive_category"][mv.id]
    cat = jnp.where(def_cat != 0, def_cat, mv.category)
    default_stat = jnp.where(cat == C.CAT_PHYSICAL, C.DEF, C.SPD)
    stat_idx = jnp.where(override != 0, override.astype(jnp.int32), default_stat)

    raw, stage = _pick_stat(dfn.stats, dfn.boosts, stat_idx)
    if def_raw is not None:
        raw = jnp.where(stat_idx == C.DEF, def_raw, raw)
    stage = jnp.where(stat_idx == C.DEF, jnp.clip(stage + def_stage_delta, -6, 6), stage)
    stage = jnp.where(is_crit, jnp.minimum(stage, 0), stage)
    stage = jnp.where(atk.ability == A.UNAWARE, 0, stage)
    # Sacred Sword and Darkest Lariat ignore every Defense change, drops too.
    stage = jnp.where(data["move_ignore_defensive"][mv.id], 0, stage)
    stat = boost_multiply(raw, stage)

    ab, it = dfn.ability, dfn.item
    # Keyed on the stat the move hits (onModifyDef / onModifySpD), so Psyshock,
    # a special move aimed at Defense, meets Fur Coat but not Assault Vest.
    phys = stat_idx == C.DEF
    # Snow raises the Defense of Ice types, Sand the Sp. Def of Rock types --
    # Showdown's `this.modify` on the stat itself, ahead of the chained
    # modifiers below, which then apply to the raised value.
    is_ice = jnp.any(dfn.types == C.ICE)
    is_rock = jnp.any(dfn.types == C.ROCK)
    weathered = ((weather == C.SNOW) & is_ice & phys) | ((weather == C.SAND) & is_rock & ~phys)
    stat = jnp.where(weathered, chain_modify(stat, m(1.5)), stat)
    mod = jnp.int32(M1)

    def apply(mod, cond, factor):
        return jnp.where(cond, chain_modify(mod, factor), mod)

    mod = apply(mod, (ab == A.FURCOAT) & phys, m(2.0))
    mod = apply(mod, (ab == A.GRASSPELT) & phys & (terrain == C.GRASSY_TERRAIN), m(1.5))
    mod = apply(mod, (ab == A.MARVELSCALE) & phys & (dfn.status != C.STATUS_NONE), m(1.5))
    mod = apply(mod, dfn.boosted_stat == stat_idx, M13)
    mod = apply(mod, (it == I.ASSAULTVEST) & ~phys, m(1.5))
    mod = apply(mod, (it == I.EVIOLITE) & dfn.nfe, m(1.5))
    # Sword of Ruin weakens everyone else's Defense, Beads of Ruin their Sp. Def.
    a_ab = atk.ability
    mod = apply(mod, phys & (a_ab == A.SWORDOFRUIN) & (ab != A.SWORDOFRUIN), m(0.75))
    mod = apply(mod, ~phys & (a_ab == A.BEADSOFRUIN) & (ab != A.BEADSOFRUIN), m(0.75))

    return jnp.maximum(chain_modify(stat, mod), 1)


# --- base power modifiers ----------------------------------------------------

def _base_power_modifiers(data, atk: Attacker, dfn: Defender, mv: MoveCtx,
                          terrain, has_secondary, analytic_ok, fainted_count,
                          bp_cb_mod, grounded_user, grounded_target,
                          target_switched_in, technician_power=None, weather=None,
                          alt_terrain=None):
    """Ability/item multipliers applied to base power before the main formula.

    With `alt_terrain`, also the chain under that terrain instead, as a second
    result: only the terrain links are worked out twice.
    """
    ab, it = atk.ability, atk.item
    mod = jnp.int32(M1)

    def apply(mod, cond, factor):
        return jnp.where(cond, chain_modify(mod, factor), mod)

    # Technician reads `technician_power` rather than `mv.base_power` so a
    # per-hit scaled power does not drag this whole chain up a rank -- see the
    # note on `calc_damage`.
    tech_power = mv.base_power if technician_power is None else technician_power
    mod = apply(mod, (ab == A.TECHNICIAN) & (tech_power <= 60), m(1.5))
    mod = apply(mod, (ab == A.TOUGHCLAWS) & has_flag(mv.flags, "contact"), M13)
    mod = apply(mod, (ab == A.STRONGJAW) & has_flag(mv.flags, "bite"), m(1.5))
    mod = apply(mod, (ab == A.MEGALAUNCHER) & data["move_pulse"][mv.id], m(1.5))
    mod = apply(mod, (ab == A.IRONFIST) & has_flag(mv.flags, "punch"), m(1.2))
    mod = apply(mod, (ab == A.SHARPNESS) & has_flag(mv.flags, "slicing"), m(1.5))
    mod = apply(mod, (ab == A.PUNKROCK) & has_flag(mv.flags, "sound"), M13)
    mod = apply(mod, (ab == A.STEELYSPIRIT) & (mv.type == C.STEEL), m(1.5))
    if weather is not None:
        # Sand Force: Rock, Ground and Steel moves in a sandstorm.
        mod = apply(mod, (ab == A.SANDFORCE) & (weather == C.SAND) &
                    ((mv.type == C.ROCK) | (mv.type == C.GROUND) | (mv.type == C.STEEL)), M13)
    # The "-ate" abilities turn Normal moves into their type for +20% power; the
    # type change itself happens in resolve_move_ctx.
    mod = apply(mod, (data["ability_ate_type"][ab] != C.TYPE_NONE) &
                (data["move_type"][mv.id] == C.NORMAL) &
                (mv.id != move_index("struggle")), m(1.2))
    # Reckless: recoil moves, and the crash moves (High Jump Kick) too.
    mod = apply(mod, (ab == A.RECKLESS) & ((data["move_recoil"][mv.id, 0] > 0) |
                                          data["move_crash_damage"][mv.id]), m(1.2))
    # Sheer Force is 5325/4096, a point above what `m(1.3)` truncates to.
    mod = apply(mod, (ab == A.SHEERFORCE) & has_secondary, 5325)
    mod = apply(mod, (ab == A.TOXICBOOST) & (mv.category == C.CAT_PHYSICAL) &
                ((atk.status == C.PSN) | (atk.status == C.TOX)), m(1.5))
    mod = apply(mod, (ab == A.FLAREBOOST) & (mv.category == C.CAT_SPECIAL) &
                (atk.status == C.BRN), m(1.5))
    # Analytic only applies when the user moves last.
    mod = apply(mod, (ab == A.ANALYTIC) & analytic_ok, M13)
    # Supreme Overlord: +10% per fallen ally, as a single 4096ths modifier.
    overlord = jnp.asarray(np.array([4096, 4506, 4915, 5325, 5734, 6144], np.int32))
    mod = jnp.where(ab == A.SUPREMEOVERLORD,
                    chain_modify(mod, overlord[jnp.clip(fainted_count, 0, 5)]), mod)

    # Type-boosting held items (Magnet, Mystic Water, ...).
    boost_type = data["item_boost_type"][it]
    mod = jnp.where(boost_type == mv.type,
                    chain_modify(mod, data["item_boost_mod"][it]), mod)
    # Items that only work for their own species, often on two types. Compared
    # directly, as the Choice items are, rather than through a table.
    for name, (nums, types) in SPECIES_BOOST_ITEMS.items():
        holder = jnp.any(jnp.stack([atk.species_num == n for n in nums]))
        typed = jnp.bool_(True) if types is None else \
            jnp.any(jnp.stack([mv.type == C.TYPE_IDX[t] for t in types]))
        mod = apply(mod, (it == getattr(I, name.upper())) & holder & typed, 4915)
    mod = apply(mod, (it == I.MUSCLEBAND) & (mv.category == C.CAT_PHYSICAL), m(1.1))
    mod = apply(mod, (it == I.WISEGLASSES) & (mv.category == C.CAT_SPECIAL), m(1.1))
    mod = apply(mod, (it == I.PUNCHINGGLOVE) & has_flag(mv.flags, "punch"), m(1.1))
    # Charge doubles the next Electric move; Dry Skin takes Fire harder.
    mod = apply(mod, atk.charged & (mv.type == C.ELECTRIC), m(2.0))
    mod = apply(mod, (dfn.ability == A.DRYSKIN) & (mv.type == C.FIRE), m(1.25))

    def under(mod, terrain):
        # Terrain boosts require the user to be grounded; Misty Terrain instead
        # weakens Dragon moves aimed at a grounded target.
        mod = apply(mod, grounded_user & (terrain == C.ELECTRIC_TERRAIN) &
                    (mv.type == C.ELECTRIC), M13)
        mod = apply(mod, grounded_user & (terrain == C.GRASSY_TERRAIN) &
                    (mv.type == C.GRASS), M13)
        mod = apply(mod, grounded_user & (terrain == C.PSYCHIC_TERRAIN) &
                    (mv.type == C.PSYCHIC), M13)
        # Grassy Terrain halves Earthquake and Bulldoze against a grounded target.
        quake = (mv.id == move_index("earthquake")) | (mv.id == move_index("bulldoze"))
        mod = apply(mod, grounded_target & (terrain == C.GRASSY_TERRAIN) & quake, m(0.5))
        mod = apply(mod, grounded_target & (terrain == C.MISTY_TERRAIN) &
                    (mv.type == C.DRAGON), m(0.5))
        # The move's own onBasePower callback, resolved once by the caller.
        return chain_modify(mod, bp_cb_mod)

    if alt_terrain is None:
        return under(mod, terrain)
    return under(mod, terrain), under(mod, alt_terrain)


# --- weather -----------------------------------------------------------------

def weather_modifier(weather, move_type, utility_umbrella, hydro_steam=False):
    """Sun/rain scaling of Fire and Water moves.

    The primal weathers only boost: a Water move in harsh sun or a Fire move in
    heavy rain never gets this far -- it fails before use (`execute_move`).
    """
    active = jnp.logical_not(utility_umbrella)
    sun = ((weather == C.SUN) | (weather == C.HARSH_SUN)) & active
    rain = ((weather == C.RAIN) | (weather == C.HEAVY_RAIN)) & active
    mod = jnp.int32(M1)
    mod = jnp.where(sun & (move_type == C.FIRE), m(1.5), mod)
    mod = jnp.where((weather == C.SUN) & active & (move_type == C.WATER),
                    jnp.where(hydro_steam, m(1.5), m(0.5)), mod)
    mod = jnp.where(rain & (move_type == C.WATER), m(1.5), mod)
    mod = jnp.where((weather == C.RAIN) & active & (move_type == C.FIRE), m(0.5), mod)
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
    # Adaptability raises 1.5x to 2x, and stacks with Tera to 2.25x -- but only
    # for a type the user has now (Showdown's `hasType`): Terastallized into
    # another type, its old types keep plain 1.5x STAB.
    has_now = jnp.where(atk.terastallized & (atk.tera_type != C.STELLAR), tera_match,
                        base_match)
    adaptable = (atk.ability == A.ADAPTABILITY) & is_stab & has_now
    normal = jnp.where(adaptable,
                       jnp.where(tera_match & base_match, m(2.25), m(2.0)), normal)
    # Stellar: 2x on your own types, a flat 1.2x on everything else.
    stellar_mod = jnp.where(base_match, m(2.0), 4915)
    return jnp.where(stellar, stellar_mod, normal).astype(jnp.int32)


# --- final modifiers ---------------------------------------------------------

def _final_modifiers(data, atk: Attacker, dfn: Defender, mv: MoveCtx, type_exp,
                     side_conditions, is_crit, berry_ok=True, full_hp=None):
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
    # Multiscale and Shadow Shield: at full HP -- for a multi-hit move only the
    # first hit finds it so (`full_hp` per hit, from the caller).
    full = (dfn.hp == dfn.maxhp) if full_hp is None else full_hp
    mod = apply(mod, ((d_ab == A.MULTISCALE) | (d_ab == A.SHADOWSHIELD)) & full, m(0.5))
    mod = apply(mod, (d_ab == A.ICESCALES) & (mv.category == C.CAT_SPECIAL), m(0.5))
    mod = apply(mod, (d_ab == A.FLUFFY) & has_flag(mv.flags, "contact"), m(0.5))
    mod = apply(mod, (d_ab == A.FLUFFY) & (mv.type == C.FIRE), m(2.0))
    mod = apply(mod, (d_ab == A.PUNKROCK) & has_flag(mv.flags, "sound"), m(0.5))

    # Items.
    mod = apply(mod, (it == I.LIFEORB), m(1.3))
    mod = apply(mod, (it == I.EXPERTBELT) & super_eff, m(1.2))
    # A resist berry halves a super-effective hit of its type (Chilan: any Normal).
    berry_type = data["item_resist_type"][d_it]
    # It only works if it gets eaten: `berry_ok` is false behind a Substitute,
    # against Unnerve, and after the first hit of a multi-hit move.
    berry_ok = berry_ok & (berry_type == mv.type) & (berry_type != C.TYPE_NONE) & \
        (super_eff | (berry_type == C.NORMAL))
    mod = apply(mod, berry_ok, m(0.5))
    return mod


# --- the formula -------------------------------------------------------------

def calc_damage(data, atk: Attacker, dfn: Defender, mv: MoveCtx, *,
                is_crit, damage_roll, weather, terrain, side_conditions,
                type_exp, bp_cb_mod=4096, grounded_user=True, grounded_target=True,
                has_secondary=None, utility_umbrella=False, analytic_ok=False,
                fainted_count=0, target_switched_in=False,
                defender_stats_source=None, technician_power=None, berry_ok=True,
                def_stage_delta=0, full_hp=None, burned=None, def_raw=None,
                sown=None):
    """Damage for one hit. `damage_roll` is 0..15, matching Showdown's `random(16)`.

    `type_exp` comes from `type_effectiveness`; immunity is handled by the caller
    so that ability-based absorption can run its own side effects.

    `technician_power` exists for compilation, not for the rules. When this is
    mapped over a move's hits, only `damage_roll` and -- for Triple Kick and
    Triple Axel -- `base_power` vary, and `vmap` batches just what depends on
    them. Technician's `base_power <= 60` test would otherwise pull the entire
    forty-step modifier chain up a rank along with the scaled power, which is the
    difference between this compiling at optimisation level 1 and not compiling
    at all. Passing the *unscaled* power keeps the chain rank-1 and is exactly
    equivalent: those two moves are the only ones that scale, and every one of
    their per-hit powers (10/20/30 and 20/40/60) is under the threshold anyway.
    """
    if defender_stats_source is None:
        defender_stats_source = dfn.stats
    if has_secondary is None:
        # Sheer Force's trigger is a property of the move, so it is read off the
        # compiled row rather than left to every caller.
        has_secondary = data["move_sheer_force"][mv.id]

    # `sown`: Seed Sower has laid Grassy Terrain since the move began (for the
    # base power; the rest of the hit still sees `terrain`).
    bp_mod = _base_power_modifiers(data, atk, dfn, mv, terrain, has_secondary,
                                   analytic_ok, fainted_count, bp_cb_mod,
                                   grounded_user, grounded_target,
                                   target_switched_in, technician_power, weather,
                                   None if sown is None else jnp.int8(C.GRASSY_TERRAIN))
    if sown is not None:
        bp_mod = jnp.where(sown, bp_mod[1], bp_mod[0])
    power = jnp.maximum(chain_modify(mv.base_power, bp_mod), 1)
    # Gen 9: a Terastallized Pokemon's moves of a type it has hit with at least
    # 60 power -- not priority moves, multi-hit moves, or moves whose power is
    # all callback (Dragon Energy). "A type it has" is Showdown's
    # `getTypes(true)`: the Tera type, or under Stellar the original types.
    declared = data["move_base_power"][mv.id].astype(jnp.int32)
    has_type_now = jnp.where(atk.tera_type == C.STELLAR, jnp.any(atk.base_types == mv.type),
                             atk.tera_type == mv.type)
    tera_floor = atk.terastallized & has_type_now & (data["move_priority"][mv.id] <= 0) & \
        (data["move_multihit"][mv.id, 1] <= 1) & jnp.logical_not(
            ((declared == 0) | (declared == 150)) & (data["move_bp_replace"][mv.id] > 0))
    power = jnp.where(tera_floor, jnp.maximum(power, 60), power)

    attack = _attack_stat(data, atk, dfn, mv, is_crit, defender_stats_source,
                          weather, terrain, target_switched_in)
    defense = _defense_stat(data, atk, dfn, mv, is_crit, weather, terrain, def_stage_delta,
                            def_raw)

    # Every quantity here is non-negative, so the divisions are `floordiv`.
    level = atk.level.astype(jnp.int32)
    base = floordiv(2 * level, 5) + 2
    base = idiv(base * power * attack, defense)
    base = floordiv(base, 50) + 2

    base = chain_modify(base, weather_modifier(weather, mv.type, utility_umbrella,
                                               mv.id == move_index("hydrosteam")))
    base = jnp.where(is_crit, floordiv(base * 3, 2), base)
    # Showdown: tr(tr(damage * (100 - random(16))) / 100)
    base = floordiv(base * (100 - damage_roll), 100)
    base = chain_modify(base, stab_modifier(atk, mv.type))

    # Type effectiveness. Showdown doubles or floor-halves one step at a time;
    # repeated floor-halving is exactly a single floor division by 2**k, so this
    # is equivalent without the loops. That matters: these were nested inside the
    # multi-hit loop, and nested batched `while` loops are what made
    # `vmap(execute_move)` take minutes to compile.
    up = jnp.clip(type_exp, 0, 6)
    down = jnp.clip(-type_exp, 0, 6)
    # Shifts rather than multiply/divide: `base` is non-negative here, and a
    # traced integer divisor is expensive for XLA to lower.
    base = jnp.left_shift(base, up)
    base = jnp.right_shift(base, down)

    # Burn halves physical damage, unless the attacker has Guts -- or the move
    # is Facade, which since Gen 6 ignores it.
    # (`burned` per hit, from the caller: a Flame Body burn partway through a
    # multi-hit move.)
    burned = ((atk.status == C.BRN) if burned is None else burned) & \
        (mv.category == C.CAT_PHYSICAL) & \
        (atk.ability != A.GUTS) & (mv.id != move_index("facade"))
    base = jnp.where(burned, chain_modify(base, m(0.5)), base)

    # One more of Showdown's ModifyDamage modifiers: Glaive Rush leaves its user
    # taking double damage until it next moves.
    field = jnp.where(dfn.glaive_rush, m(2.0), M1)
    base = chain_modify(base, chain_modify(_final_modifiers(
        data, atk, dfn, mv, type_exp, side_conditions, is_crit, berry_ok, full_hp), field))

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
                   acc_boost, eva_boost, roll, gravity, weather_acc=None):
    """True if the move connects. `roll` is uniform in [0, 100).

    `weather_acc` comes from `callbacks.modify_accuracy`: -1 means the move
    cannot miss in the current weather, -2 means no override.
    """
    base_acc = data["move_accuracy"][mv.id].astype(jnp.int32)
    if weather_acc is not None:
        base_acc = jnp.where(weather_acc != -2, weather_acc, base_acc)
    # No Guard on either side means the move always connects.
    always = (base_acc < 0) | (atk.ability == A.NOGUARD) | (dfn.ability == A.NOGUARD)

    # Accuracy and evasion share one stage table; evasion counts against you.
    # The target's evasion is ignored by moves that say so (Chip Away) and by
    # Keen Eye, Mind's Eye and Unaware users; an Unaware target ignores the
    # user's accuracy stages instead (Showdown's `hitStepAccuracy`).
    skip_eva = data["move_ignore_evasion"][mv.id] | (atk.ability == A.KEENEYE) | \
        (atk.ability == A.MINDSEYE) | (atk.ability == A.UNAWARE)
    acc_stage = jnp.where(dfn.ability == A.UNAWARE, 0, acc_boost)
    stage = jnp.clip(acc_stage - jnp.where(skip_eva, 0, eva_boost), -6, 6)
    num = jnp.where(stage >= 0, 3 + stage, 3)
    den = jnp.where(stage >= 0, 3, 3 - stage)
    acc = idiv(base_acc * num, den)

    mod = jnp.int32(M1)
    mod = jnp.where(atk.ability == A.COMPOUNDEYES, chain_modify(mod, M13), mod)
    mod = jnp.where(atk.item == I.WIDELENS, chain_modify(mod, m(1.1)), mod)
    mod = jnp.where(dfn.item == I.BRIGHTPOWDER, chain_modify(mod, m(0.9)), mod)
    # Hustle pays for its Attack with 3277/4096 accuracy on physical moves.
    mod = jnp.where((atk.ability == A.HUSTLE) & (mv.category == C.CAT_PHYSICAL),
                    chain_modify(mod, 3277), mod)
    mod = jnp.where(gravity > 0, chain_modify(mod, 6840), mod)
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
    # Aerilate / Pixilate / Refrigerate / Galvanize retype Normal moves.
    # (Not Struggle, which has no type to change.)
    ate = data["ability_ate_type"][cb_ctx.user_ability]
    mv_type = jnp.where((ate != C.TYPE_NONE) & (mv_type == C.NORMAL) &
                        (move_id != move_index("struggle")), ate, mv_type)
    # Liquid Voice runs after the other type changes: sound moves become Water.
    sound = (data["move_flags"][move_id] & (1 << C.FLAG_BITS["sound"])) != 0
    mv_type = jnp.where((cb_ctx.user_ability == A.LIQUIDVOICE) & sound,
                        jnp.int8(C.WATER), mv_type).astype(jnp.int8)
    ctx = ctx._replace(move_type=mv_type)
    base_power = cb.base_power_replace(data["move_bp_replace"][move_id], ctx)

    # Tera Blast and Photon Geyser become physical when the user's Attack is
    # higher than its Sp. Atk (boosts counted, ability/item modifiers not).
    is_terablast = data["move_bp_replace"][move_id] == BP_REPLACE_HANDLERS.index("terablast")
    is_photon = data["move_effect_cb"][move_id] == EFFECT_HANDLERS.index("photongeyser")
    # Tera Starstorm does the same, but only as Terapagos-Stellar.
    is_starstorm = (data["move_type_cb"][move_id] == TYPE_HANDLERS.index("terastarstorm")) & \
        (cb_ctx.user_species == species_index("terapagosstellar"))
    physical_switch = (((is_terablast | is_starstorm) & cb_ctx.terastallized) | is_photon) & \
                      (cb_ctx.off_atk > cb_ctx.off_spa)

    # Shell Side Arm compares the two base damages rather than the raw stats,
    # transcribed from Showdown including the truncation at each step. A tie
    # there is a coin flip (`random(2) === 0` for Physical).
    is_ssa = data["move_effect_cb"][move_id] == EFFECT_HANDLERS.index("shellsidearm")
    lvl = cb_ctx.level.astype(jnp.int32)
    step = floordiv(2 * lvl, 5) + 2
    phys_dmg = floordiv(idiv(step * 90 * cb_ctx.off_atk, cb_ctx.dfn_def), 50)
    spec_dmg = floordiv(idiv(step * 90 * cb_ctx.off_spa, cb_ctx.dfn_spd), 50)
    physical_switch = physical_switch | (is_ssa & (
        (phys_dmg > spec_dmg) | ((phys_dmg == spec_dmg) & cb_ctx.coin)))

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
    scaling = data["move_bp_replace"][move_id] == \
        BP_REPLACE_HANDLERS.index("multihit_scaling")
    power = jnp.where(scaling, mv.base_power * jnp.maximum(hit_number, 1),
                      mv.base_power)
    return mv._replace(base_power=power)
