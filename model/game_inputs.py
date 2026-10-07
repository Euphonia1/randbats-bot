"""Turn a batch of battles into GameNetwork inputs, as one player may see them.

    env = FogOfWarEnv()
    fs = env.reset_batch(jax.random.split(key, 64))
    action_logits, win_logit = net(**game_inputs(env, fs, player=0))

Your own side is shown in full. The opponent's side is what Showdown would show
you, on top of the fog wrapper's own censoring (unused moves read as empty
slots, HP as a whole percent):

- **Species and level** once that Pokemon has been on the field. While
  Illusion holds, the active one shows the Pokemon it is disguised as.
- **Stats** as a player would work them out from species and level, with
  Random Battles' standard spread. Their real stats can differ: special
  attackers get 0 Attack IVs and EVs and Trick Room sets 0 Speed, which a real
  player only finds out by fighting them.
- **Item** only once it is gone (eaten, popped, Knocked Off), which Showdown
  announces. Before that it is unknown, even where a real player could tell
  (Leftovers healing, a Choice lock), so this hides more than it needs to but
  never leaks.
- **Ability** only for a species with a single possible ability (Rotom-Wash's
  Levitate, Great Tusk's Protosynthesis), which a player knows on sight.
  Telling when any other has been revealed (Intimidate firing) needs events the
  engine does not record.
- **Tera type** once it has Terastallized.
- **Choice lock** never, since it would give the item away.

Everything else -- status, sleep and toxic counts, boosts, volatiles, the last
move used, Encore, Disable, Outrage locks, the field -- is public. Unknown
values are -1, which the embedding layers turn into zeros.

`matchups` holds damage calcs, made with the engine's own damage formula on
what you believe about the battle: your side as it is, and the opponent's
Pokemon as shown, with the stats, HP and ability above and no item. See
`_matchups` for what is calculated.

One approximation remains: the opponent's screens show their exact turns left,
which gives away Light Clay the turn they go up.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import torch

from psjax import callbacks as cb
from psjax import consts as C
from psjax.damage import calc_damage, resolve_move_ctx, type_effectiveness
from psjax.fog import FogOfWarEnv, FogState
from psjax.hooks import A
from psjax.mechanics import act, slot_get, species_stats
from psjax.moves import build_attacker, build_cb_ctx, build_defender
from psjax.teams import legal_action_mask


def game_inputs(env: FogOfWarEnv, fs: FogState, player: int,
                device: torch.device | str | None = None) -> dict:
    """GameNetwork.forward's arguments, by name, for `player` (0 or 1).

    `fs` is a batch of FogStates, as `env.reset_batch` and `env.step_batch`
    return them; for a single battle, add a batch dimension first with
    `jax.tree_util.tree_map(lambda x: x[None], fs)`.
    """
    arrays = _view(env, fs, player)
    return jax.tree_util.tree_map(lambda x: _to_torch(x, device), arrays)


def _to_torch(x, device) -> torch.Tensor:
    """IDs and counts become long, flags stay bool, everything else float32."""
    t = torch.from_numpy(np.array(x))
    if t.dtype == torch.bool:
        return t.to(device)
    if t.is_floating_point():
        return t.to(device, torch.float32)
    return t.to(device, torch.long)


@functools.partial(jax.jit, static_argnums=(0, 2))
def _view(env: FogOfWarEnv, fs: FogState, player: int) -> dict:
    """`game_inputs` as JAX arrays, before the move to torch."""
    me, them = player, 1 - player

    def one(fs: FogState) -> dict:
        st = env._censor(fs, them)  # the opponent's unused moves and exact HP

        def team(side, species, item, ability, tera_type, level, stats):
            return dict(
                species_ids=species, item_ids=item, ability_ids=ability,
                move_ids=st.moves[side], pp=st.pp[side], maxpp=st.maxpp[side],
                tera_type_ids=tera_type, terastallized=st.terastallized[side],
                hp=st.hp[side], maxhp=st.maxhp[side], status=st.status[side],
                sleep_attempts=st.sleep_attempts[side], rest_sleep=st.rest_sleep[side],
                level=level, stats=stats,
                toxic_counter=jnp.where(st.status[side] == C.TOX, st.status_turns[side], 0))

        def player_view(side, team, choice_slot):
            return dict(team=team, active_slot=st.active[side],
                        boosts=st.boosts[side], volatiles=st.volatiles[side],
                        last_move=st.last_move[side],
                        moves_since_switch=st.moves_since_switch[side],
                        choice_slot=choice_slot, encore_slot=st.encore_slot[side],
                        disabled_slot=st.disabled_slot[side],
                        locked_slot=st.locked_slot[side])

        mine = team(me, st.species[me], st.item[me], st.ability[me], st.tera_type[me],
                    st.level[me], st.stats[me, :, 1:])

        # Who the opponent's Pokemon appear to be: unseen ones are unknown, and
        # the active one wears Illusion's disguise while it holds.
        is_active = jnp.arange(C.TEAM_SIZE) == st.active[them]
        disguise = st.illusion[them].astype(jnp.int32)
        disguised = is_active & (disguise >= 0)
        shown = jnp.where(disguised, jnp.maximum(disguise, 0), jnp.arange(C.TEAM_SIZE))
        known = fs.revealed[them]
        species = jnp.where(known, st.species[them, shown], -1)
        level = jnp.where(known, st.level[them, shown], 0)
        stats = jax.vmap(lambda s, lv: species_stats(env.data, s, lv, jnp.int8(0)))(
            jnp.maximum(species, 0), level)  # (6, 6), HP first
        abilities = env.data["species_abilities"][jnp.maximum(species, 0)]
        only_ability = known & (abilities[:, 1] == 0) & (abilities[:, 2] == 0)
        ability = jnp.where(only_ability, abilities[:, 0], -1)
        theirs = team(them, species,
                      jnp.where(st.item[them] == 0, 0, -1), ability,
                      jnp.where(st.terastallized[them], st.tera_type[them], -1),
                      level, jnp.where(known[:, None], stats[:, 1:], 0))

        # The battle as you believe it to be, for the damage calcs: the
        # opponent's HP percent becomes HP out of the max you would expect.
        put = lambda field, value: field.at[them].set(value.astype(field.dtype))
        maxhp = jnp.maximum(stats[:, C.HP], 1)
        belief = st._replace(
            species=put(st.species, jnp.maximum(species, 0)), level=put(st.level, level),
            stats=put(st.stats, stats), maxhp=put(st.maxhp, maxhp),
            hp=put(st.hp, jnp.ceil(st.hp[them].astype(jnp.float32) * maxhp / 100.0)),
            item=put(st.item, jnp.zeros(C.TEAM_SIZE)),
            ability=put(st.ability, jnp.maximum(ability, 0)),
            tera_type=put(st.tera_type, jnp.where(st.terastallized[them],
                                                  st.tera_type[them], C.TYPE_NONE)))

        return dict(
            me=player_view(me, mine, st.choice_slot[me]),
            opponent=player_view(them, theirs, jnp.full_like(st.choice_slot[them], -1)),
            field=dict(weather=st.weather, terrain=st.terrain,
                       trick_room=st.trick_room, gravity=st.gravity,
                       side_conditions=jnp.stack([st.side_conditions[me],
                                                  st.side_conditions[them]]),
                       phase=st.phase,
                       force_switch=jnp.stack([st.force_switch[me],
                                               st.force_switch[them]])),
            legal_actions=legal_action_mask(env.data, fs.battle)[me],
            matchups=_matchups(env.data, belief, me, them, known))

    return jax.vmap(one)(fs)


# --- damage calcs ------------------------------------------------------------

#: Besides their active Pokemon, how many of the opponent's other revealed
#: Pokemon each of your moves is calculated against.
BENCH_TARGETS = 3


def _matchups(data, state, me: int, them: int, seen) -> dict:
    """Damage calcs for GameNetwork's action tokens, from what `me` believes.

    Each calc is (min, max): the lowest and highest damage roll, as fractions
    of the target's max HP; see `_calc`. -1 marks a calc with nothing to
    calculate against, unlike 0 for a move that does no damage. `seen` marks
    the opponent's revealed slots.

    Returns:
        moves: (4, 8) for each of your active Pokemon's moves, the calc
            against the opponent's active Pokemon, then against up to
            BENCH_TARGETS of their other revealed, living Pokemon, the first
            ones in team order: how safely they could switch into it
        tera_moves: (4, 8) the same, as if you Terastallized first
        switches: (6, 4) for each of your Pokemon, its best calc against the
            opponent's active Pokemon, then the opponent's best calc against it
            from the moves they have revealed. "Best" is the move with the
            highest max roll. -1 once fainted, or before the opponent has
            revealed a move.
    """
    M, T = C.MOVES_PER_POKEMON, C.TEAM_SIZE
    mine, theirs = state.active[me].astype(jnp.int32), state.active[them].astype(jnp.int32)
    slots, moves = jnp.arange(T), jnp.arange(M)

    # Every calc goes through one vmap, so the damage formula compiles once:
    # (attacker side, attacker slot, defender side, defender slot, move slot, Tera)
    tera_1, move_1, slot_1 = (x.ravel() for x in
                              jnp.meshgrid(jnp.arange(2), moves, slots, indexing="ij"))
    move_2, slot_2 = (x.ravel() for x in jnp.meshgrid(moves, slots, indexing="ij"))
    slot_3, move_3 = (x.ravel() for x in jnp.meshgrid(slots, moves, indexing="ij"))
    n1, n2, n3 = 2 * M * T, M * T, T * M
    full = lambda n, v: jnp.full(n, v, jnp.int32)
    low, high = jax.vmap(functools.partial(_calc, data, state))(
        jnp.concatenate([full(n1, me), full(n2, them), full(n3, me)]),
        jnp.concatenate([full(n1, mine), full(n2, theirs), slot_3]),
        jnp.concatenate([full(n1, them), full(n2, me), full(n3, them)]),
        jnp.concatenate([slot_1, slot_2, full(n3, theirs)]),
        jnp.concatenate([move_1, move_2, move_3]),
        jnp.concatenate([tera_1.astype(bool), jnp.zeros(n2 + n3, bool)]))
    split = lambda x: (x[:n1].reshape(2, M, T), x[n1:n1 + n2].reshape(M, T),
                       x[n1 + n2:].reshape(T, M))
    (low_1, low_2, low_3), (high_1, high_2, high_3) = split(low), split(high)

    # 1. Your active Pokemon's moves, without and with Tera, against their
    # active Pokemon and the first few others they have revealed
    bench = seen & (state.hp[them] > 0) & (slots != theirs)
    picks = jnp.argsort(jnp.where(bench, slots, T))[:BENCH_TARGETS]
    targets = jnp.concatenate([theirs[None], picks])
    present = jnp.concatenate([jnp.ones(1, bool), bench[picks]])
    move_calcs = jnp.stack([low_1[..., targets], high_1[..., targets]], axis=-1)
    move_calcs = jnp.where(present[:, None], move_calcs, -1.0).reshape(2, M, -1)

    # 2-3. Each of your Pokemon: its best move into their active, and their
    # active's best revealed move into it
    def best(low, high, axis):
        pick = jnp.expand_dims(jnp.argmax(high, axis=axis), axis)
        return jnp.stack([jnp.take_along_axis(low, pick, axis).squeeze(axis),
                          jnp.take_along_axis(high, pick, axis).squeeze(axis)], axis=-1)

    revealed = jnp.any(state.moves[them, theirs] >= 0)
    switch_calcs = jnp.concatenate([
        best(low_3, high_3, axis=1),
        jnp.where(revealed, best(low_2, high_2, axis=0), -1.0),
    ], axis=-1)
    switch_calcs = jnp.where((state.hp[me] > 0)[:, None], switch_calcs, -1.0)
    return dict(moves=move_calcs[0], tera_moves=move_calcs[1], switches=switch_calcs)


def _put_in(state, side, slot):
    """`state` with `slot` as `side`'s active Pokemon. Boosts and volatiles
    belong to whoever is on the field, so one brought in has none."""
    stays = slot == state.active[side]
    return state._replace(
        active=state.active.at[side].set(slot.astype(state.active.dtype)),
        boosts=state.boosts.at[side].set(jnp.where(stays, state.boosts[side], 0)),
        volatiles=state.volatiles.at[side].set(jnp.where(stays, state.volatiles[side], 0)),
        boosted_stat=state.boosted_stat.at[side].set(
            jnp.where(stays, state.boosted_stat[side], -1)))


def _calc(data, state, user, user_slot, target, target_slot, move_slot, tera):
    """One damage calc: (min, max), the lowest and highest damage roll as
    fractions of the target's max HP, capped at 2.

    Uses the engine's damage formula with no crit, and assumes the move hits;
    a multi-hit move counts its fewest hits for the min and its most for the
    max. Ignores Protect, Substitute, Focus Sash, Sturdy and Disguise. An empty
    move slot (-1, or an opponent's move not yet revealed), a status move or
    an immunity gives zeros.
    """
    state = _put_in(_put_in(state, user, user_slot), target, target_slot)
    state = state._replace(terastallized=state.terastallized.at[user, user_slot].set(
        state.terastallized[user, user_slot] | tera))
    move_id = state.moves[user, user_slot, move_slot].astype(jnp.int32)
    has_move = move_id >= 0
    move_id = jnp.maximum(move_id, 0)

    ui, ti = act(state, user), act(state, target)
    cb_ctx = build_cb_ctx(data, state, user, target, jnp.bool_(True), jnp.int32(8))
    mv = resolve_move_ctx(data, move_id, cb_ctx)
    atk = build_attacker(state, user, data)
    # Mold Breaker and moves like Sunsteel Strike ignore a breakable ability.
    u_ab, t_ab = slot_get(state.ability, user, ui), slot_get(state.ability, target, ti)
    breaks = data["ability_mold_breaker"][u_ab] | data["move_ignore_ability"][move_id]
    t_ab = jnp.where(breaks & data["ability_breakable"][t_ab], 0, t_ab).astype(t_ab.dtype)
    dfn = build_defender(data, state, target, ability=t_ab)

    type_exp, immune = type_effectiveness(
        data, mv.type, dfn.types, mv, data["move_ignore_immunity"][move_id], t_ab,
        (u_ab == A.SCRAPPY) | (u_ab == A.MINDSEYE), def_terastallized=dfn.terastallized,
        def_grounded=cb_ctx.grounded_target, def_full_hp=dfn.hp >= dfn.maxhp)
    absorbs = data["ability_absorb_type"][t_ab]  # Water Absorb, Flash Fire, ...
    immune = immune | ((absorbs == mv.type) & (absorbs >= 0))
    cb_ctx = cb_ctx._replace(type_exp=type_exp)

    def damage(roll):
        return calc_damage(
            data, atk, dfn, mv, is_crit=jnp.bool_(False), damage_roll=jnp.int32(roll),
            weather=cb_ctx.weather, terrain=state.terrain,
            side_conditions=state.side_conditions[target], type_exp=type_exp,
            bp_cb_mod=cb.base_power_modify(data["move_bp_modify"][move_id], cb_ctx),
            grounded_user=cb_ctx.grounded_user, grounded_target=cb_ctx.grounded_target,
            technician_power=mv.base_power).astype(jnp.float32)

    # Showdown's roll 0 is the highest (100%) and roll 15 the lowest (85%).
    # Seismic Toss and friends ignore the formula; a one-hit KO takes everything.
    fixed = cb.fixed_damage(data["move_dmg_cb"][move_id], cb_ctx).astype(jnp.float32)
    hits = data["move_multihit"][move_id].astype(jnp.float32)  # fewest, most
    maxhp = jnp.maximum(dfn.maxhp.astype(jnp.float32), 1.0)
    low, high = (jnp.where(data["move_ohko"][move_id], maxhp,
                           jnp.where(fixed >= 0, fixed, damage(roll)) * n)
                 for roll, n in ((15, hits[0]), (0, hits[1])))

    deals = has_move & jnp.logical_not(immune) & (mv.category != C.CAT_STATUS)
    fraction = lambda x: jnp.where(deals, jnp.minimum(x / maxhp, 2.0), 0.0)
    return fraction(low), fraction(high)
