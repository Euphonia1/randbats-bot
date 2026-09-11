/**
 * Ground truth for the damage differential test.
 *
 * Builds a real Showdown battle per case and calls `BattleActions.getDamage`,
 * once for each of the 16 damage rolls, with the crit outcome forced so the
 * only randomness left is the roll itself. The JAX engine must reproduce all
 * 16 numbers exactly.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const {Battle, Dex} = require('pokemon-showdown');

const EVS = {hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85};
const IVS = {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31};

function mkset(spec) {
  return {
    name: spec.species, species: spec.species, level: spec.level || 100,
    gender: 'N', item: spec.item || '', ability: spec.ability || '',
    moves: ['tackle'], evs: EVS, ivs: IVS, nature: 'Serious',
    teraType: spec.tera || spec.teraType || undefined,
  };
}

/** Build a fresh battle and apply the case's setup. */
function setup(c) {
  const battle = new Battle({formatid: 'gen9customgame', seed: [1, 2, 3, 4]});
  battle.setPlayer('p1', {team: [mkset(c.attacker)]});
  battle.setPlayer('p2', {team: [mkset(c.defender)]});
  // Actives are only populated once a turn begins; switch them in directly so
  // the case setup below can run before any move does.
  battle.actions.switchIn(battle.p1.pokemon[0], 0);
  battle.actions.switchIn(battle.p2.pokemon[0], 0);
  const src = battle.p1.active[0];
  const tgt = battle.p2.active[0];

  // Abilities are resolved from the set; overwrite so we can test any pairing.
  if (c.attacker.ability) src.setAbility(c.attacker.ability, null, true);
  if (c.defender.ability) tgt.setAbility(c.defender.ability, null, true);
  if (c.attacker.status) src.status = c.attacker.status;
  if (c.defender.status) tgt.status = c.defender.status;
  if (c.attacker.boosts) Object.assign(src.boosts, c.attacker.boosts);
  if (c.defender.boosts) Object.assign(tgt.boosts, c.defender.boosts);
  if (c.attacker.hpPercent) src.hp = Math.max(1, Math.floor(src.maxhp * c.attacker.hpPercent));
  if (c.defender.hpPercent) tgt.hp = Math.max(1, Math.floor(tgt.maxhp * c.defender.hpPercent));
  if (c.attacker.tera) { src.teraType = c.attacker.tera; src.terastallized = c.attacker.tera; }
  if (c.defender.tera) { tgt.teraType = c.defender.tera; tgt.terastallized = c.defender.tera; }
  if (c.weather) { battle.field.weather = c.weather; battle.field.weatherState = {id: c.weather, duration: 5}; }
  if (c.terrain) { battle.field.terrain = c.terrain; battle.field.terrainState = {id: c.terrain, duration: 5}; }
  if (c.screens) for (const s of c.screens) tgt.side.addSideCondition(s, tgt);
  // Knobs for the base-power callbacks. `newlySwitched` is set by switchIn and
  // would otherwise skew Payback / Bolt Beak, which read it directly.
  src.newlySwitched = false;
  tgt.newlySwitched = false;
  // Protosynthesis / Quark Drive are volatiles in Showdown; set them explicitly
  // so the case does not depend on switch-in ordering.
  for (const [mon, spec] of [[src, c.attacker], [tgt, c.defender]]) {
    if (!spec.boosted) continue;
    const id = mon.hasAbility('quarkdrive') ? 'quarkdrive' : 'protosynthesis';
    let best = 'atk';
    for (const st of ['atk', 'def', 'spa', 'spd', 'spe']) {
      if (mon.storedStats[st] > mon.storedStats[best]) best = st;
    }
    mon.addVolatile(id);
    if (mon.volatiles[id]) mon.volatiles[id].bestStat = best;
  }
  if (c.timesAttacked) src.timesAttacked = c.timesAttacked;
  if (c.faintedCount) src.side.totalFainted = c.faintedCount;
  if (c.targetDamaged) tgt.hurtThisTurn = 1;
  if (c.userDamaged) src.attackedBy.push({source: tgt, damage: 1, thisTurn: true, move: 'tackle', slot: tgt.getSlot()});
  return {battle, src, tgt};
}

function run(c) {
  // A fresh battle per roll: some effects fire only once per battle (the Stellar
  // Tera boost is once per type), so reusing one battle would contaminate the
  // later rolls.
  const damages = [];
  let meta = null;
  for (let roll = 0; roll < 16; roll++) {
    const {battle, src, tgt} = setup(c);
    const move = battle.dex.getActiveMove(c.move);
    move.willCrit = !!c.crit;
    battle.random = (n) => (n === 16 ? roll : 0);
    // Handlers like Unaware key off these; getDamage alone does not set them.
    battle.activePokemon = src;
    battle.activeTarget = tgt;
    battle.activeMove = move;
    let d;
    try {
      // useMove() runs these before getDamage; without them Weather Ball,
      // Terrain Pulse and Tera Blast would be measured with their base type.
      battle.singleEvent('ModifyType', move, null, src, tgt, move, move);
      battle.singleEvent('ModifyMove', move, null, src, tgt, move, move);
      battle.runEvent('ModifyType', src, tgt, move, move);
      battle.runEvent('ModifyMove', src, tgt, move, move);
      d = battle.actions.getDamage(src, tgt, move);
    } catch (e) {
      d = `ERROR:${e.message}`;
    }
    damages.push(d === false ? 'immune' : (d === undefined || d === null ? 0 : d));
    if (meta === null) {
      meta = {
        attackerStats: {...src.storedStats}, defenderStats: {...tgt.storedStats},
        attackerMaxHP: src.maxhp, defenderMaxHP: tgt.maxhp,
        actualWeather: battle.field.weather || null,
        actualTerrain: battle.field.terrain || null,
      };
    }
  }
  return {...c, ...meta, damages};
}

const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = cases.map(run);
fs.writeFileSync(process.argv[3], JSON.stringify(out, null, 1));
console.log(`ran ${out.length} damage cases -> ${process.argv[3]}`);
