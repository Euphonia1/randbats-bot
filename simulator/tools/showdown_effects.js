/**
 * Ground truth for the *effects* differential test.
 *
 * `showdown_damage.js` pins the damage numbers; this pins everything else --
 * status, boosts, hazards, residual chip and healing, switch-in abilities. Each
 * scenario runs one real Showdown turn and dumps the observable state
 * afterwards, which `tests/test_effects.py` then has to reproduce.
 *
 * Scenarios are chosen to be deterministic: 100%-chance effects, guaranteed
 * immunities, residuals. RNG is pinned anyway so accuracy and crits never vary.
 */
'use strict';
const fs = require('fs');
const {Battle} = require('pokemon-showdown');

const EVS = {hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85};
const IVS = {hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31};

function mkset(spec) {
  return {
    name: spec.species, species: spec.species, level: spec.level || 100,
    gender: spec.gender || 'N', item: spec.item || '', ability: spec.ability || '',
    moves: spec.moves && spec.moves.length ? spec.moves : ['splash'],
    evs: EVS, ivs: IVS, nature: 'Serious',
    teraType: spec.tera || undefined, happiness: 255,
  };
}

/** The state the JAX engine can be compared against.
 *
 * Showdown reorders `side.pokemon` so the active Pokemon sits at index 0, while
 * psjax keeps team slots fixed. `battle.__order` is the original array captured
 * at setup, so both engines report the same slot order.
 */
function snapshot(battle) {
  const side = (s) => {
    const team = battle.__order[s.id] || s.pokemon;
    return {
    active: s.active[0] ? team.indexOf(s.active[0]) : 0,
    hp: team.map(p => p.hp),
    maxhp: team.map(p => p.maxhp),
    status: team.map(p => p.status || ''),
    item: team.map(p => p.item || ''),
    species: team.map(p => p.species.id),
    ability: team.map(p => p.ability || ''),
    types: s.active[0] ? s.active[0].getTypes().map(t => t.toLowerCase()) : [],
    boosts: s.active[0] ? {...s.active[0].boosts} : null,
    sideConditions: Object.keys(s.sideConditions).sort(),
    spikeLayers: s.sideConditions.spikes ? s.sideConditions.spikes.layers : 0,
    toxicSpikeLayers: s.sideConditions.toxicspikes ? s.sideConditions.toxicspikes.layers : 0,
    volatiles: s.active[0] ? Object.keys(s.active[0].volatiles).sort() : [],
    subHP: s.active[0] && s.active[0].volatiles.substitute
      ? s.active[0].volatiles.substitute.hp : 0,
    };
  };
  return {
    turn: battle.turn,
    weather: battle.field.weather || '',
    terrain: battle.field.terrain || '',
    p1: side(battle.p1),
    p2: side(battle.p2),
  };
}

function run(c) {
  const battle = new Battle({formatid: 'gen9customgame', seed: [1, 2, 3, 4]});
  battle.setPlayer('p1', {team: (c.p1team || [c.p1]).map(mkset)});
  battle.setPlayer('p2', {team: (c.p2team || [c.p2]).map(mkset)});
  // gen9customgame opens on team preview.
  battle.makeChoices('default', 'default');
  // Freeze the team order before anything can switch and permute it.
  battle.__order = {p1: battle.p1.pokemon.slice(), p2: battle.p2.pokemon.slice()};

  // Pre-turn setup: statuses, HP, hazards and field that the scenario assumes.
  const apply = (side, spec) => {
    const mon = side.active[0];
    if (spec.status) mon.setStatus(spec.status, null, null, true);
    // Sleep length is otherwise rolled here, out of the scenario's control.
    if (spec.statusTurns) mon.statusState.time = spec.statusTurns;
    if (spec.hpPercent) mon.hp = Math.max(1, Math.floor(mon.maxhp * spec.hpPercent));
    if (spec.boosts) Object.assign(mon.boosts, spec.boosts);
    for (const sc of spec.sideConditions || []) side.addSideCondition(sc, mon);
    for (const v of spec.volatiles || []) mon.addVolatile(v);
  };
  apply(battle.p1, c.p1);
  apply(battle.p2, c.p2);
  // A benched team member's status (Beat Up skips the statused).
  for (const [side, key] of [[battle.p1, 'p1team'], [battle.p2, 'p2team']]) {
    (c[key] || []).forEach((spec, i) => {
      if (i > 0 && spec.status) battle.__order[side.id][i].setStatus(spec.status, null, null, true);
    });
  }
  // Scenarios that need an effect already in place before the turn runs.
  if (c.seedP2) battle.p2.active[0].addVolatile('leechseed', battle.p1.active[0]);
  if (c.weather) { battle.field.weather = c.weather; battle.field.weatherState = {id: c.weather, duration: 8}; }
  if (c.terrain) { battle.field.terrain = c.terrain; battle.field.terrainState = {id: c.terrain, duration: 8}; }
  if (c.gravity) battle.field.addPseudoWeather('gravity', battle.p1.active[0]);

  // Pin the RNG to zero. `randomChance(num, den)` is `random(den) < num`, so
  // every accuracy check and every chance-based effect succeeds -- the scenario
  // becomes deterministic. (Overriding `randomChance` itself is wrong: it is the
  // accuracy check too, so forcing it false makes every move miss.)
  battle.random = () => 0;
  // `force` also wins every `randomChance` roll -- a 30% ability, a crit, a
  // secondary -- so a scenario about one of those is deterministic too. The
  // psjax side pins its random words to zero to match.
  if (c.force) battle.forceRandomChance = true;

  const before = snapshot(battle);
  try {
    battle.makeChoices(c.p1move || 'move 1', c.p2move || 'move 1');
    // A self-switch (U-turn, Parting Shot) opens a mid-turn switch request; the
    // rest of the turn only runs once it is answered.
    if (c.p1switchAfter && battle.requestState === 'switch') {
      battle.makeChoices(`switch ${c.p1switchAfter}`, 'default');
    }
    // Scenarios that play out over several turns (Future Sight, Wish, charge
    // moves) list the later turns' choices.
    for (const [a, b] of c.turns || []) battle.makeChoices(a, b);
  } catch (e) {
    return {...c, error: String(e.message).slice(0, 200)};
  }
  return {...c, before, after: snapshot(battle)};
}

const cases = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = cases.map(run);
fs.writeFileSync(process.argv[3], JSON.stringify(out, null, 1));
const errs = out.filter(o => o.error);
console.log(`ran ${out.length} effect scenarios -> ${process.argv[3]}`);
if (errs.length) for (const e of errs) console.log(`  ERROR ${e.name}: ${e.error}`);
