/**
 * Record complete Gen 9 Random Battles for the lockstep differential test.
 *
 * `showdown_damage.js` and `showdown_effects.js` check hand-built scenarios;
 * this plays whole battles -- Showdown's own random teams, random legal choices
 * -- and records everything `tools/lockstep.py` needs to replay the same game in
 * psjax and compare the two after every decision.
 *
 * The two engines draw randomness in different orders, so their streams can
 * never be lined up. Instead every random word is pinned to one constant `w`.
 * Showdown's `random(n)` is `floor(w * n / 2**32)` and `randomChance(a, b)` is
 * `random(b) < a`, so a constant word makes every roll a fixed threshold test:
 * w = 0 hits, crits and procs everything, w near 2**32 nothing. Both engines are
 * then deterministic, and the same game can be played in each. Choices come
 * from a separate seeded generator, so they do not depend on `w`.
 *
 *   node tools/showdown_lockstep.js <battles> <out.json> [w,w,...] [seed]
 *
 * `w` values are fractions of 2**32 (default spread below), rounded so their
 * low 8 bits are zero: psjax's `uniform` reads the top 24 bits, and a word with
 * none below that is exact in both engines.
 */
'use strict';
const fs = require('fs');
const {Battle, Teams} = require('pokemon-showdown');

const N = parseInt(process.argv[2] || '20', 10);
const OUT = process.argv[3] || 'lockstep.json';
const FRACTIONS = (process.argv[4] || '0,0.05,0.2,0.36,0.5,0.62,0.72,0.97,0.9999')
  .split(',').map(Number);
const SEED = parseInt(process.argv[5] || '1', 10);
const MAX_DECISIONS = 400;

const word = f => Math.min(Math.floor(f * 2 ** 24), 2 ** 24 - 1) * 256;

/** mulberry32: the choosers' own generator, independent of the battle's. */
function chooser(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Everything psjax models, in fixed team order (see showdown_effects.js). */
function snapshot(battle) {
  const side = (s) => {
    const team = battle.__order[s.id];
    const a = s.active[0];
    const out = {
      active: a ? team.indexOf(a) : -1,
      hp: team.map(p => p.hp),
      maxhp: team.map(p => p.maxhp),
      status: team.map(p => p.status || ''),
      sleepTurns: team.map(p => (p.status === 'slp' ? p.statusState.time : 0)),
      toxicStage: team.map(p => (p.status === 'tox' ? p.statusState.stage || 0 : 0)),
      item: team.map(p => p.item || ''),
      ability: team.map(p => p.ability || ''),
      species: team.map(p => p.species.id),
      speciesBase: team.map(p => p.species.baseSpecies.toLowerCase().replace(/[^a-z0-9]/g, '')),
      tera: team.map(p => !!p.terastallized),
      pp: team.map(p => p.baseMoveSlots.map(m => m.pp)),
      sideConditions: Object.keys(s.sideConditions).sort(),
      // Layers for hazards, turns left for everything else.
      sideConditionCounts: Object.fromEntries(Object.entries(s.sideConditions)
        .map(([k, v]) => [k, v.layers || v.duration || 0])),
      spikeLayers: s.sideConditions.spikes ? s.sideConditions.spikes.layers : 0,
      toxicSpikeLayers: s.sideConditions.toxicspikes ? s.sideConditions.toxicspikes.layers : 0,
      slotConditions: Object.keys(s.slotConditions[0] || {}).sort(),
    };
    if (a) {
      out.types = a.getTypes().map(t => t.toLowerCase());
      out.boosts = {...a.boosts};
      out.volatiles = Object.keys(a.volatiles).sort();
      out.confusion = a.volatiles.confusion ? a.volatiles.confusion.time : 0;
      out.subHP = a.volatiles.substitute ? a.volatiles.substitute.hp : 0;
      out.transformed = !!a.transformed;
    }
    return out;
  };
  const req = (s) => {
    const r = s.activeRequest;
    if (battle.ended || !r || r.wait) return 'wait';
    return r.forceSwitch ? 'switch' : 'move';
  };
  return {
    turn: battle.turn,
    ended: battle.ended,
    // 0 / 1 for a side, 2 for a tie, -1 while the battle is on.
    winner: !battle.ended ? -1 : battle.winner === battle.p1.name ? 0
      : battle.winner === battle.p2.name ? 1 : 2,
    request: [req(battle.p1), req(battle.p2)],
    weather: battle.field.weather || '',
    weatherTurns: battle.field.weather ? battle.field.weatherState.duration || 0 : 0,
    terrain: battle.field.terrain || '',
    terrainTurns: battle.field.terrain ? battle.field.terrainState.duration || 0 : 0,
    pseudoWeather: Object.keys(battle.field.pseudoWeather).sort(),
    p1: side(battle.p1),
    p2: side(battle.p2),
  };
}

/** The psjax action a committed Showdown choice amounts to. */
function toAction(battle, side) {
  const act = side.choice.actions[0];
  if (!act) return 0;
  // Revival Blessing's answer names the fainted Pokemon to revive.
  if (act.choice === 'switch' || act.choice === 'instaswitch' ||
      act.choice === 'revivalblessing') {
    return 8 + battle.__order[side.id].indexOf(act.target);
  }
  if (act.choice === 'move') {
    const mon = act.pokemon;
    let slot = mon.moveSlots.findIndex(m => m.id === act.moveid);
    if (slot < 0) slot = 0;               // Struggle
    return slot + (act.terastallize ? 4 : 0);
  }
  return 0;
}

/** Every choice string the side may make, split by kind, and as psjax actions. */
function options(battle, side) {
  const req = side.activeRequest;
  if (!req || req.wait) return null;
  const switches = [], ids = [];
  // Revival Blessing's prompt picks a fainted Pokemon to revive instead.
  const reviving = !!req.forceSwitch &&
    Object.keys(side.slotConditions[0] || {}).includes('revivalblessing');
  const eligible = (p, i) => i >= side.active.length && (reviving ? p.fainted : !p.fainted);
  side.pokemon.forEach((p, i) => {
    if (eligible(p, i)) switches.push(`switch ${i + 1}`);
  });
  const switchIds = side.pokemon.filter(eligible)
    .map(p => 8 + battle.__order[side.id].indexOf(p));
  if (req.forceSwitch) return {moves: [], tera: [], switches, ids: switchIds.sort()};
  const active = req.active[0];
  const mon = side.active[0];
  const moves = [], tera = [];
  active.moves.forEach((m, i) => {
    if (m.disabled) return;
    moves.push(`move ${i + 1}`);
    // A locked move (Outrage, Recharge, Struggle) is the only entry; psjax
    // names it by its slot, or slot 0 for one the Pokemon does not know.
    const slot = Math.max(0, mon.moveSlots.findIndex(s => s.id === m.id));
    ids.push(slot);
    if (active.canTerastallize) { tera.push(`move ${i + 1} terastallize`); ids.push(slot + 4); }
  });
  const trapped = active.trapped || mon.trapped;
  if (!trapped) ids.push(...switchIds);
  return {moves, tera, switches: trapped ? [] : switches, ids: [...new Set(ids)].sort((a, b) => a - b)};
}

function pick(rand, opts) {
  const roll = rand();
  let pool;
  if (!opts.moves.length) pool = opts.switches;
  else if (!opts.switches.length) pool = roll < 0.2 && opts.tera.length ? opts.tera : opts.moves;
  else if (roll < 0.25) pool = opts.switches;
  else if (roll < 0.35 && opts.tera.length) pool = opts.tera;
  else pool = opts.moves;
  return pool.length ? pool[Math.floor(rand() * pool.length)] : 'default';
}

function play(index, w) {
  const teamSeed = [SEED, index + 1, 7, 11];
  const teams = [
    Teams.generate('gen9randombattle', {seed: [...teamSeed]}),
    Teams.generate('gen9randombattle', {seed: [teamSeed[0], teamSeed[1], 13, 17]}),
  ];
  const battle = new Battle({formatid: 'gen9randombattle', seed: [1, 2, 3, 4]});
  // Pin the battle's generator before anything rolls (gender rolls in setPlayer).
  battle.prng.rng = {next: () => w, getSeed: () => 'pinned'};
  battle.setPlayer('p1', {team: teams[0]});
  battle.setPlayer('p2', {team: teams[1]});
  battle.__order = {p1: battle.p1.pokemon.slice(), p2: battle.p2.pokemon.slice()};

  const mons = battle.__order;
  const record = {
    id: `${index}@${(w / 2 ** 32).toFixed(4)}`, battle: index, w,
    // The set's species and item, not what the leads have by now: a lead
    // Minior is Meteor from Shields Down, and a White Herb may already be
    // spent on an Intimidate.
    teams: ['p1', 'p2'].map(s => mons[s].map(p => ({
      species: p.baseSpecies.id,
      speciesBase: p.baseSpecies.baseSpecies.toLowerCase().replace(/[^a-z0-9]/g, ''),
      level: p.level, gender: p.gender,
      item: p.set.item ? battle.dex.items.get(p.set.item).id : '', ability: p.baseAbility,
      moves: p.baseMoveSlots.map(m => m.id), pp: p.baseMoveSlots.map(m => m.maxpp),
      tera: p.teraType, evs: p.set.evs, ivs: p.set.ivs,
      stats: {hp: p.maxhp, ...p.storedStats},
    }))),
    initial: snapshot(battle),
    steps: [],
  };

  const rand = chooser(index * 7919 + Math.round(w / 256));
  let logAt = battle.log.length;
  while (!battle.ended && record.steps.length < MAX_DECISIONS) {
    const sides = [battle.p1, battle.p2];
    const legal = sides.map(s => options(battle, s));
    if (!legal[0] && !legal[1]) { record.error = 'no side has a decision'; break; }
    const choices = sides.map((s, i) => {
      if (!legal[i]) return '';
      let c = pick(rand, legal[i]);
      if (!s.choose(c)) { s.clearChoice(); c = 'default'; s.choose(c); }
      return c;
    });
    const actions = sides.map(s => toAction(battle, s));
    try {
      battle.commitChoices();
    } catch (e) {
      record.error = `commit: ${String(e.message).slice(0, 200)}`;
      break;
    }
    const log = battle.log.slice(logAt).filter(l => l && !l.startsWith('|t:|') &&
      !l.startsWith('|split|') && !l.startsWith('|request|') && l !== '|');
    logAt = battle.log.length;
    record.steps.push({choices, actions, legal: legal.map(l => l && l.ids), log,
                       after: snapshot(battle)});
  }
  return record;
}

const words = FRACTIONS.map(word);
const out = [];
const t0 = Date.now();
for (let i = 0; i < N; i++) {
  for (const w of words) out.push(play(i, w));
}
fs.writeFileSync(OUT, JSON.stringify(out));
const steps = out.reduce((n, r) => n + r.steps.length, 0);
const errs = out.filter(r => r.error);
console.log(`${out.length} battles (${N} team pairs x ${words.length} words), ` +
  `${steps} decisions, ${((Date.now() - t0) / 1000).toFixed(1)}s -> ${OUT}`);
console.log(`ended: ${out.filter(r => r.steps.length && r.steps[r.steps.length - 1].after.ended).length}`);
for (const e of errs.slice(0, 10)) console.log(`  ERROR ${e.id}: ${e.error}`);
