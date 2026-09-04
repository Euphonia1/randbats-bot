/**
 * Dump declarative Pokemon Showdown Gen 9 data to JSON for the JAX engine.
 *
 * Showdown encodes most move behaviour as JS callbacks, which cannot cross into
 * JAX. We export every declarative field verbatim, plus the *names* of the
 * callbacks each effect defines, so the Python builder can tell which entries
 * are fully described by data and which need a hand-written handler.
 */
'use strict';
const fs = require('fs');
const path = require('path');
const {Dex} = require('pokemon-showdown');

const GEN = 9;
const dex = Dex.forGen(GEN);
const OUT = path.join(__dirname, '..', 'data', 'gen9_raw.json');

/** Names of function-valued properties -- these are the behaviours we must port by hand. */
function callbacks(obj) {
  const out = [];
  for (const k in obj) {
    try {
      if (typeof obj[k] === 'function') out.push(k);
    } catch (e) { /* getters that throw */ }
  }
  return out.sort();
}

/** Showdown uses `true` for "bypasses accuracy checks"; -1 keeps the field numeric. */
const acc = (a) => (a === true ? -1 : a);

function dumpMoves() {
  const out = {};
  for (const m of dex.moves.all()) {
    if (m.isNonstandard === 'CAP') continue;
    if (m.isZ || m.isMax) continue;
    out[m.id] = {
      name: m.name, num: m.num, type: m.type, category: m.category,
      basePower: m.basePower, accuracy: acc(m.accuracy), pp: m.pp,
      priority: m.priority, target: m.target, flags: m.flags || {},
      critRatio: m.critRatio ?? 1, willCrit: m.willCrit ?? null,
      drain: m.drain || null, recoil: m.recoil || null, heal: m.heal || null,
      multihit: m.multihit ?? null, multiaccuracy: !!m.multiaccuracy,
      ohko: m.ohko ?? null, damage: m.damage ?? null,
      status: m.status || null, volatileStatus: m.volatileStatus || null,
      sideCondition: m.sideCondition || null, slotCondition: m.slotCondition || null,
      weather: m.weather || null, terrain: m.terrain || null,
      pseudoWeather: m.pseudoWeather || null,
      boosts: m.boosts || null,
      self: m.self ? {
        boosts: m.self.boosts || null,
        volatileStatus: m.self.volatileStatus || null,
        sideCondition: m.self.sideCondition || null,
        chance: m.self.chance ?? null,
      } : null,
      secondaries: (m.secondaries || []).map(s => ({
        chance: s.chance ?? 100, status: s.status || null,
        volatileStatus: s.volatileStatus || null, boosts: s.boosts || null,
        self: s.self ? {boosts: s.self.boosts || null, volatileStatus: s.self.volatileStatus || null} : null,
        dustproof: !!s.dustproof, kingsrock: !!s.kingsrock,
      })),
      forceSwitch: !!m.forceSwitch, selfSwitch: m.selfSwitch ?? null,
      selfdestruct: m.selfdestruct ?? null,
      breaksProtect: !!m.breaksProtect, stallingMove: !!m.stallingMove,
      thawsTarget: !!m.thawsTarget, sleepUsable: !!m.sleepUsable,
      smartTarget: !!m.smartTarget, tracksTarget: !!m.tracksTarget,
      ignoreAbility: !!m.ignoreAbility, ignoreDefensive: !!m.ignoreDefensive,
      ignoreEvasion: !!m.ignoreEvasion, ignoreImmunity: m.ignoreImmunity ?? false,
      ignoreOffensive: !!m.ignoreOffensive,
      ignorePositiveDefensive: !!m.ignorePositiveDefensive,
      ignoreNegativeOffensive: !!m.ignoreNegativeOffensive,
      overrideOffensiveStat: m.overrideOffensiveStat || null,
      overrideOffensivePokemon: m.overrideOffensivePokemon || null,
      overrideDefensiveStat: m.overrideDefensiveStat || null,
      overrideDefensivePokemon: m.overrideDefensivePokemon || null,
      defensiveCategory: m.defensiveCategory || null,
      useSourceDefensiveAsOffensive: !!m.useSourceDefensiveAsOffensive,
      hasCrashDamage: !!m.hasCrashDamage, mindBlownRecoil: !!m.mindBlownRecoil,
      struggleRecoil: !!m.struggleRecoil, noDamageVariance: !!m.noDamageVariance,
      isFutureMove: !!m.isFutureMove, hasSheerForceBoost: !!m.hasSheerForceBoost,
      nonGhostTarget: m.nonGhostTarget || null,
      condition: m.condition ? {
        duration: m.condition.duration ?? null,
        callbacks: callbacks(m.condition),
      } : null,
      isNonstandard: m.isNonstandard || null, gen: m.gen,
      shortDesc: m.shortDesc || '',
      callbacks: callbacks(m),
    };
  }
  return out;
}

function dumpSpecies() {
  const out = {};
  for (const s of dex.species.all()) {
    if (s.isNonstandard && s.isNonstandard !== 'Unobtainable') continue;
    out[s.id] = {
      name: s.name, num: s.num, types: s.types,
      baseStats: s.baseStats, abilities: Object.values(s.abilities),
      weightkg: s.weightkg, baseSpecies: s.baseSpecies, forme: s.forme,
      nfe: !!s.nfe, gender: s.gender || '',
      requiredItem: s.requiredItem || null,
    };
  }
  return out;
}

function dumpTypechart() {
  const out = {};
  for (const t of dex.types.all()) {
    out[t.id] = {name: t.name, damageTaken: t.damageTaken, HPivs: t.HPivs || null};
  }
  return out;
}

function dumpAbilities() {
  const out = {};
  for (const a of dex.abilities.all()) {
    if (a.isNonstandard === 'CAP') continue;
    out[a.id] = {
      name: a.name, num: a.num, rating: a.rating,
      suppressWeather: !!a.suppressWeather,
      breakable: !!a.flags?.breakable, cantsuppress: !!a.flags?.cantsuppress,
      flags: a.flags || {},
      callbacks: callbacks(a),
    };
  }
  return out;
}

function dumpItems() {
  const out = {};
  for (const it of dex.items.all()) {
    if (it.isNonstandard === 'CAP') continue;
    out[it.id] = {
      name: it.name, num: it.num,
      isBerry: !!it.isBerry, isChoice: !!it.isChoice, isGem: !!it.isGem,
      megaStone: it.megaStone || null, zMove: it.zMove || null,
      naturalGift: it.naturalGift || null, fling: it.fling || null,
      onPlate: it.onPlate || null, onMemory: it.onMemory || null,
      itemUser: it.itemUser || null, boosts: it.boosts || null,
      ignoreKlutz: !!it.ignoreKlutz,
      callbacks: callbacks(it),
    };
  }
  return out;
}

function dumpConditions() {
  // Statuses, weathers, terrains, hazards -- their durations and callbacks.
  const out = {};
  const ids = ['brn','par','slp','frz','psn','tox','confusion','flinch','trapped','partiallytrapped',
    'sunnyday','raindance','sandstorm','snowscape','desolateland','primordialsea','deltastream',
    'electricterrain','grassyterrain','mistyterrain','psychicterrain','trickroom','gravity'];
  for (const id of ids) {
    const c = dex.conditions.get(id);
    if (!c || !c.exists) continue;
    out[id] = {name: c.name, duration: c.duration ?? null, callbacks: callbacks(c)};
  }
  return out;
}

function dumpRandbats() {
  const p = require.resolve('pokemon-showdown/data/random-battles/gen9/sets.json');
  return JSON.parse(fs.readFileSync(p, 'utf8'));
}

const data = {
  gen: GEN,
  showdownVersion: require('pokemon-showdown/package.json').version,
  natures: Object.fromEntries(dex.natures.all().map(n => [n.id, {name: n.name, plus: n.plus || null, minus: n.minus || null}])),
  typechart: dumpTypechart(),
  species: dumpSpecies(),
  moves: dumpMoves(),
  abilities: dumpAbilities(),
  items: dumpItems(),
  conditions: dumpConditions(),
  randbats: dumpRandbats(),
};

fs.writeFileSync(OUT, JSON.stringify(data));
const n = (o) => Object.keys(o).length;
console.log(`wrote ${OUT}`);
console.log(`  showdown v${data.showdownVersion}`);
console.log(`  types=${n(data.typechart)} species=${n(data.species)} moves=${n(data.moves)} abilities=${n(data.abilities)} items=${n(data.items)} randbats=${n(data.randbats)}`);
