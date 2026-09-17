/**
 * Benchmark Showdown's own simulator on complete Gen 9 Random Battles.
 *
 * Both sides play random legal choices, matching what `rollout_batch` does on
 * the JAX side. This drives `Battle` directly rather than going through
 * BattleStream, so it measures the engine rather than the protocol plumbing --
 * the favourable reading for Showdown, and the fair one against an engine that
 * emits no protocol at all.
 *
 *   node tools/showdown_bench.js [battles]
 */
'use strict';
const {Battle, Teams, Dex} = require('pokemon-showdown');

const N = parseInt(process.argv[2] || '100', 10);
const MAX_TURNS = 500;

/** A random legal choice for one side, from its current request. */
function choose(side) {
  // A side with nothing to decide takes an empty choice; `undefined` would
  // leave it unregistered and makeChoices would refuse to commit the turn.
  const req = side.activeRequest;
  if (!req || req.wait) return '';
  if (req.teamPreview) return 'default';

  const switches = [];
  side.pokemon.forEach((p, i) => {
    if (!p.fainted && !p.isActive) switches.push(`switch ${i + 1}`);
  });
  if (req.forceSwitch) {
    return switches.length ? switches[Math.floor(Math.random() * switches.length)]
                           : 'default';
  }
  const options = [];
  const active = req.active && req.active[0];
  if (active && active.moves) {
    active.moves.forEach((m, i) => {
      if (!m.disabled && m.pp !== 0) options.push(`move ${i + 1}`);
    });
  }
  if (!active || !active.trapped) options.push(...switches);
  if (!options.length) return 'default';
  return options[Math.floor(Math.random() * options.length)];
}

let turns = 0, finished = 0, fallbacks = 0;
const genStart = Date.now();
const teams = [];
for (let i = 0; i < N; i++) {
  teams.push([Teams.generate('gen9randombattle'), Teams.generate('gen9randombattle')]);
}
const genTime = (Date.now() - genStart) / 1000;

const t0 = Date.now();
for (let i = 0; i < N; i++) {
  const battle = new Battle({formatid: 'gen9randombattle', seed: [i, 2, 3, 4]});
  battle.setPlayer('p1', {team: teams[i][0]});
  battle.setPlayer('p2', {team: teams[i][1]});
  let guard = 0;
  while (!battle.ended && guard++ < MAX_TURNS) {
    const a = choose(battle.p1), b = choose(battle.p2);
    if (a === '' && b === '') break;
    try {
      battle.makeChoices(a, b);
    } catch (e) {
      // A random pick occasionally lands on something the request does not
      // actually allow (trapping and the like). Fall back rather than abort;
      // `fallbacks` reports how often, so it can be judged.
      fallbacks++;
      try { battle.makeChoices('default', 'default'); } catch (e2) { break; }
    }
  }
  turns += battle.turn;
  if (battle.ended) finished++;
}
const dt = (Date.now() - t0) / 1000;

console.log(JSON.stringify({
  engine: 'showdown',
  battles: N,
  finished,
  team_generation_s: +genTime.toFixed(2),
  battle_time_s: +dt.toFixed(2),
  battles_per_s: +(N / dt).toFixed(1),
  mean_turns: +(turns / N).toFixed(1),
  turns_per_s: Math.round(turns / dt),
  choice_fallbacks: fallbacks,
}, null, 1));
