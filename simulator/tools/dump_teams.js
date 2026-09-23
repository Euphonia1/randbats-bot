/**
 * Dump Gen 9 Random Battle teams from Showdown's own generator.
 *
 * Team generation is not something the JAX engine should be doing. Showdown's
 * generator is three thousand lines of sequential logic -- per-species move
 * enforcement, team-level state that accumulates as the team is built, item
 * rules keyed on the set's role and move list -- and none of it is expressible
 * as a fixed-shape traced computation. Calling the real thing and feeding the
 * result in as data gives exact fidelity for items, moves, abilities and Tera
 * types at no cost to the simulator.
 *
 * Emits newline-delimited JSON, one team per line, which
 * `tools/build_team_pool.py` compiles into `data/team_pool.npz`.
 *
 *   node tools/dump_teams.js [teams] > data/team_pool.jsonl
 */
'use strict';
const { Teams, Dex } = require('pokemon-showdown');
const dex = Dex.forFormat('gen9randombattle');

const N = parseInt(process.argv[2] || '10000', 10);

// Showdown reports progress on stderr so stdout stays a clean data stream.
const tick = Math.max(1, Math.floor(N / 20));
const t0 = Date.now();

for (let i = 0; i < N; i++) {
  const team = Teams.generate('gen9randombattle');
  // Keep only what the simulator needs to instantiate a Pokemon. Stats, types
  // and PP are derived on the Python side from the tables it already has, so
  // there is one definition of those and it is the one under test.
  const out = team.map(set => ({
    species: set.species,
    // Cosmetic formes (Florges-Blue, Vivillon-Fancy) battle as themselves but
    // share everything mechanical with the base, which is how the engine
    // indexes them. Carry the base so the compiler can fall back to it.
    base: dex.species.get(set.species).baseSpecies,
    level: set.level,
    item: set.item || '',
    ability: set.ability,
    moves: set.moves,
    tera: set.teraType,
    evs: set.evs,
    ivs: set.ivs,
  }));
  process.stdout.write(JSON.stringify(out) + '\n');
  if ((i + 1) % tick === 0) {
    const pct = Math.round(((i + 1) / N) * 100);
    process.stderr.write(`  ${pct}%  ${i + 1}/${N}\n`);
  }
}

const dt = (Date.now() - t0) / 1000;
process.stderr.write(`${N} teams in ${dt.toFixed(1)}s (${Math.round(N / dt)}/s)\n`);
