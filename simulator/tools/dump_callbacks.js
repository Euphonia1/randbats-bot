/** Print the JS source of the callbacks for the moves we hand-implement. */
'use strict';
const {Dex} = require('pokemon-showdown');
const dex = Dex.forGen(9);
const want = process.argv.slice(2);
for (const id of want) {
  const m = dex.moves.get(id);
  if (!m.exists) { console.log(`### ${id}: MISSING`); continue; }
  const cbs = [];
  for (const k in m) {
    try { if (typeof m[k] === 'function') cbs.push(k); } catch (e) {}
  }
  if (!cbs.length) continue;
  console.log(`### ${id} (bp=${m.basePower})`);
  for (const k of cbs) console.log(`  ${k}: ${String(m[k]).replace(/\n\s*/g, ' ').slice(0, 400)}`);
}
