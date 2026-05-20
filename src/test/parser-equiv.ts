// Sanity test: the hand-rolled parser must produce the same 14-dim vector as
// the reference path (JSON.parse + vectorize). Runs against every payload in
// test-data.json.

import { promises as fs } from 'node:fs';
import { DIMS } from '../quantize.ts';
import { vectorize, type Normalization, type MccRiskMap } from '../vector.ts';
import { makeParseContext, parseAndVectorize } from '../parseJson.ts';

const DATA_DIR = process.env.DATA_DIR ?? './data';

async function main(): Promise<void> {
  const [normRaw, mccRaw, tdRaw] = await Promise.all([
    fs.readFile(`${DATA_DIR}/normalization.json`, 'utf8'),
    fs.readFile(`${DATA_DIR}/mcc_risk.json`, 'utf8'),
    fs.readFile(`${DATA_DIR}/test-data.json`, 'utf8'),
  ]);
  const norm = JSON.parse(normRaw) as Normalization;
  const mccRisk = JSON.parse(mccRaw) as MccRiskMap;
  const td = JSON.parse(tdRaw);
  const entries = td.entries;

  const refVec = new Float32Array(DIMS);
  const fastVec = new Float32Array(DIMS);
  const ctx = makeParseContext(norm, mccRisk);

  let checked = 0;
  let mismatches = 0;
  const sampleMismatches: string[] = [];

  for (const entry of entries) {
    const payload = entry.request;
    vectorize(payload, refVec, { norm, mccRisk });

    const body = Buffer.from(JSON.stringify(payload), 'utf8');
    const ok = parseAndVectorize(body, body.length, fastVec, ctx);
    if (!ok) {
      mismatches++;
      if (sampleMismatches.length < 3) {
        sampleMismatches.push(`PARSE FAIL: ${payload.id}`);
      }
      continue;
    }

    let differ = false;
    for (let i = 0; i < DIMS; i++) {
      if (Math.abs(refVec[i] - fastVec[i]) > 1e-6) {
        differ = true;
        break;
      }
    }
    if (differ) {
      mismatches++;
      if (sampleMismatches.length < 3) {
        const ref = Array.from(refVec).map((v) => v.toFixed(5)).join(',');
        const fast = Array.from(fastVec).map((v) => v.toFixed(5)).join(',');
        sampleMismatches.push(`${payload.id}\n  ref=[${ref}]\n  fast=[${fast}]`);
      }
    }
    checked++;
  }

  console.log(`[parser-equiv] checked=${checked} mismatches=${mismatches}`);
  if (sampleMismatches.length > 0) {
    for (const s of sampleMismatches) console.log(s);
    process.exit(1);
  }
}

main().catch((e) => {
  console.error('[parser-equiv] FAILED:', e);
  process.exit(1);
});
