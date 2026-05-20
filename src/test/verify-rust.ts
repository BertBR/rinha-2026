// Verify the Rust kNN module against test-data.json.

import { promises as fs } from 'node:fs';
import { DIMS } from '../quantize.ts';
import { vectorize, type Normalization, type MccRiskMap } from '../vector.ts';
import { initKnn, knnFraudCount } from '../../core/index.ts';

const DATA_DIR = process.env.DATA_DIR ?? './data';

async function main(): Promise<void> {
  console.log('[verify-rust] init kNN');
  const t0 = performance.now();
  initKnn();
  console.log(`  done in ${(performance.now() - t0).toFixed(1)} ms`);

  const [normRaw, mccRaw, tdRaw] = await Promise.all([
    fs.readFile(`${DATA_DIR}/normalization.json`, 'utf8'),
    fs.readFile(`${DATA_DIR}/mcc_risk.json`, 'utf8'),
    fs.readFile(`${DATA_DIR}/test-data.json`, 'utf8'),
  ]);
  const norm = JSON.parse(normRaw) as Normalization;
  const mccRisk = JSON.parse(mccRaw) as MccRiskMap;
  const td = JSON.parse(tdRaw);
  const entries = td.entries;

  const queryFloat = new Float32Array(DIMS);

  let tp = 0, tn = 0, fp = 0, fn = 0;
  let elapsed = 0n;

  for (const entry of entries) {
    vectorize(entry.request, queryFloat, { norm, mccRisk });
    const t = process.hrtime.bigint();
    const count = knnFraudCount(
      queryFloat[0],
      queryFloat[1],
      queryFloat[2],
      queryFloat[3],
      queryFloat[4],
      queryFloat[5],
      queryFloat[6],
      queryFloat[7],
      queryFloat[8],
      queryFloat[9],
      queryFloat[10],
      queryFloat[11],
      queryFloat[12],
      queryFloat[13],
    );
    elapsed += process.hrtime.bigint() - t;

    const ourApproved = count < 3;
    const expectedApproved = entry.expected_approved;

    if (expectedApproved && ourApproved) tn++;
    else if (!expectedApproved && !ourApproved) tp++;
    else if (expectedApproved && !ourApproved) fp++;
    else fn++;
  }

  const total = tp + tn + fp + fn;
  console.log(`[verify-rust] TP=${tp} TN=${tn} FP=${fp} FN=${fn}`);
  console.log(`[verify-rust] failure rate: ${((fp + fn) / total * 100).toFixed(3)}%`);
  console.log(`[verify-rust] avg per-query: ${(Number(elapsed) / total / 1000).toFixed(1)} µs`);

  if (fp === 0 && fn === 0) {
    console.log('[verify-rust] OK: 0 FP / 0 FN — perfect detection');
  } else {
    console.log('[verify-rust] mismatch — likely NPROBE too low or quantization edge cases');
  }
}

main().catch((e) => {
  console.error('[verify-rust] FAILED:', e);
  process.exit(1);
});
