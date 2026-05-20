// Find the specific failing test case (FP) when running with the current Rust kNN.

import { promises as fs } from 'node:fs';
import { DIMS } from '../quantize.ts';
import { vectorize, type Normalization, type MccRiskMap } from '../vector.ts';
import { initKnn, knnFraudCount } from '../../core/index.ts';

const DATA_DIR = process.env.DATA_DIR ?? './data';

async function main(): Promise<void> {
  console.log('init kNN...');
  initKnn();

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

  for (const entry of entries) {
    vectorize(entry.request, queryFloat, { norm, mccRisk });
    const count = knnFraudCount(
      queryFloat[0], queryFloat[1], queryFloat[2], queryFloat[3],
      queryFloat[4], queryFloat[5], queryFloat[6], queryFloat[7],
      queryFloat[8], queryFloat[9], queryFloat[10], queryFloat[11],
      queryFloat[12], queryFloat[13],
    );
    const ourApproved = count < 3;
    const exp = entry.expected_approved;
    if (ourApproved !== exp) {
      console.log(`MISMATCH id=${entry.request.id}`);
      console.log(`  expected_approved=${exp}, expected_fraud_score=${entry.expected_fraud_score}`);
      console.log(`  our count=${count}, approved=${ourApproved}`);
      console.log(`  vector=[${Array.from(queryFloat).map(v => v.toFixed(4)).join(',')}]`);
      console.log(`  payload:`, JSON.stringify(entry.request));
    }
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
