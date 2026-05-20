// Verify our pipeline against the official test-data.json.
//
// test-data.json is the same payloads the grader uses, with pre-computed labels
// (approved=true|false) derived from exact brute-force k=5 with orig_id
// tie-break on the full 3M-vector reference set.
//
// If our local pipeline matches every label, we get the deterministic-grader
// upside: 0 FP / 0 FN, leaving ~3000 points on the detection axis no matter
// what our p99 ends up being.
//
// Requires data/vectors.bin, data/labels.bin, data/vptree.bin (run
// `make build-index` first).

import { promises as fs } from 'node:fs';
import { DIMS, quantizeOne } from '../quantize.ts';
import { TopKHeap } from '../heap.ts';
import { deserializeVpTree, searchVpTree } from '../vptree.ts';
import { vectorize, type Normalization, type MccRiskMap } from '../vector.ts';

const DATA_DIR = process.env.DATA_DIR ?? './data';

interface TestEntry {
  request: any;
  expected_approved: boolean;
  expected_fraud_score: number;
}

interface TestData {
  references_checksum_sha256?: string;
  stats?: any;
  entries: TestEntry[];
}

async function main(): Promise<void> {
  const [vBuf, lBuf, tBuf, normRaw, mccRaw, testRaw] = await Promise.all([
    fs.readFile(`${DATA_DIR}/vectors.bin`),
    fs.readFile(`${DATA_DIR}/labels.bin`),
    fs.readFile(`${DATA_DIR}/vptree.bin`),
    fs.readFile(`${DATA_DIR}/normalization.json`, 'utf8'),
    fs.readFile(`${DATA_DIR}/mcc_risk.json`, 'utf8'),
    fs.readFile(`${DATA_DIR}/test-data.json`, 'utf8'),
  ]);

  const vectors = new Int16Array(vBuf.buffer, vBuf.byteOffset, vBuf.byteLength / 2);
  const labels = new Uint8Array(lBuf.buffer, lBuf.byteOffset, lBuf.byteLength);
  const tree = deserializeVpTree(tBuf);
  const norm = JSON.parse(normRaw) as Normalization;
  const mccRisk = JSON.parse(mccRaw) as MccRiskMap;
  const td = JSON.parse(testRaw) as TestData;
  const testCases = td.entries;

  console.log(
    `[verify] vectors=${(vectors.byteLength / 1024 / 1024).toFixed(1)}MB tree=${tree.size} labels=${(labels.byteLength / 1024).toFixed(1)}KB cases=${testCases.length}`,
  );

  const queryFloat = new Float32Array(DIMS);
  const queryInt8 = new Int16Array(DIMS);
  const heap = new TopKHeap(5);

  let tp = 0;
  let tn = 0;
  let fp = 0;
  let fn = 0;
  let elapsed = 0;

  for (const entry of testCases) {
    const t0 = process.hrtime.bigint();
    vectorize(entry.request, queryFloat, { norm, mccRisk });
    quantizeOne(queryFloat, queryInt8);
    searchVpTree(tree, vectors, queryInt8, heap);
    const fraud = heap.countFrauds(labels);
    elapsed += Number(process.hrtime.bigint() - t0);

    const ourApproved = fraud < 3;
    const expectedApproved = entry.expected_approved;

    if (expectedApproved && ourApproved) tn++;
    else if (!expectedApproved && !ourApproved) tp++;
    else if (expectedApproved && !ourApproved) fp++;
    else fn++;
  }

  const total = tp + tn + fp + fn;
  const failureRate = (fp + fn) / total;
  const avgUs = elapsed / total / 1000;

  console.log(`[verify] TP=${tp} TN=${tn} FP=${fp} FN=${fn}`);
  console.log(`[verify] failure rate: ${(failureRate * 100).toFixed(3)}%`);
  console.log(`[verify] avg per-query: ${avgUs.toFixed(1)} µs`);

  if (fp !== 0 || fn !== 0) {
    console.error('[verify] DETECTION MISMATCH — debug tie-break and vector formula');
    process.exit(1);
  }
  console.log('[verify] OK: 0 FP / 0 FN');
}

main().catch((e) => {
  console.error('[verify] FAILED:', e);
  process.exit(1);
});
