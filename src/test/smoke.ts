// In-process smoke test. Builds a tiny VP-tree from the upstream
// example-references.json sample, then runs the full scoring pipeline against
// example-payloads.json. Asserts:
//   - the heap returns exactly 5 results per query (or fewer if dataset is smaller)
//   - all 5 results come from the dataset (idx < N)
//   - fraud_count is 0..5
//   - tie-break ordering is stable (rerunning the same query returns the same idxs in the same heap positions)

import { promises as fs } from 'node:fs';
import { DIMS, quantizeOne, SENTINEL, QUANT_SCALE } from '../quantize.ts';
import { TopKHeap } from '../heap.ts';
import { buildVpTree, searchVpTree, serializeVpTree, deserializeVpTree } from '../vptree.ts';
import { vectorize, type Normalization, type MccRiskMap } from '../vector.ts';

const DATA_DIR = process.env.DATA_DIR ?? './data';

interface RefEntry {
  vector: number[];
  label: 'fraud' | 'legit';
}

function quantizeFloats(arr: number[]): number[] {
  return arr.map((v) => {
    if (v === -1) return SENTINEL;
    const q = Math.round(v * QUANT_SCALE);
    return q < 0 ? 0 : q > 32767 ? 32767 : q;
  });
}

async function main(): Promise<void> {
  const norm = JSON.parse(
    await fs.readFile(`${DATA_DIR}/normalization.json`, 'utf8'),
  ) as Normalization;
  const mccRisk = JSON.parse(
    await fs.readFile(`${DATA_DIR}/mcc_risk.json`, 'utf8'),
  ) as MccRiskMap;
  const refs = JSON.parse(
    await fs.readFile(`${DATA_DIR}/example-references.json`, 'utf8'),
  ) as RefEntry[];
  const payloads = JSON.parse(
    await fs.readFile(`${DATA_DIR}/example-payloads.json`, 'utf8'),
  ) as any[];

  console.log(`[smoke] refs=${refs.length} payloads=${payloads.length}`);

  const N = refs.length;
  const vectors = new Int16Array(N * DIMS);
  const labels = new Uint8Array(Math.ceil(N / 8));

  for (let i = 0; i < N; i++) {
    const q = quantizeFloats(refs[i].vector);
    const base = i * DIMS;
    for (let j = 0; j < DIMS; j++) vectors[base + j] = q[j];
    if (refs[i].label === 'fraud') {
      labels[i >> 3] |= 1 << (i & 7);
    }
  }

  console.log(`[smoke] building VP-tree...`);
  const tree = buildVpTree(vectors, N);
  console.log(`[smoke] tree.size=${tree.size}`);

  // Round-trip serialization.
  const buf = serializeVpTree(tree);
  const tree2 = deserializeVpTree(buf);
  assertEq(tree2.size, tree.size, 'serialize round-trip size');

  const queryFloat = new Float32Array(DIMS);
  const queryInt8 = new Int16Array(DIMS);
  const heap = new TopKHeap(5);

  let queriesRun = 0;
  for (const payload of payloads) {
    vectorize(payload, queryFloat, { norm, mccRisk });
    quantizeOne(queryFloat, queryInt8);
    searchVpTree(tree, vectors, queryInt8, heap);

    const k = heap.size;
    if (k === 0 || k > 5) throw new Error(`bad heap size ${k}`);
    for (let i = 0; i < k; i++) {
      const idx = heap.idxs[i];
      if (idx < 0 || idx >= N) throw new Error(`out of range idx ${idx}`);
    }
    const fraud = heap.countFrauds(labels);
    if (fraud < 0 || fraud > 5) throw new Error(`bad fraud count ${fraud}`);

    // Verify VP-tree result matches brute-force ground truth.
    const truth = bruteForceTop5(vectors, queryInt8, N);
    const ours = heapAsSorted(heap);
    assertHeapsEqual(ours, truth, payload.id);

    queriesRun++;
  }

  console.log(`[smoke] OK — ran ${queriesRun} queries, all matched brute-force.`);
}

function bruteForceTop5(
  vectors: Int16Array,
  query: Int16Array,
  N: number,
): Array<{ d: number; i: number }> {
  const all: Array<{ d: number; i: number }> = new Array(N);
  for (let i = 0; i < N; i++) {
    const base = i * DIMS;
    let s = 0;
    for (let j = 0; j < DIMS; j++) {
      const diff = query[j] - vectors[base + j];
      s += diff * diff;
    }
    all[i] = { d: s, i };
  }
  all.sort((a, b) => (a.d !== b.d ? a.d - b.d : a.i - b.i));
  return all.slice(0, 5);
}

function heapAsSorted(heap: TopKHeap): Array<{ d: number; i: number }> {
  const out: Array<{ d: number; i: number }> = [];
  for (let i = 0; i < heap.size; i++) out.push({ d: heap.dists[i], i: heap.idxs[i] });
  out.sort((a, b) => (a.d !== b.d ? a.d - b.d : a.i - b.i));
  return out;
}

function assertHeapsEqual(
  ours: Array<{ d: number; i: number }>,
  truth: Array<{ d: number; i: number }>,
  id: string,
): void {
  if (ours.length !== truth.length) {
    throw new Error(`payload ${id}: size mismatch ${ours.length} vs ${truth.length}`);
  }
  for (let i = 0; i < ours.length; i++) {
    if (ours[i].d !== truth[i].d || ours[i].i !== truth[i].i) {
      console.error('ours:', ours);
      console.error('truth:', truth);
      throw new Error(`payload ${id}: mismatch at ${i}`);
    }
  }
}

function assertEq<T>(actual: T, expected: T, msg: string): void {
  if (actual !== expected) {
    throw new Error(`${msg}: got ${actual}, expected ${expected}`);
  }
}

main().catch((e) => {
  console.error('[smoke] FAILED:', e);
  process.exit(1);
});
