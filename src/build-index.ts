// Build-time pipeline. Reads:
//   data/references.json.gz   3M labeled vectors
//   data/normalization.json   constants (only echoed for verification)
//   data/mcc_risk.json        (not consumed here, runtime only)
//
// Emits:
//   data/vectors.bin   3M * 14 = 42 MB raw int8
//   data/labels.bin    3M / 8  = 375 KB packed bits (1 = fraud)
//   data/vptree.bin    16 + 3M * 16 = ~48 MB VP-tree
//
// Designed to run inside a Docker build stage with --max-old-space-size=4096.
// Streams via stream-json to avoid loading 284 MB of decompressed JSON into a
// single Node string.

import { createReadStream, promises as fs } from 'node:fs';
import { createGunzip } from 'node:zlib';
import parserPkg from 'stream-json/Parser.js';
import streamArrayPkg from 'stream-json/streamers/StreamArray.js';
const { parser } = parserPkg as { parser: () => any };
const { streamArray } = streamArrayPkg as { streamArray: () => any };

import { DIMS, SENTINEL, QUANT_SCALE } from './quantize.ts';
import { buildVpTree, serializeVpTree } from './vptree.ts';

const DATA_DIR = process.env.DATA_DIR ?? './data';

async function main(): Promise<void> {
  const t0 = Date.now();
  console.log(`[build-index] data dir: ${DATA_DIR}`);

  const refsPath = `${DATA_DIR}/references.json.gz`;
  const vectorsOut = `${DATA_DIR}/vectors.bin`;
  const labelsOut = `${DATA_DIR}/labels.bin`;
  const vptreeOut = `${DATA_DIR}/vptree.bin`;

  // 1. Stream-decompress + parse + quantize.
  const initialCap = 3_000_000;
  let vectors = new Int8Array(initialCap * DIMS);
  let labels = new Uint8Array(Math.ceil(initialCap / 8));
  let count = 0;
  let cap = initialCap;

  console.log(`[build-index] streaming ${refsPath}`);

  const stream = createReadStream(refsPath)
    .pipe(createGunzip())
    .pipe(parser())
    .pipe(streamArray());

  for await (const chunk of stream) {
    const { value } = chunk as { value: { vector: number[]; label: string } };
    const vec = value.vector;
    const label = value.label;

    if (count >= cap) {
      const newCap = Math.floor(cap * 1.5);
      const newVec = new Int8Array(newCap * DIMS);
      newVec.set(vectors);
      vectors = newVec;
      const newLab = new Uint8Array(Math.ceil(newCap / 8));
      newLab.set(labels);
      labels = newLab;
      cap = newCap;
    }

    const base = count * DIMS;
    for (let j = 0; j < DIMS; j++) {
      const v = vec[j];
      if (v === -1) {
        vectors[base + j] = SENTINEL;
      } else {
        let q = Math.round(v * QUANT_SCALE);
        if (q < 0) q = 0;
        if (q > 127) q = 127;
        vectors[base + j] = q;
      }
    }

    if (label === 'fraud') {
      labels[count >> 3] |= 1 << (count & 7);
    }

    count++;
    if (count % 500_000 === 0) {
      console.log(`[build-index]   parsed ${count.toLocaleString()} vectors`);
    }
  }

  const N = count;
  console.log(`[build-index] total vectors: ${N.toLocaleString()}`);

  // Truncate to actual size.
  vectors = vectors.slice(0, N * DIMS);
  labels = labels.slice(0, Math.ceil(N / 8));

  // 2. Write vectors.bin and labels.bin first (so they survive even if VP-tree
  //    build runs out of RAM).
  console.log(`[build-index] writing ${vectorsOut} (${vectors.byteLength} bytes)`);
  await fs.writeFile(vectorsOut, vectors);
  console.log(`[build-index] writing ${labelsOut} (${labels.byteLength} bytes)`);
  await fs.writeFile(labelsOut, labels);

  // 3. Build VP-tree.
  console.log(`[build-index] building VP-tree over ${N.toLocaleString()} points`);
  const tBuild = Date.now();
  const tree = buildVpTree(vectors, N);
  console.log(`[build-index]   built in ${((Date.now() - tBuild) / 1000).toFixed(1)}s, nodes=${tree.size}`);

  console.log(`[build-index] writing ${vptreeOut}`);
  const serialized = serializeVpTree(tree);
  await fs.writeFile(vptreeOut, serialized);
  console.log(`[build-index]   ${serialized.byteLength} bytes`);

  console.log(`[build-index] done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}

main().catch((err) => {
  console.error('[build-index] FAILED:', err);
  process.exit(1);
});
