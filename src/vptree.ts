// VP-tree over int16-quantized 14-dim vectors. Exact k-NN with tie-break on
// original index, matching the grader's brute-force ordering.
//
// Storage layout (in-memory + on-disk, struct-of-arrays for cache locality):
//   threshold[node] : Float32 actual Euclidean distance to the median of vp's
//                              distances to siblings  (4 * N bytes)
//   packed[node*9 .. node*9+9] : 9 bytes per node, three 24-bit fields:
//     bytes 0..2  : pointIdx (uint24, max 16M, fits 3M+)
//     bytes 3..5  : leftOffset (int24, 0xFFFFFF = -1 = no child)
//     bytes 6..8  : rightOffset (int24, same encoding)
//
// 13 bytes per node total (vs 16 in the original). At 3M nodes that saves
// 9 MB, which is critical for fitting int16 vectors + V8 baseline into the
// 165 MB per-container budget.

import { DIMS, distanceSqInt16 } from './quantize.ts';
import { TopKHeap } from './heap.ts';

const NULL_OFFSET = 0xffffff;

export interface VpTree {
  threshold: Float32Array;     // size N
  packed: Uint8Array;          // size N * 9
  size: number;
}

const STACK_DEPTH = 128;

function readUint24(buf: Uint8Array, offset: number): number {
  return buf[offset] | (buf[offset + 1] << 8) | (buf[offset + 2] << 16);
}

function writeUint24(buf: Uint8Array, offset: number, value: number): void {
  buf[offset] = value & 0xff;
  buf[offset + 1] = (value >> 8) & 0xff;
  buf[offset + 2] = (value >> 16) & 0xff;
}

export function buildVpTree(vectors: Int16Array, N: number): VpTree {
  const threshold = new Float32Array(N);
  const packed = new Uint8Array(N * 9);

  const indices = new Uint32Array(N);
  for (let i = 0; i < N; i++) indices[i] = i;

  const dists = new Float64Array(N);
  let nodeCount = 0;

  const stack: number[] = [];
  stack.push(0, N, -1, 0);

  while (stack.length > 0) {
    const side = stack.pop()!;
    const parentNode = stack.pop()!;
    const end = stack.pop()!;
    const start = stack.pop()!;

    if (start >= end) {
      if (parentNode >= 0) {
        const fieldOffset = parentNode * 9 + (side === 1 ? 3 : 6);
        writeUint24(packed, fieldOffset, NULL_OFFSET);
      }
      continue;
    }

    const nodeIdx = nodeCount++;
    if (parentNode >= 0) {
      const fieldOffset = parentNode * 9 + (side === 1 ? 3 : 6);
      writeUint24(packed, fieldOffset, nodeIdx);
    }

    const base = nodeIdx * 9;

    if (end - start === 1) {
      writeUint24(packed, base, indices[start]);
      writeUint24(packed, base + 3, NULL_OFFSET);
      writeUint24(packed, base + 6, NULL_OFFSET);
      threshold[nodeIdx] = 0;
      continue;
    }

    const vpOrigIdx = indices[start];
    writeUint24(packed, base, vpOrigIdx);
    const vpOffset = vpOrigIdx * DIMS;

    for (let i = start + 1; i < end; i++) {
      const origIdx = indices[i];
      const offset = origIdx * DIMS;
      let sum = 0;
      for (let j = 0; j < DIMS; j++) {
        const d = vectors[vpOffset + j] - vectors[offset + j];
        sum += d * d;
      }
      dists[i] = sum;
    }

    const subStart = start + 1;
    const subEnd = end;
    sortByDist(indices, dists, subStart, subEnd);

    const subLen = subEnd - subStart;
    const medianPos = subStart + (subLen >> 1);
    threshold[nodeIdx] = Math.sqrt(dists[medianPos]);

    stack.push(medianPos, subEnd, nodeIdx, 2);
    stack.push(subStart, medianPos, nodeIdx, 1);
  }

  return {
    threshold: threshold.slice(0, nodeCount),
    packed: packed.slice(0, nodeCount * 9),
    size: nodeCount,
  };
}

function sortByDist(
  indices: Uint32Array,
  dists: Float64Array,
  start: number,
  end: number,
): void {
  if (end - start <= 1) return;
  const stack: number[] = [start, end];
  while (stack.length > 0) {
    const e = stack.pop()!;
    const s = stack.pop()!;
    if (e - s <= 16) {
      for (let i = s + 1; i < e; i++) {
        const di = dists[i];
        const ii = indices[i];
        let j = i - 1;
        while (j >= s && dists[j] > di) {
          dists[j + 1] = dists[j];
          indices[j + 1] = indices[j];
          j--;
        }
        dists[j + 1] = di;
        indices[j + 1] = ii;
      }
      continue;
    }
    const mid = (s + e) >> 1;
    const pivot = dists[mid];
    let i = s;
    let j = e - 1;
    while (i <= j) {
      while (dists[i] < pivot) i++;
      while (dists[j] > pivot) j--;
      if (i <= j) {
        const td = dists[i];
        const ti = indices[i];
        dists[i] = dists[j];
        indices[i] = indices[j];
        dists[j] = td;
        indices[j] = ti;
        i++;
        j--;
      }
    }
    if (s < j) stack.push(s, j + 1);
    if (i < e) stack.push(i, e);
  }
}

const SEARCH_STACK = new Int32Array(STACK_DEPTH);

export function searchVpTree(
  tree: VpTree,
  vectors: Int16Array,
  query: Int16Array,
  heap: TopKHeap,
): void {
  heap.reset();
  const stack = SEARCH_STACK;
  let stackTop = 0;
  stack[stackTop++] = 0;

  const packed = tree.packed;
  const thresholds = tree.threshold;

  while (stackTop > 0) {
    const nodeIdx = stack[--stackTop];
    const base = nodeIdx * 9;

    // Inline uint24 read for pointIdx (hot path).
    const vpPtIdx = packed[base] | (packed[base + 1] << 8) | (packed[base + 2] << 16);
    const vpOffset = vpPtIdx * DIMS;
    const dSq = distanceSqInt16(query, vectors, vpOffset);

    heap.tryInsert(dSq, vpPtIdx);

    const left = packed[base + 3] | (packed[base + 4] << 8) | (packed[base + 5] << 16);
    const right = packed[base + 6] | (packed[base + 7] << 8) | (packed[base + 8] << 16);
    if (left === NULL_OFFSET && right === NULL_OFFSET) continue;

    const tau = thresholds[nodeIdx];
    const d = Math.sqrt(dSq);
    const gap = d - tau;
    const gapSq = gap * gap;
    const tauSq = heap.worst();

    let near: number;
    let far: number;
    if (d <= tau) {
      near = left;
      far = right;
    } else {
      near = right;
      far = left;
    }

    if (far !== NULL_OFFSET && gapSq < tauSq) {
      stack[stackTop++] = far;
    }
    if (near !== NULL_OFFSET) {
      stack[stackTop++] = near;
    }
  }
}

// Serialize to one Buffer. Header: 16 bytes
//   uint32 magic = 0x56505433  ("VPT3", compact 13-bytes/node format)
//   uint32 size
//   uint32 reserved
//   uint32 reserved
// Then:
//   threshold (Float32 * N) — 4*N bytes
//   packed    (Uint8  * 9*N) — 9*N bytes
export function serializeVpTree(tree: VpTree): Buffer {
  const N = tree.size;
  const bufSize = 16 + 4 * N + 9 * N;
  const buf = Buffer.alloc(bufSize);
  buf.writeUInt32LE(0x56505433, 0);
  buf.writeUInt32LE(N, 4);

  const f32 = new Float32Array(buf.buffer, buf.byteOffset + 16, N);
  f32.set(tree.threshold);

  // Copy packed bytes (Buffer-aligned, no alignment constraint for Uint8)
  buf.set(tree.packed, 16 + 4 * N);

  return buf;
}

export function deserializeVpTree(buf: Buffer): VpTree {
  const magic = buf.readUInt32LE(0);
  if (magic !== 0x56505433) {
    throw new Error(`bad vptree magic: 0x${magic.toString(16)} (expected VPT3)`);
  }
  const N = buf.readUInt32LE(4);
  const base = buf.byteOffset + 16;
  const thresholdView = new Float32Array(buf.buffer, base, N);
  // The packed array follows; it has no alignment constraint, but using a
  // Uint8Array view backed by the same Buffer keeps it zero-copy.
  const packedView = new Uint8Array(buf.buffer, base + 4 * N, 9 * N);
  return {
    threshold: thresholdView,
    packed: packedView,
    size: N,
  };
}

// Backward-compatible compat layer for callers that referenced the old shape.
export const NULL_NODE = NULL_OFFSET;
