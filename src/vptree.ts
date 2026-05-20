// VP-tree over int16-quantized 14-dim vectors. Exact k-NN with tie-break on
// original index, matching the grader's brute-force ordering.
//
// Storage layout (typed arrays, no objects):
//   pointIdx[node]    : Uint32  original index into vectors array
//   threshold[node]   : Float32 actual Euclidean distance (not squared) to the
//                                median of vp's distances to siblings
//   leftOffset[node]  : Int32   node index of left subtree root, -1 if none
//   rightOffset[node] : Int32   node index of right subtree root, -1 if none
//
// 16 bytes per node. 3M nodes → 48 MB. Kept simple (no bit-packing) because
// the access pattern is mostly cache-line-fetched and a packed format would
// trade saving 15 MB for several extra ALU ops per node visit.

import { DIMS, distanceSqInt16 } from './quantize.ts';
import { TopKHeap } from './heap.ts';

export interface VpTree {
  pointIdx: Uint32Array;
  threshold: Float32Array;
  leftOffset: Int32Array;
  rightOffset: Int32Array;
  size: number;
}

const STACK_DEPTH = 128;

export function buildVpTree(vectors: Int16Array, N: number): VpTree {
  const pointIdx = new Uint32Array(N);
  const threshold = new Float32Array(N);
  const leftOffset = new Int32Array(N);
  const rightOffset = new Int32Array(N);

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
        if (side === 1) leftOffset[parentNode] = -1;
        else rightOffset[parentNode] = -1;
      }
      continue;
    }

    const nodeIdx = nodeCount++;
    if (parentNode >= 0) {
      if (side === 1) leftOffset[parentNode] = nodeIdx;
      else rightOffset[parentNode] = nodeIdx;
    }

    if (end - start === 1) {
      pointIdx[nodeIdx] = indices[start];
      threshold[nodeIdx] = 0;
      leftOffset[nodeIdx] = -1;
      rightOffset[nodeIdx] = -1;
      continue;
    }

    const vpOrigIdx = indices[start];
    pointIdx[nodeIdx] = vpOrigIdx;
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
    pointIdx: pointIdx.slice(0, nodeCount),
    threshold: threshold.slice(0, nodeCount),
    leftOffset: leftOffset.slice(0, nodeCount),
    rightOffset: rightOffset.slice(0, nodeCount),
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

  const pointIdx = tree.pointIdx;
  const thresholds = tree.threshold;
  const leftOff = tree.leftOffset;
  const rightOff = tree.rightOffset;

  while (stackTop > 0) {
    const nodeIdx = stack[--stackTop];

    const vpPtIdx = pointIdx[nodeIdx];
    const vpOffset = vpPtIdx * DIMS;
    const dSq = distanceSqInt16(query, vectors, vpOffset);

    heap.tryInsert(dSq, vpPtIdx);

    const left = leftOff[nodeIdx];
    const right = rightOff[nodeIdx];
    if (left < 0 && right < 0) continue;

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

    if (far >= 0 && gapSq < tauSq) {
      stack[stackTop++] = far;
    }
    if (near >= 0) {
      stack[stackTop++] = near;
    }
  }
}

// Serialize to a single Buffer. Header: 16 bytes
//   uint32 magic = 0x56505432  ("VPT2", bumped from VPT1 for int16 format)
//   uint32 size
//   uint32 reserved
//   uint32 reserved
// Then four parallel arrays of size * 4 bytes:
//   pointIdx (uint32), threshold (float32), leftOffset (int32), rightOffset (int32)
export function serializeVpTree(tree: VpTree): Buffer {
  const N = tree.size;
  const bufSize = 16 + N * 16;
  const buf = Buffer.alloc(bufSize);
  buf.writeUInt32LE(0x56505432, 0);
  buf.writeUInt32LE(N, 4);

  const u32 = new Uint32Array(buf.buffer, buf.byteOffset + 16, N);
  u32.set(tree.pointIdx);

  const f32 = new Float32Array(buf.buffer, buf.byteOffset + 16 + N * 4, N);
  f32.set(tree.threshold);

  const i32a = new Int32Array(buf.buffer, buf.byteOffset + 16 + N * 8, N);
  i32a.set(tree.leftOffset);

  const i32b = new Int32Array(buf.buffer, buf.byteOffset + 16 + N * 12, N);
  i32b.set(tree.rightOffset);

  return buf;
}

export function deserializeVpTree(buf: Buffer): VpTree {
  const magic = buf.readUInt32LE(0);
  if (magic !== 0x56505432) {
    throw new Error(`bad vptree magic: 0x${magic.toString(16)} (expected VPT2)`);
  }
  const N = buf.readUInt32LE(4);
  const base = buf.byteOffset + 16;
  return {
    pointIdx: new Uint32Array(buf.buffer, base, N),
    threshold: new Float32Array(buf.buffer, base + N * 4, N),
    leftOffset: new Int32Array(buf.buffer, base + N * 8, N),
    rightOffset: new Int32Array(buf.buffer, base + N * 12, N),
    size: N,
  };
}
