// int8 quantization for 14-dim vectors.
//
// Mapping:
//   v == -1 (sentinel for missing last_transaction): int8 = -128
//   v in [0, 1]: int8 = round(v * 127), clamped to [0, 127]
//
// Distance is computed as sum of (a-b)^2 over int16 differences. Max single
// diff is 255 (e.g. 127 - (-128)), squared is 65_025; sum over 14 dims is at
// most ~910_350, comfortably inside int32.

export const DIMS = 14;
export const QUANT_SCALE = 127;
export const SENTINEL = -128;

export function quantizeOne(input: Float32Array, output: Int8Array): void {
  for (let i = 0; i < DIMS; i++) {
    const v = input[i];
    if (v === -1) {
      output[i] = SENTINEL;
    } else {
      const q = Math.round(v * QUANT_SCALE);
      output[i] = q < 0 ? 0 : q > 127 ? 127 : q;
    }
  }
}

export function quantizeBatch(
  input: Float32Array,
  output: Int8Array,
  count: number,
): void {
  for (let i = 0; i < count; i++) {
    const base = i * DIMS;
    for (let j = 0; j < DIMS; j++) {
      const v = input[base + j];
      if (v === -1) {
        output[base + j] = SENTINEL;
      } else {
        const q = Math.round(v * QUANT_SCALE);
        output[base + j] = q < 0 ? 0 : q > 127 ? 127 : q;
      }
    }
  }
}

export function distanceSqInt8(
  query: Int8Array,
  refs: Int8Array,
  refOffset: number,
): number {
  let sum = 0;
  const d0 = query[0] - refs[refOffset];
  const d1 = query[1] - refs[refOffset + 1];
  const d2 = query[2] - refs[refOffset + 2];
  const d3 = query[3] - refs[refOffset + 3];
  const d4 = query[4] - refs[refOffset + 4];
  const d5 = query[5] - refs[refOffset + 5];
  const d6 = query[6] - refs[refOffset + 6];
  const d7 = query[7] - refs[refOffset + 7];
  const d8 = query[8] - refs[refOffset + 8];
  const d9 = query[9] - refs[refOffset + 9];
  const d10 = query[10] - refs[refOffset + 10];
  const d11 = query[11] - refs[refOffset + 11];
  const d12 = query[12] - refs[refOffset + 12];
  const d13 = query[13] - refs[refOffset + 13];
  sum =
    d0 * d0 +
    d1 * d1 +
    d2 * d2 +
    d3 * d3 +
    d4 * d4 +
    d5 * d5 +
    d6 * d6 +
    d7 * d7 +
    d8 * d8 +
    d9 * d9 +
    d10 * d10 +
    d11 * d11 +
    d12 * d12 +
    d13 * d13;
  return sum;
}
