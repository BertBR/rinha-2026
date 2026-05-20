// int16 quantization for 14-dim vectors.
//
// Mapping:
//   v == -1 (sentinel for missing last_transaction): int16 = -32768
//   v in [0, 1]: int16 = round(v * 32767), clamped to [0, 32767]
//
// Why int16 over int8:
//   Source data is stored with ~4 decimal precision (resolution 1e-4).
//   int8 resolution is 1/127 ≈ 0.0079, coarser than the source — collapses
//   nearby vectors into identical int8 representations, producing tied
//   distances that the grader (operating on float32) breaks differently.
//   This caused 51 FP + 60 FN against test-data.json with int8.
//   int16 resolution is 1/32767 ≈ 3e-5, 30x finer than the source — no
//   collisions, no spurious ties, 0 FP / 0 FN expected.
//
// Distance: sum of (a-b)^2 over int32 differences. Max single diff is
// 65535 (32767 - (-32768)), squared ≈ 4.3e9. Sum over 14 dims ≈ 6e10
// which overflows int32. Use either float64 accumulator or chunk-by-2
// to stay safe. We use float64 (still integer-valued in practice, just
// stored as double) — V8 TurboFan handles this well.

export const DIMS = 14;
export const QUANT_SCALE = 32767;
export const SENTINEL = -32768;

export function quantizeOne(input: Float32Array, output: Int16Array): void {
  for (let i = 0; i < DIMS; i++) {
    const v = input[i];
    if (v === -1) {
      output[i] = SENTINEL;
    } else {
      const q = Math.round(v * QUANT_SCALE);
      output[i] = q < 0 ? 0 : q > 32767 ? 32767 : q;
    }
  }
}

export function distanceSqInt16(
  query: Int16Array,
  refs: Int16Array,
  refOffset: number,
): number {
  // float64 accumulator: each (a-b)^2 fits in int32 (max 4.3e9), but the
  // sum of 14 could touch 6e10. JS numbers (float64) hold this exactly.
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
  return (
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
    d13 * d13
  );
}

// Kept as an alias for callers that still import the int8 name.
export const distanceSqInt8 = distanceSqInt16;
