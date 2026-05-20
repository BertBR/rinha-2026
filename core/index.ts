// Node-side wrapper for the Rust kNN native module.
// The .node file is the librinha_knn.so renamed by the Dockerfile.

import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const native = createRequire(import.meta.url)(join(__dirname, 'rinha-knn.node'));

export const initKnn: () => void = native.initKnn;

export const knnFraudCount: (
  v0: number,
  v1: number,
  v2: number,
  v3: number,
  v4: number,
  v5: number,
  v6: number,
  v7: number,
  v8: number,
  v9: number,
  v10: number,
  v11: number,
  v12: number,
  v13: number,
) => number = native.knnFraudCount;
