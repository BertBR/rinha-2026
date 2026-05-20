import { initKnn, knnFraudCount } from './core/index.ts';

const t0 = performance.now();
initKnn();
console.log(`init in ${(performance.now() - t0).toFixed(1)} ms`);

const t1 = performance.now();
const c = knnFraudCount(0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006);
console.log(`single query: ${(performance.now() - t1).toFixed(3)} ms → fraud count ${c}`);

const N = 10000;
const t2 = performance.now();
for (let i = 0; i < N; i++) {
  knnFraudCount(0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006);
}
const elapsed = performance.now() - t2;
console.log(`bench ${N} queries: ${elapsed.toFixed(1)} ms = ${(elapsed/N*1000).toFixed(1)} µs/query`);
