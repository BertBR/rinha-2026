// Fraud detection API on uWebSockets.js (UDS-bound). Calls into the Rust
// kNN native module (./core/index.ts) for the actual top-5 search.
//
// Hot path is allocation-free:
//   - Pre-allocated Float32Array(14) for the vectorized query
//   - Pre-baked Buffer responses indexed by fraud count
//   - Hand-rolled JSON parser writes directly into the query buffer
//   - Single native call per request

import { existsSync, unlinkSync, chmodSync, promises as fs } from 'node:fs';
import uWS from 'uWebSockets.js';
import { DIMS } from './quantize.ts';
import { type Normalization, type MccRiskMap } from './vector.ts';
import { makeParseContext, parseAndVectorize } from './parseJson.ts';
import { BODY_BUFFERS } from './responses.ts';
import { initKnn, knnFraudCount } from '../core/index.ts';

const DATA_DIR = process.env.DATA_DIR ?? './data';
const UDS_PATH = process.env.UDS_PATH ?? '/sockets/api.sock';

const queryFloat = new Float32Array(DIMS);
let parseCtx: ReturnType<typeof makeParseContext>;

async function main(): Promise<void> {
  // Load config
  const [normRaw, mccRaw] = await Promise.all([
    fs.readFile(`${DATA_DIR}/normalization.json`, 'utf8'),
    fs.readFile(`${DATA_DIR}/mcc_risk.json`, 'utf8'),
  ]);
  const norm = JSON.parse(normRaw) as Normalization;
  const mccRisk = JSON.parse(mccRaw) as MccRiskMap;
  parseCtx = makeParseContext(norm, mccRisk);

  // Init native kNN (loads embedded index, ~400ms first time).
  console.log('[api] initializing native kNN');
  const t0 = performance.now();
  initKnn();
  console.log(`[api]   ready in ${(performance.now() - t0).toFixed(1)} ms`);

  // Warm V8 + native side with synthetic queries.
  const tw = performance.now();
  for (let i = 0; i < 2000; i++) {
    queryFloat[0] = (i % 100) / 100;
    queryFloat[1] = ((i * 7) % 100) / 100;
    queryFloat[2] = ((i * 13) % 100) / 100;
    queryFloat[3] = ((i * 23) % 100) / 100;
    queryFloat[4] = ((i * 31) % 100) / 100;
    queryFloat[5] = i & 1 ? -1 : 0.5;
    queryFloat[6] = i & 1 ? -1 : 0.5;
    queryFloat[7] = ((i * 5) % 100) / 100;
    queryFloat[8] = ((i * 3) % 100) / 100;
    queryFloat[9] = i & 1;
    queryFloat[10] = (i + 1) & 1;
    queryFloat[11] = (i + 2) & 1;
    queryFloat[12] = 0.3;
    queryFloat[13] = ((i * 11) % 100) / 100;
    knnFraudCount(
      queryFloat[0],
      queryFloat[1],
      queryFloat[2],
      queryFloat[3],
      queryFloat[4],
      queryFloat[5],
      queryFloat[6],
      queryFloat[7],
      queryFloat[8],
      queryFloat[9],
      queryFloat[10],
      queryFloat[11],
      queryFloat[12],
      queryFloat[13],
    );
  }
  console.log(`[api]   warmup done in ${(performance.now() - tw).toFixed(1)} ms`);

  if (existsSync(UDS_PATH)) {
    try {
      unlinkSync(UDS_PATH);
    } catch {
      // ignore
    }
  }

  const app = uWS
    .App()
    .get('/ready', (res) => {
      res.onAborted(() => {});
      res.cork(() => res.end());
    })
    .post('/fraud-score', (res) => {
      res.onAborted(() => {});
      let buf = Buffer.alloc(0);
      res.onData((chunk, isLast) => {
        const data = Buffer.from(chunk);
        buf = buf.length === 0 ? data : Buffer.concat([buf, data]);
        if (!isLast) return;
        let count = 0;
        try {
          const ok = parseAndVectorize(buf, buf.length, queryFloat, parseCtx);
          if (ok) {
            count = knnFraudCount(
              queryFloat[0],
              queryFloat[1],
              queryFloat[2],
              queryFloat[3],
              queryFloat[4],
              queryFloat[5],
              queryFloat[6],
              queryFloat[7],
              queryFloat[8],
              queryFloat[9],
              queryFloat[10],
              queryFloat[11],
              queryFloat[12],
              queryFloat[13],
            );
          }
        } catch {
          count = 0;
        }
        const respBuf = BODY_BUFFERS[count];
        res.cork(() => {
          res.writeHeader('content-type', 'application/json');
          res.end(respBuf);
        });
      });
    })
    .any('/*', (res) => {
      res.onAborted(() => {});
      res.cork(() => {
        res.writeStatus('404 Not Found').end();
      });
    });

  app.listen_unix((token: unknown) => {
    if (!token) {
      console.error('[api] failed to listen on UDS');
      process.exit(1);
    }
    try {
      chmodSync(UDS_PATH, 0o666);
    } catch {
      // ignore
    }
    console.log(`[api] listening on UDS ${UDS_PATH}`);
  }, UDS_PATH);
}

main().catch((err) => {
  console.error('[api] fatal:', err);
  process.exit(1);
});
