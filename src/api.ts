// Fraud detection API. Listens on a Unix domain socket. Two handlers:
//   GET  /ready          health check, blocks until warmup is done
//   POST /fraud-score    main detection endpoint
//
// Hot path is allocation-free:
//   - One pre-allocated Float32Array(14) for the vectorized query
//   - One pre-allocated Int8Array(14)   for the quantized query
//   - One pre-allocated TopKHeap        for the top-5 neighbors
//   - Six pre-baked response Buffers    indexed by fraud count

import { createServer, IncomingMessage, ServerResponse } from 'node:http';
import { promises as fs } from 'node:fs';
import { unlinkSync, existsSync, chmodSync } from 'node:fs';
import { DIMS } from './quantize.ts';
import { quantizeOne } from './quantize.ts';
import { TopKHeap } from './heap.ts';
import { deserializeVpTree, searchVpTree } from './vptree.ts';
import { vectorize, type Normalization, type MccRiskMap } from './vector.ts';
import {
  BODY_BUFFERS,
} from './responses.ts';

const DATA_DIR = process.env.DATA_DIR ?? './data';
const UDS_PATH = process.env.UDS_PATH ?? '/sockets/api.sock';
const WARMUP_QUERIES = Number(process.env.WARMUP_QUERIES ?? 3_000);

interface LoadedIndex {
  vectors: Int8Array;
  labels: Uint8Array;
  tree: ReturnType<typeof deserializeVpTree>;
  norm: Normalization;
  mccRisk: MccRiskMap;
}

const heap = new TopKHeap(5);
const queryFloat = new Float32Array(DIMS);
const queryInt8 = new Int8Array(DIMS);

let ready = false;
let index: LoadedIndex | null = null;

async function loadIndex(): Promise<LoadedIndex> {
  const t0 = Date.now();

  const [vBuf, lBuf, tBuf, normRaw, mccRaw] = await Promise.all([
    fs.readFile(`${DATA_DIR}/vectors.bin`),
    fs.readFile(`${DATA_DIR}/labels.bin`),
    fs.readFile(`${DATA_DIR}/vptree.bin`),
    fs.readFile(`${DATA_DIR}/normalization.json`, 'utf8'),
    fs.readFile(`${DATA_DIR}/mcc_risk.json`, 'utf8'),
  ]);

  const vectors = new Int8Array(vBuf.buffer, vBuf.byteOffset, vBuf.byteLength);
  const labels = new Uint8Array(lBuf.buffer, lBuf.byteOffset, lBuf.byteLength);
  const tree = deserializeVpTree(tBuf);
  const norm = JSON.parse(normRaw) as Normalization;
  const mccRisk = JSON.parse(mccRaw) as MccRiskMap;

  console.log(
    `[api] loaded index in ${((Date.now() - t0) / 1000).toFixed(1)}s: ` +
      `N=${tree.size.toLocaleString()}, vectors=${(vectors.byteLength / 1024 / 1024).toFixed(1)}MB, ` +
      `tree=${(tBuf.byteLength / 1024 / 1024).toFixed(1)}MB, labels=${(labels.byteLength / 1024).toFixed(1)}KB`,
  );

  return { vectors, labels, tree, norm, mccRisk };
}

function score(payload: any): number {
  vectorize(payload, queryFloat, { norm: index!.norm, mccRisk: index!.mccRisk });
  quantizeOne(queryFloat, queryInt8);
  searchVpTree(index!.tree, index!.vectors, queryInt8, heap);
  return heap.countFrauds(index!.labels);
}

function warmup(): void {
  console.log(`[api] warming up with ${WARMUP_QUERIES.toLocaleString()} queries`);
  const t0 = Date.now();

  // Synthesize a representative payload for warmup. Same shape as real traffic.
  const samplePayload: any = {
    id: 'warmup',
    transaction: {
      amount: 100.0,
      installments: 1,
      requested_at: '2026-03-11T18:45:53Z',
    },
    customer: {
      avg_amount: 200.0,
      tx_count_24h: 3,
      known_merchants: ['MERC-001', 'MERC-016'],
    },
    merchant: { id: 'MERC-016', mcc: '5411', avg_amount: 60.0 },
    terminal: { is_online: false, card_present: true, km_from_home: 5.0 },
    last_transaction: null,
  };

  for (let i = 0; i < WARMUP_QUERIES; i++) {
    // Vary the amount slightly so we touch different VP-tree paths.
    samplePayload.transaction.amount = 50 + (i % 5000);
    score(samplePayload);
  }
  console.log(`[api]   warmup done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
}

// Pre-allocated body buffer pool. Typical payload is ~500 bytes; we keep
// 2 KB per slot to absorb any reasonable variance without resizing.
const BODY_BUF_SIZE = 2048;
const BODY_POOL_SIZE = 64;
const bodyPool: Buffer[] = [];
for (let i = 0; i < BODY_POOL_SIZE; i++) {
  bodyPool.push(Buffer.allocUnsafe(BODY_BUF_SIZE));
}

function acquireBodyBuf(): Buffer {
  return bodyPool.pop() ?? Buffer.allocUnsafe(BODY_BUF_SIZE);
}

function releaseBodyBuf(b: Buffer): void {
  if (b.length === BODY_BUF_SIZE && bodyPool.length < BODY_POOL_SIZE) {
    bodyPool.push(b);
  }
}

function handleFraudScore(req: IncomingMessage, res: ServerResponse): void {
  let buf = acquireBodyBuf();
  let pos = 0;
  let overflow = false;

  req.on('data', (chunk: Buffer) => {
    const need = pos + chunk.length;
    if (need > buf.length) {
      if (!overflow) {
        // Grow once: allocate a bigger one, copy what we have, mark overflow
        // so we don't return it to the pool.
        const grown = Buffer.allocUnsafe(Math.max(need, buf.length * 2));
        buf.copy(grown, 0, 0, pos);
        // return the original pool buf since it's still pool-sized
        releaseBodyBuf(buf);
        buf = grown;
        overflow = true;
      }
    }
    chunk.copy(buf, pos);
    pos += chunk.length;
  });

  req.on('end', () => {
    let fraudCount = 0;
    try {
      const payload = JSON.parse(buf.toString('utf8', 0, pos));
      fraudCount = score(payload);
    } catch {
      // Fail safe: respond 200 with approved=true (FP weight 1 < Err weight 5).
      fraudCount = 0;
    }
    const respBuf = BODY_BUFFERS[fraudCount];
    res.writeHead(200, {
      'content-type': 'application/json',
      'content-length': respBuf.byteLength,
    });
    res.end(respBuf);
    if (!overflow) releaseBodyBuf(buf);
  });

  req.on('error', () => {
    const respBuf = BODY_BUFFERS[0];
    try {
      res.writeHead(200, {
        'content-type': 'application/json',
        'content-length': respBuf.byteLength,
      });
      res.end(respBuf);
    } catch {
      // socket already closed
    }
    if (!overflow) releaseBodyBuf(buf);
  });
}

function handleReady(_req: IncomingMessage, res: ServerResponse): void {
  if (ready) {
    res.writeHead(200, { 'content-type': 'text/plain', 'content-length': 5 });
    res.end('ready');
  } else {
    res.writeHead(503, { 'content-type': 'text/plain', 'content-length': 5 });
    res.end('busy.');
  }
}

async function main(): Promise<void> {
  index = await loadIndex();
  warmup();
  ready = true;

  let firstReq = true;
  let firstConn = true;
  const server = createServer((req, res) => {
    if (firstReq) {
      firstReq = false;
      console.log(`[api] first request: ${req.method} ${req.url}`);
    }
    if (req.method === 'POST' && req.url === '/fraud-score') {
      handleFraudScore(req, res);
    } else if (req.method === 'GET' && req.url === '/ready') {
      handleReady(req, res);
    } else {
      res.writeHead(404, { 'content-length': 0 });
      res.end();
    }
  });

  server.on('connection', () => {
    if (firstConn) {
      firstConn = false;
      console.log(`[api] first connection accepted`);
    }
  });

  server.on('clientError', (err, socket) => {
    console.log(`[api] clientError: ${(err as Error).message} code=${(err as any).code}`);
    try {
      socket.end('HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n');
    } catch {
      // ignore
    }
  });

  server.on('error', (err) => {
    console.log(`[api] server error: ${err.message}`);
  });

  if (UDS_PATH.startsWith('/')) {
    if (existsSync(UDS_PATH)) {
      try {
        unlinkSync(UDS_PATH);
      } catch {
        // ignore
      }
    }
    server.listen(UDS_PATH, () => {
      try {
        chmodSync(UDS_PATH, 0o666);
      } catch {
        // ignore
      }
      console.log(`[api] listening on UDS ${UDS_PATH}`);
    });
  } else {
    const port = Number(UDS_PATH);
    server.listen(port, '0.0.0.0', () => {
      console.log(`[api] listening on tcp ${port}`);
    });
  }
}

main().catch((err) => {
  console.error('[api] startup failed:', err);
  process.exit(1);
});
