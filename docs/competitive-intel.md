# Competitive intelligence — sources and findings

> Collected 2026-05-19 to inform the Node architecture for Rinha 2026.

## Headline finding

A Node submission won Rinha 2025 overall (`ricassiocosta`, Fastify 5 + fast-json-stringify + Redis over UDS). The 2026 challenge is more CPU-bound than 2025 (vector math vs I/O), structurally favoring Rust/C/Go more, but a well-engineered Node submission can credibly fight for #1 Node and top-10 overall.

The 2026 ceiling for Node is approximately **5400/6000 points** vs **~5800** for the top Rust submission, a gap of ~400 points (mostly from AVX2 + io_uring + zero-overhead distance math that V8 cannot replicate).

## 2026 submissions already with code (as of 2026-05-19)

### `JoaoMarcos160/rinha-de-backend-2026-node` (the only serious Node 2026 submission so far)

The reference to study. Patterns confirmed:
- raw `node:http` (no framework!) on `node:24-alpine`
- UDS upstreams
- **VP-tree** for indexing
- **int8 quantization**: 3M × 14 → 42 MB vs 168 MB float32
- Pre-allocated `Float32Array` query buffer + `Int8Array` ref buffer + K=5 heap
- All distance math integer-only after pre-scaling query × 127
- `NODE_OPTIONS="--max-old-space-size=96"`
- Build stage gunzips and converts to 3 binaries: `vectors.bin`, `labels.bin`, `vptree.bin`
- HAProxy `mode tcp`, `nbthread 2`, UDS upstreams, 2s timeouts

### `CaioDGallo/sketchy-check` (Rust top-tier playbook)

- IVF index, K=256 k-means clusters
- int16 quantization (scale=10000)
- AVX2 SIMD distance
- `io_uring` (`IOURING_QD=4096`, `ACCEPT_SQES=256`)
- 2 workers per instance
- UDS via HAProxy 3.3
- `seccomp=unconfined` (needed for io_uring)
- NPROBE=1 + bbox-repair pass for exact top-5
- **Pre-built 6 string HTTP responses** (one per fraud count)
- Tie-break on `orig_id` to match the grader exactly (this is what gives 0 FP/FN)

### Other notable

- `AndreBBM` (Rust, mmap binary index, nginx, 2 × 0.45 CPU / 160 MB)
- `GabrielDantasDs` (C with HNSW, `nginx:alpine`, under-allocates CPU)
- `DiogoThomaz` (C, full layout with bench scripts)
- `Joaopdiasventura` (Go with sharded indexes: `SHARD_ID=0/1, SHARD_COUNT=2` splits 3M across instances)
- `bulletdev` (Zig + Ruby variants)

## Rinha 2025 top-3 overall

### #1 ricassiocosta (Node.js)

p99 = 2.55 ms, profit = R$ 1.95 M

- Fastify 5 + `fast-json-stringify` + Redis 5 over **Unix socket**
- `undici` for outbound
- TypeBox schema → schema-compiled serialization
- Single Node process per container, 2 instances at 0.6 CPU / 108 MB
- nginx 0.15 CPU / 48 MB, redis 0.15 / 86 MB
- Node flags: `--no-deprecation --optimize_for_size --max_old_space_size=512`
- Internal in-memory queue worker (no external broker)
- keepaliveTimeout=50s
- nginx: `least_conn`, `keepalive 64`, `tcp_nopush`, `tcp_nodelay`, `keepalive_requests 10000`

### #2 RuanPabloCR-Rruim (C# / .NET)

Redis-backed, Redis messaging, nginx.

### #3 lucas-laurentino-go (Go fasthttp + Postgres + ZeroMQ + nginx)

- 4 API instances at 0.2 CPU / 25 MB each
- PG 0.5 / 190 MB
- nginx 0.2 / 60 MB
- WORKERS_POOL_SIZE=5, PG_MAX_CONNS=25
- `nproc/nofile=1000000` ulimits

## Rinha 2025 top Node submissions

### #1 ricassiocosta (covered above)

### #8 cristian-s-node-1 (Node + TS, p99=3.46 ms)

- **uWebSockets.js v20.52.0** + `fast-json-stringify` + `zeromq 6` IPC over UDS
- `undici 7` + custom in-process `memorydb` service over UDS
- `app.listen_unix()` (no TCP)
- Single worker thread for processor I/O
- Batching: BATCH_SIZE=100, BATCH_TIMEOUT=1000ms
- 2 backends × 0.55 CPU / 115 MB + memorydb 0.2 / 90 MB + haproxy 0.2 / 30 MB
- Node 22.18 bookworm-slim
- Runs TS directly via `node src/index.ts` (no compile step)

## Cross-edition patterns of winners

1. **Unix Domain Sockets for LB ↔ app** — universal in top-10 of every edition. Saves 100-300 µs per hop.
2. **HAProxy in `mode tcp`** with `nbthread 1-2` beats nginx by 100-300 µs in 2025. `tune.bufsize 16384`, `timeout connect 50 ms`, `default-server maxconn 10000`, `http-reuse always`.
3. **Schema-compiled serializers**: `fast-json-stringify` in 100% of top Node submissions across 2024-2025-2026.
4. **`uWebSockets.js v20.52.0+`** over Fastify for sub-3ms p99 (one path) — but raw `node:http` is the 2026 default. Both viable.
5. **`undici 6/7`** for outbound HTTP in every Node top-10. Never `node-fetch`/axios.
6. **Pre-allocated typed arrays + zero-alloc hot path** — the single biggest Node-specific trick. `vectorizeToBuffer(q, payload)` writes into an existing `Float32Array(14)`. K=5 heap is `Float32Array(5).fill(Infinity)`.
7. **Quantize the index.** int8 (JoaoMarcos: ÷4 RAM) or int16 (Caio: ÷2 RAM).
8. **Manual JSON body parse** (`Buffer.concat(chunks)` + `JSON.parse`), or hand-rolled field scanner. Saves 100-200 µs on a 500-byte payload.
9. **Node flags that matter**: `--max-old-space-size=96-128`, `--optimize-for-size`. **Avoid** `cluster`/`worker_threads` under 1 CPU — multiple sources confirm degrades latency.
10. **Two-instance symmetric topology** with `cpuset` pinning is universal. For 2026 under 1 CPU: 2× api at 0.40 + LB at 0.20 = exact 1.00 cap. Memory: 350 − 30 (LB) = 320 / 2 = 160 MB per instance hard ceiling.
11. **Build-time precomputation**: every top 2026 submission converts `references.json.gz` to a binary at Docker build. Runtime reads via `fs.readFile` once into a typed array buffer.
12. **`security_opt: seccomp=unconfined`** + memlock ulimits unbounded — required for io_uring (Caio); harmless for Node.
13. **No keepalive timeout games**: HAProxy `timeout client/server 2s` — long enough not to kill k6 but tight enough to surface deadlocks.

## Official 2026 k6 test profile

- Executor: `ramping-arrival-rate`, `startRate: 1`, `timeUnit: 1s`
- Single stage: ramp to **900 req/s over 120s**, gracefulStop 10s
- VUs: `preAllocatedVUs: 100, maxVUs: 250`
- Per-request `timeout: '2001 ms'` (anything slower = error_count, weight 5)
- DNS: roundRobin, ttl 5m
- Total test payload ~26 MB JSON pre-loaded into k6 SharedArray

**Implication:** ~108,000 requests over 120s. Only p99 (1 in 100 requests) matters for latency score. Getting p99 from 5 ms to 1.5 ms is worth +520 points; from 1.5 to 1 ms is +176. Below 1 ms saturates.

## Where Node hits a soft wall vs Rust/C/C++

1. **No AVX2 intrinsics from JS** — TurboFan generates scalar SSE at best.
2. **GC tail latency**: even with `--max-old-space-size=96` you see occasional 5-10 ms minor GC pauses that hit p99 unless every allocation on the hot path is eliminated.
3. **No `io_uring` for HTTP without a native addon** — node:http uses epoll, which is fine but adds ~50-100 µs per request vs io_uring.
4. **No native int16x8 SIMD** — your 14-dim distance is ~14 scalar ops vs Caio's 1 AVX2 256-bit-vector op.

## What this study changed in our architecture

1. Switched from HNSW to **VP-tree** (HNSW does not fit memory budget for 3M × 14 at M=16).
2. Switched from hnswlib-node to **own VP-tree implementation in TS** (~150 lines).
3. Switched from uWS to **raw node:http** (matches the 2026 Node reference, fewer surprises).
4. Added explicit **tie-break by orig_id** to the heap comparator (matches grader, gives 0 FP/FN).
5. Confirmed **HAProxy tcp mode + UDS** as LB choice (Rinha winner pattern).
6. Confirmed **no cluster/worker_threads** under 1 CPU budget.
7. Added **JIT warmup with self-pings at /ready** (10K dummy queries before serving live traffic).

## Sources (verbatim file references)

- https://github.com/ricassiocosta/backend-para-rinha-2025-node (2025 #1 overall, Fastify handler)
- https://github.com/JoaoMarcos160/rinha-de-backend-2026-node (the 2026 Node reference, VP-tree pattern)
- https://github.com/CaioDGallo/sketchy-check (the 2026 Rust algorithmic playbook — IVF + bbox repair + tie-break by orig_id)
- https://github.com/zanfranceschi/rinha-de-backend-2026/blob/main/test/test.js (k6 profile)
- https://github.com/zanfranceschi/rinha-de-backend-2025 (2025 ranking and submissions)
