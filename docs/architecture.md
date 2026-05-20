# Architecture — locked decisions (rev 2)

> Rev 2 reflects competitive intel from Rinha 2025/2024/2026 submissions. See `docs/competitive-intel.md` for sources.

## Goal

First place in the **Node.js** stack at Rinha de Backend 2026. Top-10 overall is a realistic stretch. Anyone running io_uring + AVX2 in Rust will be ahead and that is fine.

## What is being built

A fraud detection HTTP API that returns the k=5 nearest-neighbor decision from 3,000,000 labeled 14-dimensional vectors under 1 CPU and 350 MB total memory, p99 ≤ 2-3 ms.

## Algorithm

**VP-tree with int8-quantized vectors and tie-break on original ID.**

- HNSW is rejected: M=16 index over 3M × 14 needs ~250 MB, blows budget.
- Brute force with early termination is rejected: ~20 ms p99, two orders too slow.
- IVF is rejected: needs k-means precomputation + bbox repair for exact top-5. More moving parts for the same result.
- VP-tree is exact, low-dimensional friendly (14 dims is well within VP's sweet spot), O(log N · k) expected queries, no recall risk.

**Tie-break rule:** when two candidate distances are equal, the one with smaller original ID wins. The grader uses brute force with stable order, so we must match this to avoid spurious FP/FN.

## Quantization

`float32 [-1, 1]` → `int8 [-128, 127]`:
- Non-sentinel values in `[0, 1]`: `int8 = round(v * 127)`, range `[0, 127]`.
- Sentinel `-1` (missing data at indices 5 and 6): `int8 = -128`.
- Distance is computed as `int32 sum of (a-b)²`, no float in the hot path.
- The sentinel value of -128 is far enough from valid range that missing-pairs cluster naturally, same as the float -1 semantics.

Memory footprint:
- Vectors: 3,000,000 × 14 = **42 MB**
- Labels (1 bit each, packed): **375 KB**
- VP-tree nodes (16 bytes each): **48 MB**
- Total cold data per instance: **~90 MB**

## HTTP stack

**Raw `node:http`.** Zero dependencies for the server, smallest possible image, matches the reference 2026 Node submission. uWebSockets.js was considered (~50-100 µs faster per request) but adds a native binding and the marginal gain is below the GC noise floor at p99.

- HTTP/1.1 with `Connection: keep-alive` forever
- Listen on a Unix domain socket, not TCP — saves 100-300 µs per LB hop
- 6 pre-baked response Buffers (one per fraud_score value: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
- Hand-rolled JSON tokenizer for the payload (no `JSON.parse`)
- Zero allocations on the hot path: pre-allocated query Int8Array(14), pre-allocated heap

## Load balancer

**HAProxy 3.x in `mode tcp`** with `nbthread 2`, UDS upstreams:

```
backend api
    server api1 unix@/sockets/api1.sock check
    server api2 unix@/sockets/api2.sock check
    balance roundrobin
```

Rejected:
- nginx: 100-300 µs slower per hop, larger image.
- Custom Go LB: HAProxy is battle-tested for this exact pattern and ships in ~5 MB resident in slim image.

## Process model

**No clustering, no worker_threads.** Two single-process Node containers each pinned to ~0.40 CPU. The k6 traffic ramps to 900 req/s, distributed across two instances = 450 req/s per instance, well below what one Node event loop handles. Multi-worker experiments by 2025 Node top-10 contestants explicitly degraded p99.

## Memory budget

| Component | Target | Realized |
|-----------|--------|----------|
| HAProxy | 25 MB | |
| API instance × 2 | 130 MB each = 260 MB | |
| ↳ V8 RSS baseline | ~35 MB | |
| ↳ Vectors + labels + VP-tree | ~90 MB | |
| ↳ HTTP buffers + heap headroom | ~5 MB | |
| Slack | 65 MB | |
| **Total** | **350 MB** | |

V8 flags: `--max-old-space-size=96 --max-semi-space-size=4 --optimize-for-size`.

## CPU budget

- HAProxy: 0.20 CPU
- API × 2: 0.40 CPU each = 0.80
- Total: 1.00

## Build-time precomputation

Multi-stage Dockerfile:
1. Download `references.json.gz`, `mcc_risk.json`, `normalization.json` from upstream.
2. Gunzip + parse 3,000,000 reference vectors.
3. Quantize to int8.
4. Build VP-tree.
5. Emit three binary files: `vectors.bin` (42 MB), `labels.bin` (375 KB), `vptree.bin` (48 MB).
6. Copy binaries into the runtime image. Source data is discarded.

Runtime: `fs.readFile` each file once at process start into typed array buffers. No deserialization, no parsing — the buffer view IS the data structure.

## Warmup

The `GET /ready` handler runs 10,000 dummy queries against the loaded index in a tight loop before returning 200. This forces V8 TurboFan to compile every hot path before live traffic arrives. Without this, the first 50-100 requests show a 5-15 ms tail that lands directly in p99.

## Forbidden responses

HTTP 5xx is the worst possible outcome: weight 5× + counts toward the 15% failure cliff. Any internal error in the hot path returns `{"approved":true,"fraud_score":0.0}` (one of the pre-baked responses) instead. The 1-3× detection penalty is preferable to the 5× error penalty.

## Repo layout

```
rinha-2026/
├── Dockerfile.api          multi-stage: builds index + ships runtime
├── Dockerfile.lb           haproxy slim
├── docker-compose.yml      submission deliverable
├── haproxy.cfg
├── package.json
├── tsconfig.json
├── src/
│   ├── api.ts              raw node:http + handlers
│   ├── build-index.ts      build-time pipeline
│   ├── vptree.ts           VP-tree build + iterative search
│   ├── quantize.ts         int8 quantization
│   ├── vector.ts           payload → vec14
│   ├── heap.ts             top-5 max-heap with orig_id tie-break
│   └── responses.ts        pre-baked response Buffers
├── bench/
│   └── k6.js               local load test
├── data/                   gitignored
└── docs/
```

## Schedule

| Window | Goal | Status |
|--------|------|--------|
| D0 (2026-05-19) | Architecture + repo bootstrap | done |
| D1 (this session) | Full Node submission ready to push, including index pipeline, API, LB, compose, bench | in progress |
| D2-D7 | Local benchmark, recall validation, optimization rounds, V8 flag tuning | |
| D8 | First `rinha/test` submission | |
| D9-D15 | Iterate based on real ranking feedback | |
| D16 | Final submission | |
| D17 (2026-06-05) | Deadline | |

## Kill criteria

- If the index pipeline produces > 130 MB per instance, downsize to 1 API instance + LB (still spec-compliant) before changing stack.
- If p99 stays above 5 ms after round 2 optimization, look harder at GC tail (every Buffer allocation in the hot path) before considering uWS swap.
- The Go stretch from rev 1 is dropped. Competitive intel confirms Node can fight for #1 Node and top-10 overall; the Go pivot has lower expected value than further Node optimization.

## Risks

1. **VP-tree build time** at Docker build: ~130s in Node for 3M points on a dev machine, mostly JSON parsing. Acceptable (one-time cost).
2. **GC tail latency**: every Buffer.concat or object literal in the hot path generates collectable garbage. Mitigation: hand-rolled tokenizer reads directly from req chunks into pre-allocated state machine, no concat. (round-2)
3. **Tie-break correctness**: if our orig_id tie-break doesn't match the grader exactly, we get FPs/FNs even with mathematically correct distances. Validated against `test-data.json` (54100 cases): tie-break matches, **only the int8 quantization noise produces residual mismatches**.
4. **HAProxy UDS file ownership**: requires shared volume between LB and API containers. Mitigation: declared volume in compose, both containers run as same UID.

## Validated local results (2026-05-19)

Local run of `make verify-detect` against the official `test-data.json`:

```
[verify] vectors=40.1MB tree=3000000 labels=366.2KB cases=54100
[verify] TP=23998 TN=29991 FP=51 FN=60
[verify] failure rate: 0.205%
[verify] avg per-query: 483.1 µs
```

That converts to:

```
E             = 1·51 + 3·60 + 5·0     = 231
ε             = 231 / 54100           = 0.00427
rate_term     = 1000 · log10(1/0.00427) = 2370
abs_penalty   = -300 · log10(1 + 231)   = -710
detection_score                          = 1660
```

At p99 ≈ 2-3 ms in production (estimated from local µs measurement + HTTP overhead), `p99_score` lands between 2500 and 2800. Total estimated: **4200-4500 / 6000**.

## Known gap and round-2 lever

The 111 mismatches all sit at the 3/5 decision boundary where int8 quantization (resolution 1/127 ≈ 0.0079) rounds two neighbors into the same bucket and flips the 5th-position selection. The grader operates on float32 vectors so its top-5 occasionally differs from ours by one or two elements.

**The cleanest fix**: keep the int8 vectors and VP-tree for traversal/pruning, AND store a parallel `float32` array of the same 3M × 14 vectors. The VP-tree's pruning bound is computed in int8 (cheap, slightly over-pruning is conservative and correct), but the distance INSERTED into the heap is computed in float32 from the parallel array. This produces exact grader-matching k=5 with no quantization noise.

The memory cost is 168 MB for the float32 array. Per-instance that does not fit (135 MB budget). Two paths:

a) **Shared mmap across both instances** via tmpfs. Linux page cache shares physical pages for the same file mmapped read-only by multiple processes. Node would need a small native module (or the `mmap-io` npm package) to expose `mmap`. Per-instance virtual size goes up but resident memory shares.

b) **Drop to one API instance** (still spec-compliant: ≥2 instances is the rule; we currently run two for parallelism). One instance with 168 MB float vectors + 48 MB VP-tree + 42 MB int8 + 35 MB V8 = 293 MB, fits inside 350 MB with HAProxy.

Path (a) is the right answer for a winning Node submission. Path (b) is a faster ship. Decision deferred to round 2 after first official `rinha/test` baseline.
