# rinha-2026 — Node.js submission

Submission for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) (fraud detection via vector search). Targeting first place in the Node.js stack.

## Architecture

- **Raw `node:http`** server, listening on a Unix domain socket
- **VP-tree** over int8-quantized 14-dim vectors, with original-id tie-break to match the grader's brute-force ordering exactly
- **Hot path is allocation-free**: pre-allocated query buffers, six pre-baked response Buffers, no `JSON.parse` of any output
- **HAProxy 3.0** in `mode tcp` as round-robin LB over two API instances via UDS
- **Build-time index pipeline**: gunzip + parse + quantize + VP-tree build emits three binary files that the runtime image mmaps via `fs.readFile`

Full decision log in [`docs/architecture.md`](docs/architecture.md). Competitive intel that drove the design in [`docs/competitive-intel.md`](docs/competitive-intel.md).

## Repo layout

```
src/
  api.ts              raw node:http + handlers + warmup
  build-index.ts      build-time pipeline (decompress → quantize → VP-tree → write)
  vptree.ts           VP-tree build + iterative search + serialize/deserialize
  quantize.ts         int8 quantization + integer Euclidean distance
  vector.ts           payload → 14-dim float32 (char-code timestamp parsing)
  heap.ts             top-5 max-heap with (dist, idx) tie-break
  responses.ts        six pre-baked response Buffers
  test/               smoke test + detection verifier
docs/                 architecture decisions and competitive intel
bench/                local k6 harness matching the upstream test profile
Dockerfile.api        multi-stage: build-time index + runtime API
Dockerfile.lb         haproxy:3.0-alpine
docker-compose.yml    full stack (LB + 2 api instances) on bridge net, port 9999
haproxy.cfg           tcp mode, UDS upstreams, roundrobin
Makefile              data download, local build, compose up, k6 bench
```

## Constraints (from spec)

- 1 CPU, 350 MB RAM total across all services
- `bridge` network, `linux-amd64`, port 9999
- Public Docker images only

## Resource allocation (per `docker-compose.yml`)

| Service | CPU | RAM |
|---------|------|------|
| HAProxy | 0.20 | 30 MB |
| api1 | 0.40 | 135 MB |
| api2 | 0.40 | 135 MB |
| **Total** | **1.00** | **300 MB** | (50 MB headroom) |

## Memory inside one API container

| Region | Size |
|--------|------|
| V8 RSS baseline | ~35 MB |
| int8 vectors (3M × 14) | 42 MB |
| VP-tree (3M nodes × 16 B) | 48 MB |
| Packed label bitmap (3M / 8) | 0.4 MB |
| HTTP buffers + heap headroom | ~10 MB |
| **Per-instance peak** | **~135 MB** |

V8 flags: `--max-old-space-size=96 --max-semi-space-size=4 --optimize-for-size`.

## Running locally

```bash
make deps           # npm install (stream-json, @types/node)
make data           # fetch upstream reference + test data
make smoke          # in-process correctness test on a 100-vector sample
make build-index    # produce data/{vectors,labels,vptree}.bin (1-3 min)
make verify-detect  # validate 0 FP / 0 FN against official test-data.json
make up             # docker compose up (builds index inside the image, 5-10 min)
make bench          # run k6 against the stack on :9999
make down           # docker compose down -v
```

## Submission

Two branches:

- `main`: the source code (this README)
- `submission`: only the files the grader needs to run the stack — `docker-compose.yml`, `Dockerfile.api`, `Dockerfile.lb`, `haproxy.cfg`, `src/`, `package.json`, `package-lock.json`, `tsconfig.json`, `.dockerignore`

To run the official test, open an issue with `rinha/test` in the description in this repo. The Rinha engine runs the test and posts the result.

## Why these choices

Short version (full reasoning in `docs/competitive-intel.md`):

- **No HNSW**: M=16 over 3M × 14 needs ~250 MB. Blows the budget. VP-tree is exact, 14-d-friendly, and fits in 48 MB.
- **No uWebSockets.js / Fastify**: raw `node:http` matches the 2026 Node reference (`JoaoMarcos160/rinha-de-backend-2026-node`), zero deps, smallest possible image. We can switch to uWS in round 2 if p99 needs more compression.
- **No cluster/worker_threads**: 2025 Node top-10 contestants explicitly say multi-worker degrades latency under 1 CPU. Two separate single-process containers ARE the parallelism.
- **HAProxy over nginx**: 100-300 µs faster per hop in tcp+UDS mode, and the LB pattern in all Rinha top finishers across editions.
- **Six pre-baked responses**: `fraud_score` is one of six values. No `JSON.stringify` on the response path.
- **Tie-break by orig_id**: the grader uses brute-force k=5 over the full 3M reference set with stable ordering. Without matching this, ties on edge-case payloads produce spurious FP/FN. With it, we get 0 FP / 0 FN on `test-data.json` (verified by `make verify-detect`).
- **Warmup at /ready**: 10K dummy queries before the first 200 response. Kills the JIT-warming tail that would otherwise land in p99.

## License

MIT.
