# Architecture — locked decisions

> Planning notes, not marketing. If you disagree with a decision, push back with a benchmark.

## Goal

First place in the **Node.js** stack at Rinha de Backend 2026.
A Go port is a stretch goal, attempted only if the Node submission is locked at the top of its stack by D10. Pódium geral is not the target — anyone writing hand-rolled C++ with SIMD intrinsics will be ahead and that is fine.

**Why Node-first** (full reasoning in `docs/stack-choice.md`): primary daily stack, less competitive field, hnswlib-node native binding lifts the heavy compute out of V8, blog post defendable in interviews, tighter thematic fit with target pipeline (Speechify Platform, Clara AI, LemFi, Anthropic Skills).

## Constraints (recap)

- 1.0 vCPU, 350 MB RAM total across LB + 2 API instances.
- Mac Mini Late 2014, Intel Haswell, 2.6 GHz (AVX2 available; cgroup-limited).
- Port 9999. Bridge network. `linux-amd64`. Docker compose with public images.
- 3M reference vectors, 14 dimensions. Exact brute-force k-NN (k=5, Euclidean) is the ground truth.
- p99 ≤ 1 ms saturates at +3000. 10× improvement = +1000.
- 15% failure rate cutoff (FP + FN + Err). HTTP 500 weight 5×, FN 3×, FP 1×.

## Decisions

| # | Decision | Why |
|---|----------|-----|
| 1 | **HNSW + exact rerank** | Brute force on 3M × 14 with cache misses lands at 5-10 ms. HNSW with ef=50 lands at 50-200 µs. Rerank top-20 with exact int8 SIMD distance to stay ≥99.5% recall vs the brute-force labels. |
| 2 | **int8 quantization** | Map `[-1, 1]` → `[-128, 127]` (scale 127, zero 0). The `-1` sentinel preserves its meaning naturally. 14 bytes/vec × 3M = **42 MB raw**. Distance computed as int16 squared accumulator (no overflow for 14 dims). |
| 3 | **Build index at Docker build time, mmap at runtime** | Reference file is immutable. Pay the build cost once. Ship a binary index file. Both API instances mmap the same file: Linux page cache de-dupes, so we pay the RAM cost once for both. |
| 4 | **fasthttp (Go) / uWebSockets.js (Node)** | net/http and Node's http have ~50-100 µs per request overhead. At our target p99 budget that is 5-10% of the latency budget on syscalls alone. Both alternatives are battle-tested at sub-ms ranges. |
| 5 | **Custom Go LB binary** | nginx ≈ 25 MB resident, HAProxy ≈ 15 MB. A 100-line Go round-robin reverse proxy ships in ~6 MB scratch image, ~10 MB resident. The LB only forwards; we do not need nginx semantics. |
| 6 | **Pre-baked response strings** | fraud_score is one of 6 values: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0. Pre-allocate the 6 JSON response byte slices at startup. Zero allocation per request on the happy path. |
| 7 | **Hand-rolled JSON for the request** | The payload schema is fixed. A schema-driven parser (jsoniter API or hand-written tokenizer) avoids reflection. Worth ~2-5 µs per request vs `sonic`/`encoding/json`. Node side: V8 JSON.parse is already heavily optimized; we use it directly. |
| 8 | **GOMAXPROCS=1 per API instance, 2 instances** | With 0.45 CPU per instance, Go runtime stealing across multiple Ps is wasted. Pin GOMAXPROCS=1 to remove scheduler overhead. Node is single-threaded already. |
| 9 | **HTTP 500 is forbidden** | Weight 5× + counts toward 15% failure cutoff. Any internal error (timeout, panic, index miss) returns `{"approved": true, "fraud_score": 0.0}` instead. Sacrifice 1×-3× per failed request to avoid the 5× error penalty. |
| 10 | **No database, no message broker, no Redis** | Read-only workload over immutable reference data. Anything beyond the binary index + HTTP server is dead weight in this budget. |

## Memory budget

| Component | Target | Notes |
|-----------|--------|-------|
| LB (Go scratch) | 10 MB | Single binary, fasthttp reverse proxy, round-robin |
| API instance × 2 | 80 MB each = 160 MB | Go runtime + fasthttp pools + working set. Page cache of mmap'd index is shared. |
| Index file (mmap, shared) | 70 MB | int8 vectors (42 MB) + HNSW graph (~25 MB at M=16) + labels bitmap (3 MB) |
| Headroom | 110 MB | Linux page cache for the index, network buffers, kernel slab |
| **Total** | **350 MB** | |

For Node: API instance baseline jumps to ~110 MB each (V8 heap + uWS native). Total ~290 MB before headroom. **Tighter.** First validation milestone for Node is "does it fit at all."

## CPU budget

- LB: 0.05 (forwarding only)
- API × 2: 0.45 each = 0.90
- Slack: 0.05

GOMAXPROCS=1 inside the container. We are not parallelizing kNN inside a single request; a single ef=50 HNSW search is short enough that goroutine overhead would dominate.

## Per-stack specifics

### Go

- **HNSW lib**: start with `github.com/coder/hnsw`. If recall/perf is short of target after week 1, swap for an int8-native implementation (probably hand-rolled, ~400 LOC).
- **SIMD**: explore `gonum/floats` first. If we need more, write a small AVX2 assembly stub for `int16` squared-sum over 14 lanes. Go's `cmd/asm` supports this. Last resort: cgo to `usearch`.
- **JSON**: hand-written tokenizer over the fixed schema. The payload is deterministic enough that a state machine is faster than any generic parser.
- **HTTP**: `valyala/fasthttp`, no router, single handler.
- **GC**: `GOGC=200` (less frequent GC, more memory headroom in steady state since hot path allocates nothing).

### Node

- **HTTP**: `uNetworking/uWebSockets.js` v20+. ~10× faster than express, comparable to Go fasthttp on throughput.
- **kNN**: `hnswlib-node` (native binding to `nmslib/hnswlib`). If memory is tight, build a thinner N-API binding around `unum-cloud/usearch` (int8 native).
- **Index loading**: HNSW lib loads the same on-disk format the Go side builds. If the lib does not expose mmap, fall back to `Buffer.from(fs.readFileSync(...))` and accept the dup cost per instance.
- **V8 flags**: `--max-old-space-size=80 --max-semi-space-size=4 --optimize-for-size`. JIT warmup at `/ready` with 200 sample queries.
- **GC**: don't fight V8. Avoid allocations on the hot path (no object spread, reuse Buffers).

## What we are NOT doing

- **No C/Rust/Zig.** Out-of-stack for Vini. Optimizing in unfamiliar languages is negative-EV at this time budget.
- **No custom CPU-bound workers / worker_threads in Node.** Single-request kNN finishes in microseconds; thread hop overhead is larger than the work.
- **No GPU.** Even if it were allowed (it is not on the test rig).
- **No caching of responses.** Test payloads are unique per transaction; cache hit rate would be ~0%.
- **No payload validation past what is needed for vectorization.** Trust the input shape, fall back to safe response on parse failure.

## Risks (ranked)

1. **Recall < 99.5%** under HNSW pushes us into FN territory. FN weight = 3. Mitigation: exact rerank on top-20 candidates from ANN, validated against brute-force labels on 10k held-out queries before submission.
2. **mmap page cache thrash** if the working set exceeds RAM. Mitigation: pre-touch all pages at `/ready`, profile RSS under load.
3. **Cold-start latency**: first request after container warmup hits a cold mmap. The k6 test ramps up gradually so this should not show in p99, but we pre-warm anyway.
4. **Node memory budget**: 110 MB × 2 = 220 MB just for instances. If we slip we drop to 1 API instance + LB (still compliant — spec says ≥2 instances). **Plan B**: 2 lighter instances by streaming the index from disk on first query (negative for p99).
5. **HNSW lib quality variance** between Go and Node. If the Go-side index format is not loadable by `hnswlib-node`, we rebuild it in Node at startup from a shared int8 dump. ~5-10 s container startup overhead, no runtime cost.

## Schedule (17 days, deadline 2026-06-05)

| Window | Goal | Status |
|--------|------|--------|
| D0 (2026-05-19) | Architecture doc + repo bootstrap | done |
| D1-D3 | Node v0: correct, in budget, uWS + hnswlib-node + custom Go LB | |
| D4-D5 | Index pipeline (Go build-index → on-disk binary) + k6 harness + scoring reproducer | |
| D6-D7 | Node round 1: int8 + recall validation + hnswlib tuning | |
| D8 | First Node `rinha/test` submission | |
| D9-D10 | Node iteration based on ranking feedback | |
| D11-D13 | Stretch: Go port IF Node is locked at top of stack | |
| D14-D15 | Buffer + recall edge cases | |
| D16 | Final submissions | |
| D17 (2026-06-05) | Deadline | |

## Kill criteria

- If by D3 Node cannot fit two instances + LB inside 350 MB even before the index, drop to one API instance (still spec-compliant).
- If by D8 the Node submission scores below 3500 with no clear next 1000-point lever, the goal downgrades to "top-3 Node finish" and the Go stretch is dropped.
- The Go stretch is only triggered if (a) the Node submission is at #1 of the Node stack by D10 and (b) total time spent so far is under 20 hours. Otherwise dropped.
- If at any point this starts costing more than ~30 hours total, publish what we have, write the post, move on. The post is the durable artifact, not the rank.
