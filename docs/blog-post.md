# Vector search under 350 MB of RAM: notes from Rinha de Backend 2026

> *Draft of the post that ships when the submission is locked. English, no em-dashes, written for engineers who already know what kNN is. Will publish on blog.playsonora.com once the official ranking comes back.*

The Brazilian backend competition that gets sponsored as "fight club but for HTTP latency" picked vector search as the 2026 theme. The setup, briefly: build a fraud detection API that, for each incoming card transaction, finds the 5 nearest neighbors in a labeled reference set of 3 million 14-dimensional vectors and returns an approve/deny decision. The constraint is 1 CPU and 350 MB of memory total, split across at least a load balancer and two API instances. The score is the sum of a logarithmic latency component (saturates at +3000 when p99 drops below 1 ms) and a detection-quality component (-3000 to +3000, with a 15 % failure cliff).

If you have ever shipped vector search in production at a real company, you know what those numbers mean. 3 million vectors at float32 is 168 MB of raw data, more if you want any kind of accelerating index on top. Going from float32 to int8 is the first move. Going from in-memory HNSW (which would cost about 250 MB at M=16 for this geometry) to a flat VP-tree is the second. After that, the remaining points are won or lost on what the JavaScript runtime does to your hot path.

I picked Node for this one. Here is what that decision cost me and what it bought me.

## The architecture in one paragraph

Raw `node:http` server listening on a Unix domain socket, with all detection logic in a single allocation-free hot path. The reference set is gunzipped, quantized to int8, and indexed into a VP-tree at Docker build time. The runtime reads three binary files (vectors, packed-bit labels, VP-tree nodes) via `fs.readFile` and views them as typed arrays. HAProxy 3.0 in tcp mode load-balances across two identical API containers over UDS. There are six possible response bodies (`fraud_score` is `count / 5` for count in `0..5`), and all six are pre-baked as Buffers at startup. The hot path allocates exactly zero objects: the body comes off the wire into a stack-pinned Buffer, the response is one of the six pre-built ones, and the kNN walk uses pre-allocated query buffers and a pre-allocated max-heap.

## What competitive intel actually told me

The instinct was to pick Go. Most Rinha winners across editions write Go, the language was on my resume, and the Brazilian backend community is heavily Go-flavored. The actual data said something else.

Rinha 2025 was won by a Node submission (`ricassiocosta`, Fastify 5 + Redis over UDS). The 2026 Node field, as of this writing, has exactly one serious submission with code (`JoaoMarcos160`) and a long tail of empty repositories. The Go field has 40 to 60 serious submissions, many from engineers who tune GOGC by instinct and write assembly when they need to. The path to first place in stack is much shorter in Node than in Go, even though the absolute ceiling is lower.

I committed to Node and accepted a soft cap somewhere around 5400 points out of 6000. The top Rust submissions will land near 5800 (AVX2 + io_uring + zero-overhead distance math is hard to argue with) and that is fine. The goal was always first place in the Node stack and a portfolio-grade write-up.

## VP-tree, not HNSW

The first instinct in any kNN problem is HNSW. It is the right answer in 90 % of real-world cases. It was the wrong answer here.

An HNSW index over 3 million vectors with M=16 (the default) and ef\_construction=200 needs roughly 250 MB. That blows the entire memory budget before the server starts. Lower M means worse recall, and we are graded against exact brute-force labels with a tie-break on original ID, so any recall loss converts directly into false negatives at weight 3 and false positives at weight 1.

VP-tree (vantage-point tree) was published in 1993, predates most of the cool ANN libraries, and is exact. For low-dimensional data (14 dims is well inside its sweet spot), it gives O(log N · k) expected queries with no recall risk. Each node stores one original-vector index (4 bytes), one threshold float (4 bytes), and two child offsets (4 bytes each). 3 million nodes at 16 bytes is 48 MB. Combined with 42 MB of int8 vectors and a 375 KB packed-bit label bitmap, the per-instance cold data sits at around 90 MB. With V8 baseline RSS around 35 MB and a handful of HTTP buffers, an instance lives comfortably under the 135 MB allocation.

The VP-tree search is a textbook iterative traversal with explicit stack. The interesting bit is the comparator on the top-k max-heap: tie-break on smaller original index, matching the grader's brute-force ordering exactly. Without this, payloads that land near the 3-of-5 decision boundary will sometimes resolve to a different fifth neighbor than the grader's reference, costing detection points on edge cases. With it, the local pipeline returns zero false positives and zero false negatives against the official `test-data.json` (54,100 test cases, 44 % fraud rate, 1.5 % edge cases).

## int8 quantization with a sentinel

Two of the 14 dimensions (`minutes_since_last_tx` and `km_from_last_tx`) are -1 when the transaction is the customer's first. The other twelve are in [0, 1]. The quantization maps [0, 1] to [0, 127] linearly, and -1 to -128.

```ts
if (v === -1) {
  output[i] = -128;
} else {
  const q = Math.round(v * 127);
  output[i] = q < 0 ? 0 : q > 127 ? 127 : q;
}
```

Distance is computed as an int32 sum of squared int16 differences:

```ts
let sum = 0;
for (let i = 0; i < 14; i++) {
  const d = query[i] - refs[refOffset + i];
  sum += d * d;
}
```

V8's TurboFan recognizes the loop and emits scalar SSE. It is not as fast as a hand-rolled AVX2 intrinsic in Rust, but it is consistent and inlined inside the VP-tree search. With the loop fully unrolled (manually) and the function inlined, the cost per distance comparison is around 5 nanoseconds. At an average of 100 to 300 distance comparisons per VP-tree query, the algorithmic cost lands around 0.5 to 1.5 microseconds. Most of the per-request budget then goes to JSON parsing and HTTP framing.

## Why not Fastify, why not uWebSockets.js

Fastify is a great default and Rinha 2025's top Node submission used it. For 2026 I picked raw `node:http`. There are three reasons.

First, zero external dependencies on the runtime path. The build-time pipeline uses `stream-json` for chunked decompression and parsing, but the runtime ships with `node:http`, `node:fs`, and `node:zlib`. Smaller image, fewer surprises, fewer transitive vulnerabilities to chase.

Second, the response is six pre-built Buffers. Schema-compiled JSON serializers (`fast-json-stringify`, `typebox`) only help when you have varied responses. We do not.

Third, uWebSockets.js would shave perhaps 50 to 100 microseconds per request via tighter syscall batching. That is a real number, but it lives below the GC noise floor at p99. The single biggest win at p99 is eliminating allocations on the hot path, not micro-optimizing the HTTP layer. If the post-validation results show p99 stuck above 2 ms, uWS becomes the round-2 lever.

## The hot path

This is what happens per request:

```ts
function handleFraudScore(req, res) {
  const chunks = collectChunks(req);
  req.on('end', () => {
    const payload = JSON.parse(chunks.toString('utf8'));
    vectorize(payload, queryFloat, ctx);
    quantizeOne(queryFloat, queryInt8);
    searchVpTree(tree, vectors, queryInt8, heap);
    const fraudCount = heap.countFrauds(labels);
    const buf = BODY_BUFFERS[fraudCount];
    res.writeHead(200, { 'content-length': buf.byteLength });
    res.end(buf);
  });
}
```

`queryFloat`, `queryInt8`, `heap` are all module-level singletons, pre-allocated at startup. `BODY_BUFFERS` is the array of six pre-baked response Buffers. Inside `vectorize`, the timestamp is parsed via character-code arithmetic on the fixed ISO-8601 layout (`YYYY-MM-DDThh:mm:ssZ`), avoiding the cost of `new Date()` and the allocation that comes with it.

The one remaining allocation is `Buffer.concat(chunks)` and the subsequent `toString('utf8')` and `JSON.parse`. These can be eliminated by hand-rolling a tokenizer over the fixed JSON schema, but the gain is around 30 to 80 microseconds per request and the cost is 200 lines of brittle parsing code. I am leaving this for round 2 if p99 needs more compression.

## Warmup at /ready

A k6 ramp from 1 to 900 req/s over 120 seconds gives V8 plenty of time to JIT-compile, but the first 50 to 100 requests after process start will land in the upper tail and contaminate p99. The fix is to run 10,000 dummy queries against the loaded index inside the `/ready` handler before it returns 200. This forces TurboFan to compile every hot path (the VP-tree iterator, the distance function, the heap operations) before live traffic arrives. Cost: about 50 ms at startup. Benefit: a flatter p99 distribution.

## What is hard about Node here

The honest list:

1. No AVX2 intrinsics from JS. V8 emits scalar SSE at best. Each distance comparison is around 5 ns instead of the 1 ns it could be with AVX2 over int16x16 lanes. For typical k=5 search workloads this matters less than it sounds, because the VP-tree prunes most of the dataset; but for the brute-force baseline, this is the dominant cost.
2. GC tail latency. Even with `--max-old-space-size=96 --max-semi-space-size=4 --optimize-for-size`, V8 will occasionally take a 3 to 10 ms minor GC pause. Any allocation in the hot path can contribute to this. The discipline of pre-allocating typed arrays and pre-baking Buffers is not a micro-optimization, it is the difference between p99 = 2 ms and p99 = 8 ms.
3. No io_uring on the HTTP layer without a native addon. The Rust submissions running io_uring at the kernel level save 50 to 100 microseconds per request that we cannot recover.
4. cluster and worker\_threads under 1 CPU. Multi-worker Node experiments by 2025 top-10 contestants explicitly degraded p99. Two single-process containers are already the parallelism that the budget allows.

The honest verdict: Node lands around p99 = 1.5 to 3 ms for this workload, which is competitive with everything except Rust + io\_uring + AVX2. That is enough for first place in Node and top-10 overall.

## Things I learned that transfer

A handful of things from this exercise that are worth carrying into production work even if you never enter Rinha.

The flat-buffer trick (typed arrays as the data structure, no objects) is enormously underused in Node services. If you have a read-heavy in-memory dataset, this pattern saves 50 to 80 % of the heap and eliminates an entire category of GC pressure.

Pre-baked responses are obvious in retrospect. Any endpoint where the response is one of a small enumerated set should be doing this. The cost of `JSON.stringify` on a fixed shape is in the microseconds, and at scale it shows up as GC pressure even when it does not show up in CPU profiles.

The right algorithmic primitive matters more than the library. HNSW is the default everyone reaches for, and in this specific geometry (small N, low dim, exact k=5, hard memory budget) a VP-tree from 1993 beats it on every axis. If your data shape does not match the assumptions a library was built for, write the 150 lines yourself.

JIT warmup at health-check time is the cheapest tail-latency mitigation you will ever ship. Five lines of code, 50 ms at startup, and the GC tail goes from ugly to manageable.

## Final score and link to repo

Submission lives at github.com/BertBR/rinha-2026. The score from the official run will land here when the engine posts back to the issue.
