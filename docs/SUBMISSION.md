# Submission process

The Rinha engine grades whatever sits on the `submission` branch of the repo registered in [`participants/BertBR.json`](https://github.com/zanfranceschi/rinha-de-backend-2026/blob/main/participants) in the upstream repo. There is no other test path.

## One-time setup

1. Push this repo to GitHub:
   ```bash
   git remote add origin git@github.com:BertBR/rinha-2026.git
   git push -u origin main
   ```

2. Open a PR on upstream `zanfranceschi/rinha-de-backend-2026` adding `participants/BertBR.json` with the contents of `scripts/participant.json`.

## Each submission round

```bash
# from a clean main checkout
./scripts/sync-submission.sh
```

This builds (or checks out) the `submission` branch and force-pushes a clean snapshot containing only what the grader needs:

- `docker-compose.yml`
- `Dockerfile.api`, `Dockerfile.lb`, `haproxy.cfg`, `.dockerignore`
- `package.json`, `package-lock.json`, `tsconfig.json`
- `src/` (no `src/test/`)

## Triggering the official test

Open an issue on this repo with `rinha/test` in the description. The Rinha engine watches for that string, runs the test, posts the score as a comment, and closes the issue.

```bash
gh issue create -t "rinha test" -b "rinha/test"
```

The result comment will contain the JSON described in upstream `EVALUATION.md`:

```json
{
  "p99": "...",
  "scoring": {
    "breakdown": { "true_positive_detections": ..., ... },
    "p99_score":  { "value": ... },
    "detection_score": { "value": ... },
    "final_score": ...
  }
}
```

## What to do with the result

| `final_score` | Action |
|---------------|--------|
| ≥ 5400        | Lock the branch, write the blog post, watch the ranking |
| 4000 - 5400   | Identify the weaker axis (latency or detection), open a round-2 issue |
| 3000 - 4000   | Profile under k6 locally, find the GC tail or detection bug |
| < 3000        | Something is wrong with the build pipeline; check the engine logs in the issue comment |

## Round-2 levers (if needed)

Ordered by expected payoff vs cost.

1. **Hand-roll the JSON body parser** for the fixed schema. Skips `Buffer.concat`, `toString`, and `JSON.parse`. Worth 30-80 µs per request.
2. **Swap `node:http` for `uWebSockets.js`**. Saves 50-100 µs per request on syscalls. Adds a native binding to the image.
3. **Tighten V8 flags**: `--no-opt-eager` `--no-lazy` `--predictable-gc-schedule`. Trade peak throughput for steadier tail.
4. **Increase `WARMUP_QUERIES` from 10000 to 50000**. More TurboFan compilation cycles before live traffic. Costs ~150 ms at startup.
5. **Profile with `--cpu-prof`** during `make bench` and look for any unexpected allocations on the hot path.
