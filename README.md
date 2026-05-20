# rinha-2026

Submission for [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026) — fraud detection via vector search.

Two implementations on the same architecture, aiming for first place per stack:

- `go/` — Go + fasthttp + HNSW (int8 quantized)
- `node/` — Node.js + uWebSockets.js + hnswlib-node

Both share the same on-disk index format, built once at Docker build time.

## Architecture

See [`docs/architecture.md`](docs/architecture.md) for locked decisions and the memory/CPU budget.

## Layout

```
go/                   Go submission (primary)
  cmd/api/            Fraud detection API binary
  cmd/lb/             Minimal round-robin load balancer
  cmd/build-index/    Builds the binary HNSW+int8 index from references.json.gz
  internal/           hnsw, quantize, vector, api
node/                 Node submission (secondary)
bench/                Local k6 script + scoring reproducer
data/                 references.json.gz + derived files (gitignored)
docs/                 Architecture and decision log
```

## Building locally

```bash
make data           # download references.json.gz + mcc_risk.json + normalization.json
make index          # build binary index from reference data
make go-up          # docker compose up the Go submission
make node-up        # docker compose up the Node submission
make bench          # run k6 locally against the running stack
```

## Constraints

1 CPU, 350 MB RAM total. Bridge network. `linux-amd64`. Port 9999. Public Docker images only.

## License

MIT.
