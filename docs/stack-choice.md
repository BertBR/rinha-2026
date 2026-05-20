# Stack choice — why Node and not Go

Recorded for future-me. The temptation to go Go is real (it's the Rinha default, my README says backend-Go, my Digai role uses Go). This document is here to remind me why I picked Node when I revisit this in a week and start second-guessing.

## What the public evidence said

Looking at my own active (non-archived, non-fork) repos as of 2026-05-19:

**TypeScript / Node dominates the last 18 months:**
- `harmony-insights` — 2.6 MB TS, pushed 2026-05-11. Largest active project.
- `quick-whatsapp-scheduler` — 391 KB TS.
- `task-trove-finder`, `partner-product-nexus`, `recipe-ai-grocery-link`, `product-file-harvester`, `snapshot-design-lab` — all TS, all non-trivial.
- `gauntlet` (AI eval) and `grok-eval` — TS.
- LangChain.js PR #10895 merged 2026-05-18 — TS, in SDK internals (streaming, structured outputs).

**Go is older territory:**
- `teeket` — 154 KB Go, last push 2023-10. Only substantial Go.
- `RetroGamesBot-Go` — 10 KB Go, production but small.
- `go-fyne-app` — toy.

My self-assessment ("expertise = Node, basic = Go") matches the evidence.

## Why this matters for a Rinha podium attempt

| Factor | Go | Node |
|--------|-----|------|
| Daily fluency | basic | expert |
| Competitive field (historical Rinha) | crowded, 40-60 serious submissions | sparse, 5-15 |
| Raw p99 ceiling | sub-ms achievable | ~2-5 ms realistic |
| Memory budget headroom | comfortable (runtime 15-30 MB) | tight (V8 50-80 MB per process) |
| HNSW library quality | `coder/hnsw` decent, hnswlib via cgo | `hnswlib-node` is a battle-tested native binding |
| **Probability of #1 in stack** | **20-30%** | **40-60%** |
| Hours to that level | ~35h with language friction | ~25h |
| Recovery when something breaks | slow (reading unfamiliar Go) | fast (TS debugging is muscle memory) |
| Blog post depth afterward | shallow ("I used HNSW in Go") | deep, defendable in interviews (V8 hidden classes, ICs, uWS internals, N-API marshaling cost) |

## The decisive argument

The goal is **first in stack**, not first overall. Go's stack at Rinha has 40-60 serious competitors, many of them full-time Go performance engineers. Node's stack historically has ~10 serious submissions and most are Express-based.

Entering Node with **uWebSockets.js + hnswlib-node + mmap + disciplined memory** puts me in a different league within the Node stack from day zero. That's a field where I have real technical advantage, not a contested one.

Go is the opposite: I would be a generic senior backend engineer competing against people who tune GOGC by instinct and write assembly when needed.

## Strategic bonus

Target pipeline (Speechify Platform, Clara AI, LemFi, Anthropic Skills) is Node/TS-heavy on backend product surfaces. "#1 Node at Rinha 2026" + LangChain.js merged + Playsonora solo is a clean thematic trio: senior JS/TS infrastructure engineer who ships production AI. Same voice throughout.

A Go win would dilute that. The "senior backend with AI" positioning is already carried by 15 years of carrier.

## When to revisit this decision

Only if:
1. The Node submission lands at #1 of its stack by D10 with hours to spare, OR
2. Node hits a hard memory wall I cannot work around and the architecture cannot fit.

In case 1, Go is a stretch worth attempting. In case 2, I downgrade to one API instance (still spec-compliant) before switching languages.
