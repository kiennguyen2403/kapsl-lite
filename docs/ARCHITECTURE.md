# Architecture

## Runtime Components

1. Loader
- scans `.aimod` packages
- resolves model assets to cache paths
- parses scheduling metadata (priority, trigger mode, budgets)

2. Scheduler
- trigger dispatcher with priority ordering
- admission checks for memory headroom
- policy controllers:
  - thermal (`normal -> t1 -> t2 -> t3`)
  - battery (`normal -> conserve -> critical`)
  - memory (`normal -> emergency`)

3. Package Workers
- `always_running` loops for continuous inference
- `on_demand` worker pools for trigger-driven inference
- cooperative cancellation flags for preemptible workloads

4. Ingress and Trigger Bus
- manual trigger
- external input adapter (Windows named pipe, Unix socket/stream where available)
- event normalization and route-to-target dispatch

5. Inference Backends
- ONNX runtime backend
- optional LLM backend mode
- per-request options: wall timeout, token cap, cancellation signal

6. Observability (TUI)
- runtime core metrics
- per-model states
- queue depth/peak
- scheduler logs
- emergency telemetry counters

## Control Flow (High Level)

1. Startup loads package specs and registers workers.
2. Trigger dispatcher accepts events and prioritizes pending triggers.
3. Before dispatch:
- deadline check
- thermal/battery/memory policy check
- memory headroom check
4. Accepted triggers execute in target package worker.
5. Result and scheduler events are emitted to TUI + optional JSONL sink.

## Emergency Behavior

When memory emergency is active:
- on-demand effective concurrency is reduced to one active worker
- backend loads are serialized
- idle/post-run backends are unloaded aggressively
- non-realtime triggers are rejected
