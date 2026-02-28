# kapsl-runtime-lite

Edge AI execution runtime for constrained devices.  
It coordinates multiple local models with scheduler guardrails (memory, thermal, battery), event-driven triggers, and live TUI observability.

## Problem

Running multiple AI models on small edge hardware is unstable:
- memory spikes cause process kills
- thermal pressure degrades throughput unpredictably
- low-priority work can block critical tasks
- most tooling does not provide coordinated multi-model scheduling

## Solution

kapsl-lite combines three layers:

1. `.aimod` package contract  
   Each model declares memory budget, priority, trigger mode, and policy metadata.
2. Runtime scheduler  
   Admission checks, priority-aware dispatch, thermal/battery/memory actions, emergency serialization, and recovery.
3. Event trigger bus  
   Detection/manual/external input events dispatch to on-demand models asynchronously.

## Key Capabilities

- On-demand and always-running model modes
- Priority scheduling for trigger dispatch
- Queue controls (`capacity`, overflow policy, depth/peak telemetry)
- Thermal states (`thermal_t1`, `throttled`, `suspended`)
- Memory emergency mode with:
  - effective on-demand concurrency reduced to `1`
  - serialized backend load
  - aggressive on-demand backend unload
- Preemptible vs non-preemptible workload handling
- Windows named-pipe ingress + stream ingress
- TUI with model states, queue stats, scheduler log, and emergency counters

## Feature Status (Current)

- FPS targets for vision loops: supported
- Detection events triggering reasoning models: supported
- Preemptible vs non-preemptible inference workloads: supported (cooperative)
- Model priority relative to other models: supported for dispatch/load order
- Idle vs mid-generation per-model state machine: partial
- Mid-generation token state tracking: not yet complete
- Memory budget as hard OS isolation boundary: partial (scheduler contract, not cgroup/job hard limit)

## Quick Start (Windows)

```powershell
cargo build --release
.\scripts\run-edge-preset-windows.ps1 -ModelsDir ..\kapsl-runtime\models\multi -MemoryMb 3072
```

Manual trigger:

```powershell
python .\scripts\request-inference-result.py --target mistral-llm --prompt "hello"
```

## Quick Start (Linux)

```bash
cargo build --release
./target/release/kapsl run --models ../kapsl-runtime/models/multi --memory-mb 3072 --poll-ms 2000
```

## Demo Pack

- Video script: [DEMO.md](DEMO.md)
- Architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Limits and caveats: [docs/LIMITATIONS.md](docs/LIMITATIONS.md)
- Submission summary template: [submission.md](submission.md)

## Repo Layout

```text
src/        runtime core, scheduler, adapters, TUI
scripts/    run presets and inference request utilities
docs/       architecture and limitation notes
```

## Models and Assets

Large model binaries are intentionally excluded from git:
- `*.aimod`, `*.onnx`, `*.onnx_data`, `models/`

Provide your own model pack at runtime (for example `../kapsl-runtime/models/multi`).

## License

MIT. See [LICENSE](LICENSE).
