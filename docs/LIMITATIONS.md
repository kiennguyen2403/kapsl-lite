# Limitations and Caveats

## Memory Isolation

- Memory budget is enforced by scheduler admission and guardrails.
- It is not a hard OS isolation boundary by default.
- Under severe spikes or inaccurate model sizing, kernel OOM is still possible.

## Preemption Granularity

- Preemption is cooperative.
- Cancellation checks occur around inference boundaries, not guaranteed mid-kernel interruption.

## Generation State

- Runtime tracks model states (`idle/queued/running/paused`) at scheduler level.
- Full token-stream state machine (`idle/generating/paused/token-progress`) is still partial.

## Output Readability

- ONNX-only paths may return logits summaries unless full decoding pipeline is enabled.
- Multimodal quality depends on available side-stage assets in `.aimod`.

## Platform Variability

- Thermal and system telemetry quality depends on OS sensor availability.
- On unsupported sensor paths, fallback/default values may be used.

## Not in Scope for Lite Runtime

- distributed orchestration
- cloud API service layer
- hard multi-tenant isolation
- GPU-centric scheduling orchestration
