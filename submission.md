# Submission Summary Template

## Project

- Name: `kapsl-runtime-lite`
- Track: Edge AI Runtime / Systems
- Team: `kapsl`
- Repo: `https://github.com/kiennguyen2403/kapsl-lite`
- Demo Video: `<video url>`

## Problem

Edge devices running multiple AI models become unstable due to memory and thermal contention, with limited scheduling control and poor observability.

## Solution

kapsl-lite is a local runtime that coordinates multi-model workloads using:
- package-level execution contracts (`.aimod`)
- priority-aware trigger scheduling
- thermal/battery/memory guardrail policies
- event-driven asynchronous model activation
- real-time operator observability in TUI

## Why It Matters

- improves reliability of edge AI under constrained resources
- avoids uncontrolled failure modes in robotics/industrial scenarios
- enables offline operation with predictable degradation behavior

## Implemented Highlights

- multi-model trigger dispatch with queue control
- emergency memory mode with serialized load and effective concurrency reduction
- thermal tiering with load-shed and safe-mode behavior
- per-model emergency counters and scheduler audit logs

## Current Limits

- no hard OS memory isolation by default
- cooperative preemption (not guaranteed mid-kernel stop)
- token-level generation state tracking is partial

## Future Work

- per-model hard isolation via OS-level controls
- richer token-progress state machine
- expanded multimodal decode pipelines
- benchmark and chaos test suite for fault injection
