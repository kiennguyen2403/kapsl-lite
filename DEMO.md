# Demo Script (60 seconds)

Use pre-recorded segments and stitch. One continuous live take is risky.

## 0-5s: Intro

On-screen text:
- "kapsl-lite: multi-model edge runtime with graceful degradation"

## 5-12s: Boot + TUI

```powershell
.\scripts\run-edge-preset-windows.ps1 -ModelsDir ..\kapsl-runtime\models\multi -MemoryMb 12288
```

Show:
- models loaded
- memory gauge
- runtime core status

## 12-20s: Trigger Pipeline

```powershell
python .\scripts\request-inference-result.py --target mistral-llm --prompt "Describe scene"
```

Show scheduler flow:
- ingress accepted
- trigger accepted
- trigger completed

## 20-30s: Queue + Priority

Burst requests:

```powershell
1..15 | % { python .\scripts\request-inference-result.py --target mistral-llm --prompt "q$_" --timeout 8 --connect-timeout 8 }
```

Show:
- queue depth and queue peak changing
- model state transitions

## 30-44s: Memory Emergency

Run with strict memory settings to force emergency:

```powershell
$env:KAPSL_LITE_MEMORY_EMERGENCY_FREE_MIB="2048"
$env:KAPSL_LITE_MEMORY_RECOVERY_FREE_MIB="3072"
.\scripts\run-edge-preset-windows.ps1 -ModelsDir ..\kapsl-runtime\models\multi -MemoryMb 3072
```

Show:
- `status: memory_emergency`
- `worker parked`, `serialized load`, `backend unloaded`
- per-model `e=park/unload/load` counters

## 44-55s: Thermal Policy

Force thresholds below ambient:

```powershell
$env:KAPSL_LITE_THERMAL_SOFT_C="33"
$env:KAPSL_LITE_THERMAL_DEGRADED_C="35"
$env:KAPSL_LITE_THERMAL_HARD_C="40"
$env:KAPSL_LITE_THERMAL_RECOVERY_C="32"
```

Show:
- `status: thermal_t1` or `throttled`
- deferred/rejected triggers with thermal reason

## 55-60s: Close

On-screen checklist:
- multi-model scheduling
- memory/thermal guardrails
- event-driven inference
- live observability

## Cleanup

```powershell
Remove-Item Env:KAPSL_LITE_MEMORY_EMERGENCY_FREE_MIB,Env:KAPSL_LITE_MEMORY_RECOVERY_FREE_MIB,Env:KAPSL_LITE_THERMAL_SOFT_C,Env:KAPSL_LITE_THERMAL_DEGRADED_C,Env:KAPSL_LITE_THERMAL_HARD_C,Env:KAPSL_LITE_THERMAL_RECOVERY_C -ErrorAction SilentlyContinue
```
