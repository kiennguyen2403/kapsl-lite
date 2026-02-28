[CmdletBinding()]
param(
    [string]$ModelsDir = "..\kapsl-runtime\models\multi",
    [ValidateRange(64, 262144)]
    [int]$MemoryMb = 3072,
    [ValidateRange(100, 60000)]
    [int]$PollMs = 2000,
    [string]$BinaryPath = "",
    [switch]$DryRun,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,
        [Parameter(Mandatory = $true)]
        [string]$BaseDir
    )

    if ([string]::IsNullOrWhiteSpace($PathValue)) {
        throw "Path value cannot be empty."
    }

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        $candidate = $PathValue
    } else {
        $candidate = Join-Path $BaseDir $PathValue
    }

    $resolved = Resolve-Path -LiteralPath $candidate -ErrorAction SilentlyContinue
    if ($null -eq $resolved) {
        throw "Path not found: $PathValue (resolved from base '$BaseDir')"
    }
    return $resolved.Path
}

$scriptDir = Split-Path -Parent $PSCommandPath
$runtimeLiteDir = (Resolve-Path (Join-Path $scriptDir "..")).Path

$modelsResolved = Resolve-ProjectPath -PathValue $ModelsDir -BaseDir $runtimeLiteDir
if (-not (Test-Path -LiteralPath $modelsResolved -PathType Container)) {
    throw "ModelsDir must be a directory: $modelsResolved"
}

$aimodFiles = @(Get-ChildItem -LiteralPath $modelsResolved -Filter "*.aimod" -File -ErrorAction SilentlyContinue)
if ($aimodFiles.Count -eq 0) {
    throw "No .aimod files found in models directory: $modelsResolved"
}

$exeResolved = $null
if (-not [string]::IsNullOrWhiteSpace($BinaryPath)) {
    $exeResolved = Resolve-ProjectPath -PathValue $BinaryPath -BaseDir $runtimeLiteDir
} else {
    $defaultExe = Join-Path $runtimeLiteDir "target\release\kapsl.exe"
    if (Test-Path -LiteralPath $defaultExe -PathType Leaf) {
        $exeResolved = (Resolve-Path -LiteralPath $defaultExe).Path
    }
}

$env:KAPSL_LITE_MEMORY_LIMIT_MIB = "$MemoryMb"
$env:KAPSL_LITE_THERMAL_POLL_INTERVAL_MS = "$PollMs"
$env:KAPSL_LITE_INGRESS_ENABLED = "1"
$env:KAPSL_LITE_INGRESS_STREAM_ENABLED = "0"
$env:KAPSL_LITE_TRIGGER_BUS_CAPACITY = "256"
$env:KAPSL_LITE_QUEUE_LIMIT = "256"
$env:KAPSL_LITE_INFERENCE_RESULTS_PATH = (Join-Path $runtimeLiteDir "target\\inference-results.jsonl")
$env:KAPSL_LITE_LLM_DEFAULT_MAX_NEW_TOKENS = "64"
$packageTmpDir = Join-Path $runtimeLiteDir "target\\package-tmp"
$modelCacheDir = Join-Path $runtimeLiteDir "target\\model-cache"
[System.IO.Directory]::CreateDirectory($packageTmpDir) | Out-Null
[System.IO.Directory]::CreateDirectory($modelCacheDir) | Out-Null
$env:KAPSL_PACKAGE_TMP_DIR = $packageTmpDir
$env:KAPSL_LITE_MODEL_CACHE_DIR = $modelCacheDir

$runtimeArgs = @(
    "run",
    "--models", $modelsResolved,
    "--memory-mb", "$MemoryMb",
    "--poll-ms", "$PollMs"
)
if ($ExtraArgs) {
    $runtimeArgs += $ExtraArgs
}

Write-Host "[edge-preset] runtime-lite dir : $runtimeLiteDir"
Write-Host "[edge-preset] models dir       : $modelsResolved"
Write-Host "[edge-preset] memory budget    : ${MemoryMb}MiB"
Write-Host "[edge-preset] thermal poll     : ${PollMs}ms"

if ($exeResolved) {
    $launch = @($exeResolved) + $runtimeArgs
    Write-Host "[edge-preset] launcher        : binary"
    Write-Host ("[edge-preset] command         : " + ($launch -join " "))
    if ($DryRun) {
        return
    }

    & $exeResolved @runtimeArgs
    exit $LASTEXITCODE
}

$cargoArgs = @("run", "--release", "--") + $runtimeArgs
Write-Host "[edge-preset] launcher        : cargo"
Write-Host ("[edge-preset] command         : cargo " + ($cargoArgs -join " "))
if ($DryRun) {
    return
}

Push-Location $runtimeLiteDir
try {
    & cargo @cargoArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
