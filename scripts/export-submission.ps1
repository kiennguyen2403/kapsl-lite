[CmdletBinding()]
param(
    [string]$OutputDir = "..\kapsl-runtime-lite-submission",
    [switch]$Clean,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Resolve-ProjectPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$PathValue,
        [Parameter(Mandatory = $true)]
        [string]$BaseDir
    )

    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return [System.IO.Path]::GetFullPath($PathValue)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $PathValue))
}

function Ensure-Directory {
    param([Parameter(Mandatory = $true)][string]$PathValue)
    if (-not (Test-Path -LiteralPath $PathValue -PathType Container)) {
        [System.IO.Directory]::CreateDirectory($PathValue) | Out-Null
    }
}

function Copy-FileWithParents {
    param(
        [Parameter(Mandatory = $true)][string]$SourceRoot,
        [Parameter(Mandatory = $true)][string]$DestinationRoot,
        [Parameter(Mandatory = $true)][string]$SourceFile
    )
    $relative = Get-RelativePath -BasePath $SourceRoot -TargetPath $SourceFile
    $destFile = Join-Path $DestinationRoot $relative
    $destDir = Split-Path -Parent $destFile
    Ensure-Directory -PathValue $destDir
    Copy-Item -LiteralPath $SourceFile -Destination $destFile -Force
}

function Get-RelativePath {
    param(
        [Parameter(Mandatory = $true)][string]$BasePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )

    $baseFull = [System.IO.Path]::GetFullPath($BasePath)
    $targetFull = [System.IO.Path]::GetFullPath($TargetPath)

    if (-not $baseFull.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $baseFull += [System.IO.Path]::DirectorySeparatorChar
    }

    $baseUri = [Uri]::new($baseFull)
    $targetUri = [Uri]::new($targetFull)
    $relativeUri = $baseUri.MakeRelativeUri($targetUri)
    return [Uri]::UnescapeDataString($relativeUri.ToString()).Replace('/', '\')
}

function Should-IncludeFile {
    param(
        [Parameter(Mandatory = $true)][string]$RelativePath,
        [Parameter(Mandatory = $true)][string]$FileName
    )

    $normalized = $RelativePath.Replace('\', '/')

    if ($normalized -match '^target/') { return $false }
    if ($normalized -match '^\.git/') { return $false }
    if ($normalized -match '^scripts/\.venv/') { return $false }
    if ($normalized -match '/__pycache__/') { return $false }

    if ($FileName -match '\.(aimod|onnx|onnx_data|gguf|kapsl|log|jsonl)$') { return $false }
    if ($FileName -in @('.DS_Store')) { return $false }

    return $true
}

$scriptDir = Split-Path -Parent $PSCommandPath
$repoRoot = Resolve-ProjectPath -PathValue ".." -BaseDir $scriptDir
$destRoot = Resolve-ProjectPath -PathValue $OutputDir -BaseDir $repoRoot

if (-not (Test-Path -LiteralPath $repoRoot -PathType Container)) {
    throw "Failed to resolve repository root."
}

if ((Test-Path -LiteralPath $destRoot) -and $Clean) {
    Write-Host "[export] removing existing output: $destRoot"
    if (-not $DryRun) {
        Remove-Item -LiteralPath $destRoot -Recurse -Force
    }
}

Ensure-Directory -PathValue $destRoot

$rootFiles = @(
    "README.md",
    "DEMO.md",
    "submission.md",
    "LICENSE",
    ".gitignore",
    "Cargo.toml",
    "Cargo.lock"
)

$rootDirs = @(
    ".cargo",
    "src",
    "scripts",
    "docs"
)

$copied = 0
$skipped = 0

foreach ($name in $rootFiles) {
    $src = Join-Path $repoRoot $name
    if (-not (Test-Path -LiteralPath $src -PathType Leaf)) {
        Write-Host "[export] skip missing file: $name"
        continue
    }
    if (-not $DryRun) {
        Copy-FileWithParents -SourceRoot $repoRoot -DestinationRoot $destRoot -SourceFile $src
    }
    $copied++
}

foreach ($dir in $rootDirs) {
    $srcDir = Join-Path $repoRoot $dir
    if (-not (Test-Path -LiteralPath $srcDir -PathType Container)) {
        Write-Host "[export] skip missing dir: $dir"
        continue
    }

    $files = Get-ChildItem -LiteralPath $srcDir -Recurse -File
    foreach ($file in $files) {
        $relative = Get-RelativePath -BasePath $repoRoot -TargetPath $file.FullName
        if (-not (Should-IncludeFile -RelativePath $relative -FileName $file.Name)) {
            $skipped++
            continue
        }
        if (-not $DryRun) {
            Copy-FileWithParents -SourceRoot $repoRoot -DestinationRoot $destRoot -SourceFile $file.FullName
        }
        $copied++
    }
}

Write-Host "[export] source : $repoRoot"
Write-Host "[export] output : $destRoot"
Write-Host "[export] copied : $copied files"
Write-Host "[export] skipped: $skipped files (excluded by filter)"

if ($DryRun) {
    Write-Host "[export] dry-run only (no files written)"
}
