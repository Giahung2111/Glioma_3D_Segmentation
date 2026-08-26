Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:GliomaCodeRoot = (Split-Path -Parent $PSScriptRoot)
$script:GliomaProjectRoot = (Split-Path -Parent $script:GliomaCodeRoot)
$script:GliomaWorkspace = Join-Path $script:GliomaProjectRoot 'Workspace'
$condaStylePython = Join-Path $script:GliomaCodeRoot '.venv\python.exe'
$venvStylePython = Join-Path $script:GliomaCodeRoot '.venv\Scripts\python.exe'
if (Test-Path -LiteralPath $condaStylePython -PathType Leaf) {
    $script:GliomaPython = $condaStylePython
}
else {
    $script:GliomaPython = $venvStylePython
}

if (-not (Test-Path -LiteralPath $script:GliomaPython -PathType Leaf)) {
    throw "Project Python was not found at $script:GliomaPython. Create Code\.venv and install the project and External\nnUNet before running a stage."
}

$Env:nnUNet_raw = Join-Path $script:GliomaWorkspace 'nnUNet_raw'
$Env:nnUNet_preprocessed = Join-Path $script:GliomaWorkspace 'nnUNet_preprocessed'
$Env:nnUNet_results = Join-Path $script:GliomaWorkspace 'nnUNet_results'
$Env:PYTHONUNBUFFERED = '1'

foreach ($path in @(
    $script:GliomaWorkspace,
    $Env:nnUNet_raw,
    $Env:nnUNet_preprocessed,
    $Env:nnUNet_results,
    (Join-Path $script:GliomaWorkspace 'reports'),
    (Join-Path $script:GliomaWorkspace 'telemetry'),
    (Join-Path $script:GliomaWorkspace 'predictions'),
    (Join-Path $script:GliomaWorkspace 'cache')
)) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}

function Write-GliomaStage {
    param(
        [Parameter(Mandatory = $true)][string]$Number,
        [Parameter(Mandatory = $true)][string]$Name
    )
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Host "[$timestamp] [$Number] $Name" -ForegroundColor Cyan
}

function Invoke-GliomaPython {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)
    & $script:GliomaPython @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python stage failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Invoke-GliomaScript {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptName,
        [string[]]$Arguments = @()
    )
    $target = Join-Path $PSScriptRoot $ScriptName
    & $target @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$ScriptName failed with exit code $LASTEXITCODE"
    }
}

function New-GliomaExperimentId {
    param(
        [ValidateSet('prelim', 'fullcv', 'benchmark', 'system')][string]$Kind = 'prelim',
        [ValidateRange(0, 4)][int]$Fold = 0
    )
    $output = & $script:GliomaPython '-m' 'glioma_seg.backends.nnunet.backend' `
        '--project-root' $script:GliomaProjectRoot `
        'new-experiment' '--kind' $Kind '--fold' $Fold
    if ($LASTEXITCODE -ne 0) {
        throw 'Unable to allocate a unique experiment ID.'
    }
    return ($output | Select-Object -Last 1).Trim()
}

function Get-GliomaReportDirectory {
    param([Parameter(Mandatory = $true)][string]$ExperimentId)
    $path = Join-Path (Join-Path $script:GliomaWorkspace 'reports') $ExperimentId
    New-Item -ItemType Directory -Path $path -Force | Out-Null
    return $path
}

function Get-GliomaTrainer {
    param([Parameter(Mandatory = $true)][string]$ExperimentId)
    $reportDirectory = Get-GliomaReportDirectory -ExperimentId $ExperimentId
    $manifestPath = Join-Path $reportDirectory 'experiment.json'
    if (Test-Path -LiteralPath $manifestPath -PathType Leaf) {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        if ($null -ne $manifest.trainer -and $manifest.trainer -match '^nnUNetTrainer_(20|50)epochs$') {
            return [string]$manifest.trainer
        }
    }
    $benchmarkPath = Join-Path $reportDirectory 'benchmark_summary.json'
    if (-not (Test-Path -LiteralPath $benchmarkPath -PathType Leaf)) {
        throw "Neither a selected trainer nor benchmark summary exists for $ExperimentId."
    }
    $benchmark = Get-Content -LiteralPath $benchmarkPath -Raw | ConvertFrom-Json
    if ($null -eq $benchmark.recommended_preliminary_trainer) {
        throw 'The official five-epoch benchmark did not yield a safe preliminary trainer recommendation.'
    }
    return [string]$benchmark.recommended_preliminary_trainer
}
