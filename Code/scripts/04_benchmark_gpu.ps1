param(
    [string]$ExperimentId = '',
    [switch]$Resume
)

. (Join-Path $PSScriptRoot '_common.ps1')
Write-GliomaStage -Number '5/8' -Name 'Official Five-Epoch GPU Benchmark'

if ([string]::IsNullOrWhiteSpace($ExperimentId)) {
    $ExperimentId = New-GliomaExperimentId -Kind benchmark
}
$reportDirectory = Get-GliomaReportDirectory -ExperimentId $ExperimentId
$summaryPath = Join-Path $reportDirectory 'benchmark_summary.json'
if (Test-Path -LiteralPath $summaryPath -PathType Leaf) {
    if (-not $Resume) {
        throw "Benchmark summary already exists for $ExperimentId. Use -Resume to retain it, or use a new experiment ID; existing benchmark evidence is never overwritten implicitly."
    }
    $gpuSummaryPath = Join-Path $reportDirectory 'benchmark_gpu_summary.json'
    if (-not (Test-Path -LiteralPath $gpuSummaryPath -PathType Leaf)) {
        throw "Benchmark summary exists but GPU telemetry summary is missing: $gpuSummaryPath"
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    if ($summary.experiment_id -ne $ExperimentId -or $summary.dataset_id -ne 501 -or $summary.configuration -ne '3d_fullres' -or $summary.fold -ne 0) {
        throw "Existing benchmark metadata does not match this experiment/dataset/configuration/fold: $summaryPath"
    }
    if ([string]$summary.recommended_preliminary_trainer -notmatch '^nnUNetTrainer_(20|50)epochs$') {
        throw "Existing benchmark did not produce a safe preliminary trainer recommendation: $summaryPath"
    }
    Write-Host "Existing completed benchmark retained: $summaryPath" -ForegroundColor Yellow
    return
}

Invoke-GliomaPython -Arguments @(
    '-m', 'glioma_seg.backends.nnunet.backend',
    '--project-root', $script:GliomaProjectRoot,
    'benchmark',
    '--experiment-id', $ExperimentId
)
if (-not (Test-Path -LiteralPath $summaryPath -PathType Leaf)) {
    throw "Benchmark command returned without creating its summary: $summaryPath"
}
Write-Host "Benchmark passed: $summaryPath" -ForegroundColor Green
