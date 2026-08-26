param(
    [string]$ExperimentId = '',
    [ValidateSet('', 'nnUNetTrainer_20epochs', 'nnUNetTrainer_50epochs')][string]$Trainer = '',
    [switch]$Resume,
    [switch]$AllowLowGpuUtilization
)

. (Join-Path $PSScriptRoot '_common.ps1')
Write-GliomaStage -Number '6/8' -Name 'Preliminary Fold-0 Training'

if ([string]::IsNullOrWhiteSpace($ExperimentId)) {
    throw 'Pass the experiment ID produced before the benchmark; training never invents a disconnected result ID.'
}
$reportDirectory = Get-GliomaReportDirectory -ExperimentId $ExperimentId
$benchmarkPath = Join-Path $reportDirectory 'benchmark_summary.json'
$gpuPath = Join-Path $reportDirectory 'benchmark_gpu_summary.json'
if (-not (Test-Path -LiteralPath $benchmarkPath -PathType Leaf)) {
    throw "Official five-epoch benchmark summary missing: $benchmarkPath"
}
$benchmark = Get-Content -LiteralPath $benchmarkPath -Raw | ConvertFrom-Json
$recommendedTrainer = [string]$benchmark.recommended_preliminary_trainer
if ([string]::IsNullOrWhiteSpace($Trainer)) {
    $Trainer = $recommendedTrainer
}
if ($Trainer -notmatch '^nnUNetTrainer_(20|50)epochs$') {
    throw 'Benchmark did not produce a valid official 20/50-epoch trainer recommendation.'
}
if ($Trainer -ne $recommendedTrainer) {
    throw "Requested trainer $Trainer conflicts with the benchmark selection $recommendedTrainer. A standard preliminary baseline must follow the recorded selection rule."
}
if ($benchmark.experiment_id -ne $ExperimentId -or $benchmark.dataset_id -ne 501 -or $benchmark.configuration -ne '3d_fullres' -or $benchmark.fold -ne 0) {
    throw "Benchmark metadata does not match experiment $ExperimentId / dataset 501 / 3d_fullres / Fold 0."
}
if ($null -eq $benchmark.fastest_epoch_seconds -or [double]$benchmark.fastest_epoch_seconds -le 0 -or $null -eq $benchmark.linear_runtime_estimates_seconds.'50' -or [double]$benchmark.linear_runtime_estimates_seconds.'50' -le 0) {
    throw 'Benchmark timing is incomplete; diagnose the benchmark before long training.'
}
if (-not (Test-Path -LiteralPath $gpuPath -PathType Leaf)) {
    throw "Benchmark GPU telemetry summary missing: $gpuPath"
}
$gpu = Get-Content -LiteralPath $gpuPath -Raw | ConvertFrom-Json
if ([int]$gpu.samples -lt 1) { throw 'Benchmark recorded no GPU telemetry samples.' }
if ($null -eq $gpu.dedicated_memory_total_mb -or [double]$gpu.dedicated_memory_total_mb -lt 10000.0) {
    throw "Benchmark did not record the expected dedicated GPU memory: $($gpu.dedicated_memory_total_mb) MiB."
}
if (-not $AllowLowGpuUtilization -and [double]$gpu.mean_gpu_utilization_percent -lt 20.0) {
    throw "Mean benchmark GPU utilization was only $($gpu.mean_gpu_utilization_percent)%. Diagnose the data-loader/GPU path, or explicitly pass -AllowLowGpuUtilization after review."
}

$modelRoot = Join-Path (Join-Path $Env:nnUNet_results 'Dataset501_BraTS2023GLI') "$Trainer`__nnUNetPlans__3d_fullres"
$foldDirectory = Join-Path $modelRoot 'fold_0'
$finalCheckpoint = Join-Path $foldDirectory 'checkpoint_final.pth'
$validationSummary = Join-Path $foldDirectory 'validation\summary.json'
if ($Resume -and (Test-Path -LiteralPath $finalCheckpoint -PathType Leaf) -and (Test-Path -LiteralPath $validationSummary -PathType Leaf)) {
    $ownerPath = Join-Path $foldDirectory 'glioma_experiment_owner.json'
    if (-not (Test-Path -LiteralPath $ownerPath -PathType Leaf)) {
        throw "Completed trainer output has no project ownership manifest; refusing stale reuse: $ownerPath"
    }
    $owner = Get-Content -LiteralPath $ownerPath -Raw | ConvertFrom-Json
    if ($owner.experiment_id -ne $ExperimentId -or $owner.dataset_id -ne 501 -or $owner.configuration -ne '3d_fullres' -or $owner.fold -ne 0 -or $owner.trainer -ne $Trainer) {
        throw "Completed trainer output belongs to another experiment or configuration: $ownerPath"
    }
    Write-Host "Completed training and validation artifacts already exist; retaining them: $foldDirectory" -ForegroundColor Yellow
    return
}

Write-Host 'Running the required pre-training unit-test gate...' -ForegroundColor Cyan
Invoke-GliomaPython -Arguments @(
    '-m', 'pytest',
    (Join-Path $script:GliomaCodeRoot 'tests'),
    '-q'
)

$arguments = @(
    '-m', 'glioma_seg.backends.nnunet.backend',
    '--project-root', $script:GliomaProjectRoot,
    'train',
    '--experiment-id', $ExperimentId,
    '--fold', '0',
    '--trainer', $Trainer,
    '--config', (Join-Path $script:GliomaCodeRoot 'configs\experiments\nnunet_preliminary_fold0.yaml')
)
if ($Resume) { $arguments += '--continue' }
if ($AllowLowGpuUtilization) { $arguments += '--allow-low-gpu-utilization' }
Invoke-GliomaPython -Arguments $arguments
Write-Host "Preliminary Fold-0 training completed with $Trainer." -ForegroundColor Green
