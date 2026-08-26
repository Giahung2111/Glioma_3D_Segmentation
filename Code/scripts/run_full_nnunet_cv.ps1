param(
    [Parameter(Mandatory = $true)][switch]$ConfirmFullBaseline,
    [string]$ExperimentId = '',
    [switch]$Resume
)

. (Join-Path $PSScriptRoot '_common.ps1')
if (-not $ConfirmFullBaseline) {
    throw 'Full five-fold training is intentionally opt-in. Re-run with -ConfirmFullBaseline after reviewing compute/storage readiness.'
}
if ([string]::IsNullOrWhiteSpace($ExperimentId)) {
    $ExperimentId = New-GliomaExperimentId -Kind fullcv -Fold 0
}
Write-Host "FULL SCIENTIFIC BASELINE: $ExperimentId" -ForegroundColor Magenta
Write-Host 'Trainer: official default nnUNetTrainer; folds: 0,1,2,3,4; --npz enabled.'

$datasetName = 'Dataset501_BraTS2023GLI'
$modelRoot = Join-Path (Join-Path $Env:nnUNet_results $datasetName) 'nnUNetTrainer__nnUNetPlans__3d_fullres'
foreach ($fold in 0..4) {
    $foldDirectory = Join-Path $modelRoot "fold_$fold"
    $finalCheckpoint = Join-Path $foldDirectory 'checkpoint_final.pth'
    if (Test-Path -LiteralPath $finalCheckpoint -PathType Leaf) {
        Write-Host "Fold $fold already has checkpoint_final.pth; retaining it." -ForegroundColor Yellow
        continue
    }
    $arguments = @(
        '-m', 'glioma_seg.backends.nnunet.backend',
        '--project-root', $script:GliomaProjectRoot,
        'train',
        '--experiment-id', $ExperimentId,
        '--fold', [string]$fold,
        '--trainer', 'nnUNetTrainer',
        '--config', (Join-Path $script:GliomaCodeRoot 'configs\experiments\nnunet_full_cv.yaml')
    )
    $latestCheckpoint = Join-Path $foldDirectory 'checkpoint_latest.pth'
    if ($Resume -and (Test-Path -LiteralPath $latestCheckpoint -PathType Leaf)) {
        $arguments += '--continue'
    }
    Invoke-GliomaPython -Arguments $arguments
}

$crossvalOutput = Join-Path $modelRoot 'crossval_results_folds_0_1_2_3_4'
if (-not (Test-Path -LiteralPath $crossvalOutput -PathType Container) -or -not (Get-ChildItem -LiteralPath $crossvalOutput -Force -ErrorAction SilentlyContinue)) {
    Invoke-GliomaPython -Arguments @(
        '-m', 'glioma_seg.backends.nnunet.backend',
        '--project-root', $script:GliomaProjectRoot,
        'accumulate-crossval',
        '--experiment-id', $ExperimentId,
        '--output-dir', $crossvalOutput
    )
}
else {
    Write-Host "Accumulated CV output already exists and is non-empty; retaining it: $crossvalOutput" -ForegroundColor Yellow
}

Write-Host 'All five default-protocol folds and official CV accumulation are complete.' -ForegroundColor Green
Write-Host 'This script is never called by run_preliminary_pipeline.ps1.'
