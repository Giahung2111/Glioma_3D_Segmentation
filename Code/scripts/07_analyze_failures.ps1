param(
    [string]$ExperimentId = '',
    [ValidateSet('', 'nnUNetTrainer_20epochs', 'nnUNetTrainer_50epochs')][string]$Trainer = '',
    [ValidateRange(1, 100)][int]$TopN = 5,
    [ValidateRange(1, 100)][int]$MaxCases = 15
)

. (Join-Path $PSScriptRoot '_common.ps1')
Write-GliomaStage -Number '7/8' -Name 'Failure Analysis & Visualizations'

if ([string]::IsNullOrWhiteSpace($ExperimentId)) { throw 'ExperimentId is required.' }
if ([string]::IsNullOrWhiteSpace($Trainer)) { $Trainer = Get-GliomaTrainer -ExperimentId $ExperimentId }
$reportDirectory = Get-GliomaReportDirectory -ExperimentId $ExperimentId
$datasetName = 'Dataset501_BraTS2023GLI'
$rawDataset = Join-Path $Env:nnUNet_raw $datasetName
$labelsDirectory = Join-Path $rawDataset 'labelsTr'
$validationPredictions = Join-Path (Join-Path (Join-Path $Env:nnUNet_results $datasetName) "$Trainer`__nnUNetPlans__3d_fullres") 'fold_0\validation'
$metricsPerCase = Join-Path $reportDirectory 'metrics_per_case.csv'
$validationManifest = Join-Path $reportDirectory 'data_validation.json'
if (-not (Test-Path -LiteralPath $validationManifest -PathType Leaf)) {
    $validationManifest = Join-Path (Join-Path $script:GliomaWorkspace 'reports') 'data_validation.json'
}
if (-not (Test-Path -LiteralPath $validationManifest -PathType Leaf)) { throw 'Training validation manifest is missing.' }
$validationReport = Get-Content -LiteralPath $validationManifest -Raw | ConvertFrom-Json
$rawTrainingDirectory = [string]$validationReport.dataset_root
if ([string]::IsNullOrWhiteSpace($rawTrainingDirectory) -or
    -not (Test-Path -LiteralPath $rawTrainingDirectory -PathType Container)) {
    throw "Validated raw training directory is unavailable: $rawTrainingDirectory"
}
foreach ($requiredPath in @($labelsDirectory, $validationPredictions, $metricsPerCase)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required failure-analysis artifact missing: $requiredPath"
    }
}

Invoke-GliomaPython -Arguments @(
    '-m', 'glioma_seg.analysis.failure_analysis',
    '--ground-truth-dir', $labelsDirectory,
    '--prediction-dir', $validationPredictions,
    '--metrics-per-case-csv', $metricsPerCase,
    '--output-dir', $reportDirectory,
    '--top-n', [string]$TopN,
    '--max-cases', [string]$MaxCases
)

Invoke-GliomaPython -Arguments @(
    '-m', 'glioma_seg.visualization.overlays',
    '--raw-training-dir', $rawTrainingDirectory,
    '--ground-truth-dir', $labelsDirectory,
    '--prediction-dir', $validationPredictions,
    '--failure-cases-csv', (Join-Path $reportDirectory 'failure_cases.csv'),
    '--metrics-per-case-csv', $metricsPerCase,
    '--output-dir', (Join-Path $reportDirectory 'figures'),
    '--max-cases', [string]$MaxCases
)
$failureRankings = Join-Path $reportDirectory 'failure_rankings.csv'
$failureCases = Join-Path $reportDirectory 'failure_cases.csv'
$figuresDirectory = Join-Path $reportDirectory 'figures'
$figuresManifest = Join-Path $figuresDirectory 'figures_manifest.csv'
foreach ($requiredOutput in @($failureRankings, $failureCases, $figuresManifest)) {
    if (-not (Test-Path -LiteralPath $requiredOutput -PathType Leaf)) {
        throw "Failure-analysis output missing: $requiredOutput"
    }
}
Invoke-GliomaPython -Arguments @(
    '-m', 'glioma_seg.backends.nnunet.backend',
    '--project-root', $script:GliomaProjectRoot,
    'record-artifacts',
    '--experiment-id', $ExperimentId,
    '--artifact', "failure_rankings=$failureRankings",
    '--artifact', "failure_cases=$failureCases",
    '--artifact', "figures=$figuresDirectory",
    '--artifact', "figures_manifest=$figuresManifest"
)
Write-Host "Failure analysis passed: $reportDirectory" -ForegroundColor Green
