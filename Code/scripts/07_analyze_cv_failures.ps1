param(
    [Parameter(Mandatory = $true)][string]$ExperimentId,
    [Parameter(Mandatory = $true)][string]$AccumulatedPredictionDir,
    [ValidateRange(1, 100)][int]$TopN = 5,
    [ValidateRange(1, 100)][int]$MaxCases = 15,
    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot '_common.ps1')
Write-GliomaStage -Number 'CV-ANALYSIS' -Name 'Five-Fold Failure Analysis and Visualizations'

$reportDirectory = Get-GliomaReportDirectory -ExperimentId $ExperimentId
$datasetName = 'Dataset501_BraTS2023GLI'
$rawDataset = Join-Path $Env:nnUNet_raw $datasetName
$labelsDirectory = Join-Path $rawDataset 'labelsTr'
$metricsPerCase = Join-Path $reportDirectory 'metrics_per_case.csv'
$validationManifest = Join-Path $reportDirectory 'data_validation.json'
if (-not (Test-Path -LiteralPath $validationManifest -PathType Leaf)) {
    $validationManifest = Join-Path (Join-Path $script:GliomaWorkspace 'reports') 'data_validation.json'
}
if (-not (Test-Path -LiteralPath $validationManifest -PathType Leaf)) {
    throw 'Training validation manifest is missing.'
}
$validationReport = Get-Content -LiteralPath $validationManifest -Raw | ConvertFrom-Json
$rawTrainingDirectory = [string]$validationReport.dataset_root
if ([string]::IsNullOrWhiteSpace($rawTrainingDirectory) -or
    -not (Test-Path -LiteralPath $rawTrainingDirectory -PathType Container)) {
    throw "Validated raw training directory is unavailable: $rawTrainingDirectory"
}
foreach ($requiredPath in @($labelsDirectory, $AccumulatedPredictionDir, $metricsPerCase)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required cross-validation failure-analysis artifact is missing: $requiredPath"
    }
}

$failureRankings = Join-Path $reportDirectory 'failure_rankings.csv'
$failureCases = Join-Path $reportDirectory 'failure_cases.csv'
$figuresDirectory = Join-Path $reportDirectory 'figures'
$figuresManifest = Join-Path $figuresDirectory 'figures_manifest.csv'
$requiredOutputs = @($failureRankings, $failureCases, $figuresManifest)
$presentOutputs = @($requiredOutputs | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
$reuseOutputs = $presentOutputs.Count -eq $requiredOutputs.Count
if ($reuseOutputs) {
    $figureRows = @(Import-Csv -LiteralPath $figuresManifest)
    $orientationRows = @(
        $figureRows | Where-Object {
            [string]$_.orientation_convention -ne 'RAS canonical axial; anterior/face up; neurological L/R'
        }
    )
    $invalidFigurePathRows = @()
    foreach ($figureRow in $figureRows) {
        $recordedFigurePath = [string]$figureRow.figure_path
        if ([string]::IsNullOrWhiteSpace($recordedFigurePath) -or
            [IO.Path]::IsPathRooted($recordedFigurePath) -or
            -not (Test-Path -LiteralPath (Join-Path $figuresDirectory $recordedFigurePath) -PathType Leaf)) {
            $invalidFigurePathRows += $figureRow
        }
    }
    if ($figureRows.Count -lt 1 -or
        $orientationRows.Count -gt 0 -or
        $invalidFigurePathRows.Count -gt 0) {
        $reuseOutputs = $false
    }
}
if ($reuseOutputs -and -not $Resume) {
    throw 'Complete CV failure-analysis artifacts already exist. Use -Resume to verify and retain them.'
}
if ($presentOutputs.Count -gt 0 -and -not $reuseOutputs -and -not $Resume) {
    throw 'Partial CV failure-analysis artifacts exist. Use -Resume to repair the owned derived outputs.'
}

if ($reuseOutputs) {
    Write-Host 'Complete CV failure rankings and figures retained.' -ForegroundColor Yellow
}
else {
    $stageStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $stagingDirectory = Join-Path (Join-Path $script:GliomaWorkspace 'cache') "$ExperimentId\failure_analysis_staging_$stageStamp"
    $stagingFigures = Join-Path $stagingDirectory 'figures'
    New-Item -ItemType Directory -Path $stagingDirectory -Force | Out-Null
    Invoke-GliomaPython -Arguments @(
        '-m', 'glioma_seg.analysis.failure_analysis',
        '--ground-truth-dir', $labelsDirectory,
        '--prediction-dir', $AccumulatedPredictionDir,
        '--metrics-per-case-csv', $metricsPerCase,
        '--output-dir', $stagingDirectory,
        '--top-n', [string]$TopN,
        '--max-cases', [string]$MaxCases
    )

    $stagingFailureCases = Join-Path $stagingDirectory 'failure_cases.csv'
    $stagingFailureRankings = Join-Path $stagingDirectory 'failure_rankings.csv'
    Invoke-GliomaPython -Arguments @(
        '-m', 'glioma_seg.visualization.overlays',
        '--raw-training-dir', $rawTrainingDirectory,
        '--ground-truth-dir', $labelsDirectory,
        '--prediction-dir', $AccumulatedPredictionDir,
        '--failure-cases-csv', $stagingFailureCases,
        '--metrics-per-case-csv', $metricsPerCase,
        '--output-dir', $stagingFigures,
        '--max-cases', [string]$MaxCases
    )

    $stagingFiguresManifest = Join-Path $stagingFigures 'figures_manifest.csv'
    foreach ($stagingOutput in @(
        $stagingFailureRankings,
        $stagingFailureCases,
        $stagingFiguresManifest
    )) {
        if (-not (Test-Path -LiteralPath $stagingOutput -PathType Leaf)) {
            throw "Staged CV failure analysis did not create: $stagingOutput"
        }
    }
    $stagedFigureRows = @(Import-Csv -LiteralPath $stagingFiguresManifest)
    $invalidOrientationRows = @(
        $stagedFigureRows | Where-Object {
            [string]$_.orientation_convention -ne 'RAS canonical axial; anterior/face up; neurological L/R'
        }
    )
    $invalidStagedFigurePathRows = @()
    foreach ($stagedFigureRow in $stagedFigureRows) {
        $recordedStagedFigurePath = [string]$stagedFigureRow.figure_path
        if ([string]::IsNullOrWhiteSpace($recordedStagedFigurePath) -or
            [IO.Path]::IsPathRooted($recordedStagedFigurePath) -or
            -not (Test-Path -LiteralPath (Join-Path $stagingFigures $recordedStagedFigurePath) -PathType Leaf)) {
            $invalidStagedFigurePathRows += $stagedFigureRow
        }
    }
    if ($stagedFigureRows.Count -lt 1 -or
        $invalidOrientationRows.Count -gt 0 -or
        $invalidStagedFigurePathRows.Count -gt 0) {
        throw 'Staged figures did not prove portable paths and RAS canonical axial, anterior/face-up orientation.'
    }

    $preserveStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $preservedDirectory = Join-Path (Join-Path $script:GliomaWorkspace 'cache') "$ExperimentId\derived_incomplete_$preserveStamp"
    New-Item -ItemType Directory -Path $preservedDirectory -Force | Out-Null
    foreach ($derivedFile in @($failureRankings, $failureCases)) {
        if (Test-Path -LiteralPath $derivedFile) {
            Move-Item -LiteralPath $derivedFile -Destination (Join-Path $preservedDirectory (Split-Path -Leaf $derivedFile))
        }
    }
    if (Test-Path -LiteralPath $figuresDirectory) {
        Move-Item -LiteralPath $figuresDirectory -Destination (Join-Path $preservedDirectory 'figures')
    }
    Move-Item -LiteralPath $stagingFailureRankings -Destination $failureRankings
    Move-Item -LiteralPath $stagingFailureCases -Destination $failureCases
    Move-Item -LiteralPath $stagingFigures -Destination $figuresDirectory
}

foreach ($requiredOutput in $requiredOutputs) {
    if (-not (Test-Path -LiteralPath $requiredOutput -PathType Leaf)) {
        throw "Cross-validation failure analysis did not create: $requiredOutput"
    }
}
$finalFigureRows = @(Import-Csv -LiteralPath $figuresManifest)
$invalidFinalFigureRows = @()
foreach ($finalFigureRow in $finalFigureRows) {
    $recordedFinalFigurePath = [string]$finalFigureRow.figure_path
    if ([string]::IsNullOrWhiteSpace($recordedFinalFigurePath) -or
        [IO.Path]::IsPathRooted($recordedFinalFigurePath) -or
        -not (Test-Path -LiteralPath (Join-Path $figuresDirectory $recordedFinalFigurePath) -PathType Leaf)) {
        $invalidFinalFigureRows += $finalFigureRow
    }
}
if ($finalFigureRows.Count -lt 1 -or $invalidFinalFigureRows.Count -gt 0) {
    throw 'Final figure manifest is not portable or references missing figure files.'
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
Write-Host "Five-fold failure analysis passed: $reportDirectory" -ForegroundColor Green
