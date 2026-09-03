param(
    [Parameter(Mandatory = $true)][string]$ExperimentId,
    [ValidateSet('nnUNetTrainer_100epochs')][string]$Trainer = 'nnUNetTrainer_100epochs',
    [Parameter(Mandatory = $true)][string]$AccumulatedPredictionDir,
    [switch]$Resume
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot '_common.ps1')
Write-GliomaStage -Number 'CV-EVAL' -Name 'Five-Fold Semantic and Official Evaluation'

$reportDirectory = Get-GliomaReportDirectory -ExperimentId $ExperimentId
$datasetName = 'Dataset501_BraTS2023GLI'
$rawDataset = Join-Path $Env:nnUNet_raw $datasetName
$preprocessedDataset = Join-Path $Env:nnUNet_preprocessed $datasetName
$modelDirectory = Join-Path (Join-Path $Env:nnUNet_results $datasetName) "$Trainer`__nnUNetPlans__3d_fullres"
$labelsDirectory = Join-Path $rawDataset 'labelsTr'
$datasetJson = Join-Path $rawDataset 'dataset.json'
$splitsJson = Join-Path $preprocessedDataset 'splits_final.json'

foreach ($requiredPath in @(
    $labelsDirectory,
    $datasetJson,
    $splitsJson,
    $modelDirectory,
    $AccumulatedPredictionDir
)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required cross-validation evaluation input is missing: $requiredPath"
    }
}

# Time one complete five-model prediction on an explicit, immutable ten-case
# subset. This timing is separate from out-of-fold scoring and always uses TTA off.
$inferenceCaseLimit = 10
$allCaseIds = @(
    Get-ChildItem -LiteralPath (Join-Path $rawDataset 'imagesTr') -File -Filter '*_0000.nii.gz' |
        ForEach-Object { $_.Name -replace '_0000\.nii\.gz$', '' } |
        Sort-Object
)
if ($allCaseIds.Count -ne 1251) {
    throw "Expected 1,251 converted training cases before timing, found $($allCaseIds.Count)."
}
$timingCases = @($allCaseIds | Select-Object -First $inferenceCaseLimit)
$timingInput = Join-Path (Join-Path $script:GliomaWorkspace 'cache') "$ExperimentId\cv_timing_input_n$inferenceCaseLimit"
New-Item -ItemType Directory -Path $timingInput -Force | Out-Null
$expectedTimingFiles = @()
foreach ($caseId in $timingCases) {
    foreach ($channel in @('0000', '0001', '0002', '0003')) {
        $fileName = "$caseId`_$channel.nii.gz"
        $expectedTimingFiles += $fileName
        $source = Join-Path (Join-Path $rawDataset 'imagesTr') $fileName
        $destination = Join-Path $timingInput $fileName
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            throw "Five-fold timing source is missing: $source"
        }
        if (Test-Path -LiteralPath $destination -PathType Leaf) {
            if ((Get-Item -LiteralPath $destination).Length -ne (Get-Item -LiteralPath $source).Length) {
                throw "Conflicting five-fold timing input exists: $destination"
            }
            continue
        }
        try {
            New-Item -ItemType HardLink -Path $destination -Target $source -ErrorAction Stop | Out-Null
        }
        catch {
            Copy-Item -LiteralPath $source -Destination $destination
        }
    }
}
$actualTimingFiles = @(
    Get-ChildItem -LiteralPath $timingInput -File -Filter '*.nii.gz' |
        ForEach-Object { $_.Name }
)
if (@(Compare-Object -ReferenceObject $expectedTimingFiles -DifferenceObject $actualTimingFiles).Count -gt 0) {
    throw "Five-fold timing input inventory is not exactly four modalities for ten cases: $timingInput"
}

$timingOutputBase = Join-Path (Join-Path $script:GliomaWorkspace 'predictions') "$ExperimentId\five_fold_tta_off_timing_n$inferenceCaseLimit"
$timingOutput = $timingOutputBase
$inferenceRuntime = Join-Path $reportDirectory 'inference_runtime.json'
$runFreshTiming = $true
if ((Test-Path -LiteralPath $timingOutput -PathType Container) -and
    @(Get-ChildItem -LiteralPath $timingOutput -Force).Count -gt 0) {
    if (-not $Resume) {
        throw "Five-fold timing output already exists. Use -Resume to verify/reuse it: $timingOutput"
    }
    & $script:GliomaPython @(
        '-m', 'glioma_seg.evaluation.inference_audit',
        '--input-dir', $timingInput,
        '--prediction-dir', $timingOutput,
        '--runtime-json', $inferenceRuntime
    )
    if ($LASTEXITCODE -eq 0) {
        $runFreshTiming = $false
        Write-Host 'Verified complete five-fold TTA-off inference timing retained.' -ForegroundColor Yellow
    }
    else {
        $retryStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
        $timingOutput = "${timingOutputBase}_fresh_$retryStamp"
        Write-Warning "Existing timing output is incomplete/non-comparable; it was retained and a fresh run will use $timingOutput"
    }
}
if ($runFreshTiming) {
    Invoke-GliomaPython -Arguments @(
        '-m', 'glioma_seg.backends.nnunet.backend',
        '--project-root', $script:GliomaProjectRoot,
        'predict',
        '--experiment-id', $ExperimentId,
        '--input-dir', $timingInput,
        '--output-dir', $timingOutput,
        '--fold', '0',
        '--fold', '1',
        '--fold', '2',
        '--fold', '3',
        '--fold', '4',
        '--trainer', $Trainer
    )
    Invoke-GliomaPython -Arguments @(
        '-m', 'glioma_seg.evaluation.inference_audit',
        '--input-dir', $timingInput,
        '--prediction-dir', $timingOutput,
        '--runtime-json', $inferenceRuntime,
        '--finalize-fresh-run'
    )
}
$timingRecord = Get-Content -LiteralPath $inferenceRuntime -Raw | ConvertFrom-Json
if ((@($timingRecord.folds) -join ',') -ne '0,1,2,3,4' -or
    [string]$timingRecord.tta_state -ne 'OFF' -or
    [int]$timingRecord.number_of_cases -ne $inferenceCaseLimit -or
    -not [bool]$timingRecord.timing_comparable) {
    throw "Five-fold inference runtime did not pass its fold/TTA/case/comparability audit: $inferenceRuntime"
}

$semanticOutputs = @(
    (Join-Path $reportDirectory 'metrics_per_case.csv'),
    (Join-Path $reportDirectory 'metrics_summary.csv'),
    (Join-Path $reportDirectory 'metrics_summary.json'),
    (Join-Path $reportDirectory 'evaluation_protocol.json'),
    (Join-Path $reportDirectory 'crossval_metrics_by_fold.csv'),
    (Join-Path $reportDirectory 'crossval_summary.json'),
    (Join-Path $reportDirectory 'crossval_integrity.json')
)
$presentSemanticOutputs = @(
    $semanticOutputs | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
)
$reuseSemanticOutputs = $presentSemanticOutputs.Count -eq $semanticOutputs.Count
if ($reuseSemanticOutputs) {
    if (-not $Resume) {
        throw 'Complete cross-validation metric artifacts already exist. Use -Resume to verify and retain them.'
    }
    $summary = Get-Content -LiteralPath (Join-Path $reportDirectory 'crossval_summary.json') -Raw | ConvertFrom-Json
    $integrity = Get-Content -LiteralPath (Join-Path $reportDirectory 'crossval_integrity.json') -Raw | ConvertFrom-Json
    $summaryCountProperty = $summary.PSObject.Properties['total_cases']
    $integrityValidProperty = $integrity.PSObject.Properties['valid']
    if ($null -eq $summaryCountProperty -or [int]$summaryCountProperty.Value -ne 1251 -or
        $null -eq $integrityValidProperty -or -not [bool]$integrityValidProperty.Value) {
        $reuseSemanticOutputs = $false
    }
}

if ($reuseSemanticOutputs) {
    Write-Host 'Verified complete 1,251-case semantic CV metrics retained.' -ForegroundColor Yellow
}
else {
    if ($presentSemanticOutputs.Count -gt 0 -and -not $Resume) {
        throw 'Partial cross-validation metric artifacts exist. Use -Resume to let the owned experiment repair them.'
    }
    $crossvalArguments = @(
        '-m', 'glioma_seg.evaluation.crossval',
        '--ground-truth-dir', $labelsDirectory,
        '--model-dir', $modelDirectory,
        '--accumulated-prediction-dir', $AccumulatedPredictionDir,
        '--splits-json', $splitsJson,
        '--dataset-json', $datasetJson,
        '--output-dir', $reportDirectory,
        '--expected-case-count', '1251',
        '--require-probabilities'
    )
    if ($Resume -and $presentSemanticOutputs.Count -gt 0) {
        $crossvalArguments += '--overwrite'
    }
    Invoke-GliomaPython -Arguments $crossvalArguments
}

foreach ($requiredOutput in $semanticOutputs) {
    if (-not (Test-Path -LiteralPath $requiredOutput -PathType Leaf)) {
        throw "Cross-validation evaluator did not create: $requiredOutput"
    }
}

$officialRoot = Join-Path $script:GliomaProjectRoot 'External\BraTS-2023-Metrics'
$officialPython = Join-Path $script:GliomaWorkspace 'cache\brats2023_metrics_env\python.exe'
$officialStatus = Join-Path $reportDirectory 'official_brats_metrics_status.json'
$officialSummary = Join-Path $reportDirectory 'official_lesionwise_metrics_summary.csv'
$officialSummaryJson = Join-Path $reportDirectory 'official_lesionwise_metrics_summary.json'
$officialPerCase = Join-Path $reportDirectory 'official_lesionwise_metrics_per_case.csv'
$officialOutputs = @($officialSummary, $officialSummaryJson, $officialPerCase)
$presentOfficialOutputs = @(
    $officialOutputs | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
)
$reuseOfficialOutputs = $false
if ($presentOfficialOutputs.Count -eq $officialOutputs.Count -and
    (Test-Path -LiteralPath $officialStatus -PathType Leaf)) {
    $status = Get-Content -LiteralPath $officialStatus -Raw | ConvertFrom-Json
    $reuseOfficialOutputs = (
        [bool]$status.available -and
        [int]$status.case_count -eq 1251 -and
        [string]$status.version_or_commit -eq '43c905242b2eecf421d4ab2da7af8ece9777d322'
    )
    if ($reuseOfficialOutputs -and -not $Resume) {
        throw 'Complete official CV metric artifacts already exist. Use -Resume to verify and retain them.'
    }
}
elseif ($presentOfficialOutputs.Count -gt 0) {
    if (-not $Resume) {
        throw 'Partial official lesion-wise artifacts exist; use -Resume to preserve and rebuild these derived outputs.'
    }
}

if ($reuseOfficialOutputs) {
    Write-Host 'Verified complete pinned official lesion-wise CV metrics retained.' -ForegroundColor Yellow
}
else {
    $staleOfficialArtifacts = @(
        $officialOutputs + @(
            $officialStatus,
            (Join-Path $reportDirectory 'official_brats_evaluator.log')
        ) | Where-Object { Test-Path -LiteralPath $_ }
    )
    if ($staleOfficialArtifacts.Count -gt 0) {
        if (-not $Resume) {
            throw 'Existing official derived artifacts are not a verified complete set. Use -Resume to preserve and rebuild them.'
        }
        $officialPreserveStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
        $officialPreserveDirectory = Join-Path (Join-Path $script:GliomaWorkspace 'cache') "$ExperimentId\official_incomplete_$officialPreserveStamp"
        New-Item -ItemType Directory -Path $officialPreserveDirectory -Force | Out-Null
        foreach ($staleOfficialArtifact in $staleOfficialArtifacts) {
            Move-Item -LiteralPath $staleOfficialArtifact -Destination (Join-Path $officialPreserveDirectory (Split-Path -Leaf $staleOfficialArtifact))
        }
        Write-Warning "Unverified official derived artifacts were preserved at $officialPreserveDirectory"
    }
}

if (-not $reuseOfficialOutputs -and
    (Test-Path -LiteralPath $officialRoot -PathType Container) -and
        (Test-Path -LiteralPath $officialPython -PathType Leaf)) {
    & $script:GliomaPython @(
        '-m', 'glioma_seg.evaluation.official_runner',
        '--ground-truth-dir', $labelsDirectory,
        '--prediction-dir', $AccumulatedPredictionDir,
        '--output-dir', $reportDirectory,
        '--official-root', $officialRoot,
        '--python', $officialPython
    )
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'Pinned official lesion-wise evaluation was unavailable. Semantic CV metrics remain valid; inspect official_brats_metrics_status.json and the official log.'
    }
}
elseif (-not $reuseOfficialOutputs) {
    Invoke-GliomaPython -Arguments @(
        '-m', 'glioma_seg.evaluation.brats2023_official',
        '--output-dir', $reportDirectory,
        '--unavailable-reason', 'The pinned official BraTS-2023-Metrics checkout or its isolated compatibility Python was unavailable; standard semantic 5-fold Dice/HD95 were computed separately.'
    )
}
if (-not (Test-Path -LiteralPath $officialStatus -PathType Leaf)) {
    throw 'Official metric availability status was not recorded.'
}
$verifiedOfficialStatus = Get-Content -LiteralPath $officialStatus -Raw | ConvertFrom-Json
if (-not [bool]$verifiedOfficialStatus.available -or
    [int]$verifiedOfficialStatus.case_count -ne 1251 -or
    [string]$verifiedOfficialStatus.version_or_commit -ne '43c905242b2eecf421d4ab2da7af8ece9777d322') {
    throw "The final five-fold report requires successful pinned official lesion-wise metrics for all 1,251 cases. See $officialStatus"
}
foreach ($officialOutput in $officialOutputs) {
    if (-not (Test-Path -LiteralPath $officialOutput -PathType Leaf)) {
        throw "Pinned official evaluation reported success but an output is missing: $officialOutput"
    }
}

$recordArguments = @(
    '-m', 'glioma_seg.backends.nnunet.backend',
    '--project-root', $script:GliomaProjectRoot,
    'record-artifacts',
    '--experiment-id', $ExperimentId,
    '--artifact', "metrics_summary=$(Join-Path $reportDirectory 'metrics_summary.csv')",
    '--artifact', "metrics_summary_json=$(Join-Path $reportDirectory 'metrics_summary.json')",
    '--artifact', "metrics_per_case=$(Join-Path $reportDirectory 'metrics_per_case.csv')",
    '--artifact', "evaluation_protocol=$(Join-Path $reportDirectory 'evaluation_protocol.json')",
    '--artifact', "crossval_metrics_by_fold=$(Join-Path $reportDirectory 'crossval_metrics_by_fold.csv')",
    '--artifact', "crossval_summary=$(Join-Path $reportDirectory 'crossval_summary.json')",
    '--artifact', "crossval_integrity=$(Join-Path $reportDirectory 'crossval_integrity.json')",
    '--artifact', "crossval_predictions=$AccumulatedPredictionDir",
    '--artifact', "timing_predictions_tta_off=$timingOutput",
    '--artifact', "inference_runtime=$inferenceRuntime",
    '--artifact', "official_status=$officialStatus"
)
if ($officialOutputs.Count -eq @(
        $officialOutputs | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
    ).Count) {
    $recordArguments += @(
        '--artifact', "official_metrics_summary=$officialSummary",
        '--artifact', "official_metrics_summary_json=$officialSummaryJson",
        '--artifact', "official_metrics_per_case=$officialPerCase"
    )
}
Invoke-GliomaPython -Arguments $recordArguments
Write-Host "Five-fold evaluation passed: $reportDirectory" -ForegroundColor Green
