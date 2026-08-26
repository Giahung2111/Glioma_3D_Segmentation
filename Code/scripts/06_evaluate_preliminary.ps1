param(
    [string]$ExperimentId = '',
    [ValidateSet('', 'nnUNetTrainer_20epochs', 'nnUNetTrainer_50epochs')][string]$Trainer = '',
    [ValidateRange(1, 1000)][int]$InferenceCaseLimit = 10,
    [switch]$ResumeInference
)

. (Join-Path $PSScriptRoot '_common.ps1')
Write-GliomaStage -Number '7/8' -Name 'Fold-0 Inference & Evaluation'

if ([string]::IsNullOrWhiteSpace($ExperimentId)) { throw 'ExperimentId is required.' }
if ([string]::IsNullOrWhiteSpace($Trainer)) { $Trainer = Get-GliomaTrainer -ExperimentId $ExperimentId }
$reportDirectory = Get-GliomaReportDirectory -ExperimentId $ExperimentId
$datasetName = 'Dataset501_BraTS2023GLI'
$rawDataset = Join-Path $Env:nnUNet_raw $datasetName
$preprocessedDataset = Join-Path $Env:nnUNet_preprocessed $datasetName
$splitsPath = Join-Path $preprocessedDataset 'splits_final.json'
$datasetJsonPath = Join-Path $rawDataset 'dataset.json'
$labelsDirectory = Join-Path $rawDataset 'labelsTr'
$modelDirectory = Join-Path (Join-Path $Env:nnUNet_results $datasetName) "$Trainer`__nnUNetPlans__3d_fullres"
$validationPredictions = Join-Path (Join-Path $modelDirectory 'fold_0') 'validation'

foreach ($requiredPath in @($splitsPath, $datasetJsonPath, $labelsDirectory, $validationPredictions)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) { throw "Required evaluation artifact missing: $requiredPath" }
}

# Re-check the output reconstruction contract immediately before scoring. For
# nnU-Net region training, dictionary order WT,TC,ET plus class order 2,1,3 is
# what reconstructs raw BraTS 2023 labels (ED=2, NCR=1, ET=3).
$datasetJson = Get-Content -LiteralPath $datasetJsonPath -Raw | ConvertFrom-Json
$regionNames = @($datasetJson.labels.PSObject.Properties.Name)
if (($regionNames -join ',') -ne 'background,whole_tumor,tumor_core,enhancing_tumor') {
    throw "Unsafe dataset.json region order: $($regionNames -join ',')"
}
if ((@($datasetJson.labels.whole_tumor) -join ',') -ne '1,2,3' -or
    (@($datasetJson.labels.tumor_core) -join ',') -ne '1,3' -or
    [int]$datasetJson.labels.enhancing_tumor -ne 3 -or
    (@($datasetJson.regions_class_order) -join ',') -ne '2,1,3') {
    throw 'dataset.json does not implement BraTS 2023 WT/TC/ET reconstruction to labels 2/1/3.'
}

# Create a source-preserving timing subset from Fold 0 only. Hardlinks are tried
# first; verified source files are copied only when hardlinks are unavailable.
$splits = Get-Content -LiteralPath $splitsPath -Raw | ConvertFrom-Json
$validationCases = @($splits[0].val | ForEach-Object { [string]$_ })
if ($validationCases.Count -ne @($validationCases | Sort-Object -Unique).Count) {
    throw 'Fold 0 validation split contains duplicate case IDs.'
}
$splitOverlap = @($splits[0].train | Where-Object { $validationCases -contains [string]$_ })
if ($splitOverlap.Count -gt 0) { throw "Fold 0 train/validation leakage detected: $($splitOverlap[0])" }
$timingCases = @($validationCases | Sort-Object | Select-Object -First $InferenceCaseLimit)
if ($timingCases.Count -lt 1) { throw 'Fold 0 contains no validation cases.' }
$timingInput = Join-Path (Join-Path $script:GliomaWorkspace 'cache') "$ExperimentId\fold0_timing_input_n$($timingCases.Count)"
New-Item -ItemType Directory -Path $timingInput -Force | Out-Null
$expectedTimingFiles = @()
foreach ($caseId in $timingCases) {
    foreach ($channel in '0000', '0001', '0002', '0003') {
        $fileName = "$caseId`_$channel.nii.gz"
        $expectedTimingFiles += $fileName
        $source = Join-Path (Join-Path $rawDataset 'imagesTr') $fileName
        $destination = Join-Path $timingInput $fileName
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Timing source missing: $source" }
        if (Test-Path -LiteralPath $destination -PathType Leaf) {
            if ((Get-Item -LiteralPath $destination).Length -ne (Get-Item -LiteralPath $source).Length) {
                throw "Conflicting timing input exists: $destination"
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
$actualTimingFiles = @(Get-ChildItem -LiteralPath $timingInput -File -Filter '*.nii.gz' | ForEach-Object { $_.Name })
$timingInventoryDifference = @(Compare-Object -ReferenceObject $expectedTimingFiles -DifferenceObject $actualTimingFiles)
if ($timingInventoryDifference.Count -gt 0) {
    throw "Timing input inventory is not exactly four modalities for the selected Fold-0 cases: $timingInput"
}

$timingOutputBase = Join-Path (Join-Path $script:GliomaWorkspace 'predictions') "$ExperimentId\fold0_tta_off_timing_n$($timingCases.Count)"
$timingOutput = $timingOutputBase
$inferenceRuntime = Join-Path $reportDirectory 'inference_runtime.json'
$runFreshTiming = $true
if ((Test-Path -LiteralPath $timingOutput -PathType Container) -and
    @(Get-ChildItem -LiteralPath $timingOutput -Force).Count -gt 0) {
    if (-not $ResumeInference) {
        throw "Timing output already exists. Use -ResumeInference to verify/reuse it: $timingOutput"
    }
    & $script:GliomaPython @(
        '-m', 'glioma_seg.evaluation.inference_audit',
        '--input-dir', $timingInput,
        '--prediction-dir', $timingOutput,
        '--runtime-json', $inferenceRuntime
    )
    if ($LASTEXITCODE -eq 0) {
        $runFreshTiming = $false
        Write-Host 'Reusing a verified fresh, complete TTA-off timing run.' -ForegroundColor Yellow
    }
    else {
        # A resumed partial run has a biased timing denominator. Preserve it,
        # then make a new complete timing run in a unique directory.
        $retryStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
        $timingOutput = "${timingOutputBase}_fresh_$retryStamp"
        Write-Warning "Existing timing run is incomplete/non-comparable; using fresh output $timingOutput"
    }
}
if ($runFreshTiming) {
    $predictArguments = @(
        '-m', 'glioma_seg.backends.nnunet.backend',
        '--project-root', $script:GliomaProjectRoot,
        'predict',
        '--experiment-id', $ExperimentId,
        '--input-dir', $timingInput,
        '--output-dir', $timingOutput,
        '--fold', '0',
        '--trainer', $Trainer
    )
    # NNUNetV2Backend defaults to disable_tta=True and build_predict emits
    # the official --disable_tta flag. The audit below refuses any other state.
    Invoke-GliomaPython -Arguments $predictArguments
    Invoke-GliomaPython -Arguments @(
        '-m', 'glioma_seg.evaluation.inference_audit',
        '--input-dir', $timingInput,
        '--prediction-dir', $timingOutput,
        '--runtime-json', $inferenceRuntime,
        '--finalize-fresh-run'
    )
}

$semanticMetricArtifacts = @(
    (Join-Path $reportDirectory 'metrics_summary.csv'),
    (Join-Path $reportDirectory 'metrics_summary.json'),
    (Join-Path $reportDirectory 'metrics_per_case.csv'),
    (Join-Path $reportDirectory 'evaluation_protocol.json')
)
$presentSemanticMetricArtifacts = @($semanticMetricArtifacts | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
$reuseSemanticMetrics = $false
if ($presentSemanticMetricArtifacts.Count -gt 0) {
    if (-not $ResumeInference) {
        throw 'Semantic metric artifacts already exist. Use -ResumeInference to verify/reuse or repair them.'
    }
    if ($presentSemanticMetricArtifacts.Count -eq $semanticMetricArtifacts.Count) {
        $existingProtocol = Get-Content -LiteralPath (Join-Path $reportDirectory 'evaluation_protocol.json') -Raw | ConvertFrom-Json
        $existingRows = @(Import-Csv -LiteralPath (Join-Path $reportDirectory 'metrics_per_case.csv'))
        $existingSummaryRows = @(Import-Csv -LiteralPath (Join-Path $reportDirectory 'metrics_summary.csv'))
        $recordedPredictionDirectory = [IO.Path]::GetFullPath([string]$existingProtocol.prediction_dir)
        if ([int]$existingProtocol.case_count -eq $validationCases.Count -and
            [string]$existingProtocol.prediction_tta_state -eq 'DEFAULT_MIRRORING' -and
            $recordedPredictionDirectory -eq [IO.Path]::GetFullPath($validationPredictions) -and
            $existingRows.Count -eq $validationCases.Count -and
            $existingSummaryRows.Count -eq 2 -and
            (@($existingSummaryRows.metric) -join ',') -eq 'Dice,HD95') {
            $reuseSemanticMetrics = $true
            Write-Host 'Reusing verified complete Fold-0 semantic metric artifacts.' -ForegroundColor Yellow
        }
    }
}
if (-not $reuseSemanticMetrics) {
    Invoke-GliomaPython -Arguments @(
        '-m', 'glioma_seg.evaluation.evaluate',
        '--ground-truth-dir', $labelsDirectory,
        '--prediction-dir', $validationPredictions,
        '--output-dir', $reportDirectory,
        '--splits-json', $splitsPath,
        '--fold', '0',
        '--prediction-provenance', 'nnU-Net perform_actual_validation output (fold_0/validation)',
        '--prediction-tta-state', 'DEFAULT_MIRRORING'
    )
}

# Attempt the separately pinned, unmodified official evaluator. It is optional:
# a failure leaves an explicit status artifact and never fabricates official CSVs.
$officialRoot = Join-Path $script:GliomaProjectRoot 'External\BraTS-2023-Metrics'
$officialPython = Join-Path $script:GliomaWorkspace 'cache\brats2023_metrics_env\python.exe'
$officialStatus = Join-Path $reportDirectory 'official_brats_metrics_status.json'
$officialSummary = Join-Path $reportDirectory 'official_lesionwise_metrics_summary.csv'
$officialSummaryJson = Join-Path $reportDirectory 'official_lesionwise_metrics_summary.json'
$officialPerCase = Join-Path $reportDirectory 'official_lesionwise_metrics_per_case.csv'
$officialMetricArtifacts = @($officialSummary, $officialSummaryJson, $officialPerCase)
$presentOfficialMetricArtifacts = @($officialMetricArtifacts | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf })
$reuseOfficialMetrics = $false
if ($presentOfficialMetricArtifacts.Count -gt 0) {
    if ($presentOfficialMetricArtifacts.Count -ne $officialMetricArtifacts.Count -or
        -not (Test-Path -LiteralPath $officialStatus -PathType Leaf)) {
        throw 'Official metric artifacts are incomplete; refusing to mix partial or stale outputs.'
    }
    $existingOfficialStatus = Get-Content -LiteralPath $officialStatus -Raw | ConvertFrom-Json
    if (-not [bool]$existingOfficialStatus.available -or
        [string]$existingOfficialStatus.version_or_commit -ne '43c905242b2eecf421d4ab2da7af8ece9777d322' -or
        [int]$existingOfficialStatus.case_count -ne $validationCases.Count) {
        throw 'Existing official metric artifacts do not match the pinned evaluator and current Fold-0 case count.'
    }
    $reuseOfficialMetrics = $true
    Write-Host 'Reusing complete pinned official lesion-wise metric artifacts.' -ForegroundColor Yellow
}
if (-not $reuseOfficialMetrics -and
    (Test-Path -LiteralPath $officialRoot -PathType Container) -and
    (Test-Path -LiteralPath $officialPython -PathType Leaf)) {
    & $script:GliomaPython @(
        '-m', 'glioma_seg.evaluation.official_runner',
        '--ground-truth-dir', $labelsDirectory,
        '--prediction-dir', $validationPredictions,
        '--output-dir', $reportDirectory,
        '--official-root', $officialRoot,
        '--python', $officialPython,
        '--splits-json', $splitsPath,
        '--fold', '0'
    )
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'Pinned official lesion-wise evaluation was unavailable; standard semantic metrics remain valid. See the official status/log artifacts.'
    }
}
elseif (-not $reuseOfficialMetrics) {
    Invoke-GliomaPython -Arguments @(
        '-m', 'glioma_seg.evaluation.brats2023_official',
        '--output-dir', $reportDirectory,
        '--unavailable-reason', 'The pinned official BraTS-2023-Metrics checkout or its isolated compatibility Python was not available; standard semantic Dice/HD95 were computed separately.'
    )
}
if (-not (Test-Path -LiteralPath $officialStatus -PathType Leaf)) {
    throw 'Official metric availability status was not recorded.'
}
foreach ($requiredOutput in @(
    (Join-Path $reportDirectory 'metrics_summary.csv'),
    (Join-Path $reportDirectory 'metrics_summary.json'),
    (Join-Path $reportDirectory 'metrics_per_case.csv'),
    (Join-Path $reportDirectory 'evaluation_protocol.json'),
    $inferenceRuntime
)) {
    if (-not (Test-Path -LiteralPath $requiredOutput -PathType Leaf)) {
        throw "Evaluation output missing: $requiredOutput"
    }
}
Invoke-GliomaPython -Arguments @(
    '-m', 'glioma_seg.backends.nnunet.backend',
    '--project-root', $script:GliomaProjectRoot,
    'record-artifacts',
    '--experiment-id', $ExperimentId,
    '--artifact', "metrics_summary=$(Join-Path $reportDirectory 'metrics_summary.csv')",
    '--artifact', "metrics_summary_json=$(Join-Path $reportDirectory 'metrics_summary.json')",
    '--artifact', "metrics_per_case=$(Join-Path $reportDirectory 'metrics_per_case.csv')",
    '--artifact', "evaluation_protocol=$(Join-Path $reportDirectory 'evaluation_protocol.json')",
    '--artifact', "standard_metric_predictions=$validationPredictions",
    '--artifact', "timing_predictions_tta_off=$timingOutput",
    '--artifact', "inference_runtime=$inferenceRuntime",
    '--artifact', "official_status=$officialStatus"
)
if ((Test-Path -LiteralPath $officialSummary -PathType Leaf) -and
    (Test-Path -LiteralPath $officialSummaryJson -PathType Leaf) -and
    (Test-Path -LiteralPath $officialPerCase -PathType Leaf)) {
    Invoke-GliomaPython -Arguments @(
        '-m', 'glioma_seg.backends.nnunet.backend',
        '--project-root', $script:GliomaProjectRoot,
        'record-artifacts',
        '--experiment-id', $ExperimentId,
        '--artifact', "official_metrics_summary=$officialSummary",
        '--artifact', "official_metrics_summary_json=$officialSummaryJson",
        '--artifact', "official_metrics_per_case=$officialPerCase"
    )
}
Write-Host "Fold-0 evaluation passed: $reportDirectory" -ForegroundColor Green
