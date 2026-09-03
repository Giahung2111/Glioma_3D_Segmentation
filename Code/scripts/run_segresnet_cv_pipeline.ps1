<#
.SYNOPSIS
Runs the owned MONAI SegResNet pipeline without activating a shell environment.

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Code\scripts\run_segresnet_cv_pipeline.ps1 -ConfirmRun

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Code\scripts\run_segresnet_cv_pipeline.ps1 -SmokeTest -ConfirmRun

.EXAMPLE
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Code\scripts\run_segresnet_cv_pipeline.ps1 -ExperimentId <EXACT_ID> -Resume -ConfirmRun

.NOTES
The full default is folds 0..4, 100 epochs per fold, sequentially. SmokeTest uses
8 real training cases and 2 real validation cases from canonical fold 0, forces a
stop at epoch 1, resumes to epoch 3, and completes evaluation/report finalization.
#>
param(
    [string]$ExperimentId = '',
    [switch]$Resume,
    [switch]$ConfirmRun,
    [switch]$SmokeTest,
    [switch]$PreflightOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot '_research_models_common.ps1')

$providedExperimentId = -not [string]::IsNullOrWhiteSpace($ExperimentId)
$experimentKind = if ($SmokeTest) { 'smoke' } else { 'fullcv' }
$folds = if ($SmokeTest) { @(0) } else { @(0, 1, 2, 3, 4) }
$expectedCaseCount = if ($SmokeTest) { 2 } else { 1251 }
$targetEpochs = if ($SmokeTest) { 3 } else { 100 }
$quickTrainCases = 8
$quickValidationCases = 2
$pipelineSucceeded = $false
$reportDirectory = $null

if (-not $ConfirmRun) {
    throw (
        'This GPU workflow is opt-in. Re-run with -ConfirmRun after reviewing the ' +
        'SegResNet protocol and available disk space.'
    )
}
if ($Resume -and -not $providedExperimentId) {
    throw (
        '-Resume requires the exact -ExperimentId printed by the original run. ' +
        'The pipeline never guesses an experiment.'
    )
}
if ($providedExperimentId) {
    Assert-ResearchExperimentId -ExperimentId $ExperimentId
}
else {
    $ExperimentId = New-SegResNetExperimentId -Kind $experimentKind
}

$reportCandidate = Get-ResearchReportDirectory -ExperimentId $ExperimentId
$experimentManifest = Join-Path $reportCandidate 'experiment.json'
if ($Resume -and -not (Test-Path -LiteralPath $experimentManifest -PathType Leaf)) {
    throw (
        "Resume manifest does not exist: $experimentManifest. " +
        'Check the exact ID; no new experiment was created from a resume typo.'
    )
}
if ($providedExperimentId -and -not $Resume -and
    (Test-Path -LiteralPath $reportCandidate -PathType Container) -and
    @(Get-ChildItem -LiteralPath $reportCandidate -Force).Count -gt 0) {
    throw (
        "Experiment $ExperimentId already contains artifacts. Use -Resume only for that " +
        'same run, or omit -ExperimentId to allocate a new experiment.'
    )
}

try {
    Enter-ResearchGpuLocks -ExperimentId $ExperimentId
    Assert-NoActiveResearchModelProcess
    Enable-ResearchSleepPrevention

    $bannerTitle = if ($SmokeTest) {
        "SEGRESNET REAL-DATA RESUME SMOKE TEST | $ExperimentId"
    }
    else {
        "MONAI SEGRESNET 100-EPOCH 5-FOLD CV | $ExperimentId"
    }
    Write-ResearchBanner -Title $bannerTitle -Color Magenta
    Write-Host "Python (direct, no activation): $script:ResearchPython"
    Write-Host 'Dataset: Dataset501_BraTS2023GLI'
    Write-Host "Experiment kind: $experimentKind"
    Write-Host "Folds: $($folds -join ', ')"
    Write-Host "Target epochs: $targetEpochs per fold"
    Write-Host 'TTA: OFF; canonical ET/TC/WT probabilities retained'
    Write-Host 'Windows automatic system sleep: prevented until this process exits'

    Write-ResearchStage -Number '1/12' -Name 'Initialize Owned Experiment'
    Invoke-ResearchPython -Arguments @(
        '-m', 'glioma_seg.backends.segresnet.backend',
        '--project-root', $script:ResearchProjectRoot,
        'initialize',
        '--experiment-id', $ExperimentId,
        '--kind', $experimentKind
    )
    $reportDirectory = Get-ResearchReportDirectory -ExperimentId $ExperimentId -Create
    $experimentManifest = Join-Path $reportDirectory 'experiment.json'
    $experiment = Get-Content -LiteralPath $experimentManifest -Raw | ConvertFrom-Json
    if ([string]$experiment.experiment_id -ne $ExperimentId -or
        [string]$experiment.backend -ne 'segresnet' -or
        [string]$experiment.experiment_kind -ne $experimentKind) {
        throw "Experiment identity/kind mismatch in $experimentManifest"
    }
    $recordedModelConfig = [string]$experiment.model_config
    $recordedSplitSource = [string]$experiment.split.source
    foreach ($ownedInput in @(
        [pscustomobject]@{
            Name = 'SegResNet model configuration'
            Path = $recordedModelConfig
            ExpectedHash = [string]$experiment.model_config_sha256
        },
        [pscustomobject]@{
            Name = 'canonical five-fold split'
            Path = $recordedSplitSource
            ExpectedHash = [string]$experiment.split.sha256
        }
    )) {
        if (-not (Test-Path -LiteralPath $ownedInput.Path -PathType Leaf)) {
            throw "$($ownedInput.Name) disappeared after experiment initialization: $($ownedInput.Path)"
        }
        $actualHash = (Get-FileHash -LiteralPath $ownedInput.Path -Algorithm SHA256).Hash
        if ($actualHash -ne $ownedInput.ExpectedHash) {
            throw (
                "$($ownedInput.Name) changed after experiment initialization: " +
                "expected=$($ownedInput.ExpectedHash), actual=$actualHash"
            )
        }
    }
    $logsDirectory = Join-Path $reportDirectory 'logs'
    New-Item -ItemType Directory -Path $logsDirectory -Force | Out-Null

    Write-ResearchStage -Number '2/12' -Name 'Pinned Environment and GPU System Check'
    Invoke-ResearchPython -Arguments @(
        '-m', 'glioma_seg.backends.segresnet.backend',
        '--project-root', $script:ResearchProjectRoot,
        'system-check',
        '--experiment-id', $ExperimentId,
        '--output', (Join-Path $reportDirectory 'environment.json')
    ) -LogPath (Join-Path $logsDirectory 'system_check.log')
    Assert-OfficialBraTSEvaluatorReady

    Write-ResearchStage -Number '3/12' -Name 'Complete Project Test Gate'
    Invoke-ResearchPython -Arguments @(
        '-m', 'pytest',
        (Join-Path $script:ResearchCodeRoot 'tests'),
        '-q'
    ) `
        -LogPath (Join-Path $logsDirectory 'test_suite.log') `
        -PythonExecutable $script:ResearchProjectPython

    Write-ResearchStage -Number '4/12' -Name 'Raw Data Validation (Reuse or Run)'
    Invoke-ResearchDataValidation `
        -ExperimentId $ExperimentId `
        -ReportDirectory $reportDirectory `
        -LogPath (Join-Path $logsDirectory 'data_validation.log')
    Write-SegResNetPreprocessingEvidence -ReportDirectory $reportDirectory

    Write-ResearchStage -Number '5/12' -Name 'Unchanged-Recipe GPU Memory Preflight'
    $memoryOutput = Join-Path $reportDirectory 'memory_preflight.json'
    $reuseMemoryGate = $false
    if ($Resume -and (Test-Path -LiteralPath $memoryOutput -PathType Leaf)) {
        try {
            $memoryEvidence = Get-Content -LiteralPath $memoryOutput -Raw | ConvertFrom-Json
            $reuseMemoryGate = (
                [bool]$memoryEvidence.valid -and
                [string]$memoryEvidence.model -eq [string]$experiment.model_id -and
                (@($memoryEvidence.crop_size) -join ',') -eq '224,224,144' -and
                [int]$memoryEvidence.batch_size -eq 1 -and
                [bool]$memoryEvidence.amp -and
                (@($memoryEvidence.output_shape) -join ',') -eq '1,3,224,224,144' -and
                [double]$memoryEvidence.peak_reserved_mb -gt 0.0 -and
                [double]$memoryEvidence.peak_reserved_mb -le [double]$memoryEvidence.total_vram_mb
            )
        }
        catch {
            $reuseMemoryGate = $false
        }
    }
    if ($reuseMemoryGate) {
        Write-Host 'Reused the valid experiment-local GPU memory preflight.' -ForegroundColor Green
    }
    else {
        # memory-preflight currently writes a smoke fold_manifest. A dedicated
        # namespace prevents that probe from colliding with the real full-CV Fold 0.
        $memoryExperimentId = if ($SmokeTest) {
            $ExperimentId
        }
        else {
            "${ExperimentId}_memory_gate"
        }
        Invoke-ResearchPython -Arguments @(
            '-m', 'glioma_seg.backends.segresnet.backend',
            '--project-root', $script:ResearchProjectRoot,
            'memory-preflight',
            '--experiment-id', $memoryExperimentId,
            '--fold', '0',
            '--output', $memoryOutput
        ) -LogPath (Join-Path $logsDirectory 'memory_preflight.log')
    }

    if ($PreflightOnly) {
        Write-ResearchBanner -Title 'PREFLIGHT COMPLETE - NO TRAINING FOLD WAS STARTED' -Color Green
        Write-Host "Experiment ID retained: $ExperimentId"
        $resumeCommand = (
            ".\Code\scripts\run_segresnet_cv_pipeline.ps1 " +
            "-ExperimentId $ExperimentId -Resume -ConfirmRun"
        )
        if ($SmokeTest) { $resumeCommand += ' -SmokeTest' }
        Write-Host $resumeCommand -ForegroundColor Yellow
        $pipelineSucceeded = $true
        return
    }

    Write-ResearchStage -Number '6/12' -Name 'Sequential Fold Training, Audit, and Resume'
    $resultRoot = Get-SegResNetResultDirectory -ExperimentId $ExperimentId
    foreach ($fold in $folds) {
        $foldDirectory = Join-Path $resultRoot "fold_$fold"
        $foldReportDirectory = Join-Path $reportDirectory "folds\fold_$fold"
        $auditPath = Join-Path $foldReportDirectory 'artifact_audit.json'
        $audit = Invoke-SegResNetFoldAudit `
            -FoldDirectory $foldDirectory `
            -OutputPath $auditPath
        if ([bool]$audit.Report.valid -and [bool]$audit.Report.complete) {
            Write-ResearchBanner `
                -Title "FOLD $fold VERIFIED - REUSING COMPLETE ARTIFACTS" `
                -Color Green
            Sync-SegResNetFoldEvidence `
                -ExperimentId $ExperimentId `
                -Fold $fold `
                -ReportDirectory $reportDirectory
            continue
        }

        $foldManifestPath = Join-Path $foldDirectory 'fold_manifest.json'
        if (Test-Path -LiteralPath $foldManifestPath -PathType Leaf) {
            $foldManifest = Get-Content -LiteralPath $foldManifestPath -Raw | ConvertFrom-Json
            if ([string]$foldManifest.experiment_id -ne $ExperimentId -or
                [int]$foldManifest.fold -ne $fold -or
                [bool]$foldManifest.smoke -ne [bool]$SmokeTest -or
                [int]$foldManifest.target_epochs -ne $targetEpochs) {
                throw "Fold $fold manifest ownership/configuration mismatch: $foldManifestPath"
            }
        }

        if ([bool]$audit.Report.safe_to_resume -and -not $Resume) {
            throw (
                "Fold $fold has an owner-matched checkpoint. Re-run this exact experiment " +
                'with -Resume; the pipeline will not restart it silently.'
            )
        }

        if ($SmokeTest) {
            if ([bool]$audit.Report.safe_to_resume) {
                Write-ResearchBanner `
                    -Title 'SMOKE RESUME LEG - CONTINUING VERIFIED EPOCH-1 CHECKPOINT TO EPOCH 3' `
                    -Color Yellow
                Invoke-ResearchPython -Arguments @(
                    '-m', 'glioma_seg.backends.segresnet.backend',
                    '--project-root', $script:ResearchProjectRoot,
                    'train-fold',
                    '--experiment-id', $ExperimentId,
                    '--fold', '0',
                    '--epochs', '3',
                    '--smoke',
                    '--quick-train-cases', [string]$quickTrainCases,
                    '--quick-validation-cases', [string]$quickValidationCases,
                    '--resume'
                ) -LogPath (Join-Path $logsDirectory 'train_fold_0_smoke_resume.log')
            }
            else {
                $unexpected = @()
                if (Test-Path -LiteralPath $foldDirectory -PathType Container) {
                    $unexpected = @(
                        Get-ChildItem -LiteralPath $foldDirectory -Force |
                            Where-Object { $_.Name -ne 'fold_manifest.json' }
                    )
                }
                if ($unexpected.Count -gt 0) {
                    throw (
                        "Fold 0 has non-resumable partial artifacts. The backend has no safe " +
                        "archive/restart API; inspect $foldDirectory and use a new experiment."
                    )
                }
                Write-ResearchBanner `
                    -Title 'SMOKE LEG 1/2 - RUN TO EPOCH 1, THEN STOP INTENTIONALLY' `
                    -Color Magenta
                Invoke-ResearchPython -Arguments @(
                    '-m', 'glioma_seg.backends.segresnet.backend',
                    '--project-root', $script:ResearchProjectRoot,
                    'train-fold',
                    '--experiment-id', $ExperimentId,
                    '--fold', '0',
                    '--epochs', '3',
                    '--smoke',
                    '--quick-train-cases', [string]$quickTrainCases,
                    '--quick-validation-cases', [string]$quickValidationCases,
                    '--stop-after-epoch', '1'
                ) -LogPath (Join-Path $logsDirectory 'train_fold_0_smoke_forced_stop.log')
                $stoppedAudit = Invoke-SegResNetFoldAudit `
                    -FoldDirectory $foldDirectory `
                    -OutputPath $auditPath
                $stoppedRuntime = Get-Content -LiteralPath (
                    Join-Path $foldDirectory 'runtime.json'
                ) -Raw | ConvertFrom-Json
                if (-not [bool]$stoppedAudit.Report.safe_to_resume -or
                    [bool]$stoppedAudit.Report.complete -or
                    -not [bool]$stoppedRuntime.stopped_for_resume_test -or
                    [int]$stoppedRuntime.number_of_epochs -ne 1 -or
                    [int]$stoppedRuntime.target_epochs -ne 3) {
                    throw 'Smoke forced-stop gate did not prove an epoch-1 resumable checkpoint.'
                }
                Write-ResearchBanner `
                    -Title 'SMOKE LEG 2/2 - VERIFIED RESUME FROM EPOCH 1 TO EPOCH 3' `
                    -Color Magenta
                Invoke-ResearchPython -Arguments @(
                    '-m', 'glioma_seg.backends.segresnet.backend',
                    '--project-root', $script:ResearchProjectRoot,
                    'train-fold',
                    '--experiment-id', $ExperimentId,
                    '--fold', '0',
                    '--epochs', '3',
                    '--smoke',
                    '--quick-train-cases', [string]$quickTrainCases,
                    '--quick-validation-cases', [string]$quickValidationCases,
                    '--resume'
                ) -LogPath (Join-Path $logsDirectory 'train_fold_0_smoke_resume.log')
            }
        }
        else {
            if (-not [bool]$audit.Report.safe_to_resume -and
                (Test-Path -LiteralPath $foldDirectory -PathType Container)) {
                $unexpected = @(
                    Get-ChildItem -LiteralPath $foldDirectory -Force |
                        Where-Object { $_.Name -ne 'fold_manifest.json' }
                )
                if ($unexpected.Count -gt 0) {
                    throw (
                        "Fold $fold has non-resumable partial artifacts. The backend has no " +
                        "safe archive/restart API; inspect $foldDirectory and use a new experiment."
                    )
                }
            }
            Write-ResearchBanner `
                -Title "TRAINING SEGRESNET FOLD $($fold + 1)/5 | 100 EPOCHS" `
                -Color Magenta
            $trainArguments = @(
                '-m', 'glioma_seg.backends.segresnet.backend',
                '--project-root', $script:ResearchProjectRoot,
                'train-fold',
                '--experiment-id', $ExperimentId,
                '--fold', [string]$fold,
                '--epochs', '100'
            )
            if ([bool]$audit.Report.safe_to_resume) {
                $trainArguments += '--resume'
                Write-Host "Fold $fold will resume its verified full-state checkpoint." -ForegroundColor Yellow
            }
            Invoke-ResearchPython -Arguments $trainArguments `
                -LogPath (Join-Path $logsDirectory "train_fold_$fold.log")
        }

        $completedAudit = Invoke-SegResNetFoldAudit `
            -FoldDirectory $foldDirectory `
            -OutputPath $auditPath
        if ($completedAudit.ExitCode -ne 0 -or
            -not [bool]$completedAudit.Report.valid -or
            -not [bool]$completedAudit.Report.complete) {
            throw "Fold $fold returned from training but failed final artifact audit: $auditPath"
        }
        Sync-SegResNetFoldEvidence `
            -ExperimentId $ExperimentId `
            -Fold $fold `
            -ReportDirectory $reportDirectory
        Write-Host "Fold $fold checkpoint, masks, and probabilities are verified." -ForegroundColor Green
    }

    $labelsDirectory = Join-Path $script:ResearchWorkspace (
        'nnUNet_raw\Dataset501_BraTS2023GLI\labelsTr'
    )
    $splitPath = Join-Path $script:ResearchWorkspace (
        'nnUNet_preprocessed\Dataset501_BraTS2023GLI\splits_final.json'
    )
    $predictionDirectory = $null
    $analysisGroundTruth = $labelsDirectory

    Write-ResearchStage -Number '7/12' -Name 'OOF Assembly and Semantic Evaluation'
    if ($SmokeTest) {
        $smokeFoldDirectory = Join-Path $resultRoot 'fold_0'
        $predictionDirectory = Join-Path $smokeFoldDirectory 'predictions'
        $smokeArguments = @(
            '-m', 'glioma_seg.evaluation.smoke',
            '--fold-manifest', (Join-Path $smokeFoldDirectory 'fold_manifest.json'),
            '--ground-truth-dir', $labelsDirectory,
            '--prediction-dir', $predictionDirectory,
            '--output-dir', $reportDirectory
        )
        if ($Resume) { $smokeArguments += '--overwrite' }
        Invoke-ResearchPython -Arguments $smokeArguments `
            -LogPath (Join-Path $logsDirectory 'smoke_evaluation.log')
        $analysisGroundTruth = Join-Path $reportDirectory 'smoke_ground_truth'
    }
    else {
        $oofArguments = @(
            '-m', 'glioma_seg.backends.segresnet.backend',
            '--project-root', $script:ResearchProjectRoot,
            'assemble-oof',
            '--experiment-id', $ExperimentId
        )
        foreach ($fold in $folds) { $oofArguments += @('--fold', [string]$fold) }
        Invoke-ResearchPython -Arguments $oofArguments `
            -LogPath (Join-Path $logsDirectory 'assemble_oof.log')
        $predictionDirectory = Join-Path $resultRoot 'oof\predictions'

        $manifestArguments = @(
            '-m', 'glioma_seg.backends.segresnet.backend',
            '--project-root', $script:ResearchProjectRoot,
            'write-evaluation-manifest',
            '--experiment-id', $ExperimentId
        )
        foreach ($fold in $folds) { $manifestArguments += @('--fold', [string]$fold) }
        Invoke-ResearchPython -Arguments $manifestArguments
        $crossvalManifest = Join-Path $reportDirectory 'crossval_artifact_manifest.json'
        $evaluationArguments = @(
            '-m', 'glioma_seg.evaluation.model_crossval',
            '--ground-truth-dir', $labelsDirectory,
            '--splits-json', $splitPath,
            '--artifact-manifest', $crossvalManifest,
            '--output-dir', $reportDirectory,
            '--expected-case-count', '1251'
        )
        if ($Resume) { $evaluationArguments += '--overwrite' }
        Invoke-ResearchPython -Arguments $evaluationArguments `
            -LogPath (Join-Path $logsDirectory 'model_crossval_evaluation.log')
    }

    Write-ResearchStage -Number '8/12' -Name 'Pinned Official BraTS Lesion-Wise Evaluation'
    $officialPaths = @(
        (Join-Path $reportDirectory 'official_brats_metrics_status.json'),
        (Join-Path $reportDirectory 'official_brats_evaluator.log'),
        (Join-Path $reportDirectory 'official_lesionwise_metrics_per_case.csv'),
        (Join-Path $reportDirectory 'official_lesionwise_metrics_summary.csv'),
        (Join-Path $reportDirectory 'official_lesionwise_metrics_summary.json')
    )
    $reuseOfficial = $false
    if ($Resume -and (Test-Path -LiteralPath $officialPaths[0] -PathType Leaf)) {
        try {
            $officialStatus = Get-Content -LiteralPath $officialPaths[0] -Raw | ConvertFrom-Json
            $officialCases = @(Import-Csv -LiteralPath $officialPaths[2])
            $officialSummary = @(Import-Csv -LiteralPath $officialPaths[3])
            $officialSummaryJson = Get-Content -LiteralPath $officialPaths[4] -Raw |
                ConvertFrom-Json
            $reuseOfficial = (
                [bool]$officialStatus.available -and
                [int]$officialStatus.case_count -eq $expectedCaseCount -and
                [string]$officialStatus.version_or_commit -eq (
                    '43c905242b2eecf421d4ab2da7af8ece9777d322'
                ) -and
                (@($officialStatus.region_order) -join ',') -eq 'ET,TC,WT' -and
                @($officialPaths | Where-Object {
                    -not (Test-Path -LiteralPath $_ -PathType Leaf)
                }).Count -eq 0 -and
                $officialCases.Count -eq $expectedCaseCount -and
                @($officialCases.case_id | Select-Object -Unique).Count -eq $expectedCaseCount -and
                (@($officialSummary.metric | Sort-Object) -join ',') -eq 'Dice,HD95' -and
                @($officialSummary | Where-Object {
                    [int]$_.total_cases -ne $expectedCaseCount
                }).Count -eq 0 -and
                [int]$officialSummaryJson.case_count -eq $expectedCaseCount -and
                (@($officialSummaryJson.region_order) -join ',') -eq 'ET,TC,WT'
            )
        }
        catch {
            $reuseOfficial = $false
        }
    }
    if ($reuseOfficial) {
        Write-Host 'Reused complete pinned official lesion-wise metrics.' -ForegroundColor Green
    }
    else {
        if ($Resume) {
            Move-ResearchDerivedArtifacts `
                -ExperimentId $ExperimentId `
                -ReportDirectory $reportDirectory `
                -Label 'official_metrics' `
                -Paths $officialPaths
        }
        Invoke-ResearchPython -Arguments @(
            '-m', 'glioma_seg.evaluation.official_runner',
            '--ground-truth-dir', $analysisGroundTruth,
            '--prediction-dir', $predictionDirectory,
            '--output-dir', $reportDirectory,
            '--official-root', $script:ResearchOfficialMetricsRoot,
            '--python', $script:ResearchOfficialMetricsPython
        ) -LogPath (Join-Path $logsDirectory 'official_evaluation_command.log')
    }

    Write-ResearchStage -Number '9/12' -Name 'Backend-Neutral Failure Statistics'
    $failureStatisticsJson = Join-Path $reportDirectory 'failure_statistics.json'
    $failureStatisticsCsv = Join-Path $reportDirectory 'failure_statistics_per_case_region.csv'
    $failureStatisticsComplete = $false
    if ($Resume -and
        (Test-Path -LiteralPath $failureStatisticsJson -PathType Leaf) -and
        (Test-Path -LiteralPath $failureStatisticsCsv -PathType Leaf)) {
        try {
            $failureStatistics = Get-Content -LiteralPath $failureStatisticsJson -Raw |
                ConvertFrom-Json
            $failureRows = @(Import-Csv -LiteralPath $failureStatisticsCsv)
            $failurePairs = @($failureRows | ForEach-Object {
                "$($_.case_id)|$($_.region)"
            } | Select-Object -Unique)
            $currentMetricsHash = (Get-FileHash -LiteralPath (
                Join-Path $reportDirectory 'metrics_per_case.csv'
            ) -Algorithm SHA256).Hash
            $currentIntegrityHash = (Get-FileHash -LiteralPath (
                Join-Path $reportDirectory 'crossval_integrity.json'
            ) -Algorithm SHA256).Hash
            $failureStatisticsComplete = (
                [int]$failureStatistics.case_count -eq $expectedCaseCount -and
                [int]$failureStatistics.case_region_count -eq (3 * $expectedCaseCount) -and
                (@($failureStatistics.regions) -join ',') -eq 'ET,TC,WT' -and
                [string]$failureStatistics.sources.metrics_per_case_sha256 -eq (
                    $currentMetricsHash
                ) -and
                [string]$failureStatistics.sources.crossval_integrity_sha256 -eq (
                    $currentIntegrityHash
                ) -and
                $failureRows.Count -eq (3 * $expectedCaseCount) -and
                $failurePairs.Count -eq (3 * $expectedCaseCount) -and
                @($failureRows | Where-Object {
                    [string]$_.region -notin @('ET', 'TC', 'WT')
                }).Count -eq 0
            )
        }
        catch {
            $failureStatisticsComplete = $false
        }
    }
    if ($failureStatisticsComplete) {
        Write-Host 'Reused complete failure-statistics evidence.' -ForegroundColor Green
    }
    else {
        if ($Resume) {
            Move-ResearchDerivedArtifacts `
                -ExperimentId $ExperimentId `
                -ReportDirectory $reportDirectory `
                -Label 'failure_statistics' `
                -Paths @($failureStatisticsJson, $failureStatisticsCsv)
        }
        Invoke-ResearchPython -Arguments @(
            '-m', 'glioma_seg.analysis.failure_statistics',
            '--ground-truth-dir', $analysisGroundTruth,
            '--prediction-dir', $predictionDirectory,
            '--metrics-csv', (Join-Path $reportDirectory 'metrics_per_case.csv'),
            '--integrity-json', (Join-Path $reportDirectory 'crossval_integrity.json'),
            '--output-json', $failureStatisticsJson,
            '--output-csv', $failureStatisticsCsv,
            '--expected-case-count', [string]$expectedCaseCount
        ) -LogPath (Join-Path $logsDirectory 'failure_statistics.log')
    }

    $failureStageName = if ($SmokeTest) {
        'Real-Data Smoke Failure Ranking'
    }
    else {
        'Representative Failure Ranking and Clinical-Orientation Figures'
    }
    Write-ResearchStage -Number '10/12' -Name $failureStageName
    $failureRankings = Join-Path $reportDirectory 'failure_rankings.csv'
    $failureCases = Join-Path $reportDirectory 'failure_cases.csv'
    $figuresDirectory = Join-Path $reportDirectory 'figures'
    $figuresManifest = Join-Path $figuresDirectory 'figures_manifest.csv'
    $failureAnalysisComplete = (
        (Test-Path -LiteralPath $failureRankings -PathType Leaf) -and
        (Test-Path -LiteralPath $failureCases -PathType Leaf)
    )
    if ($failureAnalysisComplete) {
        $rankingRows = @(Import-Csv -LiteralPath $failureRankings)
        $failureRows = @(Import-Csv -LiteralPath $failureCases)
        $failureAnalysisComplete = $rankingRows.Count -gt 0 -and $failureRows.Count -gt 0
    }
    if ($failureAnalysisComplete -and -not $SmokeTest) {
        $failureAnalysisComplete = (
            (Test-Path -LiteralPath $figuresManifest -PathType Leaf) -and
            @(Get-ChildItem -LiteralPath $figuresDirectory -File -Filter '*.png').Count -gt 0
        )
        if ($failureAnalysisComplete) {
            $figureRows = @(Import-Csv -LiteralPath $figuresManifest)
            $failureAnalysisComplete = (
                $figureRows.Count -gt 0 -and
                @($figureRows | Where-Object {
                    [string]$_.orientation_convention -ne (
                        'RAS canonical axial; anterior/face up; neurological L/R'
                    )
                }).Count -eq 0
            )
        }
    }
    if ($failureAnalysisComplete -and $Resume) {
        Write-Host 'Reused verified failure-ranking artifacts.' -ForegroundColor Green
    }
    else {
        $failureOutputs = @($failureRankings, $failureCases)
        if (-not $SmokeTest) { $failureOutputs += $figuresDirectory }
        if ($Resume) {
            Move-ResearchDerivedArtifacts `
                -ExperimentId $ExperimentId `
                -ReportDirectory $reportDirectory `
                -Label 'failure_analysis' `
                -Paths $failureOutputs
        }
        $stage = Join-Path $script:ResearchCache (
            "$ExperimentId\failure_analysis_$((Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmss_fff'))"
        )
        $stageFigures = Join-Path $stage 'figures'
        $rankingDepth = if ($SmokeTest) { '2' } else { '5' }
        $representativeLimit = if ($SmokeTest) { '2' } else { '15' }
        New-Item -ItemType Directory -Path $stage -Force | Out-Null
        Invoke-ResearchPython -Arguments @(
            '-m', 'glioma_seg.analysis.failure_analysis',
            '--ground-truth-dir', $analysisGroundTruth,
            '--prediction-dir', $predictionDirectory,
            '--metrics-per-case-csv', (Join-Path $reportDirectory 'metrics_per_case.csv'),
            '--output-dir', $stage,
            '--top-n', $rankingDepth,
            '--max-cases', $representativeLimit
        ) -LogPath (Join-Path $logsDirectory 'failure_ranking.log')
        $stageRankingRows = @(Import-Csv -LiteralPath (
            Join-Path $stage 'failure_rankings.csv'
        ))
        $stageFailureRows = @(Import-Csv -LiteralPath (
            Join-Path $stage 'failure_cases.csv'
        ))
        if ($stageRankingRows.Count -lt 1 -or $stageFailureRows.Count -lt 1) {
            throw 'Failure ranking did not create non-empty, artifact-backed case records.'
        }
        if (-not $SmokeTest) {
            $dataValidation = Get-Content -LiteralPath (
                Join-Path $reportDirectory 'data_validation.json'
            ) -Raw | ConvertFrom-Json
            Invoke-ResearchPython -Arguments @(
                '-m', 'glioma_seg.visualization.overlays',
                '--raw-training-dir', ([string]$dataValidation.dataset_root),
                '--ground-truth-dir', $analysisGroundTruth,
                '--prediction-dir', $predictionDirectory,
                '--failure-cases-csv', (Join-Path $stage 'failure_cases.csv'),
                '--metrics-per-case-csv', (Join-Path $reportDirectory 'metrics_per_case.csv'),
                '--output-dir', $stageFigures,
                '--max-cases', $representativeLimit
            ) -LogPath (Join-Path $logsDirectory 'representative_figures.log')
            $stageManifest = Join-Path $stageFigures 'figures_manifest.csv'
            $stageRows = @(Import-Csv -LiteralPath $stageManifest)
            if ($stageRows.Count -lt 1 -or
                @($stageRows | Where-Object {
                    [string]$_.orientation_convention -ne (
                        'RAS canonical axial; anterior/face up; neurological L/R'
                    )
                }).Count -gt 0) {
                throw 'Representative figures did not prove the required clinical orientation.'
            }
        }
        Move-Item -LiteralPath (Join-Path $stage 'failure_rankings.csv') `
            -Destination $failureRankings
        Move-Item -LiteralPath (Join-Path $stage 'failure_cases.csv') `
            -Destination $failureCases
        if (-not $SmokeTest) {
            Move-Item -LiteralPath $stageFigures -Destination $figuresDirectory
        }
    }

    Write-ResearchStage -Number '11/12' -Name 'Training, GPU, and Inference Telemetry Aggregation'
    Write-SegResNetTelemetryAggregation `
        -ExperimentId $ExperimentId `
        -ReportDirectory $reportDirectory `
        -Folds $folds `
        -Smoke ([bool]$SmokeTest)

    Write-ResearchStage -Number '12/12' -Name 'Artifact-Backed Report and Model Bundle'
    $reportArguments = @(
        '-m', 'glioma_seg.reporting.report',
        '--output-dir', $reportDirectory,
        '--experiment-json', (Join-Path $reportDirectory 'experiment.json'),
        '--metrics-summary-csv', (Join-Path $reportDirectory 'metrics_summary.csv'),
        '--environment-json', (Join-Path $reportDirectory 'environment.json'),
        '--runtime-json', (Join-Path $reportDirectory 'runtime.json'),
        '--inference-runtime-json', (Join-Path $reportDirectory 'inference_runtime.json'),
        '--gpu-summary-json', (Join-Path $reportDirectory 'gpu_summary.json'),
        '--data-validation-json', (Join-Path $reportDirectory 'data_validation.json'),
        '--preprocessing-artifacts-json', (Join-Path $reportDirectory 'preprocessing_artifacts.json'),
        '--official-status-json', (Join-Path $reportDirectory 'official_brats_metrics_status.json'),
        '--official-summary-csv', (Join-Path $reportDirectory 'official_lesionwise_metrics_summary.csv'),
        '--evaluation-protocol-json', (Join-Path $reportDirectory 'evaluation_protocol.json'),
        '--failure-cases-csv', (Join-Path $reportDirectory 'failure_cases.csv')
    )
    if (-not $SmokeTest) {
        $reportArguments += @(
            '--figures-dir', (Join-Path $reportDirectory 'figures'),
            '--figures-manifest-csv', (Join-Path $reportDirectory 'figures\figures_manifest.csv'),
            '--crossval-summary-json', (Join-Path $reportDirectory 'crossval_summary.json')
        )
    }
    Invoke-ResearchPython -Arguments $reportArguments `
        -LogPath (Join-Path $logsDirectory 'report_generation.log')

    $bundleArguments = @(
        '-m', 'glioma_seg.reporting.model_bundle',
        '--experiment-dir', $reportDirectory,
        '--expected-case-count', [string]$expectedCaseCount,
        '--expected-folds'
    )
    foreach ($fold in $folds) { $bundleArguments += [string]$fold }
    # Do not tee this command into the report directory: the finalizer hashes
    # every existing log while it runs, so a simultaneously-open empty log
    # would invalidate its own audit.
    Invoke-ResearchPython -Arguments $bundleArguments
    if (-not (Test-Path -LiteralPath (
        Join-Path $reportDirectory 'report_manifest.json'
    ) -PathType Leaf)) {
        throw 'Model report finalizer completed without report_manifest.json.'
    }

    $pipelineSucceeded = $true
    Write-ResearchBanner -Title 'SEGRESNET PIPELINE COMPLETED AND VERIFIED' -Color Green
    Write-Host "Experiment: $ExperimentId"
    Write-Host "Report bundle: $reportDirectory"
}
finally {
    Disable-ResearchSleepPrevention
    Exit-ResearchGpuLocks
    if ($pipelineSucceeded) {
        Write-Host 'Windows sleep policy restored; GPU pipeline locks released.' -ForegroundColor Green
    }
    else {
        $resumeHint = (
            ".\Code\scripts\run_segresnet_cv_pipeline.ps1 " +
            "-ExperimentId $ExperimentId -Resume -ConfirmRun"
        )
        if ($SmokeTest) { $resumeHint += ' -SmokeTest' }
        Write-Warning (
            'Pipeline stopped before verified completion. Windows sleep policy was restored and ' +
            "GPU locks were released. Resume only with: $resumeHint"
        )
    }
}
