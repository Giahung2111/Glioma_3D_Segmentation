param(
    [string]$ExperimentId = '',
    [switch]$Resume,
    [Parameter(Mandatory = $true)][switch]$ConfirmRun,
    [switch]$PreflightOnly,
    [switch]$IncludeOfficialValidation,
    [switch]$AllowLowGpuUtilization
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot '_common.ps1')

$trainer = 'nnUNetTrainer_100epochs'
$datasetName = 'Dataset501_BraTS2023GLI'
$folds = @(0, 1, 2, 3, 4)
$experimentConfig = Join-Path $script:GliomaCodeRoot 'configs\experiments\nnunet_100epoch_cv.yaml'
$providedExperimentId = -not [string]::IsNullOrWhiteSpace($ExperimentId)
$lockHandle = $null
$sleepPreventionActive = $false
$pipelineSucceeded = $false

function Write-CvBanner {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [ConsoleColor]$Color = [ConsoleColor]::Cyan
    )
    $line = '=' * 72
    Write-Host $line -ForegroundColor $Color
    Write-Host $Title -ForegroundColor $Color
    Write-Host $line -ForegroundColor $Color
}

function Test-ActiveNnunetTraining {
    try {
        $processes = @(
            Get-CimInstance -ClassName Win32_Process -ErrorAction Stop |
                Where-Object {
                    $_.ProcessId -ne $PID -and
                    $null -ne $_.CommandLine -and
                    (
                        $_.CommandLine -match '(?i)nnUNetv2_train' -or
                        $_.CommandLine -match '(?i)glioma_seg\.backends\.nnunet\.backend.+\btrain\b'
                    )
                }
        )
    }
    catch {
        throw "Unable to verify that no other nnU-Net training process is active: $($_.Exception.Message)"
    }
    if ($processes.Count -gt 0) {
        $description = ($processes | ForEach-Object {
            "PID=$($_.ProcessId) Name=$($_.Name)"
        }) -join ', '
        throw "Another nnU-Net training process is already active ($description). No second training process was started."
    }
}

function Invoke-FoldAudit {
    param(
        [Parameter(Mandatory = $true)][int]$Fold,
        [Parameter(Mandatory = $true)][string]$OutputPath
    )
    $outputParent = Split-Path -Parent $OutputPath
    New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
    & $script:GliomaPython @(
        '-m', 'glioma_seg.backends.nnunet.backend',
        '--project-root', $script:GliomaProjectRoot,
        'audit-fold',
        '--experiment-id', $ExperimentId,
        '--fold', [string]$Fold,
        '--trainer', $trainer,
        '--require-probabilities',
        '--output', $OutputPath
    ) | Out-Host
    $auditExitCode = $LASTEXITCODE
    if (-not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
        throw "Fold $Fold audit returned exit code $auditExitCode without creating $OutputPath"
    }
    $audit = Get-Content -LiteralPath $OutputPath -Raw | ConvertFrom-Json
    return [pscustomobject]@{
        ExitCode = $auditExitCode
        Report = $audit
    }
}

function Test-AccumulatedCrossvalOutput {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return $false }
    foreach ($requiredName in @('dataset.json', 'plans.json', 'summary.json')) {
        if (-not (Test-Path -LiteralPath (Join-Path $Path $requiredName) -PathType Leaf)) {
            return $false
        }
    }
    $labelsDirectory = Join-Path (Join-Path $Env:nnUNet_raw $datasetName) 'labelsTr'
    if (-not (Test-Path -LiteralPath $labelsDirectory -PathType Container)) { return $false }
    $expected = @(
        Get-ChildItem -LiteralPath $labelsDirectory -File -Filter '*.nii.gz' |
            ForEach-Object { $_.Name } |
            Sort-Object
    )
    $actual = @(
        Get-ChildItem -LiteralPath $Path -File -Filter '*.nii.gz' |
            ForEach-Object { $_.Name } |
            Sort-Object
    )
    if ($expected.Count -ne 1251 -or $actual.Count -ne 1251) { return $false }
    return @(Compare-Object -ReferenceObject $expected -DifferenceObject $actual).Count -eq 0
}

function Move-IncompleteDerivedDirectory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $preservedPath = "$Path.incomplete.$stamp"
    Move-Item -LiteralPath $Path -Destination $preservedPath
    Write-Warning "Incomplete derived cross-validation output was preserved at $preservedPath"
}

if (-not $ConfirmRun) {
    throw 'This compute-intensive workflow is opt-in. Re-run with -ConfirmRun after reviewing the printed protocol.'
}
if ($Resume -and -not $providedExperimentId) {
    throw '-Resume requires the exact -ExperimentId printed by the original run; the pipeline never guesses an experiment.'
}
if ($providedExperimentId -and $ExperimentId -notmatch '^[A-Za-z0-9_-]+$') {
    throw 'ExperimentId may contain only letters, digits, underscore, and hyphen.'
}
if (-not (Test-Path -LiteralPath $experimentConfig -PathType Leaf)) {
    throw "Pinned experiment configuration is missing: $experimentConfig"
}

$lockDirectory = Join-Path $script:GliomaWorkspace 'cache\locks'
New-Item -ItemType Directory -Path $lockDirectory -Force | Out-Null
$lockPath = Join-Path $lockDirectory 'Dataset501_BraTS2023GLI_nnunet_100epoch_cv.lock'

try {
    try {
        $lockHandle = [System.IO.File]::Open(
            $lockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    }
    catch {
        throw "Another 100-epoch CV pipeline holds the exclusive lock at $lockPath. Wait for it to finish; a second pipeline was not started."
    }

    Test-ActiveNnunetTraining

    if ([string]::IsNullOrWhiteSpace($ExperimentId)) {
        $ExperimentId = New-GliomaExperimentId -Kind fullcv -Fold 0
    }
    $reportCandidate = Join-Path (Join-Path $script:GliomaWorkspace 'reports') $ExperimentId
    if ($Resume -and
        -not (Test-Path -LiteralPath (Join-Path $reportCandidate 'experiment.json') -PathType Leaf)) {
        throw "Resume experiment manifest does not exist: $(Join-Path $reportCandidate 'experiment.json'). Check the exact ID; no new experiment was created from a resume typo."
    }
    $reportDirectory = Get-GliomaReportDirectory -ExperimentId $ExperimentId
    if ($providedExperimentId -and -not $Resume -and
        (Test-Path -LiteralPath $reportDirectory -PathType Container) -and
        @(Get-ChildItem -LiteralPath $reportDirectory -Force).Count -gt 0) {
        throw "Experiment $ExperimentId already contains artifacts. Use -Resume only for that same run, or omit -ExperimentId to create a new run."
    }

    $lockText = @(
        "pid=$PID",
        "experiment_id=$ExperimentId",
        "trainer=$trainer",
        "started_utc=$([DateTime]::UtcNow.ToString('o'))"
    ) -join "`n"
    $lockBytes = [System.Text.Encoding]::UTF8.GetBytes("$lockText`n")
    $lockHandle.SetLength(0)
    $lockHandle.Write($lockBytes, 0, $lockBytes.Length)
    $lockHandle.Flush($true)

    if ($null -eq ('GliomaCvPowerState' -as [type])) {
        Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class GliomaCvPowerState
{
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint SetThreadExecutionState(uint flags);
    public static uint PreventSystemSleep() { return SetThreadExecutionState(0x80000001); }
    public static uint RestoreDefaults() { return SetThreadExecutionState(0x80000000); }
}
'@
    }
    if ([GliomaCvPowerState]::PreventSystemSleep() -eq 0) {
        throw 'Windows sleep prevention could not be enabled; training was not started.'
    }
    $sleepPreventionActive = $true

    Write-CvBanner -Title "100-EPOCH COMPUTE-LIMITED nnU-NET 5-FOLD CV | $ExperimentId" -Color Magenta
    Write-Host 'Dataset: Dataset501_BraTS2023GLI (1,251 training cases)'
    Write-Host 'Trainer: official nnUNetTrainer_100epochs'
    Write-Host 'Folds: 0, 1, 2, 3, 4 (sequential on one GPU)'
    Write-Host 'Validation probabilities: ENABLED (--npz)'
    Write-Host 'Classification: compute-limited experiment, not the standard 1000-epoch baseline'
    Write-Host "Report directory: $reportDirectory"
    Write-Host 'Windows automatic system sleep: prevented until this process exits'

    Write-CvBanner -Title 'PREPARATION AND SAFETY GATES'
    & (Join-Path $PSScriptRoot '00_system_check.ps1') -ExperimentId $ExperimentId

    Write-Host '[TEST GATE] Running the complete project test suite before GPU training...' -ForegroundColor Cyan
    Invoke-GliomaPython -Arguments @(
        '-m', 'pytest',
        (Join-Path $script:GliomaCodeRoot 'tests'),
        '-q'
    )

    & (Join-Path $PSScriptRoot '01_validate_raw_data.ps1') `
        -ExperimentId $ExperimentId `
        -IncludeOfficialValidation:$IncludeOfficialValidation
    & (Join-Path $PSScriptRoot '02_prepare_nnunet.ps1') `
        -ExperimentId $ExperimentId `
        -IncludeOfficialValidation:$IncludeOfficialValidation
    & (Join-Path $PSScriptRoot '03_plan_and_preprocess.ps1') -ExperimentId $ExperimentId
    & (Join-Path $PSScriptRoot '04_benchmark_gpu.ps1') `
        -ExperimentId $ExperimentId `
        -Resume:$Resume

    $foldAudits = @{}
    foreach ($fold in $folds) {
        $foldReportDirectory = Join-Path $reportDirectory "folds\fold_$fold"
        $auditPath = Join-Path $foldReportDirectory 'artifact_audit.json'
        $auditResult = Invoke-FoldAudit -Fold $fold -OutputPath $auditPath
        $foldAudits[$fold] = $auditResult.Report

        if ([bool]$auditResult.Report.complete -and [bool]$auditResult.Report.valid) {
            Write-Host "[PREFLIGHT] Fold $fold is already complete and verified." -ForegroundColor Green
            continue
        }

        if ([bool]$auditResult.Report.safe_to_restart) {
            if (-not $Resume) {
                throw "Fold $fold is owned by this experiment but has no checkpoint. Re-run with -Resume so its partial directory can be preserved and restarted safely."
            }
            $archiveOutput = Join-Path $foldReportDirectory (
                "restart_archive_$((Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmss_fff')).json"
            )
            Write-Warning "Fold $fold has no resumable checkpoint. Preserving its owned partial output before a fresh restart."
            Invoke-GliomaPython -Arguments @(
                '-m', 'glioma_seg.backends.nnunet.backend',
                '--project-root', $script:GliomaProjectRoot,
                'archive-restart-fold',
                '--experiment-id', $ExperimentId,
                '--fold', [string]$fold,
                '--trainer', $trainer,
                '--output', $archiveOutput
            )
            $auditResult = Invoke-FoldAudit -Fold $fold -OutputPath $auditPath
            $foldAudits[$fold] = $auditResult.Report
            if ([bool]$auditResult.Report.safe_to_resume -or
                [bool]$auditResult.Report.safe_to_restart) {
                throw "Fold $fold remained resume/restart-owned after archive; inspect $archiveOutput before continuing."
            }
            Write-Host "Fold $fold partial output was preserved; readiness will validate a fresh restart. Evidence: $archiveOutput" -ForegroundColor Yellow
        }

        $readinessArguments = @(
            '-m', 'glioma_seg.backends.nnunet.backend',
            '--project-root', $script:GliomaProjectRoot,
            'readiness',
            '--experiment-id', $ExperimentId,
            '--fold', [string]$fold,
            '--trainer', $trainer,
            '--output', (Join-Path $foldReportDirectory 'readiness.json')
        )
        if ($Resume -and [bool]$auditResult.Report.safe_to_resume) {
            $readinessArguments += '--continue'
        }
        if ($AllowLowGpuUtilization) {
            $readinessArguments += '--allow-low-gpu-utilization'
        }
        Invoke-GliomaPython -Arguments $readinessArguments
    }

    if ($PreflightOnly) {
        Write-CvBanner -Title 'PREFLIGHT COMPLETE — NO 100-EPOCH FOLD WAS STARTED' -Color Green
        Write-Host "Experiment ID retained for the real run: $ExperimentId"
        Write-Host 'Start/resume with:'
        Write-Host ".\Code\scripts\run_nnunet_cv_pipeline.ps1 -ExperimentId $ExperimentId -Resume -ConfirmRun" -ForegroundColor Yellow
        $pipelineSucceeded = $true
        return
    }

    foreach ($fold in $folds) {
        $foldPosition = $fold + 1
        $foldReportDirectory = Join-Path $reportDirectory "folds\fold_$fold"
        $auditPath = Join-Path $foldReportDirectory 'artifact_audit.json'
        $auditResult = Invoke-FoldAudit -Fold $fold -OutputPath $auditPath
        if ([bool]$auditResult.Report.complete -and [bool]$auditResult.Report.valid) {
            Write-CvBanner -Title "FOLD $foldPosition/5 VERIFIED — REUSING COMPLETE ARTIFACTS" -Color Green
            continue
        }

        Write-CvBanner -Title "TRAINING FOLD $foldPosition/5 | 100 EPOCHS | --npz ENABLED" -Color Magenta
        $trainArguments = @(
            '-m', 'glioma_seg.backends.nnunet.backend',
            '--project-root', $script:GliomaProjectRoot,
            'train',
            '--experiment-id', $ExperimentId,
            '--fold', [string]$fold,
            '--trainer', $trainer,
            '--config', $experimentConfig
        )
        if ($Resume -and [bool]$auditResult.Report.safe_to_resume) {
            $trainArguments += '--continue'
            Write-Host "Fold $fold will continue from its verified owner-matched checkpoint." -ForegroundColor Yellow
        }
        if ($AllowLowGpuUtilization) {
            $trainArguments += '--allow-low-gpu-utilization'
        }
        Invoke-GliomaPython -Arguments $trainArguments

        $completedAudit = Invoke-FoldAudit -Fold $fold -OutputPath $auditPath
        if ($completedAudit.ExitCode -ne 0 -or
            -not [bool]$completedAudit.Report.complete -or
            -not [bool]$completedAudit.Report.valid) {
            throw "Fold $fold returned from training but failed its final checkpoint/validation/NPZ ownership audit. See $auditPath"
        }
        $verifiedValidationCases = [int]$completedAudit.Report.expected_validation_cases
        Write-Host "Fold $fold is complete and its checkpoint, $verifiedValidationCases validation cases, NPZ and PKL artifacts are verified." -ForegroundColor Green
    }

    Write-CvBanner -Title 'AGGREGATING FIVE-FOLD TRAINING RUNTIME AND GPU TELEMETRY'
    Invoke-GliomaPython -Arguments @(
        '-m', 'glioma_seg.reporting.crossval',
        '--experiment-dir', $reportDirectory,
        '--expected-epochs-per-fold', '100'
    )
    foreach ($aggregateOutput in @(
        (Join-Path $reportDirectory 'runtime.json'),
        (Join-Path $reportDirectory 'gpu_summary.json'),
        (Join-Path $reportDirectory 'fold_training_summary.csv'),
        (Join-Path $reportDirectory 'cv_runtime_summary.json'),
        (Join-Path $reportDirectory 'cv_gpu_summary.json')
    )) {
        if (-not (Test-Path -LiteralPath $aggregateOutput -PathType Leaf)) {
            throw "Five-fold telemetry aggregation output is missing: $aggregateOutput"
        }
    }

    Write-CvBanner -Title 'ACCUMULATING OUT-OF-FOLD PREDICTIONS'
    $modelRoot = Join-Path (Join-Path $Env:nnUNet_results $datasetName) "$trainer`__nnUNetPlans__3d_fullres"
    $crossvalOutput = Join-Path $modelRoot 'crossval_results_folds_0_1_2_3_4'
    if (Test-AccumulatedCrossvalOutput -Path $crossvalOutput) {
        Write-Host "Verified accumulated CV output retained: $crossvalOutput" -ForegroundColor Green
    }
    else {
        if (Test-Path -LiteralPath $crossvalOutput) {
            if (-not $Resume) {
                throw "Accumulated CV output exists but is incomplete: $crossvalOutput. Re-run this experiment with -Resume to preserve and repair it."
            }
            Move-IncompleteDerivedDirectory -Path $crossvalOutput
        }
        Invoke-GliomaPython -Arguments @(
            '-m', 'glioma_seg.backends.nnunet.backend',
            '--project-root', $script:GliomaProjectRoot,
            'accumulate-crossval',
            '--experiment-id', $ExperimentId,
            '--output-dir', $crossvalOutput,
            '--trainer', $trainer
        )
        if (-not (Test-AccumulatedCrossvalOutput -Path $crossvalOutput)) {
            throw "Official CV accumulation did not create an exact 1,251-case output: $crossvalOutput"
        }
        Write-Host "Verified 1,251-case accumulated CV output: $crossvalOutput" -ForegroundColor Green
    }

    Write-CvBanner -Title 'CROSS-VALIDATION EVALUATION'
    & (Join-Path $PSScriptRoot '06_evaluate_cv.ps1') `
        -ExperimentId $ExperimentId `
        -Trainer $trainer `
        -AccumulatedPredictionDir $crossvalOutput `
        -Resume:$Resume

    Write-CvBanner -Title 'CROSS-VALIDATION FAILURE ANALYSIS AND FIGURES'
    & (Join-Path $PSScriptRoot '07_analyze_cv_failures.ps1') `
        -ExperimentId $ExperimentId `
        -AccumulatedPredictionDir $crossvalOutput `
        -Resume:$Resume

    Write-CvBanner -Title 'FINAL REPORT BUNDLE'
    & (Join-Path $PSScriptRoot '08_generate_report.ps1') `
        -ExperimentId $ExperimentId `
        -CrossvalSummaryJson (Join-Path $reportDirectory 'crossval_summary.json')

    $pipelineSucceeded = $true
    Write-CvBanner -Title '100-EPOCH FIVE-FOLD PIPELINE COMPLETE' -Color Green
    Write-Host "Experiment: $ExperimentId"
    Write-Host "Summary: $(Join-Path $reportDirectory 'summary.md')"
    Write-Host "Discussion deck source: $(Join-Path $reportDirectory 'weekly_discussion.md')"
    Write-Host "Zip/send this report directory: $reportDirectory"
}
catch {
    Write-CvBanner -Title 'PIPELINE STOPPED SAFELY' -Color Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    if (-not [string]::IsNullOrWhiteSpace($ExperimentId)) {
        Write-Host 'Check the referenced log, then resume with the exact same experiment ID:' -ForegroundColor Yellow
        Write-Host ".\Code\scripts\run_nnunet_cv_pipeline.ps1 -ExperimentId $ExperimentId -Resume -ConfirmRun" -ForegroundColor Yellow
    }
    throw
}
finally {
    if ($sleepPreventionActive) {
        [void][GliomaCvPowerState]::RestoreDefaults()
        Write-Host 'Windows sleep policy was restored to its previous default behavior.' -ForegroundColor DarkGray
    }
    if ($null -ne $lockHandle) {
        $lockHandle.Dispose()
    }
    if (-not $pipelineSucceeded -and -not [string]::IsNullOrWhiteSpace($ExperimentId)) {
        Write-Host "All completed checkpoints and derived artifacts were retained for experiment $ExperimentId." -ForegroundColor DarkYellow
    }
}
