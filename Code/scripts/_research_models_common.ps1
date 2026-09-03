Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$script:ResearchCodeRoot = Split-Path -Parent $PSScriptRoot
$script:ResearchProjectRoot = Split-Path -Parent $script:ResearchCodeRoot
$script:ResearchWorkspace = Join-Path $script:ResearchProjectRoot 'Workspace'
$script:ResearchReports = Join-Path $script:ResearchWorkspace 'reports'
$script:ResearchModelResults = Join-Path $script:ResearchWorkspace 'model_results'
$script:ResearchCache = Join-Path $script:ResearchWorkspace 'cache'
$script:ResearchPython = Join-Path $script:ResearchCodeRoot '.venv-models\python.exe'
$script:ResearchProjectPython = Join-Path $script:ResearchCodeRoot '.venv\python.exe'
$script:ResearchOfficialMetricsPython = Join-Path (
    Join-Path $script:ResearchCache 'brats2023_metrics_env'
) 'python.exe'
$script:ResearchOfficialMetricsRoot = Join-Path (
    Join-Path $script:ResearchProjectRoot 'External'
) 'BraTS-2023-Metrics'
$script:ResearchGpuLockHandles = [System.Collections.Generic.List[System.IO.FileStream]]::new()
$script:ResearchSleepPreventionActive = $false

if (-not (Test-Path -LiteralPath $script:ResearchPython -PathType Leaf)) {
    throw (
        "The isolated research-model Python was not found at $script:ResearchPython. " +
        'Run Code\scripts\setup_research_models_env.ps1 first. Manual activation is not used.'
    )
}
if (-not (Test-Path -LiteralPath $script:ResearchProjectPython -PathType Leaf)) {
    throw "The project test Python was not found at $script:ResearchProjectPython."
}

$Env:PYTHONUNBUFFERED = '1'
$Env:PYTHONUTF8 = '1'

foreach ($directory in @(
    $script:ResearchWorkspace,
    $script:ResearchReports,
    $script:ResearchModelResults,
    $script:ResearchCache,
    (Join-Path $script:ResearchCache 'locks')
)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

function Write-ResearchStage {
    param(
        [Parameter(Mandatory = $true)][string]$Number,
        [Parameter(Mandatory = $true)][string]$Name,
        [ConsoleColor]$Color = [ConsoleColor]::Cyan
    )
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Host "[$timestamp] [$Number] $Name" -ForegroundColor $Color
}

function Write-ResearchBanner {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [ConsoleColor]$Color = [ConsoleColor]::Cyan
    )
    $line = '=' * 78
    Write-Host $line -ForegroundColor $Color
    Write-Host $Title -ForegroundColor $Color
    Write-Host $line -ForegroundColor $Color
}

function Assert-ResearchExperimentId {
    param([Parameter(Mandatory = $true)][string]$ExperimentId)
    if ($ExperimentId -notmatch '^[A-Za-z0-9_-]+$') {
        throw 'ExperimentId may contain only letters, digits, underscore, and hyphen.'
    }
}

function Get-ResearchReportDirectory {
    param(
        [Parameter(Mandatory = $true)][string]$ExperimentId,
        [switch]$Create
    )
    $path = Join-Path $script:ResearchReports $ExperimentId
    if ($Create) {
        New-Item -ItemType Directory -Path $path -Force | Out-Null
    }
    return $path
}

function Get-SegResNetResultDirectory {
    param([Parameter(Mandatory = $true)][string]$ExperimentId)
    return Join-Path (Join-Path $script:ResearchModelResults 'segresnet') $ExperimentId
}

function Get-MedNeXtResultDirectory {
    param([Parameter(Mandatory = $true)][string]$ExperimentId)
    return Join-Path (Join-Path $script:ResearchModelResults 'mednext') $ExperimentId
}

function Invoke-ResearchPython {
    param(
        [Parameter(Mandatory = $true)][string[]]$Arguments,
        [string]$LogPath = '',
        [string]$PythonExecutable = $script:ResearchPython
    )
    # Windows PowerShell converts redirected native stderr into non-terminating
    # ErrorRecord objects. With the project-wide Stop policy, harmless Python
    # warnings would otherwise abort a successful process before LASTEXITCODE
    # can be inspected. Keep stderr visible/logged and gate only on exit code.
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        if ([string]::IsNullOrWhiteSpace($LogPath)) {
            & $PythonExecutable @Arguments
            $exitCode = $LASTEXITCODE
        }
        else {
            $logParent = Split-Path -Parent $LogPath
            New-Item -ItemType Directory -Path $logParent -Force | Out-Null
            & $PythonExecutable @Arguments 2>&1 |
                Tee-Object -FilePath $LogPath |
                Out-Host
            $exitCode = $LASTEXITCODE
        }
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) {
        throw (
            "Python failed with exit code ${exitCode} using ${PythonExecutable}: " +
            ($Arguments -join ' ')
        )
    }
}

function Assert-OfficialBraTSEvaluatorReady {
    $expectedCommit = '43c905242b2eecf421d4ab2da7af8ece9777d322'
    if (-not (Test-Path -LiteralPath $script:ResearchOfficialMetricsPython -PathType Leaf)) {
        throw (
            'The pinned BraTS evaluator compatibility Python is missing: ' +
            "$script:ResearchOfficialMetricsPython. Recreate the existing evaluator " +
            'environment before starting GPU training.'
        )
    }
    if (-not (Test-Path -LiteralPath (
        Join-Path $script:ResearchOfficialMetricsRoot 'metrics.py'
    ) -PathType Leaf)) {
        throw "The official BraTS evaluator checkout is incomplete: $script:ResearchOfficialMetricsRoot"
    }
    $actualCommit = @(
        & git -C $script:ResearchOfficialMetricsRoot rev-parse HEAD 2>&1
    ) | Select-Object -Last 1
    if ($LASTEXITCODE -ne 0 -or [string]$actualCommit -ne $expectedCommit) {
        throw (
            "Official BraTS evaluator commit mismatch: expected=$expectedCommit, " +
            "actual=$actualCommit"
        )
    }
    $trackedChanges = @(
        & git -C $script:ResearchOfficialMetricsRoot status --porcelain --untracked-files=no 2>&1
    )
    if ($LASTEXITCODE -ne 0 -or $trackedChanges.Count -gt 0) {
        throw 'The official BraTS evaluator has tracked changes or cannot be audited.'
    }
    & $script:ResearchOfficialMetricsPython -c (
        'import sys; sys.path.insert(0, sys.argv[1]); import metrics; ' +
        'print(metrics.__file__)'
    ) $script:ResearchOfficialMetricsRoot | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw (
            'The isolated BraTS evaluator Python cannot import its pinned dependencies: ' +
            $script:ResearchOfficialMetricsPython
        )
    }
}

function New-SegResNetExperimentId {
    param([Parameter(Mandatory = $true)][ValidateSet('fullcv', 'smoke')][string]$Kind)
    $output = @(
        & $script:ResearchPython @(
            '-m', 'glioma_seg.backends.segresnet.backend',
            '--project-root', $script:ResearchProjectRoot,
            'new-experiment', '--kind', $Kind
        )
    )
    if ($LASTEXITCODE -ne 0 -or $output.Count -lt 1) {
        throw 'Unable to allocate a SegResNet experiment ID.'
    }
    $experimentId = [string]($output | Select-Object -Last 1)
    Assert-ResearchExperimentId -ExperimentId $experimentId.Trim()
    return $experimentId.Trim()
}

function New-MedNeXtExperimentId {
    param([Parameter(Mandatory = $true)][ValidateSet('fullcv', 'smoke')][string]$Kind)
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        $output = @(
            & $script:ResearchPython @(
                '-m', 'glioma_seg.backends.mednext.backend',
                '--project-root', $script:ResearchProjectRoot,
                'new-experiment', '--kind', $Kind
            )
        )
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0 -or $output.Count -lt 1) {
        throw 'Unable to allocate a MedNeXt experiment ID.'
    }
    $experimentId = [string]($output | Select-Object -Last 1)
    Assert-ResearchExperimentId -ExperimentId $experimentId.Trim()
    return $experimentId.Trim()
}

function Invoke-SegResNetFoldAudit {
    param(
        [Parameter(Mandatory = $true)][string]$FoldDirectory,
        [Parameter(Mandatory = $true)][string]$OutputPath
    )
    New-Item -ItemType Directory -Path (Split-Path -Parent $OutputPath) -Force | Out-Null
    & $script:ResearchPython @(
        '-m', 'glioma_seg.backends.segresnet.backend',
        '--project-root', $script:ResearchProjectRoot,
        'audit-fold',
        '--fold-dir', $FoldDirectory,
        '--output', $OutputPath
    ) | Out-Host
    $exitCode = $LASTEXITCODE
    if ($exitCode -notin @(0, 2)) {
        throw "SegResNet fold audit failed unexpectedly with exit code $exitCode"
    }
    if (-not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
        throw "SegResNet fold audit did not create $OutputPath"
    }
    $report = Get-Content -LiteralPath $OutputPath -Raw | ConvertFrom-Json
    return [pscustomobject]@{
        ExitCode = $exitCode
        Report = $report
    }
}

function Invoke-MedNeXtFoldAudit {
    param(
        [Parameter(Mandatory = $true)][string]$FoldDirectory,
        [Parameter(Mandatory = $true)][string]$OutputPath
    )
    New-Item -ItemType Directory -Path (Split-Path -Parent $OutputPath) -Force | Out-Null
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $script:ResearchPython @(
            '-m', 'glioma_seg.backends.mednext.backend',
            '--project-root', $script:ResearchProjectRoot,
            'audit-fold',
            '--fold-dir', $FoldDirectory,
            '--output', $OutputPath
        ) | Out-Host
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -notin @(0, 2)) {
        throw "MedNeXt fold audit failed unexpectedly with exit code $exitCode"
    }
    if (-not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
        throw "MedNeXt fold audit did not create $OutputPath"
    }
    $report = Get-Content -LiteralPath $OutputPath -Raw | ConvertFrom-Json
    return [pscustomobject]@{
        ExitCode = $exitCode
        Report = $report
    }
}

function Assert-NoActiveResearchModelProcess {
    try {
        $processes = @(
            Get-CimInstance -ClassName Win32_Process -ErrorAction Stop |
                Where-Object {
                    $_.ProcessId -ne $PID -and
                    $null -ne $_.CommandLine -and
                    (
                        $_.CommandLine -match '(?i)nnUNetv2_(train|predict)' -or
                        $_.CommandLine -match (
                            '(?i)glioma_seg\.backends\.(segresnet|mednext)\.' +
                            'backend.+\b(train-fold|train)\b'
                        )
                    )
                }
        )
    }
    catch {
        throw "Unable to inspect active model processes: $($_.Exception.Message)"
    }
    if ($processes.Count -gt 0) {
        $description = ($processes | ForEach-Object {
            "PID=$($_.ProcessId) Name=$($_.Name)"
        }) -join ', '
        throw "Another GPU model process is active ($description). No second run was started."
    }
}

function Enter-ResearchGpuLocks {
    param(
        [Parameter(Mandatory = $true)][string]$ExperimentId,
        [ValidateSet('segresnet', 'mednext')][string]$Backend = 'segresnet'
    )
    $lockDirectory = Join-Path $script:ResearchCache 'locks'
    $lockPaths = @(
        (Join-Path $lockDirectory 'research_models_gpu.lock'),
        (Join-Path $lockDirectory 'Dataset501_BraTS2023GLI_nnunet_100epoch_cv.lock')
    )
    try {
        foreach ($lockPath in $lockPaths) {
            $handle = [System.IO.File]::Open(
                $lockPath,
                [System.IO.FileMode]::OpenOrCreate,
                [System.IO.FileAccess]::ReadWrite,
                [System.IO.FileShare]::None
            )
            $script:ResearchGpuLockHandles.Add($handle)
            $lockText = @(
                "pid=$PID",
                "experiment_id=$ExperimentId",
                "backend=$Backend",
                "started_utc=$([DateTime]::UtcNow.ToString('o'))"
            ) -join "`n"
            $bytes = [System.Text.Encoding]::UTF8.GetBytes("$lockText`n")
            $handle.SetLength(0)
            $handle.Write($bytes, 0, $bytes.Length)
            $handle.Flush($true)
        }
    }
    catch {
        Exit-ResearchGpuLocks
        throw (
            'A GPU pipeline lock is already held. Wait for the active nnU-Net/research-model ' +
            "run to finish. Details: $($_.Exception.Message)"
        )
    }
}

function Exit-ResearchGpuLocks {
    foreach ($handle in $script:ResearchGpuLockHandles) {
        if ($null -ne $handle) {
            $handle.Dispose()
        }
    }
    $script:ResearchGpuLockHandles.Clear()
}

function Enable-ResearchSleepPrevention {
    if ($null -eq ('GliomaResearchPowerState' -as [type])) {
        Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class GliomaResearchPowerState
{
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint SetThreadExecutionState(uint flags);
    public static uint PreventSystemSleep() { return SetThreadExecutionState(0x80000001); }
    public static uint RestoreDefaults() { return SetThreadExecutionState(0x80000000); }
}
'@
    }
    if ([GliomaResearchPowerState]::PreventSystemSleep() -eq 0) {
        throw 'Windows sleep prevention could not be enabled; GPU work was not started.'
    }
    $script:ResearchSleepPreventionActive = $true
}

function Disable-ResearchSleepPrevention {
    if ($script:ResearchSleepPreventionActive) {
        [void][GliomaResearchPowerState]::RestoreDefaults()
        $script:ResearchSleepPreventionActive = $false
    }
}

function Invoke-ResearchDataValidation {
    param(
        [Parameter(Mandatory = $true)][string]$ExperimentId,
        [Parameter(Mandatory = $true)][string]$ReportDirectory,
        [Parameter(Mandatory = $true)][string]$LogPath
    )
    $globalJson = Join-Path $script:ResearchReports 'data_validation.json'
    $globalCsv = Join-Path $script:ResearchReports 'data_validation.csv'
    $reuse = $false
    if ((Test-Path -LiteralPath $globalJson -PathType Leaf) -and
        (Test-Path -LiteralPath $globalCsv -PathType Leaf)) {
        try {
            $existing = Get-Content -LiteralPath $globalJson -Raw | ConvertFrom-Json
            $reuse = (
                [bool]$existing.valid -and
                [string]$existing.dataset_kind -eq 'training' -and
                [int]$existing.expected_case_count -eq 1251 -and
                [int]$existing.actual_case_count -eq 1251 -and
                [int]$existing.valid_case_count -eq 1251 -and
                (Test-Path -LiteralPath ([string]$existing.dataset_root) -PathType Container)
            )
        }
        catch {
            $reuse = $false
        }
    }
    $reportJson = Join-Path $ReportDirectory 'data_validation.json'
    $reportCsv = Join-Path $ReportDirectory 'data_validation.csv'
    if ($reuse) {
        Copy-Item -LiteralPath $globalJson -Destination $reportJson -Force
        Copy-Item -LiteralPath $globalCsv -Destination $reportCsv -Force
        Write-Host 'Reused the complete 1,251-case raw-data validation evidence.' -ForegroundColor Green
        return
    }
    Invoke-ResearchPython -Arguments @(
        '-m', 'glioma_seg.data.validate',
        '--data-root', (Join-Path $script:ResearchProjectRoot 'Datasets'),
        '--kind', 'training',
        '--output-json', $reportJson,
        '--output-csv', $reportCsv,
        '--expected-training-cases', '1251'
    ) -LogPath $LogPath
    $validated = Get-Content -LiteralPath $reportJson -Raw | ConvertFrom-Json
    if (-not [bool]$validated.valid -or [int]$validated.valid_case_count -ne 1251) {
        throw "Raw-data validation is not complete: $reportJson"
    }
    Copy-Item -LiteralPath $reportJson -Destination $globalJson -Force
    Copy-Item -LiteralPath $reportCsv -Destination $globalCsv -Force
}

function Write-SegResNetPreprocessingEvidence {
    param([Parameter(Mandatory = $true)][string]$ReportDirectory)
    $rawDataset = Join-Path $script:ResearchWorkspace 'nnUNet_raw\Dataset501_BraTS2023GLI'
    $images = Join-Path $rawDataset 'imagesTr'
    $labels = Join-Path $rawDataset 'labelsTr'
    $splitPath = Join-Path $script:ResearchWorkspace (
        'nnUNet_preprocessed\Dataset501_BraTS2023GLI\splits_final.json'
    )
    foreach ($required in @($images, $labels, $splitPath)) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw "Shared converted-dataset evidence is missing: $required"
        }
    }
    $labelCount = @(Get-ChildItem -LiteralPath $labels -File -Filter '*.nii.gz').Count
    $imageCount = @(Get-ChildItem -LiteralPath $images -File -Filter '*.nii.gz').Count
    if ($labelCount -ne 1251 -or $imageCount -ne 5004) {
        throw (
            'Converted BraTS inventory mismatch: ' +
            "labels=$labelCount/1251, modalities=$imageCount/5004"
        )
    }
    # Do not wrap ConvertFrom-Json in @(...). Windows PowerShell 5.1 emits a
    # top-level JSON array as one no-enumerate Object[]; wrapping that result
    # creates an outer one-element array and makes a valid five-fold file look
    # as though it contains only one fold.
    $splits = Get-Content -LiteralPath $splitPath -Raw | ConvertFrom-Json
    if ($splits.Count -ne 5) {
        throw "Canonical split must contain five folds: $splitPath"
    }
    $validationCounts = @($splits | ForEach-Object { @($_.val).Count })
    if (@(Compare-Object $validationCounts @(251, 250, 250, 250, 250)).Count -ne 0) {
        throw "Canonical split validation counts are invalid: $($validationCounts -join ',')"
    }
    $evidence = [ordered]@{
        valid = $true
        dataset_name = 'Dataset501_BraTS2023GLI'
        preprocessing_scope = (
            'Shared lossless BraTS-to-project NIfTI conversion inventory; ' +
            'SegResNet model-specific MONAI transforms are applied at load time.'
        )
        checks = @(
            [ordered]@{
                name = 'converted modality and label inventory'
                ok = $true
                detail = '1,251 labels and 5,004 modality files'
            },
            [ordered]@{
                name = 'official five-fold split'
                ok = $true
                detail = (
                    'official seed=12345, fold_sizes=[(1000, 251), (1001, 250), ' +
                    '(1001, 250), (1001, 250), (1001, 250)]'
                )
            }
        )
        details = [ordered]@{
            raw_dataset_dir = [IO.Path]::GetFullPath($rawDataset)
            expected_case_count = 1251
            case_count = $labelCount
            modality_file_count = $imageCount
            splits_created = $true
            splits_file = [IO.Path]::GetFullPath($splitPath)
            splits_sha256 = (Get-FileHash -LiteralPath $splitPath -Algorithm SHA256).Hash.ToLowerInvariant()
            validation_case_counts = $validationCounts
        }
    }
    Write-ResearchJsonAtomic `
        -Path (Join-Path $ReportDirectory 'preprocessing_artifacts.json') `
        -Value $evidence
}

function Sync-SegResNetFoldEvidence {
    param(
        [Parameter(Mandatory = $true)][string]$ExperimentId,
        [Parameter(Mandatory = $true)][int]$Fold,
        [Parameter(Mandatory = $true)][string]$ReportDirectory
    )
    $source = Join-Path (Get-SegResNetResultDirectory -ExperimentId $ExperimentId) "fold_$Fold"
    $destination = Join-Path $ReportDirectory "folds\fold_$Fold"
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    foreach ($name in @(
        'fold_manifest.json',
        'runtime.json',
        'gpu_summary.json',
        'gpu_samples.csv',
        'train_history.json',
        'validation_summary.json'
    )) {
        $sourcePath = Join-Path $source $name
        if (Test-Path -LiteralPath $sourcePath -PathType Leaf) {
            Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $destination $name) -Force
        }
    }
    foreach ($segment in Get-ChildItem -LiteralPath $source -File -Filter 'gpu_samples_segment_*') {
        Copy-Item -LiteralPath $segment.FullName `
            -Destination (Join-Path $destination $segment.Name) -Force
    }
}

function Sync-MedNeXtFoldEvidence {
    param(
        [Parameter(Mandatory = $true)][string]$ExperimentId,
        [Parameter(Mandatory = $true)][int]$Fold,
        [Parameter(Mandatory = $true)][string]$ReportDirectory
    )
    $source = Join-Path (Get-MedNeXtResultDirectory -ExperimentId $ExperimentId) "fold_$Fold"
    $destination = Join-Path $ReportDirectory "folds\fold_$Fold"
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    foreach ($name in @(
        'fold_manifest.json',
        'runtime.json',
        'gpu_summary.json',
        'gpu_samples.csv',
        'train_history.json',
        'validation_summary.json'
    )) {
        $sourcePath = Join-Path $source $name
        if (Test-Path -LiteralPath $sourcePath -PathType Leaf) {
            Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $destination $name) -Force
        }
    }
    foreach ($segment in Get-ChildItem -LiteralPath $source -File -Filter 'gpu_samples_segment_*') {
        Copy-Item -LiteralPath $segment.FullName `
            -Destination (Join-Path $destination $segment.Name) -Force
    }
}

function Write-ResearchJsonAtomic {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value
    )
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $fileName = [IO.Path]::GetFileName($Path)
    # A killed process can leave only our own hidden atomic-write siblings.
    # Recover them before creating the next unique pair. The experiment lock
    # prevents another writer for the same report directory.
    foreach ($suffix in @('tmp', 'bak')) {
        Get-ChildItem -LiteralPath $parent -File -Filter ".$fileName.*.$suffix" |
            Remove-Item -Force
    }
    $temporary = Join-Path $parent ".$fileName.$([Guid]::NewGuid().ToString('N')).tmp"
    $backup = Join-Path $parent ".$fileName.$([Guid]::NewGuid().ToString('N')).bak"
    $json = ($Value | ConvertTo-Json -Depth 30) + "`n"
    [IO.File]::WriteAllText($temporary, $json, [Text.UTF8Encoding]::new($false))
    try {
        if (Test-Path -LiteralPath $Path -PathType Leaf) {
            # File.Replace is atomic on the same volume. Windows PowerShell 5.1
            # cannot reliably bind a null backup path, so use a real temporary
            # backup and delete it after the replacement succeeds.
            [IO.File]::Replace($temporary, $Path, $backup, $true)
        }
        else {
            [IO.File]::Move($temporary, $Path)
        }
    }
    finally {
        foreach ($transient in @($temporary, $backup)) {
            if (Test-Path -LiteralPath $transient -PathType Leaf) {
                Remove-Item -LiteralPath $transient -Force
            }
        }
    }
}

function Write-SegResNetTelemetryAggregation {
    param(
        [Parameter(Mandatory = $true)][string]$ExperimentId,
        [Parameter(Mandatory = $true)][string]$ReportDirectory,
        [Parameter(Mandatory = $true)][int[]]$Folds,
        [Parameter(Mandatory = $true)][bool]$Smoke
    )
    $totalInferenceSeconds = 0.0
    $totalCases = 0
    $inferenceSources = @()
    $inferenceCaseIds = @()
    foreach ($fold in $Folds) {
        Sync-SegResNetFoldEvidence `
            -ExperimentId $ExperimentId `
            -Fold $fold `
            -ReportDirectory $ReportDirectory
        $foldReport = Join-Path $ReportDirectory "folds\fold_$fold"
        $validationPath = Join-Path $foldReport 'validation_summary.json'
        if (-not (Test-Path -LiteralPath $validationPath -PathType Leaf)) {
            throw "Fold $fold validation telemetry is missing: $validationPath"
        }
        $validation = Get-Content -LiteralPath $validationPath -Raw | ConvertFrom-Json
        if (-not [bool]$validation.valid -or [int]$validation.case_count -lt 1) {
            throw "Fold $fold validation telemetry is invalid: $validationPath"
        }
        $totalInferenceSeconds += [double]$validation.inference_total_seconds
        $totalCases += [int]$validation.case_count
        $inferenceSources += $validationPath
        $inferenceCaseIds += @($validation.case_ids | ForEach-Object { [string]$_ })
    }
    if ($inferenceCaseIds.Count -ne $totalCases -or
        @($inferenceCaseIds | Select-Object -Unique).Count -ne $totalCases) {
        throw 'Inference timing case IDs are missing or duplicated across folds.'
    }
    if ($Smoke) {
        $smokeFold = [int]$Folds[0]
        $smokeFoldReport = Join-Path $ReportDirectory "folds\fold_$smokeFold"
        Copy-Item -LiteralPath (Join-Path $smokeFoldReport 'runtime.json') `
            -Destination (Join-Path $ReportDirectory 'runtime.json') -Force
        Copy-Item -LiteralPath (Join-Path $smokeFoldReport 'gpu_summary.json') `
            -Destination (Join-Path $ReportDirectory 'gpu_summary.json') -Force
    }
    else {
        Invoke-ResearchPython -Arguments @(
            '-m', 'glioma_seg.reporting.crossval',
            '--experiment-dir', $ReportDirectory,
            '--expected-epochs-per-fold', '100'
        ) -LogPath (Join-Path $ReportDirectory 'logs\telemetry_aggregation.log')
    }
    $experiment = Get-Content -LiteralPath (Join-Path $ReportDirectory 'experiment.json') `
        -Raw | ConvertFrom-Json
    $inference = [ordered]@{
        stage = if ($Smoke) { 'segresnet_smoke_validation_inference' } else { 'segresnet_five_fold_oof_inference' }
        backend = 'segresnet'
        model_id = [string]$experiment.model_id
        total_seconds = $totalInferenceSeconds
        number_of_cases = $totalCases
        mean_seconds_per_case = $totalInferenceSeconds / $totalCases
        case_ids = $inferenceCaseIds
        timing_scope = 'fresh_complete_run'
        timing_details = 'validation sliding-window inference and original-space export'
        timing_comparable = $true
        tta_state = 'OFF'
        source_validation_summaries = $inferenceSources
    }
    Write-ResearchJsonAtomic `
        -Path (Join-Path $ReportDirectory 'inference_runtime.json') `
        -Value $inference
}

function Write-MedNeXtTelemetryAggregation {
    param(
        [Parameter(Mandatory = $true)][string]$ExperimentId,
        [Parameter(Mandatory = $true)][string]$ReportDirectory,
        [Parameter(Mandatory = $true)][int[]]$Folds,
        [Parameter(Mandatory = $true)][bool]$Smoke
    )
    $totalInferenceSeconds = 0.0
    $totalCases = 0
    $inferenceSources = @()
    $inferenceCaseIds = @()
    foreach ($fold in $Folds) {
        Sync-MedNeXtFoldEvidence `
            -ExperimentId $ExperimentId `
            -Fold $fold `
            -ReportDirectory $ReportDirectory
        $foldReport = Join-Path $ReportDirectory "folds\fold_$fold"
        $validationPath = Join-Path $foldReport 'validation_summary.json'
        if (-not (Test-Path -LiteralPath $validationPath -PathType Leaf)) {
            throw "Fold $fold validation telemetry is missing: $validationPath"
        }
        $validation = Get-Content -LiteralPath $validationPath -Raw | ConvertFrom-Json
        if (-not [bool]$validation.valid -or [int]$validation.case_count -lt 1) {
            throw "Fold $fold validation telemetry is invalid: $validationPath"
        }
        $totalInferenceSeconds += [double]$validation.inference_total_seconds
        $totalCases += [int]$validation.case_count
        $inferenceSources += $validationPath
        $inferenceCaseIds += @($validation.case_ids | ForEach-Object { [string]$_ })
    }
    if ($totalCases -lt 1 -or $inferenceCaseIds.Count -ne $totalCases -or
        @($inferenceCaseIds | Select-Object -Unique).Count -ne $totalCases) {
        throw 'Inference timing case IDs are missing or duplicated across folds.'
    }
    if ($Smoke) {
        $smokeFold = [int]$Folds[0]
        $smokeFoldReport = Join-Path $ReportDirectory "folds\fold_$smokeFold"
        Copy-Item -LiteralPath (Join-Path $smokeFoldReport 'runtime.json') `
            -Destination (Join-Path $ReportDirectory 'runtime.json') -Force
        Copy-Item -LiteralPath (Join-Path $smokeFoldReport 'gpu_summary.json') `
            -Destination (Join-Path $ReportDirectory 'gpu_summary.json') -Force
    }
    else {
        Invoke-ResearchPython -Arguments @(
            '-m', 'glioma_seg.reporting.crossval',
            '--experiment-dir', $ReportDirectory,
            '--expected-epochs-per-fold', '100'
        ) -LogPath (Join-Path $ReportDirectory 'logs\telemetry_aggregation.log')
    }
    $experiment = Get-Content -LiteralPath (Join-Path $ReportDirectory 'experiment.json') `
        -Raw | ConvertFrom-Json
    $inference = [ordered]@{
        stage = if ($Smoke) { 'mednext_smoke_validation_inference' } else { 'mednext_five_fold_oof_inference' }
        backend = 'mednext'
        model_id = [string]$experiment.model_id
        total_seconds = $totalInferenceSeconds
        number_of_cases = $totalCases
        mean_seconds_per_case = $totalInferenceSeconds / $totalCases
        case_ids = $inferenceCaseIds
        timing_scope = 'fresh_complete_run'
        timing_details = 'official MedNeXt sliding-window validation, NIfTI/NPZ export, and native validation evaluation'
        timing_comparable = $true
        tta_state = 'OFF'
        source_validation_summaries = $inferenceSources
    }
    Write-ResearchJsonAtomic `
        -Path (Join-Path $ReportDirectory 'inference_runtime.json') `
        -Value $inference
}

function Move-ResearchDerivedArtifacts {
    param(
        [Parameter(Mandatory = $true)][string]$ExperimentId,
        [Parameter(Mandatory = $true)][string]$ReportDirectory,
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string[]]$Paths
    )
    $existing = @($Paths | Where-Object { Test-Path -LiteralPath $_ })
    if ($existing.Count -eq 0) { return }
    $resolvedReport = [IO.Path]::GetFullPath($ReportDirectory).TrimEnd('\') + '\'
    $archive = Join-Path (
        Join-Path $script:ResearchCache $ExperimentId
    ) "derived_archive_$((Get-Date).ToUniversalTime().ToString('yyyyMMdd_HHmmss_fff'))\$Label"
    New-Item -ItemType Directory -Path $archive -Force | Out-Null
    foreach ($path in $existing) {
        $resolved = [IO.Path]::GetFullPath($path)
        if (-not $resolved.StartsWith($resolvedReport, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to archive a derived artifact outside the experiment report: $resolved"
        }
        Move-Item -LiteralPath $resolved -Destination (Join-Path $archive (Split-Path -Leaf $resolved))
    }
    Write-Warning "Incomplete derived $Label artifacts were preserved at $archive"
}
