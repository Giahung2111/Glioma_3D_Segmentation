param(
    [string]$ExperimentId = '',
    [switch]$Force
)

. (Join-Path $PSScriptRoot '_common.ps1')
Write-GliomaStage -Number '4/8' -Name 'Planning & Preprocessing'

if ([string]::IsNullOrWhiteSpace($ExperimentId)) {
    $ExperimentId = New-GliomaExperimentId -Kind prelim
}
$datasetDirectory = Join-Path $Env:nnUNet_preprocessed 'Dataset501_BraTS2023GLI'
$rawDatasetDirectory = Join-Path $Env:nnUNet_raw 'Dataset501_BraTS2023GLI'
$reportDirectory = Get-GliomaReportDirectory -ExperimentId $ExperimentId
$artifactReport = Join-Path $reportDirectory 'preprocessing_artifacts.json'
$lockPath = Join-Path $script:GliomaWorkspace 'cache\nnunet_preprocess.lock'
$preprocessingLock = $null
try {
    try {
        $preprocessingLock = [System.IO.File]::Open(
            $lockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    }
    catch {
        throw "Another project preprocessing audit/run holds the exclusive lock at $lockPath. Wait for it to finish; a second preprocessing process was not started."
    }

    $lockText = "pid=$PID`nstarted_utc=$([DateTime]::UtcNow.ToString('o'))`n"
    $lockBytes = [System.Text.Encoding]::UTF8.GetBytes($lockText)
    $preprocessingLock.SetLength(0)
    $preprocessingLock.Write($lockBytes, 0, $lockBytes.Length)
    $preprocessingLock.Flush($true)

    if (-not $Force) {
        $artifactAuditArguments = @(
            '-m', 'glioma_seg.backends.nnunet.artifacts',
            '--raw-dataset-dir', $rawDatasetDirectory,
            '--preprocessed-dataset-dir', $datasetDirectory,
            '--dataset-name', 'Dataset501_BraTS2023GLI',
            '--configuration', '3d_fullres',
            '--plans-name', 'nnUNetPlans',
            '--expected-case-count', '1251',
            '--ensure-splits',
            '--output', $artifactReport
        )
        & $script:GliomaPython @artifactAuditArguments
        $artifactAuditExitCode = $LASTEXITCODE
        if ($artifactAuditExitCode -eq 0) {
            Write-Host 'Complete official preprocessing artifacts already exist; leaving them unchanged. Use -Force only for an explicit rerun.' -ForegroundColor Yellow
            return
        }
        Write-Host 'Preprocessing is absent or incomplete; checking for an active official process.' -ForegroundColor Yellow
    }

    try {
        $activePreprocessing = @(
            Get-CimInstance -ClassName Win32_Process -ErrorAction Stop |
                Where-Object {
                    $_.ProcessId -ne $PID -and
                    $null -ne $_.CommandLine -and
                    (
                        $_.CommandLine -match '(?i)nnUNetv2_plan_and_preprocess' -or
                        $_.CommandLine -match '(?i)glioma_seg\.backends\.nnunet\.backend.+\bpreprocess\b'
                    )
                }
        )
    }
    catch {
        throw "Unable to verify that no nnU-Net preprocessing process is active. A second process was not started: $($_.Exception.Message)"
    }
    if ($activePreprocessing.Count -gt 0) {
        $activeDescription = ($activePreprocessing | ForEach-Object {
            "PID=$($_.ProcessId) $($_.Name)"
        }) -join ', '
        throw "nnU-Net preprocessing is already active ($activeDescription). Wait for it to finish; a second process was not started."
    }

    Invoke-GliomaPython -Arguments @(
        '-m', 'glioma_seg.backends.nnunet.backend',
        '--project-root', $script:GliomaProjectRoot,
        'preprocess',
        '--experiment-id', $ExperimentId
    )
    Write-Host "Planning and preprocessing passed for Dataset501_BraTS2023GLI." -ForegroundColor Green
}
finally {
    if ($null -ne $preprocessingLock) {
        $preprocessingLock.Dispose()
    }
}
