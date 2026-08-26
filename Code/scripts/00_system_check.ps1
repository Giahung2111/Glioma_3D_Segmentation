param([string]$ExperimentId = '')

. (Join-Path $PSScriptRoot '_common.ps1')
Write-GliomaStage -Number '1/8' -Name 'System Check'

if ([string]::IsNullOrWhiteSpace($ExperimentId)) {
    $ExperimentId = New-GliomaExperimentId -Kind system
}
$reportDirectory = Get-GliomaReportDirectory -ExperimentId $ExperimentId
$environmentPath = Join-Path $reportDirectory 'environment.json'
$lockPath = Join-Path $script:GliomaCodeRoot 'requirements-lock.txt'

Invoke-GliomaPython -Arguments @(
    '-m', 'glioma_seg.backends.nnunet.backend',
    '--project-root', $script:GliomaProjectRoot,
    'system-check',
    '--output', $environmentPath,
    '--lock-output', $lockPath
)

$report = Get-Content -LiteralPath $environmentPath -Raw | ConvertFrom-Json
$criticalFailures = [System.Collections.Generic.List[string]]::new()
if (-not $report.torch_details.installed) { $criticalFailures.Add('PyTorch is not installed in Code\.venv.') }
if ($report.torch_details.installed -and -not $report.torch_details.cuda_available) { $criticalFailures.Add('torch.cuda.is_available() is false.') }
if ([string]$report.torch_details.version -like '2.9*') { $criticalFailures.Add('PyTorch 2.9.* is rejected because of the documented 3D convolution + AMP regression.') }
if (-not $report.nnunet.installed) { $criticalFailures.Add('nnunetv2 is not installed.') }
if (-not $report.nnunet.editable) { $criticalFailures.Add('nnunetv2 is not installed in editable mode.') }
if (-not $report.safety_checks.official_upstream_commit) { $criticalFailures.Add('External\nnUNet is not at the pinned upstream commit.') }
if (-not $report.safety_checks.upstream_unmodified) { $criticalFailures.Add('External\nnUNet has local source changes; it cannot be used as the standard baseline.') }
$gpuAvailableProperty = $report.gpu_details.PSObject.Properties['available']
if ($null -ne $gpuAvailableProperty -and -not [bool]$gpuAvailableProperty.Value) {
    $criticalFailures.Add('NVIDIA GPU telemetry is unavailable.')
}
if ([int64]$report.disk.free_bytes -lt 50GB) { $criticalFailures.Add('Less than 50 GiB is free on the project drive.') }

if ($criticalFailures.Count -gt 0) {
    Write-Host 'SYSTEM CHECK FAILED' -ForegroundColor Red
    $criticalFailures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    throw "System is not ready. See $environmentPath"
}

Write-Host "System check passed. Environment: $environmentPath" -ForegroundColor Green
Write-Host "Dependency lock: $lockPath"
