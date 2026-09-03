param(
    [string]$ExperimentId = '',
    [string]$CrossvalSummaryJson = ''
)

. (Join-Path $PSScriptRoot '_common.ps1')
Write-GliomaStage -Number '8/8' -Name 'Report Generation'

if ([string]::IsNullOrWhiteSpace($ExperimentId)) { throw 'ExperimentId is required.' }
$reportDirectory = Get-GliomaReportDirectory -ExperimentId $ExperimentId
if (-not [string]::IsNullOrWhiteSpace($CrossvalSummaryJson)) {
    $CrossvalSummaryJson = [IO.Path]::GetFullPath($CrossvalSummaryJson)
    if (-not (Test-Path -LiteralPath $CrossvalSummaryJson -PathType Leaf)) {
        throw "Cross-validation report summary is missing: $CrossvalSummaryJson"
    }
}
$required = @(
    (Join-Path $reportDirectory 'experiment.json'),
    (Join-Path $reportDirectory 'metrics_summary.csv')
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required report input missing: $path" }
}
$globalReportDirectory = Join-Path $script:GliomaWorkspace 'reports'
Invoke-GliomaPython -Arguments @(
    '-m', 'glioma_seg.reporting.bundle',
    '--workspace-reports', $globalReportDirectory,
    '--experiment-dir', $reportDirectory,
    '--phase', 'prepare'
)

$arguments = @(
    '-m', 'glioma_seg.reporting.report',
    '--output-dir', $reportDirectory,
    '--experiment-json', (Join-Path $reportDirectory 'experiment.json'),
    '--metrics-summary-csv', (Join-Path $reportDirectory 'metrics_summary.csv')
)
$optionalArguments = @(
    @('--environment-json', (Join-Path $reportDirectory 'environment.json')),
    @('--runtime-json', (Join-Path $reportDirectory 'runtime.json')),
    @('--inference-runtime-json', (Join-Path $reportDirectory 'inference_runtime.json')),
    @('--gpu-summary-json', (Join-Path $reportDirectory 'gpu_summary.json')),
    @('--data-validation-json', (Join-Path $reportDirectory 'data_validation.json')),
    @('--official-validation-json', (Join-Path $reportDirectory 'official_validation_data_validation.json')),
    @('--preprocessing-artifacts-json', (Join-Path $reportDirectory 'preprocessing_artifacts.json')),
    @('--official-status-json', (Join-Path $reportDirectory 'official_brats_metrics_status.json')),
    @('--official-summary-csv', (Join-Path $reportDirectory 'official_lesionwise_metrics_summary.csv')),
    @('--evaluation-protocol-json', (Join-Path $reportDirectory 'evaluation_protocol.json')),
    @('--failure-cases-csv', (Join-Path $reportDirectory 'failure_cases.csv')),
    @('--figures-dir', (Join-Path $reportDirectory 'figures')),
    @('--figures-manifest-csv', (Join-Path $reportDirectory 'figures\figures_manifest.csv'))
)
foreach ($pair in $optionalArguments) {
    if (Test-Path -LiteralPath $pair[1]) { $arguments += $pair }
}
if (-not [string]::IsNullOrWhiteSpace($CrossvalSummaryJson)) {
    # Append directly. Wrapping one nested pair in @() causes Windows PowerShell
    # 5.1 to flatten it into two strings, silently dropping this CLI argument.
    $arguments += @('--crossval-summary-json', $CrossvalSummaryJson)
}
Invoke-GliomaPython -Arguments $arguments

$summaryPath = Join-Path $reportDirectory 'summary.md'
$weeklyPath = Join-Path $reportDirectory 'weekly_discussion.md'
foreach ($path in @($summaryPath, $weeklyPath)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Report generator did not create: $path" }
    $text = Get-Content -LiteralPath $path -Raw
    if ($text -match '(?m)\bTODO\b|<placeholder>|\.\.\.') { throw "Report contains a placeholder token: $path" }
}
Invoke-GliomaPython -Arguments @(
    '-m', 'glioma_seg.reporting.bundle',
    '--workspace-reports', $globalReportDirectory,
    '--experiment-dir', $reportDirectory,
    '--phase', 'finalize'
)
Write-Host "Summary: $summaryPath" -ForegroundColor Green
Write-Host "Weekly discussion: $weeklyPath" -ForegroundColor Green
