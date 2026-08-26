param(
    [string]$ExperimentId = '',
    [switch]$IncludeOfficialValidation
)

. (Join-Path $PSScriptRoot '_common.ps1')
Write-GliomaStage -Number '2/8' -Name 'Dataset Validation'

if ([string]::IsNullOrWhiteSpace($ExperimentId)) {
    $ExperimentId = New-GliomaExperimentId -Kind prelim
}
$dataRoot = Join-Path $script:GliomaProjectRoot 'Datasets'
$globalReportDirectory = Join-Path $script:GliomaWorkspace 'reports'
$reportDirectory = Get-GliomaReportDirectory -ExperimentId $ExperimentId
$trainingJson = Join-Path $globalReportDirectory 'data_validation.json'
$trainingCsv = Join-Path $globalReportDirectory 'data_validation.csv'

Invoke-GliomaPython -Arguments @(
    '-m', 'glioma_seg.data.validate',
    '--data-root', $dataRoot,
    '--kind', 'training',
    '--output-json', $trainingJson,
    '--output-csv', $trainingCsv,
    '--expected-training-cases', '1251'
)
Copy-Item -LiteralPath $trainingJson -Destination (Join-Path $reportDirectory 'data_validation.json') -Force
Copy-Item -LiteralPath $trainingCsv -Destination (Join-Path $reportDirectory 'data_validation.csv') -Force

if ($IncludeOfficialValidation) {
    $officialJson = Join-Path $globalReportDirectory 'official_validation_data_validation.json'
    $officialCsv = Join-Path $globalReportDirectory 'official_validation_data_validation.csv'
    Invoke-GliomaPython -Arguments @(
        '-m', 'glioma_seg.data.validate',
        '--data-root', $dataRoot,
        '--kind', 'validation',
        '--output-json', $officialJson,
        '--output-csv', $officialCsv,
        '--expected-validation-cases', '219'
    )
    Copy-Item -LiteralPath $officialJson -Destination (Join-Path $reportDirectory 'official_validation_data_validation.json') -Force
    Copy-Item -LiteralPath $officialCsv -Destination (Join-Path $reportDirectory 'official_validation_data_validation.csv') -Force
}

Write-Host "Raw training validation passed: $trainingJson" -ForegroundColor Green
