param(
    [string]$ExperimentId = '',
    [switch]$IncludeOfficialValidation,
    [switch]$ForceCopy
)

. (Join-Path $PSScriptRoot '_common.ps1')
Write-GliomaStage -Number '3/8' -Name 'nnU-Net Conversion'

if ([string]::IsNullOrWhiteSpace($ExperimentId)) {
    $ExperimentId = New-GliomaExperimentId -Kind prelim
}
$dataRoot = Join-Path $script:GliomaProjectRoot 'Datasets'
$globalReportDirectory = Join-Path $script:GliomaWorkspace 'reports'
$validationJson = Join-Path $globalReportDirectory 'data_validation.json'
if (-not (Test-Path -LiteralPath $validationJson -PathType Leaf)) {
    throw "Validated training manifest missing: $validationJson. Run 01_validate_raw_data.ps1 first."
}
$reportDirectory = Get-GliomaReportDirectory -ExperimentId $ExperimentId
$conversionReport = Join-Path $reportDirectory 'nnunet_conversion.json'
$arguments = @(
    '-m', 'glioma_seg.data.nnunet_conversion',
    '--data-root', $dataRoot,
    '--output-root', $Env:nnUNet_raw,
    '--validation-json', $validationJson,
    '--dataset-config', (Join-Path $script:GliomaCodeRoot 'configs\datasets\brats2023_gli.yaml'),
    '--report-json', $conversionReport,
    '--expected-training-cases', '1251'
)
if ($IncludeOfficialValidation) {
    $officialJson = Join-Path $globalReportDirectory 'official_validation_data_validation.json'
    if (-not (Test-Path -LiteralPath $officialJson -PathType Leaf)) {
        throw "Official validation manifest missing: $officialJson"
    }
    $arguments += @('--include-validation', '--official-validation-json', $officialJson)
}
if ($ForceCopy) { $arguments += '--force-copy' }

Invoke-GliomaPython -Arguments $arguments
Write-Host "nnU-Net raw dataset is ready under $Env:nnUNet_raw" -ForegroundColor Green
