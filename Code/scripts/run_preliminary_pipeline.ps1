param(
    [string]$ExperimentId = '',
    [switch]$Resume,
    [switch]$IncludeOfficialValidation,
    [switch]$AllowLowGpuUtilization
)

. (Join-Path $PSScriptRoot '_common.ps1')
if ([string]::IsNullOrWhiteSpace($ExperimentId)) {
    $ExperimentId = New-GliomaExperimentId -Kind prelim -Fold 0
}
Write-Host "Experiment: $ExperimentId" -ForegroundColor Magenta

& (Join-Path $PSScriptRoot '00_system_check.ps1') -ExperimentId $ExperimentId
if ($LASTEXITCODE -ne 0) { throw 'System check failed.' }

& (Join-Path $PSScriptRoot '01_validate_raw_data.ps1') `
    -ExperimentId $ExperimentId `
    -IncludeOfficialValidation:$IncludeOfficialValidation
if ($LASTEXITCODE -ne 0) { throw 'Dataset validation failed.' }

& (Join-Path $PSScriptRoot '02_prepare_nnunet.ps1') `
    -ExperimentId $ExperimentId `
    -IncludeOfficialValidation:$IncludeOfficialValidation
if ($LASTEXITCODE -ne 0) { throw 'nnU-Net conversion failed.' }

& (Join-Path $PSScriptRoot '03_plan_and_preprocess.ps1') -ExperimentId $ExperimentId
if ($LASTEXITCODE -ne 0) { throw 'Planning/preprocessing failed.' }

& (Join-Path $PSScriptRoot '04_benchmark_gpu.ps1') -ExperimentId $ExperimentId -Resume:$Resume
if ($LASTEXITCODE -ne 0) { throw 'GPU benchmark failed.' }

& (Join-Path $PSScriptRoot '05_train_preliminary.ps1') `
    -ExperimentId $ExperimentId `
    -Resume:$Resume `
    -AllowLowGpuUtilization:$AllowLowGpuUtilization
if ($LASTEXITCODE -ne 0) { throw 'Preliminary training failed.' }

& (Join-Path $PSScriptRoot '06_evaluate_preliminary.ps1') `
    -ExperimentId $ExperimentId `
    -ResumeInference:$Resume
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed.' }

& (Join-Path $PSScriptRoot '07_analyze_failures.ps1') -ExperimentId $ExperimentId
if ($LASTEXITCODE -ne 0) { throw 'Failure analysis failed.' }

& (Join-Path $PSScriptRoot '08_generate_report.ps1') -ExperimentId $ExperimentId
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }

$reportDirectory = Get-GliomaReportDirectory -ExperimentId $ExperimentId
Write-Host ('=' * 40) -ForegroundColor Green
Write-Host 'PRELIMINARY BASELINE COMPLETE' -ForegroundColor Green
Write-Host ('=' * 40) -ForegroundColor Green
Write-Host "Experiment: $ExperimentId"
Write-Host 'Dataset: BraTS 2023 Adult Glioma Pre-Treatment'
Write-Host 'Model: nnU-Net v2 3d_fullres'
Write-Host 'Fold: 0'
Write-Host "Report: $(Join-Path $reportDirectory 'summary.md')"
Write-Host "Checkpoint metadata: $(Join-Path $reportDirectory 'experiment.json')"
Write-Host ('=' * 40) -ForegroundColor Green
