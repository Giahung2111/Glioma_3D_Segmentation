param()

. (Join-Path $PSScriptRoot '_common.ps1')

Write-Host "nnUNet_raw=$Env:nnUNet_raw"
Write-Host "nnUNet_preprocessed=$Env:nnUNet_preprocessed"
Write-Host "nnUNet_results=$Env:nnUNet_results"
Write-Host "Project Python=$script:GliomaPython"
if ($null -ne $Env:CONDA_PREFIX -and
    [IO.Path]::GetFullPath($Env:CONDA_PREFIX) -eq [IO.Path]::GetFullPath((Join-Path $script:GliomaCodeRoot '.venv'))) {
    Write-Host 'Conda project environment is active.' -ForegroundColor Green
}
else {
    Write-Host 'The project scripts do not require manual activation; they call Code\.venv\python.exe directly.' -ForegroundColor Yellow
    Write-Host 'For interactive Python/nnU-Net commands, initialize Conda and activate Code\.venv first (see Code\docs\pipeline.md).'
}
Write-Host 'Environment variables are set for this PowerShell process. Dot-source this script to retain them in the calling terminal.'
