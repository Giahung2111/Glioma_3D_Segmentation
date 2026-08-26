param()

. (Join-Path $PSScriptRoot '_common.ps1')

Write-Host "nnUNet_raw=$Env:nnUNet_raw"
Write-Host "nnUNet_preprocessed=$Env:nnUNet_preprocessed"
Write-Host "nnUNet_results=$Env:nnUNet_results"
Write-Host 'Environment variables are set for this PowerShell process. Dot-source this script to retain them in the calling terminal.'
