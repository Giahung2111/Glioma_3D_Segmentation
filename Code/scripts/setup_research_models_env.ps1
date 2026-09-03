param(
    [switch]$Recreate
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$codeRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent $codeRoot
$environmentRoot = Join-Path $codeRoot '.venv-models'
$python = Join-Path $environmentRoot 'python.exe'
$requirements = Join-Path $codeRoot 'requirements-research-models.txt'
$lockFile = Join-Path $codeRoot 'requirements-research-models-lock.txt'
$mednextRoot = Join-Path $projectRoot 'External\MedNeXt'
$monaiRoot = Join-Path $projectRoot 'External\MONAI'
$modelZooRoot = Join-Path $projectRoot 'External\MONAI-model-zoo'

$expectedMednextCommit = '0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba'
$expectedMonaiCommit = '46a5272196a6c2590ca2589029eed8e4d56ff008'
$expectedModelZooCommit = 'b9e4d04bb2a073110bde9e5c05c9690241e938b6'

function Assert-CleanPinnedRepository {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$ExpectedCommit,
        [Parameter(Mandatory = $true)][string]$Name
    )
    if (-not (Test-Path -LiteralPath (Join-Path $Path '.git') -PathType Container)) {
        throw "$Name checkout is missing or is not a Git repository: $Path"
    }
    $actual = (& git -C $Path rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $actual -ne $ExpectedCommit) {
        throw "$Name must be pinned at $ExpectedCommit; found $actual"
    }
    $trackedChanges = @(& git -C $Path status --porcelain --untracked-files=no)
    if ($LASTEXITCODE -ne 0 -or $trackedChanges.Count -gt 0) {
        throw "$Name contains tracked modifications. Upstream source must remain unchanged: $Path"
    }
}

Assert-CleanPinnedRepository -Path $mednextRoot -ExpectedCommit $expectedMednextCommit -Name 'MedNeXt'
Assert-CleanPinnedRepository -Path $monaiRoot -ExpectedCommit $expectedMonaiCommit -Name 'MONAI 1.4.0'
Assert-CleanPinnedRepository -Path $modelZooRoot -ExpectedCommit $expectedModelZooCommit -Name 'MONAI Model Zoo'

if ($Recreate -and (Test-Path -LiteralPath $environmentRoot)) {
    $resolvedCode = [IO.Path]::GetFullPath($codeRoot)
    $resolvedEnvironment = [IO.Path]::GetFullPath($environmentRoot)
    if (-not $resolvedEnvironment.StartsWith($resolvedCode + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to recreate an environment outside Code: $resolvedEnvironment"
    }
    & conda env remove --prefix $environmentRoot --yes
    if ($LASTEXITCODE -ne 0) { throw 'Conda could not remove the existing research-model environment.' }
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-Host "Creating isolated research-model environment: $environmentRoot" -ForegroundColor Cyan
    & conda create --prefix $environmentRoot --yes python=3.10.18 pip
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed.' }
}

& $python -m pip install --disable-pip-version-check --upgrade pip==24.3.1
if ($LASTEXITCODE -ne 0) { throw 'Pinned pip bootstrap failed.' }

Write-Host 'Installing the Windows-verified CUDA 12.1 compatibility stack...' -ForegroundColor Cyan
& $python -m pip install --disable-pip-version-check `
    --index-url https://download.pytorch.org/whl/cu121 `
    torch==2.5.1 torchvision==0.20.1
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }

& $python -m pip install --disable-pip-version-check -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'Pinned research dependency installation failed.' }

# Both editable installs are project/upstream boundaries: MedNeXt supplies the
# official model/trainer; glioma-seg supplies orchestration and evaluation.
& $python -m pip install --disable-pip-version-check --no-deps -e $mednextRoot
if ($LASTEXITCODE -ne 0) { throw 'Editable MedNeXt installation failed.' }
& $python -m pip install --disable-pip-version-check -e "$codeRoot[dev]"
if ($LASTEXITCODE -ne 0) { throw 'Editable glioma-seg installation failed.' }

& $python -c @'
import importlib.metadata as md
import sys
import torch
import monai
import nnunet_mednext

assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.startswith('2.5.1+cu121'), torch.__version__
assert monai.__version__ == '1.4.0', monai.__version__
assert md.version('mednextv1') == '1.7.0', md.version('mednextv1')
assert torch.cuda.is_available(), 'CUDA is not available to PyTorch'
print(f'Python={sys.version.split()[0]}')
print(f'PyTorch={torch.__version__}; CUDA={torch.version.cuda}; GPU={torch.cuda.get_device_name(0)}')
print('MONAI=%s; MedNeXt=%s' % (monai.__version__, md.version('mednextv1')))
'@
if ($LASTEXITCODE -ne 0) { throw 'Research-model environment verification failed.' }

& $python -m pip freeze --all | Set-Content -LiteralPath $lockFile -Encoding utf8
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $lockFile -PathType Leaf)) {
    throw 'Dependency lock generation failed.'
}

Write-Host 'Research-model environment is ready.' -ForegroundColor Green
Write-Host "Python: $python"
Write-Host "Dependency lock: $lockFile"
Write-Host 'No manual activation is required by the model pipeline scripts.' -ForegroundColor Yellow
