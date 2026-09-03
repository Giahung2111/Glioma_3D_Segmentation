from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMMON_SCRIPT = ROOT / "Code" / "scripts" / "_research_models_common.ps1"
SEGRESNET_SCRIPT = ROOT / "Code" / "scripts" / "run_segresnet_cv_pipeline.ps1"
MEDNEXT_SCRIPT = ROOT / "Code" / "scripts" / "run_mednext_cv_pipeline.ps1"


@pytest.mark.parametrize("script", (COMMON_SCRIPT, SEGRESNET_SCRIPT, MEDNEXT_SCRIPT))
def test_research_model_powershell_has_no_parse_errors(script: Path) -> None:
    executable = shutil.which("powershell.exe") or shutil.which("pwsh")
    if executable is None:
        pytest.skip("PowerShell parser is unavailable on this platform")
    escaped = str(script.resolve()).replace("'", "''")
    command = (
        "$errors=$null;$tokens=$null;"
        f"[Management.Automation.Language.Parser]::ParseFile('{escaped}',"
        "[ref]$tokens,[ref]$errors)|Out-Null;"
        "if($errors.Count){$errors|ForEach-Object{Write-Error $_.Message};exit 1}"
    )

    completed = subprocess.run(
        (executable, "-NoProfile", "-NonInteractive", "-Command", command),
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_segresnet_one_command_contract_is_explicit_and_fail_closed() -> None:
    common = COMMON_SCRIPT.read_text(encoding="utf-8")
    pipeline = SEGRESNET_SCRIPT.read_text(encoding="utf-8")

    assert "$script:ResearchPython" in common
    assert ".venv-models\\python.exe" in common
    assert "source/bin/activate" not in pipeline
    assert "[switch]$ConfirmRun" in pipeline
    assert "[Parameter(Mandatory = $true)][switch]$ConfirmRun" not in pipeline
    assert "$folds = if ($SmokeTest) { @(0) } else { @(0, 1, 2, 3, 4) }" in pipeline
    assert "$targetEpochs = if ($SmokeTest) { 3 } else { 100 }" in pipeline
    assert "'--stop-after-epoch', '1'" in pipeline
    assert "'--resume'" in pipeline
    assert "'--failure-cases-csv'" in pipeline
    assert "'glioma_seg.reporting.model_bundle'" in pipeline
    assert "'model_bundle.log'" not in pipeline
    assert (
        "$splits = Get-Content -LiteralPath $splitPath -Raw | ConvertFrom-Json"
        in common
    )
    assert (
        "$splits = @(Get-Content -LiteralPath $splitPath -Raw | ConvertFrom-Json)"
        not in common
    )
    assert "[IO.File]::Replace($temporary, $Path, $backup, $true)" in common
    assert "[IO.File]::Replace($temporary, $Path, $null)" not in common
    assert "-Filter \".$fileName.*.$suffix\"" in common


def test_official_metrics_environment_is_verified_before_gpu_training() -> None:
    common = COMMON_SCRIPT.read_text(encoding="utf-8")
    pipeline = SEGRESNET_SCRIPT.read_text(encoding="utf-8")

    assert "brats2023_metrics_env" in common
    assert "43c905242b2eecf421d4ab2da7af8ece9777d322" in common
    assert "function Assert-OfficialBraTSEvaluatorReady" in common
    assert "'--python', $script:ResearchOfficialMetricsPython" in pipeline
    assert pipeline.index("Assert-OfficialBraTSEvaluatorReady") < pipeline.index(
        "Write-ResearchStage -Number '6/12'"
    )


def test_mednext_one_command_contract_is_explicit_and_fail_closed() -> None:
    common = COMMON_SCRIPT.read_text(encoding="utf-8")
    pipeline = MEDNEXT_SCRIPT.read_text(encoding="utf-8")

    assert "source/bin/activate" not in pipeline
    assert "[switch]$ConfirmRun" in pipeline
    assert "$folds = if ($SmokeTest) { @(0) } else { @(0, 1, 2, 3, 4) }" in pipeline
    assert "$targetEpochs = if ($SmokeTest) { 3 } else { 100 }" in pipeline
    assert "$quickTrainCases = 8" in pipeline
    assert "$quickValidationCases = 2" in pipeline
    assert "Task951_BraTS2023GLISmoke" in pipeline
    assert "Task501_BraTS2023GLI" in pipeline
    assert '"${ExperimentId}_memory_gate"' in pipeline
    assert "'--stop-after-epoch', '1'" in pipeline
    assert "'glioma_seg.backends.mednext.backend'" in pipeline
    assert "'glioma_seg.reporting.model_bundle'" in pipeline
    assert "'model_bundle.log'" not in pipeline
    assert "Enter-ResearchGpuLocks -ExperimentId $ExperimentId -Backend mednext" in pipeline
    assert "function Get-MedNeXtResultDirectory" in common
    assert "function Invoke-MedNeXtFoldAudit" in common
    assert "function Write-MedNeXtTelemetryAggregation" in common
    assert pipeline.index("Assert-OfficialBraTSEvaluatorReady") < pipeline.index(
        "Write-ResearchStage -Number '6/12'"
    )
    assert pipeline.index("'preprocess'") < pipeline.index(
        "Write-ResearchStage -Number '6/12'"
    )
    assert pipeline.index("'memory-preflight'") < pipeline.index(
        "Write-ResearchStage -Number '6/12'"
    )
