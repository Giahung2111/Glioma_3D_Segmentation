"""Issue and verify the mandatory MedNeXt real-data smoke-test gate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from glioma_seg.backends.mednext.config import EXPECTED_MEDNEXT_COMMIT, load_recipe
from glioma_seg.monitoring.timing import write_json_atomic
from glioma_seg.reporting.model_bundle import audit_model_report_bundle
from glioma_seg.utils.hashing import sha256_file

SMOKE_GATE_SCHEMA = "glioma_mednext_smoke_gate_v1"
SMOKE_GATE_FILENAME = "mednext_smoke_gate.json"


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _git_head(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    return completed.stdout.strip()


def _paths(project_root: Path) -> dict[str, Path]:
    root = project_root.resolve()
    return {
        "root": root,
        "model_config": root / "Code" / "configs" / "models" / "mednext.yaml",
        "split": root
        / "Workspace"
        / "nnUNet_preprocessed"
        / "Dataset501_BraTS2023GLI"
        / "splits_final.json",
        "reports": root / "Workspace" / "reports",
        "upstream": root / "External" / "MedNeXt",
    }


def issue_smoke_gate(project_root: Path, experiment_id: str) -> Path:
    """Publish a global receipt only after the complete smoke bundle audits."""

    paths = _paths(project_root)
    report_dir = paths["reports"] / experiment_id
    manifest_path = report_dir / "report_manifest.json"
    published = _load_object(manifest_path)
    audited = audit_model_report_bundle(
        report_dir,
        expected_case_count=2,
        expected_folds=(0,),
    )
    if published != audited:
        raise ValueError("Published smoke report manifest no longer matches its artifacts")
    if (
        audited.get("valid") is not True
        or audited.get("experiment_kind") != "smoke"
        or audited.get("backend") != "mednext"
        or audited.get("model_id") != "mednext_v1_s_kernel3"
        or audited.get("is_final_baseline") is not False
    ):
        raise ValueError("Report bundle is not a valid non-final MedNeXt smoke test")

    recipe = load_recipe(paths["model_config"])
    environment_path = report_dir / "environment.json"
    environment = _load_object(environment_path)
    memory_path = report_dir / "memory_preflight.json"
    memory = _load_object(memory_path)
    if (
        memory.get("valid") is not True
        or memory.get("model") != recipe.model_id
        or memory.get("patch_size") != [128, 128, 128]
        or memory.get("official_loss_and_augmentation") is not True
        or memory.get("dedicated_vram_fit") is not True
    ):
        raise ValueError("Smoke memory preflight is missing, incompatible, or did not fit")
    upstream_commit = _git_head(paths["upstream"])
    if upstream_commit != EXPECTED_MEDNEXT_COMMIT:
        raise ValueError(
            f"MedNeXt commit mismatch: expected={EXPECTED_MEDNEXT_COMMIT}, actual={upstream_commit}"
        )

    receipt = {
        "schema": SMOKE_GATE_SCHEMA,
        "valid": True,
        "backend": "mednext",
        "model_id": recipe.model_id,
        "smoke_experiment_id": experiment_id,
        "smoke_report_directory": str(report_dir.resolve()),
        "report_manifest": str(manifest_path.resolve()),
        "report_manifest_sha256": sha256_file(manifest_path),
        "model_config_sha256": recipe.source_sha256,
        "canonical_split": str(paths["split"].resolve()),
        "canonical_split_sha256": sha256_file(paths["split"]),
        "upstream_commit": upstream_commit,
        "environment_json_sha256": sha256_file(environment_path),
        "memory_preflight_sha256": sha256_file(memory_path),
        "gpu_name": environment.get("gpu_name", environment.get("gpu")),
        "gpu_vram_mb": environment.get("gpu_vram_mb"),
        "torch_version": environment.get("torch_version", environment.get("torch")),
        "cuda_runtime": environment.get("cuda_runtime", environment.get("cuda")),
        "forced_stop_resume_verified_by_report_audit": True,
        "issued_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    destination = paths["reports"] / SMOKE_GATE_FILENAME
    write_json_atomic(destination, receipt)
    return destination


def verify_smoke_gate(
    project_root: Path,
    *,
    current_environment: Path | None = None,
) -> dict[str, Any]:
    """Fail closed if code, split, GPU identity, or smoke evidence changed."""

    paths = _paths(project_root)
    gate_path = paths["reports"] / SMOKE_GATE_FILENAME
    receipt = _load_object(gate_path)
    if (
        receipt.get("schema") != SMOKE_GATE_SCHEMA
        or receipt.get("valid") is not True
        or receipt.get("backend") != "mednext"
        or receipt.get("model_id") != "mednext_v1_s_kernel3"
    ):
        raise ValueError("MedNeXt smoke gate identity is invalid")
    recipe = load_recipe(paths["model_config"])
    checks = {
        "model_config": recipe.source_sha256 == receipt.get("model_config_sha256"),
        "canonical_split": sha256_file(paths["split"]) == receipt.get("canonical_split_sha256"),
        "upstream_commit": _git_head(paths["upstream"]) == receipt.get("upstream_commit"),
    }
    manifest_path = Path(str(receipt["report_manifest"])).resolve()
    checks["report_manifest_hash"] = manifest_path.is_file() and sha256_file(
        manifest_path
    ) == receipt.get("report_manifest_sha256")
    report_dir = Path(str(receipt["smoke_report_directory"])).resolve()
    audited = audit_model_report_bundle(
        report_dir,
        expected_case_count=2,
        expected_folds=(0,),
    )
    checks["report_bundle"] = (
        audited.get("valid") is True and audited.get("experiment_kind") == "smoke"
    )
    checks["expected_upstream_commit"] = receipt.get("upstream_commit") == EXPECTED_MEDNEXT_COMMIT
    if current_environment is not None:
        current = _load_object(current_environment.resolve())
        checks["same_gpu_name"] = current.get("gpu_name", current.get("gpu")) == receipt.get(
            "gpu_name"
        )
        checks["same_gpu_vram"] = current.get("gpu_vram_mb") == receipt.get("gpu_vram_mb")
    failed = sorted(name for name, valid in checks.items() if not valid)
    if failed:
        raise ValueError(f"MedNeXt smoke gate failed checks: {failed}")
    return {
        "valid": True,
        "gate": str(gate_path.resolve()),
        "smoke_experiment_id": receipt["smoke_experiment_id"],
        "checks": checks,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    issue = subparsers.add_parser("issue")
    issue.add_argument("--experiment-id", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--current-environment", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "issue":
        destination = issue_smoke_gate(args.project_root, args.experiment_id)
        result: dict[str, Any] = {"valid": True, "gate": str(destination.resolve())}
    elif args.command == "verify":
        result = verify_smoke_gate(
            args.project_root,
            current_environment=args.current_environment,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SMOKE_GATE_FILENAME",
    "SMOKE_GATE_SCHEMA",
    "issue_smoke_gate",
    "verify_smoke_gate",
]
