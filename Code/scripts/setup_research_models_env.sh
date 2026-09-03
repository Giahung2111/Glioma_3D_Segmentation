#!/usr/bin/env bash
# Create the isolated Linux environments used by the MedNeXt/SegResNet research pipelines.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
PROJECT_ROOT="$(cd -- "$CODE_ROOT/.." && pwd -P)"
MODEL_ENV="$CODE_ROOT/.venv-models"
EVALUATOR_ENV="$CODE_ROOT/.venv-brats-metrics"
EVALUATOR_PYTHON_VERSION="3.9.23"
RECREATE=0
BOOTSTRAP_PYTHON="$(command -v python3 || command -v python || true)"
EVALUATOR_BOOTSTRAP=""

usage() {
    cat <<'EOF'
Usage:
  bash Code/scripts/setup_research_models_env.sh [options]

Options:
  --recreate                 Recreate both owned environments after path checks.
  --python PATH              Python 3.10 executable for the model environment.
  --evaluator-python PATH    Python 3.9 executable for the official evaluator.
  -h, --help                 Show this help.

This script never uses sudo and never installs into JupyterHub's shared Python.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

while (($#)); do
    case "$1" in
        --recreate)
            RECREATE=1
            shift
            ;;
        --python)
            (($# >= 2)) || die "--python requires a path"
            BOOTSTRAP_PYTHON="$2"
            shift 2
            ;;
        --evaluator-python)
            (($# >= 2)) || die "--evaluator-python requires a path"
            EVALUATOR_BOOTSTRAP="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

[[ "$(id -u)" -ne 0 ]] || die "Run as your normal JupyterHub user, not root/sudo."
[[ -n "$BOOTSTRAP_PYTHON" && -x "$BOOTSTRAP_PYTHON" ]] || die "Python was not found."
[[ -d "$PROJECT_ROOT/Workspace" ]] || die \
    "Workspace is missing. Create the persistent Workspace symlink before setup."
[[ -d "$PROJECT_ROOT/Datasets" ]] || die \
    "Datasets is missing. Create the persistent Datasets symlink before setup."
case "$MODEL_ENV" in
    /mnt/*) die "The Python environment must be on a Linux-native filesystem, not /mnt/*." ;;
esac

for command_name in git nvidia-smi; do
    command -v "$command_name" >/dev/null 2>&1 || die "$command_name is required."
done

assert_pinned_repository() {
    local path="$1"
    local expected="$2"
    local name="$3"
    [[ -d "$path" ]] || die "$name checkout is missing: $path"
    [[ "$(git -C "$path" rev-parse --is-inside-work-tree 2>/dev/null)" == "true" ]] || \
        die "$name is not a Git checkout: $path"
    local actual
    actual="$(git -C "$path" rev-parse HEAD)"
    [[ "$actual" == "$expected" ]] || \
        die "$name commit mismatch: expected=$expected actual=$actual"
    [[ -z "$(git -C "$path" status --porcelain --untracked-files=no)" ]] || \
        die "$name has tracked modifications: $path"
}

assert_pinned_repository \
    "$PROJECT_ROOT/External/MedNeXt" \
    "0b78ed869fbd1cc2fd38754d2f8519f1b72d43ba" \
    "MedNeXt"
assert_pinned_repository \
    "$PROJECT_ROOT/External/BraTS-2023-Metrics" \
    "43c905242b2eecf421d4ab2da7af8ece9777d322" \
    "BraTS 2023 Metrics"
assert_pinned_repository \
    "$PROJECT_ROOT/External/MONAI" \
    "46a5272196a6c2590ca2589029eed8e4d56ff008" \
    "MONAI"
assert_pinned_repository \
    "$PROJECT_ROOT/External/MONAI-model-zoo" \
    "b9e4d04bb2a073110bde9e5c05c9690241e938b6" \
    "MONAI Model Zoo"

WORKSPACE="$(cd -- "$PROJECT_ROOT/Workspace" && pwd -P)"
LOCK_DIR="$WORKSPACE/environment_locks"
mkdir -p -- "$LOCK_DIR" "$WORKSPACE/cache"

remove_owned_environment() {
    local target="$1"
    local owner="$2"
    [[ -e "$target" ]] || return 0
    local resolved_target resolved_owner
    resolved_target="$(readlink -f -- "$target")"
    resolved_owner="$(readlink -f -- "$owner")"
    case "$resolved_target" in
        "$resolved_owner"/*) ;;
        *) die "Refusing to recreate environment outside owned root: $resolved_target" ;;
    esac
    rm -rf --one-file-system -- "$resolved_target"
}

if ((RECREATE)); then
    remove_owned_environment "$MODEL_ENV" "$CODE_ROOT"
    remove_owned_environment "$EVALUATOR_ENV" "$CODE_ROOT"
fi

if [[ ! -x "$MODEL_ENV/bin/python" ]]; then
    echo "Creating model environment: $MODEL_ENV"
    "$BOOTSTRAP_PYTHON" -m venv "$MODEL_ENV"
fi
MODEL_PYTHON="$MODEL_ENV/bin/python"
"$MODEL_PYTHON" -c \
    'import sys; assert sys.version_info[:2] == (3, 10), sys.version' || \
    die "The model environment must use Python 3.10."

"$MODEL_PYTHON" -m pip install --disable-pip-version-check --upgrade \
    pip==24.3.1 setuptools wheel
"$MODEL_PYTHON" -m pip install --disable-pip-version-check \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.5.1 torchvision==0.20.1
"$MODEL_PYTHON" -m pip install --disable-pip-version-check \
    -r "$CODE_ROOT/requirements-research-models.txt"
"$MODEL_PYTHON" -m pip install --disable-pip-version-check --no-deps \
    -e "$PROJECT_ROOT/External/MedNeXt"
"$MODEL_PYTHON" -m pip install --disable-pip-version-check \
    -e "$CODE_ROOT[dev]"

"$MODEL_PYTHON" - <<'PY'
import importlib.metadata as md
import sys

import monai
import torch
import nnunet_mednext

assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.startswith("2.5.1+cu121"), torch.__version__
assert monai.__version__ == "1.4.0", monai.__version__
assert md.version("mednextv1") == "1.7.0", md.version("mednextv1")
assert torch.cuda.is_available(), "CUDA is not available to PyTorch"
print(f"Python={sys.version.split()[0]}")
print(f"PyTorch={torch.__version__}; CUDA={torch.version.cuda}")
print(f"GPU={torch.cuda.get_device_name(0)}")
print(f"MONAI={monai.__version__}; MedNeXt={md.version('mednextv1')}")
PY

if [[ ! -x "$EVALUATOR_ENV/bin/python" ]]; then
    if [[ -n "$EVALUATOR_BOOTSTRAP" ]]; then
        [[ -x "$EVALUATOR_BOOTSTRAP" ]] || die \
            "Evaluator Python is not executable: $EVALUATOR_BOOTSTRAP"
        "$EVALUATOR_BOOTSTRAP" -c \
            'import sys; assert sys.version_info[:2] == (3, 9), sys.version' || \
            die "--evaluator-python must be Python 3.9."
        "$EVALUATOR_BOOTSTRAP" -m venv "$EVALUATOR_ENV"
    elif command -v python3.9 >/dev/null 2>&1; then
        python3.9 -m venv "$EVALUATOR_ENV"
    elif command -v conda >/dev/null 2>&1; then
        conda create --prefix "$EVALUATOR_ENV" --yes \
            "python=$EVALUATOR_PYTHON_VERSION" pip
    elif command -v micromamba >/dev/null 2>&1; then
        micromamba create --prefix "$EVALUATOR_ENV" --yes \
            "python=$EVALUATOR_PYTHON_VERSION" pip
    else
        die "Python 3.9/conda/micromamba is required for the pinned official BraTS evaluator. Ask the administrator for a user environment; do not use sudo."
    fi
fi

EVALUATOR_PYTHON="$EVALUATOR_ENV/bin/python"
"$EVALUATOR_PYTHON" -c \
    'import sys; assert sys.version_info[:2] == (3, 9), sys.version' || \
    die "The official evaluator environment must use Python 3.9."
"$EVALUATOR_PYTHON" -m pip install --disable-pip-version-check --upgrade \
    pip==24.3.1 setuptools wheel
"$EVALUATOR_PYTHON" -m pip install --disable-pip-version-check \
    -r "$PROJECT_ROOT/External/BraTS-2023-Metrics/requirements.txt"
"$EVALUATOR_PYTHON" - "$PROJECT_ROOT/External/BraTS-2023-Metrics" <<'PY'
import sys

root = sys.argv[1]
sys.path.insert(0, root)
import metrics

print(f"Official BraTS evaluator import: {metrics.__file__}")
PY

"$MODEL_PYTHON" -m pip freeze --all > \
    "$LOCK_DIR/research_models_linux_freeze.txt"
"$EVALUATOR_PYTHON" -m pip freeze --all > \
    "$LOCK_DIR/brats2023_metrics_linux_freeze.txt"

echo
echo "Research-model Linux environments are ready."
echo "Model Python: $MODEL_PYTHON"
echo "Evaluator Python: $EVALUATOR_PYTHON"
echo "Evaluator Python pin: $EVALUATOR_PYTHON_VERSION"
echo "Generated freezes: $LOCK_DIR"
echo "No activation is required by the Linux pipeline runner."
