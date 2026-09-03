#!/usr/bin/env bash
# Complete Linux/JupyterHub MedNeXt-S-k3 pipeline: bootstrap, smoke/full CV,
# evaluation, failure analysis, telemetry, and final report bundle.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CODE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
PROJECT_ROOT="$(cd -- "$CODE_ROOT/.." && pwd -P)"
MODEL_PYTHON="$CODE_ROOT/.venv-models/bin/python"

EXPERIMENT_ID=""
EXPERIMENT_ID_PROVIDED=0
RESUME=0
CONFIRM_RUN=0
SMOKE_TEST=0
PREFLIGHT_ONLY=0
PREPROCESSING_THREADS=8
PIPELINE_SUCCEEDED=0
REPORT_DIR=""

usage() {
    cat <<'EOF'
Usage:
  bash Code/scripts/run_mednext_cv_pipeline.sh --smoke-test --confirm-run
  bash Code/scripts/run_mednext_cv_pipeline.sh --confirm-run
  bash Code/scripts/run_mednext_cv_pipeline.sh --experiment-id EXACT_ID --resume [--smoke-test] --confirm-run

Options:
  --experiment-id ID          Exact experiment ID to resume.
  --resume                    Resume only owner-matched artifacts for that ID.
  --confirm-run               Required opt-in before any GPU work.
  --smoke-test                Run the mandatory 3-epoch real-data smoke pipeline.
  --preflight-only            Stop after preprocessing and GPU memory preflight.
  --preprocessing-threads N   Official preprocessing workers (default: 8).
  -h, --help                  Show this help.

The full default is 100 epochs x folds 0..4, sequentially. It is refused until
a complete 3-epoch smoke pipeline has passed on the same GPU and current code.
The script uses Code/.venv-models directly; shell activation is not required.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

stage() {
    echo
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$1] $2"
}

banner() {
    echo
    echo "============================================================================="
    echo "$1"
    echo "============================================================================="
}

run_logged() {
    local log_path="$1"
    shift
    mkdir -p -- "$(dirname -- "$log_path")"
    set +e
    "$@" 2>&1 | tee -a "$log_path"
    local command_status="${PIPESTATUS[0]}"
    set -e
    return "$command_status"
}

json_value() {
    local path="$1"
    local dotted_key="$2"
    "$MODEL_PYTHON" -c '
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
for key in sys.argv[2].split("."):
    value = value[key]
if isinstance(value, bool):
    print("true" if value else "false")
elif value is None:
    print("null")
else:
    print(value)
' "$path" "$dotted_key"
}

archive_report_artifacts() {
    local label="$1"
    shift
    local existing=()
    local path
    for path in "$@"; do
        [[ -e "$path" || -L "$path" ]] && existing+=("$path")
    done
    ((${#existing[@]})) || return 0
    local archive="$WORKSPACE/cache/$EXPERIMENT_ID/derived_archive_$(date -u '+%Y%m%d_%H%M%S_%3N')/$label"
    mkdir -p -- "$archive"
    for path in "${existing[@]}"; do
        case "$path" in
            "$REPORT_DIR"/*) ;;
            *) die "Refusing to archive an artifact outside this report: $path" ;;
        esac
        mv -- "$path" "$archive/$(basename -- "$path")"
    done
    echo "Preserved previous derived $label artifacts at: $archive"
}

audit_fold() {
    local fold_dir="$1"
    local output="$2"
    mkdir -p -- "$(dirname -- "$output")"
    set +e
    "$MODEL_PYTHON" -m glioma_seg.backends.mednext.backend \
        --project-root "$PROJECT_ROOT" audit-fold \
        --fold-dir "$fold_dir" --output "$output"
    local status=$?
    set -e
    [[ "$status" -eq 0 || "$status" -eq 2 ]] || \
        die "Fold audit crashed with exit code $status: $fold_dir"
}

on_exit() {
    local status=$?
    if [[ "$PIPELINE_SUCCEEDED" -eq 1 ]]; then
        echo "Pipeline lock released."
        return
    fi
    if [[ -n "$EXPERIMENT_ID" ]]; then
        local hint="bash Code/scripts/run_mednext_cv_pipeline.sh --experiment-id $EXPERIMENT_ID --resume --confirm-run"
        [[ "$SMOKE_TEST" -eq 1 ]] && hint+=" --smoke-test"
        echo >&2
        echo "Pipeline stopped before verified completion." >&2
        echo "Resume only with this exact command from the repository root:" >&2
        echo "$hint" >&2
    fi
    return "$status"
}
trap on_exit EXIT

while (($#)); do
    case "$1" in
        --experiment-id)
            (($# >= 2)) || die "--experiment-id requires a value"
            EXPERIMENT_ID="$2"
            EXPERIMENT_ID_PROVIDED=1
            shift 2
            ;;
        --resume)
            RESUME=1
            shift
            ;;
        --confirm-run)
            CONFIRM_RUN=1
            shift
            ;;
        --smoke-test)
            SMOKE_TEST=1
            shift
            ;;
        --preflight-only)
            PREFLIGHT_ONLY=1
            shift
            ;;
        --preprocessing-threads)
            (($# >= 2)) || die "--preprocessing-threads requires an integer"
            PREPROCESSING_THREADS="$2"
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

[[ "$CONFIRM_RUN" -eq 1 ]] || die \
    "This GPU workflow is opt-in. Re-run with --confirm-run after reviewing it."
[[ "$PREPROCESSING_THREADS" =~ ^[1-9][0-9]*$ ]] || die \
    "--preprocessing-threads must be a positive integer."
((PREPROCESSING_THREADS <= 64)) || die "Use at most 64 preprocessing threads."
[[ "$RESUME" -eq 0 || "$EXPERIMENT_ID_PROVIDED" -eq 1 ]] || die \
    "--resume requires the exact --experiment-id printed by the original run."
if [[ "$EXPERIMENT_ID_PROVIDED" -eq 1 ]]; then
    [[ "$EXPERIMENT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{5,127}$ ]] || die \
        "Experiment ID contains unsafe characters: $EXPERIMENT_ID"
fi

[[ "$(id -u)" -ne 0 ]] || die "Run as your normal JupyterHub user, never sudo/root."
[[ -x "$MODEL_PYTHON" ]] || die \
    "Model environment is missing. Run: bash Code/scripts/setup_research_models_env.sh"
[[ -d "$PROJECT_ROOT/Datasets" ]] || die "Datasets link/directory is missing."
[[ -d "$PROJECT_ROOT/Workspace" ]] || die "Workspace link/directory is missing."

WORKSPACE="$(cd -- "$PROJECT_ROOT/Workspace" && pwd -P)"
DATASETS="$(cd -- "$PROJECT_ROOT/Datasets" && pwd -P)"
REPORTS="$WORKSPACE/reports"
RESULTS="$WORKSPACE/model_results/mednext"
OFFICIAL_ROOT="$PROJECT_ROOT/External/BraTS-2023-Metrics"
EVALUATOR_PYTHON="$CODE_ROOT/.venv-brats-metrics/bin/python"
RAW_DATASET="$WORKSPACE/nnUNet_raw/Dataset501_BraTS2023GLI"
LABELS_DIR="$RAW_DATASET/labelsTr"
SPLIT_PATH="$WORKSPACE/nnUNet_preprocessed/Dataset501_BraTS2023GLI/splits_final.json"
GLOBAL_VALIDATION_JSON="$REPORTS/data_validation.json"
GLOBAL_VALIDATION_CSV="$REPORTS/data_validation.csv"
BOOTSTRAP_LOGS="$REPORTS/bootstrap_logs"

case "$CODE_ROOT/.venv-models" in
    /mnt/*) die "The model environment must be on Linux-native storage, not /mnt/*." ;;
esac
for command_name in git nvidia-smi flock tee; do
    command -v "$command_name" >/dev/null 2>&1 || die "$command_name is required."
done
[[ -x "$EVALUATOR_PYTHON" ]] || die \
    "Official evaluator environment is missing. Re-run the setup script."

mkdir -p -- "$REPORTS" "$RESULTS" "$BOOTSTRAP_LOGS" "$CODE_ROOT/.venv-models/.pipeline-locks"
exec 9>"$CODE_ROOT/.venv-models/.pipeline-locks/mednext_gpu.lock"
flock -n 9 || die "Another MedNeXt pipeline owns the GPU lock."

active_compute="$(nvidia-smi --query-compute-apps=pid,process_name \
    --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
[[ -z "$active_compute" ]] || die \
    "Another CUDA compute process is already using the GPU: $active_compute"

EXPERIMENT_KIND="fullcv"
FOLDS=(0 1 2 3 4)
EXPECTED_CASE_COUNT=1251
TARGET_EPOCHS=100
if [[ "$SMOKE_TEST" -eq 1 ]]; then
    EXPERIMENT_KIND="smoke"
    FOLDS=(0)
    EXPECTED_CASE_COUNT=2
    TARGET_EPOCHS=3
fi

banner "MEDNEXT LINUX/JUPYTERHUB PIPELINE | $EXPERIMENT_KIND"
echo "Repository (real path): $PROJECT_ROOT"
echo "Dataset storage: $DATASETS"
echo "Workspace storage: $WORKSPACE"
echo "Python (direct; activation not required): $MODEL_PYTHON"
echo "Model: official MedNeXt-S-k3; patch 128x128x128; TTA OFF"
echo "Folds: ${FOLDS[*]}; epochs per fold: $TARGET_EPOCHS"

stage "BOOTSTRAP" "Validate all 1,251 raw cases and create shared converted inputs"
validation_reusable=0
if [[ -f "$GLOBAL_VALIDATION_JSON" && -f "$GLOBAL_VALIDATION_CSV" ]]; then
    if "$MODEL_PYTHON" -c '
import json, pathlib, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
ok = (x.get("valid") is True and x.get("dataset_kind") == "training" and
      x.get("expected_case_count") == 1251 and x.get("actual_case_count") == 1251 and
      x.get("valid_case_count") == 1251 and pathlib.Path(x["dataset_root"]).is_dir())
raise SystemExit(0 if ok else 1)
' "$GLOBAL_VALIDATION_JSON"; then
        validation_reusable=1
        echo "Reusing verified 1,251-case raw-data validation evidence."
    fi
fi
if [[ "$validation_reusable" -eq 0 ]]; then
    run_logged "$BOOTSTRAP_LOGS/data_validation.log" \
        "$MODEL_PYTHON" -m glioma_seg.data.validate \
        --data-root "$DATASETS" --kind training \
        --output-json "$GLOBAL_VALIDATION_JSON" \
        --output-csv "$GLOBAL_VALIDATION_CSV" \
        --expected-training-cases 1251
fi
[[ "$(json_value "$GLOBAL_VALIDATION_JSON" valid)" == "true" ]] || die \
    "Raw-data validation did not pass. No model work was started."
[[ "$(json_value "$GLOBAL_VALIDATION_JSON" valid_case_count)" == "1251" ]] || die \
    "Raw dataset is incomplete; expected 1,251 valid cases."

run_logged "$BOOTSTRAP_LOGS/nnunet_conversion.log" \
    "$MODEL_PYTHON" -m glioma_seg.data.nnunet_conversion \
    --data-root "$DATASETS" \
    --output-root "$WORKSPACE/nnUNet_raw" \
    --validation-json "$GLOBAL_VALIDATION_JSON" \
    --dataset-config "$CODE_ROOT/configs/datasets/brats2023_gli.yaml" \
    --report-json "$REPORTS/mednext_bootstrap_nnunet_conversion.json" \
    --expected-training-cases 1251

run_logged "$BOOTSTRAP_LOGS/canonical_split.log" \
    "$MODEL_PYTHON" -m glioma_seg.data.canonical_splits \
    --labels-dir "$LABELS_DIR" --output "$SPLIT_PATH"

if [[ -z "$EXPERIMENT_ID" ]]; then
    EXPERIMENT_ID="$($MODEL_PYTHON -m glioma_seg.backends.mednext.backend \
        --project-root "$PROJECT_ROOT" new-experiment --kind "$EXPERIMENT_KIND" | tail -n 1)"
fi
[[ "$EXPERIMENT_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{5,127}$ ]] || die \
    "Backend returned an unsafe experiment ID: $EXPERIMENT_ID"
REPORT_DIR="$REPORTS/$EXPERIMENT_ID"
EXPERIMENT_MANIFEST="$REPORT_DIR/experiment.json"
if [[ "$RESUME" -eq 1 && ! -f "$EXPERIMENT_MANIFEST" ]]; then
    die "Resume manifest does not exist: $EXPERIMENT_MANIFEST"
fi
if [[ "$EXPERIMENT_ID_PROVIDED" -eq 1 && "$RESUME" -eq 0 && -d "$REPORT_DIR" ]] && \
        find "$REPORT_DIR" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    die "Experiment already has artifacts. Use --resume with the same ID or omit the ID."
fi

stage "1/12" "Initialize owned experiment"
"$MODEL_PYTHON" -m glioma_seg.backends.mednext.backend \
    --project-root "$PROJECT_ROOT" initialize \
    --experiment-id "$EXPERIMENT_ID" --kind "$EXPERIMENT_KIND"
LOGS="$REPORT_DIR/logs"
mkdir -p -- "$LOGS"
cp -- "$GLOBAL_VALIDATION_JSON" "$REPORT_DIR/data_validation.json"
cp -- "$GLOBAL_VALIDATION_CSV" "$REPORT_DIR/data_validation.csv"

stage "2/12" "Pinned environment, official sources, and GPU system check"
run_logged "$LOGS/system_check.log" \
    "$MODEL_PYTHON" -m glioma_seg.backends.mednext.backend \
    --project-root "$PROJECT_ROOT" system-check \
    --experiment-id "$EXPERIMENT_ID" --output "$REPORT_DIR/environment.json"
expected_official_commit="43c905242b2eecf421d4ab2da7af8ece9777d322"
actual_official_commit="$(git -C "$OFFICIAL_ROOT" rev-parse HEAD)"
[[ "$actual_official_commit" == "$expected_official_commit" ]] || die \
    "Official BraTS evaluator commit mismatch."
[[ -z "$(git -C "$OFFICIAL_ROOT" status --porcelain --untracked-files=no)" ]] || die \
    "Official BraTS evaluator has tracked changes."
"$EVALUATOR_PYTHON" -c \
    'import sys; sys.path.insert(0, sys.argv[1]); import metrics; print(metrics.__file__)' \
    "$OFFICIAL_ROOT"

if [[ "$SMOKE_TEST" -eq 0 && "$PREFLIGHT_ONLY" -eq 0 ]]; then
    "$MODEL_PYTHON" -m glioma_seg.backends.mednext.smoke_gate \
        --project-root "$PROJECT_ROOT" verify \
        --current-environment "$REPORT_DIR/environment.json"
fi

stage "3/12" "MedNeXt-compatible project test gate"
run_logged "$LOGS/test_suite.log" \
    "$MODEL_PYTHON" -m pytest "$CODE_ROOT/tests" -q \
    --ignore="$CODE_ROOT/tests/test_nnunet_artifacts.py"

stage "4/12" "Publish complete raw-data validation evidence"
echo "Validated 1,251/1,251 BraTS training cases."

stage "5/12" "Official preprocessing and real-data GPU memory preflight"
MEMORY_GATE_ID="$EXPERIMENT_ID"
if [[ "$SMOKE_TEST" -eq 0 ]]; then
    MEMORY_GATE_ID="${EXPERIMENT_ID}_memory_gate"
fi
run_logged "$LOGS/preprocess_task951.log" \
    "$MODEL_PYTHON" -m glioma_seg.backends.mednext.backend \
    --project-root "$PROJECT_ROOT" preprocess \
    --experiment-id "$MEMORY_GATE_ID" --smoke \
    --threads "$PREPROCESSING_THREADS" --quick-train-cases 8 --quick-validation-cases 2
if [[ "$SMOKE_TEST" -eq 0 ]]; then
    run_logged "$LOGS/preprocess_task501.log" \
        "$MODEL_PYTHON" -m glioma_seg.backends.mednext.backend \
        --project-root "$PROJECT_ROOT" preprocess \
        --experiment-id "$EXPERIMENT_ID" \
        --threads "$PREPROCESSING_THREADS" --quick-train-cases 8 --quick-validation-cases 2
fi

MEMORY_OUTPUT="$REPORT_DIR/memory_preflight.json"
reuse_memory=0
if [[ "$RESUME" -eq 1 && -f "$MEMORY_OUTPUT" ]]; then
    if "$MODEL_PYTHON" -c '
import json, sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
ok=(x.get("valid") is True and x.get("model")=="mednext_v1_s_kernel3" and
    x.get("patch_size")==[128,128,128] and
    x.get("official_loss_and_augmentation") is True and
    float(x.get("peak_reserved_mb",0))>0 and x.get("dedicated_vram_fit") is True)
raise SystemExit(0 if ok else 1)
' "$MEMORY_OUTPUT"; then
        reuse_memory=1
        echo "Reusing valid experiment-local GPU memory preflight."
    fi
fi
if [[ "$reuse_memory" -eq 0 ]]; then
    run_logged "$LOGS/memory_preflight.log" \
        "$MODEL_PYTHON" -m glioma_seg.backends.mednext.backend \
        --project-root "$PROJECT_ROOT" memory-preflight \
        --experiment-id "$MEMORY_GATE_ID" --output "$MEMORY_OUTPUT"
fi
[[ "$(json_value "$MEMORY_OUTPUT" valid)" == "true" ]] || die \
    "MedNeXt memory preflight failed. No training fold was started."
[[ "$(json_value "$MEMORY_OUTPUT" dedicated_vram_fit)" == "true" ]] || die \
    "MedNeXt did not fit dedicated VRAM. No training fold was started."

MEMORY_PREPROCESSING="$REPORTS/$MEMORY_GATE_ID/preprocessing_artifacts.json"
"$MODEL_PYTHON" -c '
import hashlib, json, os, pathlib, sys, tempfile
report, primary, preflight, memory, prep = map(pathlib.Path, sys.argv[1:])
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()
experiment=json.load(open(report/"experiment.json", encoding="utf-8"))
value={
 "valid":True, "primary_experiment_id":primary.name,
 "preflight_experiment_id":preflight.name, "task_name":"Task951_BraTS2023GLISmoke",
 "source_fold":0, "training_cases":8, "validation_cases":2,
 "target_epochs_for_smoke":3, "model_id":experiment["model_id"],
 "model_config_sha256":experiment["model_config_sha256"],
 "memory_preflight_json":str(memory.resolve()), "memory_preflight_sha256":digest(memory),
 "preprocessing_artifacts_json":str(prep.resolve()),
 "preprocessing_artifacts_sha256":digest(prep),
 "scope":"One real forward, official loss, backward, and optimizer step using the unchanged MedNeXt-S-k3 128x128x128 plan on the deterministic real-data smoke task."
}
destination=report/"memory_preflight_context.json"
fd,name=tempfile.mkstemp(prefix=".memory_preflight_context.", suffix=".tmp", dir=report)
try:
    with os.fdopen(fd,"w",encoding="utf-8",newline="\n") as f:
        json.dump(value,f,indent=2,ensure_ascii=False); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(name,destination)
finally:
    pathlib.Path(name).unlink(missing_ok=True)
' "$REPORT_DIR" "$REPORT_DIR" "$REPORTS/$MEMORY_GATE_ID" \
    "$MEMORY_OUTPUT" "$MEMORY_PREPROCESSING"

if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
    banner "PREFLIGHT COMPLETE - NO TRAINING FOLD WAS STARTED"
    echo "Experiment ID: $EXPERIMENT_ID"
    hint="bash Code/scripts/run_mednext_cv_pipeline.sh --experiment-id $EXPERIMENT_ID --resume --confirm-run"
    [[ "$SMOKE_TEST" -eq 1 ]] && hint+=" --smoke-test"
    echo "Continue with: $hint"
    PIPELINE_SUCCEEDED=1
    exit 0
fi

stage "6/12" "Sequential fold training, deep audit, and safe resume"
RESULT_ROOT="$RESULTS/$EXPERIMENT_ID"
for fold in "${FOLDS[@]}"; do
    FOLD_DIR="$RESULT_ROOT/fold_$fold"
    FOLD_REPORT="$REPORT_DIR/folds/fold_$fold"
    AUDIT_PATH="$FOLD_REPORT/artifact_audit.json"
    audit_fold "$FOLD_DIR" "$AUDIT_PATH"
    if [[ "$(json_value "$AUDIT_PATH" valid)" == "true" && \
          "$(json_value "$AUDIT_PATH" complete)" == "true" ]]; then
        banner "FOLD $fold VERIFIED - REUSING COMPLETE ARTIFACTS"
        continue
    fi
    SAFE_TO_RESUME="$(json_value "$AUDIT_PATH" safe_to_resume)"
    if [[ "$SAFE_TO_RESUME" == "true" && "$RESUME" -eq 0 ]]; then
        die "Fold $fold has an owner-matched checkpoint. Re-run with --resume."
    fi

    if [[ "$SMOKE_TEST" -eq 1 ]]; then
        if [[ "$SAFE_TO_RESUME" == "true" ]]; then
            banner "SMOKE RESUME LEG - CONTINUING VERIFIED CHECKPOINT TO EPOCH 3"
            run_logged "$LOGS/train_fold_0_smoke_resume.log" \
                "$MODEL_PYTHON" -m glioma_seg.backends.mednext.backend \
                --project-root "$PROJECT_ROOT" train-fold \
                --experiment-id "$EXPERIMENT_ID" --fold 0 --epochs 3 --smoke \
                --quick-train-cases 8 --quick-validation-cases 2 --resume
        else
            unexpected=""
            if [[ -d "$FOLD_DIR" ]]; then
                unexpected="$(find "$FOLD_DIR" -mindepth 1 -maxdepth 1 \
                    ! -name fold_manifest.json -print -quit)"
            fi
            [[ -z "$unexpected" ]] || die \
                "Smoke fold has non-resumable partial artifacts: $FOLD_DIR"
            banner "SMOKE LEG 1/2 - INTENTIONAL STOP AFTER EPOCH 1"
            run_logged "$LOGS/train_fold_0_smoke_forced_stop.log" \
                "$MODEL_PYTHON" -m glioma_seg.backends.mednext.backend \
                --project-root "$PROJECT_ROOT" train-fold \
                --experiment-id "$EXPERIMENT_ID" --fold 0 --epochs 3 --smoke \
                --quick-train-cases 8 --quick-validation-cases 2 --stop-after-epoch 1
            audit_fold "$FOLD_DIR" "$AUDIT_PATH"
            [[ "$(json_value "$AUDIT_PATH" safe_to_resume)" == "true" ]] || die \
                "The epoch-1 forced-stop checkpoint did not pass resume audit."
            [[ "$(json_value "$AUDIT_PATH" complete)" == "false" ]] || die \
                "The forced-stop leg was incorrectly marked complete."
            banner "SMOKE LEG 2/2 - VERIFIED RESUME FROM EPOCH 1 TO EPOCH 3"
            run_logged "$LOGS/train_fold_0_smoke_resume.log" \
                "$MODEL_PYTHON" -m glioma_seg.backends.mednext.backend \
                --project-root "$PROJECT_ROOT" train-fold \
                --experiment-id "$EXPERIMENT_ID" --fold 0 --epochs 3 --smoke \
                --quick-train-cases 8 --quick-validation-cases 2 --resume
        fi
    else
        if [[ "$SAFE_TO_RESUME" != "true" && -d "$FOLD_DIR" ]]; then
            unexpected="$(find "$FOLD_DIR" -mindepth 1 -maxdepth 1 \
                ! -name fold_manifest.json -print -quit)"
            [[ -z "$unexpected" ]] || die \
                "Fold $fold contains non-resumable partial artifacts: $FOLD_DIR"
        fi
        banner "TRAINING MEDNEXT FOLD $((fold + 1))/5 | 100 EPOCHS"
        train_command=(
            "$MODEL_PYTHON" -m glioma_seg.backends.mednext.backend
            --project-root "$PROJECT_ROOT" train-fold
            --experiment-id "$EXPERIMENT_ID" --fold "$fold" --epochs 100
        )
        if [[ "$SAFE_TO_RESUME" == "true" ]]; then
            train_command+=(--resume)
            echo "Fold $fold will resume its verified full-state checkpoint."
        fi
        run_logged "$LOGS/train_fold_$fold.log" "${train_command[@]}"
    fi

    audit_fold "$FOLD_DIR" "$AUDIT_PATH"
    [[ "$(json_value "$AUDIT_PATH" valid)" == "true" && \
       "$(json_value "$AUDIT_PATH" complete)" == "true" ]] || die \
        "Fold $fold returned from training but failed final artifact audit."
    echo "Fold $fold checkpoint, masks, probabilities, and telemetry are verified."
done

PREDICTION_DIR=""
ANALYSIS_GROUND_TRUTH="$LABELS_DIR"
stage "7/12" "OOF assembly and semantic evaluation"
if [[ "$SMOKE_TEST" -eq 1 ]]; then
    PREDICTION_DIR="$RESULT_ROOT/fold_0/predictions"
    smoke_eval=(
        "$MODEL_PYTHON" -m glioma_seg.evaluation.smoke
        --fold-manifest "$RESULT_ROOT/fold_0/fold_manifest.json"
        --ground-truth-dir "$LABELS_DIR" --prediction-dir "$PREDICTION_DIR"
        --output-dir "$REPORT_DIR"
    )
    [[ "$RESUME" -eq 1 ]] && smoke_eval+=(--overwrite)
    run_logged "$LOGS/smoke_evaluation.log" "${smoke_eval[@]}"
    ANALYSIS_GROUND_TRUTH="$REPORT_DIR/smoke_ground_truth"
else
    oof_command=(
        "$MODEL_PYTHON" -m glioma_seg.backends.mednext.backend
        --project-root "$PROJECT_ROOT" assemble-oof --experiment-id "$EXPERIMENT_ID"
    )
    manifest_command=(
        "$MODEL_PYTHON" -m glioma_seg.backends.mednext.backend
        --project-root "$PROJECT_ROOT" write-evaluation-manifest
        --experiment-id "$EXPERIMENT_ID"
    )
    for fold in "${FOLDS[@]}"; do
        oof_command+=(--fold "$fold")
        manifest_command+=(--fold "$fold")
    done
    run_logged "$LOGS/assemble_oof.log" "${oof_command[@]}"
    "${manifest_command[@]}"
    PREDICTION_DIR="$RESULT_ROOT/oof/predictions"
    evaluation_command=(
        "$MODEL_PYTHON" -m glioma_seg.evaluation.model_crossval
        --ground-truth-dir "$LABELS_DIR" --splits-json "$SPLIT_PATH"
        --artifact-manifest "$REPORT_DIR/crossval_artifact_manifest.json"
        --output-dir "$REPORT_DIR" --expected-case-count 1251
    )
    [[ "$RESUME" -eq 1 ]] && evaluation_command+=(--overwrite)
    run_logged "$LOGS/model_crossval_evaluation.log" "${evaluation_command[@]}"
fi

stage "8/12" "Pinned official BraTS lesion-wise evaluation"
official_paths=(
    "$REPORT_DIR/official_brats_metrics_status.json"
    "$REPORT_DIR/official_brats_evaluator.log"
    "$REPORT_DIR/official_lesionwise_metrics_per_case.csv"
    "$REPORT_DIR/official_lesionwise_metrics_summary.csv"
    "$REPORT_DIR/official_lesionwise_metrics_summary.json"
)
[[ "$RESUME" -eq 0 ]] || archive_report_artifacts official_metrics "${official_paths[@]}"
run_logged "$LOGS/official_evaluation_command.log" \
    "$MODEL_PYTHON" -m glioma_seg.evaluation.official_runner \
    --ground-truth-dir "$ANALYSIS_GROUND_TRUTH" --prediction-dir "$PREDICTION_DIR" \
    --output-dir "$REPORT_DIR" --official-root "$OFFICIAL_ROOT" \
    --python "$EVALUATOR_PYTHON"

stage "9/12" "Backend-neutral failure statistics"
FAILURE_STATS_JSON="$REPORT_DIR/failure_statistics.json"
FAILURE_STATS_CSV="$REPORT_DIR/failure_statistics_per_case_region.csv"
[[ "$RESUME" -eq 0 ]] || archive_report_artifacts failure_statistics \
    "$FAILURE_STATS_JSON" "$FAILURE_STATS_CSV"
run_logged "$LOGS/failure_statistics.log" \
    "$MODEL_PYTHON" -m glioma_seg.analysis.failure_statistics \
    --ground-truth-dir "$ANALYSIS_GROUND_TRUTH" --prediction-dir "$PREDICTION_DIR" \
    --metrics-csv "$REPORT_DIR/metrics_per_case.csv" \
    --integrity-json "$REPORT_DIR/crossval_integrity.json" \
    --output-json "$FAILURE_STATS_JSON" --output-csv "$FAILURE_STATS_CSV" \
    --expected-case-count "$EXPECTED_CASE_COUNT"

stage "10/12" "Failure ranking and correctly oriented representative figures"
FAILURE_RANKINGS="$REPORT_DIR/failure_rankings.csv"
FAILURE_CASES="$REPORT_DIR/failure_cases.csv"
FIGURES_DIR="$REPORT_DIR/figures"
failure_outputs=("$FAILURE_RANKINGS" "$FAILURE_CASES")
[[ "$SMOKE_TEST" -eq 1 ]] || failure_outputs+=("$FIGURES_DIR")
[[ "$RESUME" -eq 0 ]] || archive_report_artifacts failure_analysis "${failure_outputs[@]}"
FAILURE_STAGE="$WORKSPACE/cache/$EXPERIMENT_ID/failure_analysis_$(date -u '+%Y%m%d_%H%M%S_%3N')"
mkdir -p -- "$FAILURE_STAGE"
RANKING_DEPTH=5
REPRESENTATIVE_LIMIT=15
if [[ "$SMOKE_TEST" -eq 1 ]]; then
    RANKING_DEPTH=2
    REPRESENTATIVE_LIMIT=2
fi
run_logged "$LOGS/failure_ranking.log" \
    "$MODEL_PYTHON" -m glioma_seg.analysis.failure_analysis \
    --ground-truth-dir "$ANALYSIS_GROUND_TRUTH" --prediction-dir "$PREDICTION_DIR" \
    --metrics-per-case-csv "$REPORT_DIR/metrics_per_case.csv" \
    --output-dir "$FAILURE_STAGE" --top-n "$RANKING_DEPTH" \
    --max-cases "$REPRESENTATIVE_LIMIT"
if [[ "$SMOKE_TEST" -eq 0 ]]; then
    RAW_TRAINING_DIR="$(json_value "$REPORT_DIR/data_validation.json" dataset_root)"
    run_logged "$LOGS/representative_figures.log" \
        "$MODEL_PYTHON" -m glioma_seg.visualization.overlays \
        --raw-training-dir "$RAW_TRAINING_DIR" \
        --ground-truth-dir "$ANALYSIS_GROUND_TRUTH" --prediction-dir "$PREDICTION_DIR" \
        --failure-cases-csv "$FAILURE_STAGE/failure_cases.csv" \
        --metrics-per-case-csv "$REPORT_DIR/metrics_per_case.csv" \
        --output-dir "$FAILURE_STAGE/figures" --max-cases "$REPRESENTATIVE_LIMIT"
    mv -- "$FAILURE_STAGE/figures" "$FIGURES_DIR"
fi
mv -- "$FAILURE_STAGE/failure_rankings.csv" "$FAILURE_RANKINGS"
mv -- "$FAILURE_STAGE/failure_cases.csv" "$FAILURE_CASES"

stage "11/12" "Training, GPU, and inference telemetry aggregation"
runtime_command=(
    "$MODEL_PYTHON" -m glioma_seg.reporting.model_runtime
    --project-root "$PROJECT_ROOT" --experiment-id "$EXPERIMENT_ID"
)
for fold in "${FOLDS[@]}"; do runtime_command+=(--fold "$fold"); done
[[ "$SMOKE_TEST" -eq 0 ]] || runtime_command+=(--smoke)
run_logged "$LOGS/telemetry_aggregation.log" "${runtime_command[@]}"

stage "12/12" "Artifact-backed summary and final report-bundle audit"
[[ "$RESUME" -eq 0 ]] || archive_report_artifacts previous_report_manifest \
    "$REPORT_DIR/report_manifest.json"
report_command=(
    "$MODEL_PYTHON" -m glioma_seg.reporting.report
    --output-dir "$REPORT_DIR" --experiment-json "$REPORT_DIR/experiment.json"
    --metrics-summary-csv "$REPORT_DIR/metrics_summary.csv"
    --environment-json "$REPORT_DIR/environment.json"
    --runtime-json "$REPORT_DIR/runtime.json"
    --inference-runtime-json "$REPORT_DIR/inference_runtime.json"
    --gpu-summary-json "$REPORT_DIR/gpu_summary.json"
    --data-validation-json "$REPORT_DIR/data_validation.json"
    --preprocessing-artifacts-json "$REPORT_DIR/preprocessing_artifacts.json"
    --official-status-json "$REPORT_DIR/official_brats_metrics_status.json"
    --official-summary-csv "$REPORT_DIR/official_lesionwise_metrics_summary.csv"
    --evaluation-protocol-json "$REPORT_DIR/evaluation_protocol.json"
    --failure-cases-csv "$REPORT_DIR/failure_cases.csv"
)
if [[ "$SMOKE_TEST" -eq 0 ]]; then
    report_command+=(
        --figures-dir "$FIGURES_DIR"
        --figures-manifest-csv "$FIGURES_DIR/figures_manifest.csv"
        --crossval-summary-json "$REPORT_DIR/crossval_summary.json"
    )
fi
run_logged "$LOGS/report_generation.log" "${report_command[@]}"

bundle_command=(
    "$MODEL_PYTHON" -m glioma_seg.reporting.model_bundle
    --experiment-dir "$REPORT_DIR" --expected-case-count "$EXPECTED_CASE_COUNT"
    --expected-folds
)
for fold in "${FOLDS[@]}"; do bundle_command+=("$fold"); done
"${bundle_command[@]}"

if [[ "$SMOKE_TEST" -eq 1 ]]; then
    [[ "$(json_value "$REPORT_DIR/report_manifest.json" is_final_baseline)" == "false" ]] || \
        die "Smoke report was incorrectly marked as a final baseline."
    "$MODEL_PYTHON" -m glioma_seg.backends.mednext.smoke_gate \
        --project-root "$PROJECT_ROOT" issue --experiment-id "$EXPERIMENT_ID"
else
    [[ "$(json_value "$REPORT_DIR/report_manifest.json" is_final_baseline)" == "true" ]] || \
        die "Full five-fold report was not marked as a final baseline."
fi

PIPELINE_SUCCEEDED=1
banner "MEDNEXT PIPELINE COMPLETED AND VERIFIED"
echo "Experiment ID: $EXPERIMENT_ID"
echo "Final report bundle: $REPORT_DIR"
