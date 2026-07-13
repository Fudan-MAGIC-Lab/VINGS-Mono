#!/usr/bin/env bash
set -uo pipefail

cd /home/jetson/VINGS-Mono
source /home/jetson/miniconda3/etc/profile.d/conda.sh
conda activate vings_jetson

EVIDENCE_DIR=reports/tensorrt_fp16/droid_cnet_smallcity50_online_20260712
OUTPUT_ROOT=output/smallcity_droid_cnet_online_20260712
PYTORCH_PREFIX=jetson_smallcity_gt50_pytorch_droid_cnet_online_t10
TENSORRT_PREFIX=jetson_smallcity_gt50_tensorrt_droid_cnet_online_t10
PYTORCH_OUTPUT=$OUTPUT_ROOT/pytorch
TENSORRT_OUTPUT=$OUTPUT_ROOT/tensorrt
CNET_ENGINE=engines/tensorrt/droid/droid_cnet_b1_344x616_fp16.plan

if [ -e "$EVIDENCE_DIR" ]; then
  printf 'Refusing to overwrite evidence directory: %s\n' "$EVIDENCE_DIR" >&2
  exit 2
fi
mkdir -p "$EVIDENCE_DIR"
cp "$0" "$EVIDENCE_DIR/driver.sh"

git status --short > "$EVIDENCE_DIR/git_status.txt"
git submodule status --recursive > "$EVIDENCE_DIR/git_submodule_status.txt"
git rev-parse HEAD > "$EVIDENCE_DIR/git_head.txt"
python --version > "$EVIDENCE_DIR/python_version.txt" 2>&1
python -m pip freeze > "$EVIDENCE_DIR/pip_freeze.txt"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
  > "$EVIDENCE_DIR/torch_version.txt" 2>&1
python -c "import sys; sys.path.append('/usr/lib/python3.8/dist-packages'); import tensorrt; print(tensorrt.__version__)" \
  > "$EVIDENCE_DIR/tensorrt_version.txt" 2>&1
PYTHONPATH=scripts python -c \
  "from acceleration.trt_engine import TensorRTEngine; r=TensorRTEngine('$CNET_ENGINE'); print(r.metadata['engine_sha256'], r.metadata['bindings'])" \
  > "$EVIDENCE_DIR/cnet_engine_preflight.txt" 2>&1 || exit $?

check_processes() {
  local label=$1
  if pgrep -af '[s]cripts/run.py|[t]egrastats' \
      > "$EVIDENCE_DIR/processes_before_${label}.txt"; then
    printf 'Residual benchmark process exists before %s\n' "$label" >&2
    return 3
  fi
}

verify_backend() {
  local label=$1
  local run_dir=$2
  python - "$label" "$run_dir/runtime_profile_metadata.json" <<'PY'
import json
import sys

label, path = sys.argv[1:]
metadata = json.load(open(path))
status = metadata["droid_cnet"]
expected_backend = "torch" if label == "pytorch" else "tensorrt"
if status["actual_backend"] != expected_backend:
    raise SystemExit(f"actual backend mismatch: {status}")
if status["fallback_reason"] is not None:
    raise SystemExit(f"unexpected fallback: {status}")
if label == "tensorrt" and not status["strict"]:
    raise SystemExit(f"TensorRT candidate was not strict: {status}")
print(json.dumps(status, sort_keys=True))
PY
}

run_variant() {
  local label=$1
  local prefix=$2
  local output_parent=$3
  shift 3
  local backend_args=("$@")
  local status_file="$EVIDENCE_DIR/status_${label}.txt"
  local tegra_pid=""

  check_processes "$label" || return $?
  if [ -e "$output_parent" ]; then
    printf 'status=2 reason=output_exists output=%s\n' "$output_parent" \
      > "$status_file"
    return 2
  fi
  mkdir -p "$output_parent"

  cleanup() {
    if [ -n "$tegra_pid" ]; then
      kill "$tegra_pid" 2>/dev/null || true
      wait "$tegra_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  tegrastats --interval 1000 > "$EVIDENCE_DIR/tegrastats_${label}.log" 2>&1 &
  tegra_pid=$!

  local started
  started=$(date +%s)
  python scripts/run.py configs/hierarchical/smallcity.yaml \
    --prefix "$prefix" \
    --dataset-root data/smallcity_subset_50/small_city \
    --output-dir "$output_parent" \
    --frontend-weight ckpts/droid.pth \
    --frontend-save-buffer 64 \
    --no-vis \
    --profile-runtime \
    --export-eval \
    --export-eval-interval 1 \
    --skip-save-ply \
    --training-iters 10 \
    --enable-mapping-budget \
    --enable-jetson-pruning \
    --enable-pixel-budget \
    --enable-metric-depth-schedule \
    --metric-depth-mode keyframe \
    --metric-depth-warmup 8 \
    --metric-depth-keyframe-min-interval 3 \
    --metric-depth-keyframe-force-interval 10 \
    --metric-depth-high-motion-ratio 3.0 \
    --metric-depth-scale 0.75 \
    --enable-jetson-motion-gate \
    --motion-gate-backend vpi_cpp \
    --motion-gate-threshold 12.0 \
    --motion-gate-force-interval 8 \
    --motion-gate-resize 96,160 \
    --motion-gate-grid-size 4 \
    --motion-gate-vpi-levels 1 \
    --motion-gate-vpi-quality low \
    "${backend_args[@]}" \
    > "$EVIDENCE_DIR/run_${label}.log" 2>&1
  local run_status=$?
  local elapsed=$(($(date +%s)-started))

  cleanup
  trap - EXIT INT TERM
  tegra_pid=""

  if [ "$run_status" -ne 0 ]; then
    printf 'status=%s elapsed_s=%s\n' "$run_status" "$elapsed" \
      > "$status_file"
    return "$run_status"
  fi

  local run_dir
  run_dir=$(find "$output_parent" -mindepth 1 -maxdepth 1 -type d \
    -name "*${prefix}*" -print | sort | tail -n 1)
  if [ -z "$run_dir" ]; then
    printf 'status=4 elapsed_s=%s reason=run_dir_missing\n' "$elapsed" \
      > "$status_file"
    return 4
  fi
  printf '%s\n' "$run_dir" > "$EVIDENCE_DIR/run_dir_${label}.txt"

  verify_backend "$label" "$run_dir" \
    > "$EVIDENCE_DIR/backend_${label}.json" 2>&1
  local backend_status=$?
  if [ "$backend_status" -ne 0 ]; then
    printf 'status=0 backend_status=%s elapsed_s=%s\n' \
      "$backend_status" "$elapsed" > "$status_file"
    return "$backend_status"
  fi

  python scripts/profiling/evaluate_smallcity_run.py \
    --run-dir "$run_dir" \
    --dataset-root data/smallcity_subset_50/small_city \
    --output "$run_dir/smallcity_eval_metrics.md" \
    > "$EVIDENCE_DIR/evaluate_${label}.log" 2>&1
  local eval_status=$?
  printf 'status=%s backend_status=%s eval_status=%s elapsed_s=%s\n' \
    "$run_status" "$backend_status" "$eval_status" "$elapsed" > "$status_file"
  return "$eval_status"
}

run_variant pytorch "$PYTORCH_PREFIX" "$PYTORCH_OUTPUT"
PYTORCH_STATUS=$?
if [ "$PYTORCH_STATUS" -ne 0 ]; then
  exit "$PYTORCH_STATUS"
fi

run_variant tensorrt "$TENSORRT_PREFIX" "$TENSORRT_OUTPUT" \
  --droid-cnet-backend tensorrt \
  --droid-cnet-engine engines/tensorrt/droid/droid_cnet_b1_344x616_fp16.plan \
  --tensorrt-strict
TENSORRT_STATUS=$?
if [ "$TENSORRT_STATUS" -ne 0 ]; then
  exit "$TENSORRT_STATUS"
fi

PYTORCH_RUN=$(cat "$EVIDENCE_DIR/run_dir_pytorch.txt")
TENSORRT_RUN=$(cat "$EVIDENCE_DIR/run_dir_tensorrt.txt")
python scripts/profiling/compare_tracking_runs.py \
  --run "pytorch=$PYTORCH_RUN" \
  --run "tensorrt=$TENSORRT_RUN" \
  --output-dir "$EVIDENCE_DIR/tracking_comparison" \
  > "$EVIDENCE_DIR/compare_tracking.log" 2>&1
COMPARE_STATUS=$?
printf 'status=%s\n' "$COMPARE_STATUS" > "$EVIDENCE_DIR/compare_status.txt"
exit "$COMPARE_STATUS"
