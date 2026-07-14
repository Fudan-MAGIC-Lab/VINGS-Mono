#!/usr/bin/env bash

set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
shared_root="$repo_root"
if [ "$(basename "$(dirname "$repo_root")")" = ".worktrees" ]; then
  shared_root="$(cd "$repo_root/../.." && pwd)"
fi

EVIDENCE_DIR="${VINGS_PHASE0_EVIDENCE_DIR:-$shared_root/reports/cuda_graph_dba/phase0_20260714}"
output_parent="${VINGS_PHASE0_OUTPUT_DIR:-$shared_root/output/cuda_graph_dba_phase0_20260714}"

if [ -e "$EVIDENCE_DIR" ]; then
  printf 'Evidence directory already exists: %s\n' "$EVIDENCE_DIR" >&2
  exit 2
fi
if [ -e "$output_parent" ]; then
  printf 'Output directory already exists: %s\n' "$output_parent" >&2
  exit 2
fi

mkdir -p "$EVIDENCE_DIR" "$output_parent"
cd "$repo_root"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

tegrastats_pid=""

stop_tegrastats() {
  if [ -n "$tegrastats_pid" ]; then
    kill "$tegrastats_pid" 2>/dev/null || true
    wait "$tegrastats_pid" 2>/dev/null || true
    tegrastats_pid=""
  fi
}

cleanup() {
  stop_tegrastats
}
trap cleanup EXIT INT TERM

check_residual_processes() {
  local label=$1
  if pgrep -af '[s]cripts/run.py|[t]egrastats' \
      > "$EVIDENCE_DIR/processes_before_${label}.txt"; then
    printf 'Residual benchmark process exists before %s\n' "$label" >&2
    return 3
  fi
}

snapshot_environment() {
  git status --short > "$EVIDENCE_DIR/git_status.txt"
  git submodule status --recursive > "$EVIDENCE_DIR/git_submodule_status.txt"
  git rev-parse HEAD > "$EVIDENCE_DIR/git_head.txt"
  python --version > "$EVIDENCE_DIR/python_version.txt" 2>&1
  python -m pip freeze > "$EVIDENCE_DIR/pip_freeze.txt"
  python - <<'PY' > "$EVIDENCE_DIR/torch_cuda.txt"
import torch

print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name())
PY
}

run_variant() {
  local label=$1
  local profiled=$2
  local variant_output=$3
  shift 3

  check_residual_processes "$label" || return $?
  mkdir -p "$variant_output"

  tegrastats --interval 1000 \
    > "$EVIDENCE_DIR/tegrastats_${label}.log" 2>&1 &
  tegrastats_pid=$!
  local started
  started=$(date +%s)

  "$@" > "$EVIDENCE_DIR/run_${label}.log" 2>&1
  local run_status=$?

  local finished
  finished=$(date +%s)
  stop_tegrastats
  printf 'status=%s elapsed_s=%s\n' \
    "$run_status" "$((finished - started))" \
    > "$EVIDENCE_DIR/status_${label}.txt"
  if [ "$run_status" -ne 0 ]; then
    return "$run_status"
  fi

  mapfile -t run_dirs < <(
    find "$variant_output" -mindepth 1 -maxdepth 1 -type d | sort
  )
  if [ "${#run_dirs[@]}" -ne 1 ]; then
    printf 'Expected one run directory for %s, found %s\n' \
      "$label" "${#run_dirs[@]}" >&2
    return 4
  fi
  local run_dir=${run_dirs[0]}
  printf '%s\n' "$run_dir" > "$EVIDENCE_DIR/run_dir_${label}.txt"

  if [ "$profiled" = "1" ]; then
    local details="$run_dir/runtime_profile_details.jsonl"
    if [ ! -s "$details" ]; then
      printf 'Missing runtime profile details for %s: %s\n' \
        "$label" "$details" >&2
      return 5
    fi
    if ! grep -q '"kind": "dba_signature"' "$details"; then
      printf 'No DBA signatures recorded for %s\n' "$label" >&2
      return 6
    fi
  fi
}

evaluate_smallcity() {
  local label=$1
  local dataset_root=$2
  local run_dir
  run_dir=$(cat "$EVIDENCE_DIR/run_dir_${label}.txt")
  python scripts/profiling/evaluate_smallcity_run.py \
    --run-dir "$run_dir" \
    --dataset-root "$dataset_root" \
    --output "$run_dir/smallcity_eval_metrics.md" \
    > "$EVIDENCE_DIR/evaluate_${label}.log" 2>&1
}

snapshot_environment

smallcity50_root="$shared_root/data/smallcity_subset_50/small_city"
smallcity200_root="$shared_root/data/smallcity_subset_200/small_city"
hotel_root="$shared_root/data/hotel"
frontend_weight="$shared_root/ckpts/droid.pth"
lightglue_root="$shared_root/ckpts/lightglue"

for required in \
  "$frontend_weight" \
  "$lightglue_root/superpoint.onnx" \
  "$lightglue_root/superpoint_lightglue.onnx"; do
  if [ ! -e "$required" ]; then
    printf 'Missing required file: %s\n' "$required" >&2
    exit 7
  fi
done
for required_dir in \
  "$smallcity50_root" \
  "$smallcity200_root" \
  "$hotel_root/nosky_color"; do
  if [ ! -d "$required_dir" ]; then
    printf 'Missing required dataset directory: %s\n' "$required_dir" >&2
    exit 7
  fi
done

run_variant \
  smallcity50_control 0 "$output_parent/smallcity50_control" \
  python scripts/run.py configs/hierarchical/smallcity.yaml \
    --prefix cuda_graph_dba_phase0_smallcity50_control \
    --dataset-root "$smallcity50_root" \
    --output-dir "$output_parent/smallcity50_control" \
    --frontend-weight "$frontend_weight" \
    --frontend-image-size 344,616 \
    --frontend-save-buffer 64 \
    --droid-update-backend torch \
    --export-eval --export-eval-interval 1 \
    --no-vis --skip-save-ply \
    --training-iters 10 \
    --enable-mapping-budget --enable-jetson-pruning --enable-pixel-budget \
    --enable-metric-depth-schedule --metric-depth-mode keyframe \
    --metric-depth-keyframe-min-interval 3 \
    --metric-depth-keyframe-force-interval 10 \
    --metric-depth-high-motion-ratio 3.0 \
    --metric-depth-scale 0.75 \
    --enable-jetson-motion-gate --motion-gate-backend vpi_cpp \
    --motion-gate-threshold 12.0 --motion-gate-force-interval 8 \
    --motion-gate-resize 96,160 || exit $?
evaluate_smallcity smallcity50_control "$smallcity50_root" || exit $?

run_variant \
  smallcity50 1 "$output_parent/smallcity50" \
  python scripts/run.py configs/hierarchical/smallcity.yaml \
    --prefix cuda_graph_dba_phase0_smallcity50 \
    --dataset-root "$smallcity50_root" \
    --output-dir "$output_parent/smallcity50" \
    --frontend-weight "$frontend_weight" \
    --frontend-image-size 344,616 \
    --frontend-save-buffer 64 \
    --droid-update-backend torch \
    --profile-runtime \
    --export-eval --export-eval-interval 1 \
    --no-vis --skip-save-ply \
    --training-iters 10 \
    --enable-mapping-budget --enable-jetson-pruning --enable-pixel-budget \
    --enable-metric-depth-schedule --metric-depth-mode keyframe \
    --metric-depth-keyframe-min-interval 3 \
    --metric-depth-keyframe-force-interval 10 \
    --metric-depth-high-motion-ratio 3.0 \
    --metric-depth-scale 0.75 \
    --enable-jetson-motion-gate --motion-gate-backend vpi_cpp \
    --motion-gate-threshold 12.0 --motion-gate-force-interval 8 \
    --motion-gate-resize 96,160 || exit $?
evaluate_smallcity smallcity50 "$smallcity50_root" || exit $?

run_variant \
  smallcity200 1 "$output_parent/smallcity200" \
  python scripts/run.py configs/hierarchical/smallcity.yaml \
    --prefix cuda_graph_dba_phase0_smallcity200 \
    --dataset-root "$smallcity200_root" \
    --output-dir "$output_parent/smallcity200" \
    --frontend-weight "$frontend_weight" \
    --frontend-image-size 344,616 \
    --frontend-save-buffer 64 \
    --droid-update-backend torch \
    --profile-runtime \
    --export-eval --export-eval-interval 1 \
    --no-vis --skip-save-ply \
    --training-iters 10 \
    --enable-mapping-budget --enable-jetson-pruning --enable-pixel-budget \
    --enable-metric-depth-schedule --metric-depth-mode keyframe \
    --metric-depth-keyframe-min-interval 3 \
    --metric-depth-keyframe-force-interval 10 \
    --metric-depth-high-motion-ratio 3.0 \
    --metric-depth-scale 0.75 \
    --enable-jetson-motion-gate --motion-gate-backend vpi_cpp \
    --motion-gate-threshold 12.0 --motion-gate-force-interval 8 \
    --motion-gate-resize 96,160 || exit $?
evaluate_smallcity smallcity200 "$smallcity200_root" || exit $?

run_variant \
  hotel344 1 "$output_parent/hotel344" \
  python scripts/run.py configs/rtg/hotel.yaml \
    --prefix cuda_graph_dba_phase0_hotel344 \
    --dataset-root "$hotel_root" \
    --output-dir "$output_parent/hotel344" \
    --frontend-weight "$frontend_weight" \
    --lightglue-weight-dir "$lightglue_root" \
    --loop-onnx-provider cpu \
    --frontend-image-size 344,616 \
    --frontend-save-buffer 512 \
    --frontend-iters1 3 --frontend-iters2 1 \
    --droid-update-backend torch \
    --profile-runtime \
    --export-eval --export-eval-interval 1 \
    --no-vis --skip-save-ply \
    --training-iters 30 \
    --adaptive-runtime \
    --enable-mapping-budget --enable-jetson-pruning --enable-pixel-budget \
    --enable-metric-depth-schedule --metric-depth-mode keyframe \
    --metric-depth-keyframe-min-interval 3 \
    --metric-depth-keyframe-force-interval 10 \
    --metric-depth-high-motion-ratio 3.0 \
    --metric-depth-scale 0.75 || exit $?

printf 'phase0_status=complete\n' > "$EVIDENCE_DIR/status.txt"
