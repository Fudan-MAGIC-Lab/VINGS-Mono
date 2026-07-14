#!/usr/bin/env bash

set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
shared_root="$repo_root"
if [ "$(basename "$(dirname "$repo_root")")" = ".worktrees" ]; then
  shared_root="$(cd "$repo_root/../.." && pwd)"
fi

EVIDENCE_DIR="${VINGS_PHASE05_EVIDENCE_DIR:-$shared_root/reports/cuda_graph_dba/phase05_20260714}"
output_parent="${VINGS_PHASE05_OUTPUT_DIR:-$shared_root/output/cuda_graph_dba_phase05_20260714}"

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
  git submodule status --recursive \
    > "$EVIDENCE_DIR/git_submodule_status.txt"
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
    print("memory", torch.cuda.mem_get_info())
PY
}

verify_weighted_sample_coverage() {
  local details=$1
  local label=$2
  python - "$details" "$label" <<'PY' \
    > "$EVIDENCE_DIR/coverage_${label}.txt"
import json
import sys
from collections import Counter


def signature_key(payload):
    return (
        int(payload["active_edges"]),
        int(payload["ba_edges"]),
        int(payload["source_poses"]),
        int(payload["pose_window"]),
        bool(payload["use_inactive"]),
        bool(payload["upsample"]),
        str(payload["dtype"]),
    )


details, label = sys.argv[1:]
observed = Counter()
sampled = set()
with open(details) as handle:
    for line in handle:
        row = json.loads(line)
        if row.get("kind") == "dba_signature":
            observed[signature_key(row["payload"])] += 1
        elif row.get("kind") == "dba_memory_sample":
            sampled.add(signature_key(row["payload"]["signature"]))
total = sum(observed.values())
covered = sum(count for key, count in observed.items() if key in sampled)
weighted_sample_coverage = covered / total if total else 0.0
print("label", label)
print("observed_calls", total)
print("sampled_signatures", len(sampled))
print("weighted_sample_coverage", weighted_sample_coverage)
if weighted_sample_coverage < 0.95:
    raise SystemExit(8)
PY
}

run_variant() {
  local label=$1
  local variant_output=$2
  shift 2

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
  local details="$run_dir/runtime_profile_details.jsonl"
  printf '%s\n' "$run_dir" > "$EVIDENCE_DIR/run_dir_${label}.txt"
  printf '%s\n' "$details" > "$EVIDENCE_DIR/details_${label}.txt"

  if [ ! -s "$details" ]; then
    printf 'Missing runtime profile details for %s: %s\n' \
      "$label" "$details" >&2
    return 5
  fi
  if ! grep -q '"kind": "dba_signature"' "$details"; then
    printf 'No DBA signatures recorded for %s\n' "$label" >&2
    return 6
  fi
  if ! grep -q '"kind": "dba_memory_sample"' "$details"; then
    printf 'No DBA memory samples recorded for %s\n' "$label" >&2
    return 7
  fi
  verify_weighted_sample_coverage "$details" "$label"
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

calibrate_and_plan() {
  local short_name=$1
  local boundary=$2
  local calibration="$EVIDENCE_DIR/phase05_${short_name}_memory_calibration.json"
  local report="$EVIDENCE_DIR/phase05_${short_name}_memory_calibration.md"
  local manifest="$EVIDENCE_DIR/phase05_${short_name}_bucket_manifest.json"
  local manifest_report="$EVIDENCE_DIR/phase05_${short_name}_bucket_manifest.md"
  local smallcity_details
  local hotel_details
  smallcity_details=$(cat "$EVIDENCE_DIR/details_smallcity200.txt")
  hotel_details=$(cat "$EVIDENCE_DIR/details_hotel344.txt")

  python scripts/profiling/calibrate_dba_workspace.py \
    --details "$smallcity_details" \
    --details "$hotel_details" \
    --capture-boundary "$boundary" \
    --output-calibration "$calibration" \
    --output-report "$report" \
    > "$EVIDENCE_DIR/calibrate_${short_name}.log" 2>&1
  local calibration_status=$?
  if [ ! -s "$calibration" ]; then
    printf 'Calibration did not produce an artifact for %s\n' \
      "$boundary" >&2
    return "$calibration_status"
  fi
  if [ "$calibration_status" -ne 0 ]; then
    printf '%s=NO-GO\n' "$boundary" \
      >> "$EVIDENCE_DIR/boundary_status.txt"
    return 0
  fi

  python scripts/profiling/plan_dba_buckets.py \
    --details "$smallcity_details" \
    --details "$hotel_details" \
    --memory-calibration "$calibration" \
    --capture-boundary "$boundary" \
    --output-manifest "$manifest" \
    --output-report "$manifest_report" \
    --target-coverage 0.90 \
    --max-buckets 6 \
    --max-padding-ratio 0.35 \
    --max-workspace-mb 1024 \
    > "$EVIDENCE_DIR/plan_${short_name}.log" 2>&1 || return $?
  printf '%s=VALID\n' "$boundary" \
    >> "$EVIDENCE_DIR/boundary_status.txt"
}

snapshot_environment

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
    exit 9
  fi
done
for required_dir in \
  "$smallcity200_root" \
  "$hotel_root/nosky_color"; do
  if [ ! -d "$required_dir" ]; then
    printf 'Missing required dataset directory: %s\n' \
      "$required_dir" >&2
    exit 9
  fi
done

run_variant \
  smallcity200 "$output_parent/smallcity200" \
  python scripts/run.py configs/hierarchical/smallcity.yaml \
    --prefix cuda_graph_dba_phase05_smallcity200 \
    --dataset-root "$smallcity200_root" \
    --output-dir "$output_parent/smallcity200" \
    --frontend-weight "$frontend_weight" \
    --frontend-image-size 344,616 \
    --frontend-save-buffer 64 \
    --droid-update-backend torch \
    --profile-runtime \
    --profile-dba-memory \
    --profile-dba-memory-samples-per-signature 2 \
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
  hotel344 "$output_parent/hotel344" \
  python scripts/run.py configs/rtg/hotel.yaml \
    --prefix cuda_graph_dba_phase05_hotel344 \
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
    --profile-dba-memory \
    --profile-dba-memory-samples-per-signature 2 \
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

: > "$EVIDENCE_DIR/boundary_status.txt"
calibrate_and_plan corr corr_update_aggregation || exit $?
calibrate_and_plan update update_aggregation_only || exit $?

printf 'phase05_status=complete\n' > "$EVIDENCE_DIR/status.txt"
