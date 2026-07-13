#!/usr/bin/env bash
set -uo pipefail

cd /home/jetson/VINGS-Mono
source /home/jetson/miniconda3/etc/profile.d/conda.sh
conda activate vings_jetson

PROCESS_SNAPSHOT=reports/tensorrt_fp16/processes_before_droid_encoder_build.txt
if pgrep -af '[s]cripts/run.py|[t]rtexec|[t]egrastats' > "$PROCESS_SNAPSHOT"; then
  printf 'A conflicting benchmark/build process already exists.\n' >&2
  exit 3
fi

build_encoder() {
  local encoder=$1
  local onnx=engines/tensorrt/droid/droid_${encoder}_b1_344x616.onnx
  local plan=engines/tensorrt/droid/droid_${encoder}_b1_344x616_fp16.plan
  local log=reports/tensorrt_fp16/droid_${encoder}_trtexec_build.log
  local resource_log=reports/tensorrt_fp16/droid_${encoder}_trtexec_tegrastats.log
  local status=reports/tensorrt_fp16/droid_${encoder}_trtexec_status.txt
  local tegra_pid=""

  if [ ! -f "$onnx" ]; then
    printf 'status=4 reason=onnx_missing\n' > "$status"
    return 4
  fi
  if [ -e "$plan" ]; then
    printf 'status=2 reason=plan_exists\n' > "$status"
    return 2
  fi

  cleanup() {
    if [ -n "$tegra_pid" ]; then
      kill "$tegra_pid" 2>/dev/null || true
      wait "$tegra_pid" 2>/dev/null || true
    fi
  }
  trap cleanup EXIT INT TERM
  tegrastats --interval 1000 > "$resource_log" 2>&1 &
  tegra_pid=$!

  local started
  started=$(date +%s)
  /usr/src/tensorrt/bin/trtexec \
    --onnx="$onnx" \
    --saveEngine="$plan" \
    --fp16 \
    --memPoolSize=workspace:1024MiB \
    --buildOnly \
    --verbose \
    --dumpLayerInfo \
    > "$log" 2>&1
  local build_status=$?
  local elapsed=$(($(date +%s)-started))

  cleanup
  trap - EXIT INT TERM
  tegra_pid=""
  printf 'status=%s elapsed_s=%s\n' "$build_status" "$elapsed" > "$status"
  return "$build_status"
}

if [ "$#" -eq 1 ]; then
  case "$1" in
    fnet|cnet) ;;
    *) printf 'unsupported encoder: %s\n' "$1" >&2; exit 5 ;;
  esac
  build_encoder "$1"
  exit $?
elif [ "$#" -gt 1 ]; then
  printf 'usage: %s [fnet|cnet]\n' "$0" >&2
  exit 5
fi

build_encoder fnet
FNET_STATUS=$?
if [ "$FNET_STATUS" -ne 0 ]; then
  exit "$FNET_STATUS"
fi

build_encoder cnet
CNET_STATUS=$?
exit "$CNET_STATUS"
