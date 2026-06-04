#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/cn_mirror_hooks.sh"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: hooks/pip-download-hook.sh <pip download args>" >&2
  echo "Example: hooks/pip-download-hook.sh torchvision==0.16.1 -d /tmp/wheels" >&2
  exit 1
fi

pip_download_with_cn_mirror "$@"
