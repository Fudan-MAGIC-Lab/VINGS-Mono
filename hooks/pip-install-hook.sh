#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/cn_mirror_hooks.sh"

if [[ "$#" -eq 0 ]]; then
  echo "Usage: hooks/pip-install-hook.sh <pip install args>" >&2
  echo "Example: hooks/pip-install-hook.sh torchvision==0.16.1" >&2
  exit 1
fi

pip_with_cn_mirror "$@"
