#!/usr/bin/env bash

# Source this file to enable mirror-first install helpers.
# Example:
#   source hooks/cn_mirror_hooks.sh
#   pipi install torchvision==0.16.1

set -euo pipefail

# Mirror order is intentionally explicit so we can fail over quickly.
CN_PIP_MIRRORS=(
  "https://pypi.tuna.tsinghua.edu.cn/simple"
  "https://mirrors.ustc.edu.cn/pypi/web/simple"
  "https://mirrors.aliyun.com/pypi/simple"
)

PIP_FALLBACK_INDEX="https://pypi.org/simple"

_pip_try_install() {
  local index_url="$1"
  shift

  python -m pip install \
    --index-url "$index_url" \
    --default-timeout "${PIP_TIMEOUT:-1000}" \
    --retries "${PIP_RETRIES:-20}" \
    "$@"
}

_pip_try_download() {
  local index_url="$1"
  shift

  python -m pip download \
    --index-url "$index_url" \
    --default-timeout "${PIP_TIMEOUT:-1000}" \
    --retries "${PIP_RETRIES:-20}" \
    "$@"
}

pip_with_cn_mirror() {
  local ok=0
  local mirror

  for mirror in "${CN_PIP_MIRRORS[@]}"; do
    echo "[mirror-hook] Trying pip install via: ${mirror}" >&2
    if _pip_try_install "$mirror" "$@"; then
      ok=1
      break
    fi
    echo "[mirror-hook] Failed on ${mirror}, trying next mirror..." >&2
  done

  if [[ "$ok" -eq 0 ]]; then
    echo "[mirror-hook] Mirrors failed, fallback to official: ${PIP_FALLBACK_INDEX}" >&2
    _pip_try_install "$PIP_FALLBACK_INDEX" "$@"
  fi
}

pip_download_with_cn_mirror() {
  local ok=0
  local mirror

  for mirror in "${CN_PIP_MIRRORS[@]}"; do
    echo "[mirror-hook] Trying pip download via: ${mirror}" >&2
    if _pip_try_download "$mirror" "$@"; then
      ok=1
      break
    fi
    echo "[mirror-hook] Failed on ${mirror}, trying next mirror..." >&2
  done

  if [[ "$ok" -eq 0 ]]; then
    echo "[mirror-hook] Mirrors failed, fallback to official: ${PIP_FALLBACK_INDEX}" >&2
    _pip_try_download "$PIP_FALLBACK_INDEX" "$@"
  fi
}

# Short aliases for day-to-day use in this shell session.
alias pipi='pip_with_cn_mirror'
alias pipd='pip_download_with_cn_mirror'
