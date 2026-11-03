#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${AUTODL_TMP:-/root/autodl-tmp}"
MIN_FREE_GIB="${MIN_FREE_GIB:-8}"

df -h /
if [[ -d "${DATA_ROOT}" ]]; then
  df -h "${DATA_ROOT}"
  du -sh "${DATA_ROOT}"/* 2>/dev/null | sort -h || true
else
  printf 'warning: data root does not exist: %s\n' "${DATA_ROOT}" >&2
fi

free_kib="$(df -Pk "${DATA_ROOT}" 2>/dev/null | awk 'NR==2 {print $4}')"
if [[ -z "${free_kib}" ]]; then
  printf 'warning: could not determine free space for %s\n' "${DATA_ROOT}" >&2
  exit 1
fi

minimum_kib=$((MIN_FREE_GIB * 1024 * 1024))
if (( free_kib < minimum_kib )); then
  printf 'error: less than %s GiB free on %s\n' "${MIN_FREE_GIB}" "${DATA_ROOT}" >&2
  exit 1
fi

printf 'storage check passed: at least %s GiB free\n' "${MIN_FREE_GIB}"

