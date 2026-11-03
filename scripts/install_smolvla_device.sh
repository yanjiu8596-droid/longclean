#!/usr/bin/env bash
set -euo pipefail

workspace="${1:-/root/autodl-tmp/tableclear-workspace}"
project="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${workspace}/.kinex/devices/smolvla"

if [[ -e "${destination}" ]]; then
  printf 'Refusing to overwrite existing device: %s\n' "${destination}" >&2
  exit 1
fi

mkdir -p "${workspace}/.kinex/devices" "${workspace}/models"
cp -r "${project}/kinex_devices/smolvla" "${destination}"

printf 'Installed SmolVLA device at %s\n' "${destination}"
printf 'Download weights with:\n'
printf '  source /etc/network_turbo\n'
printf '  .kinex/model-python/bin/python %s --output %s\n' \
  "${project}/scripts/download_smolvla.py" \
  "${workspace}/models/smolvla_libero"

