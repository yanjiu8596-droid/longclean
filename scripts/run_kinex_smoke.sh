#!/usr/bin/env bash
set -euo pipefail

workspace="${1:-/root/autodl-tmp/tableclear-workspace}"
project="${2:-/root/autodl-tmp/longclean}"
output="${workspace}/results/smoke-$(date -u +%Y%m%dT%H%M%SZ)"

mkdir -p "${output}"
export ROBOENV_ENDPOINT="${ROBOENV_ENDPOINT:-http://127.0.0.1:8123}"
export PATH="/root/autodl-tmp/kinex-data/node/current/bin:/root/.local/bin:${PATH}"
prompt="$(<"${project}/configs/tableclear_agent_prompt.md")"

cd "${workspace}"
kinex -p --mode json "${prompt}" \
  >"${output}/kinex-events.jsonl" \
  2>"${output}/kinex.stderr.log"

printf 'Kinex smoke output: %s\n' "${output}"

