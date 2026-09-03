#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
temporary="$(mktemp -d)"
trap 'rm -rf -- "${temporary}"' EXIT
log="${temporary}/retry.log"

if DVCS_APT_CONFIG_DIR="${temporary}/apt" \
   DVCS_APT_COMMAND=/bin/false \
   DVCS_APT_RETRY_ATTEMPTS=2 \
   DVCS_APT_RETRY_DELAY_BASE=0 \
   /bin/sh "${root}/scripts/install-system-packages.sh" \
     20260801T000000Z test-package >"${log}" 2>&1; then
    echo "failing APT fixture unexpectedly succeeded" >&2
    exit 1
fi
grep -F 'apt-get attempt 1/2 failed; retrying in 0s: update' "${log}" >/dev/null
grep -F 'apt-get failed after 2 attempts: update' "${log}" >/dev/null
grep -F 'Acquire::Retries "3";' \
  "${temporary}/apt/80dvcs-network-retries" >/dev/null
grep -F 'Acquire::https::Timeout "45";' \
  "${temporary}/apt/80dvcs-network-retries" >/dev/null
echo "system-package retry verification: PASS"
