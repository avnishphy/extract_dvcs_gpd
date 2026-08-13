#!/usr/bin/env bash
set -euo pipefail
image="${1:?image reference required}"
output="${2:-artifacts/sbom.spdx.json}"
mkdir -p "$(dirname -- "${output}")"
if command -v syft >/dev/null 2>&1; then
    exec syft "${image}" -o "spdx-json=${output}"
fi
echo "syft is required to generate the release SBOM" >&2
exit 69
