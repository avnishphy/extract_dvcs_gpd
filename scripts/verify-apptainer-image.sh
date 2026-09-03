#!/usr/bin/env bash
set -euo pipefail

engine="${1:?Apptainer command is required}"
image="${2:?SIF path is required}"
expected_version="${3:?expected distribution version is required}"
expected_definition="${4:-}"

[[ -s "${image}" ]] || {
    echo "missing or empty SIF: ${image}" >&2
    exit 65
}

"${engine}" inspect --json "${image}" 2>/dev/null | python3 -c '
import json, sys

expected = sys.argv[1]
document = json.load(sys.stdin)
labels = document["data"]["attributes"]["labels"]
if labels.get("org.opencontainers.image.title") != "extract-dvcs-gpd":
    raise SystemExit("unexpected Apptainer image title")
if labels.get("org.opencontainers.image.version") != expected:
    raise SystemExit("unexpected Apptainer image version")
' "${expected_version}"

if [[ -n "${expected_definition}" ]]; then
    expected_hash="$(sha256sum "${expected_definition}" | awk '{print $1}')"
    embedded_hash="$(
        "${engine}" exec --cleanenv "${image}" sha256sum \
          /opt/dvcs/app/containers/apptainer.def | awk '{print $1}'
    )"
    [[ "${embedded_hash}" == "${expected_hash}" ]] || {
        echo "embedded Apptainer definition does not match current source" >&2
        exit 65
    }
fi
