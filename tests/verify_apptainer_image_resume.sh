#!/usr/bin/env bash
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
temp="$(mktemp -d)"
trap 'rm -rf -- "${temp}"' EXIT
mkdir -p "${temp}/bin"
printf 'sif\n' > "${temp}/candidate.sif"
printf 'definition-a\n' > "${temp}/current.def"

cat > "${temp}/bin/apptainer" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
case "$1" in
    inspect)
        cat <<JSON
{"data":{"attributes":{"labels":{"org.opencontainers.image.title":"extract-dvcs-gpd","org.opencontainers.image.version":"${FAKE_IMAGE_VERSION:-0.3.0}"}}}}
JSON
        ;;
    exec)
        sha256sum "${FAKE_EMBEDDED_DEFINITION}"
        ;;
    *) exit 64 ;;
esac
EOF
chmod +x "${temp}/bin/apptainer"

PATH="${temp}/bin:${PATH}" \
  FAKE_EMBEDDED_DEFINITION="${temp}/current.def" \
  "${root}/scripts/verify-apptainer-image.sh" \
  apptainer "${temp}/candidate.sif" 0.3.0 "${temp}/current.def"

if PATH="${temp}/bin:${PATH}" FAKE_IMAGE_VERSION=0.2.0 \
  "${root}/scripts/verify-apptainer-image.sh" \
  apptainer "${temp}/candidate.sif" 0.3.0 >/dev/null 2>&1; then
    echo "mismatched label version was accepted" >&2
    exit 1
fi

printf 'definition-b\n' > "${temp}/stale.def"
if PATH="${temp}/bin:${PATH}" \
  FAKE_EMBEDDED_DEFINITION="${temp}/stale.def" \
  "${root}/scripts/verify-apptainer-image.sh" \
  apptainer "${temp}/candidate.sif" 0.3.0 \
  "${temp}/current.def" >/dev/null 2>&1; then
    echo "stale embedded Apptainer definition was accepted" >&2
    exit 1
fi

grep -F 'resuming verification of completed partial SIF' \
  "${root}/install.sh" >/dev/null
grep -F 'another installer is already running' "${root}/install.sh" >/dev/null
grep -F 'install.lock.d' "${root}/install.sh" >/dev/null
if grep -F 'flock -n' "${root}/install.sh" >/dev/null; then
    echo "installer still uses unreliable NFS advisory locking" >&2
    exit 1
fi
echo "Apptainer image resume checks passed"
