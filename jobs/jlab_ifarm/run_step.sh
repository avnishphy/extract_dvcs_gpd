#!/usr/bin/env bash
set -euo pipefail
step="${1:?step required}"; shift
job_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${job_dir}/resources.env"
[[ "${JLAB_ACCOUNT}" != REQUIRED_* ]] || { echo "set JLAB_ACCOUNT in resources.env" >&2; exit 64; }
[[ -d "${DVCS_REPOSITORY}" ]] || { echo "invalid DVCS_REPOSITORY" >&2; exit 66; }
export DVCS_WORKSPACE DVCS_RESULTS DVCS_CACHE DVCS_DATABASE
provenance="${DVCS_RESULTS}/slurm/${SLURM_JOB_ID:?}/${step}"
mkdir -p "${provenance}"
env | sort > "${provenance}/environment.txt"
scontrol show job "${SLURM_JOB_ID}" > "${provenance}/job.txt"
(lscpu || true) > "${provenance}/lscpu.txt" 2>&1
(nvidia-smi -q || true) > "${provenance}/nvidia-smi.txt" 2>&1
git -C "${DVCS_REPOSITORY}" rev-parse HEAD > "${provenance}/distribution-commit.txt"
cp "${DVCS_REPOSITORY}/provenance/images.lock.json" "${provenance}/images.lock.json"
set +e
"${DVCS_REPOSITORY}/dvcs" "$@" > >(tee "${provenance}/stdout.log") 2> >(tee "${provenance}/stderr.log" >&2)
status=$?
set -e
printf '%s\n' "${status}" > "${provenance}/exit-code.txt"
exit "${status}"
