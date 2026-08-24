#!/usr/bin/env bash
set -euo pipefail

: "${SWIF_JOB_WORK_DIR:?SWIF2 work directory is required}"
: "${SLURM_JOB_ID:?Slurm job ID is required}"
corpus="${1:?corpus}" image="${2:?image}" archive_output="${3:?output archive}"
summary_output="${4:?summary output}" heartbeat_seconds="${5:-300}"; shift 5
[[ "${heartbeat_seconds}" =~ ^[0-9]+$ ]] || exit 64
(($# >= 2)) || { echo "at least two batch archives are required" >&2; exit 64; }
batch_archives=("$@")
cd "${SWIF_JOB_WORK_DIR}"
[[ -s "${image}" && ! -L "${image}" ]] || exit 66
for archive in "${batch_archives[@]}"; do
    [[ -s "${archive}" && ! -L "${archive}" ]] || {
        echo "staged batch archive is missing or unsafe: ${archive}" >&2
        exit 66
    }
done
mkdir -p workspace/.corpus_exports results cache database
sha256sum "${image}" "${batch_archives[@]}" > staged-inputs-sha256.txt
image_sha="$(sha256sum "${image}" | awk '{print $1}')"
container=(
    /usr/bin/apptainer run --cleanenv
    --env DVCS_ACCELERATOR=cpu --env DVCS_PROGRESS=0
    --env "DVCS_IMAGE_DIGEST=sha256:${image_sha}"
    --env DVCS_WORKSPACE_ROOT=/workspace
    --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
    --env "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}"
    --bind "${SWIF_JOB_WORK_DIR}/workspace:/workspace"
    --bind "${SWIF_JOB_WORK_DIR}/results:/results"
    --bind "${SWIF_JOB_WORK_DIR}/cache:/cache"
    --bind "${SWIF_JOB_WORK_DIR}/database:/database:ro"
    "${image}"
)
batch_names=()
index=0
for archive in "${batch_archives[@]}"; do
    name="batch-${index}"
    cp --reflink=auto "${archive}" "workspace/.corpus_exports/${name}.tar.gz"
    "${container[@]}" corpus-checkpoint-import "${name}" "${name}" \
        > "import-${index}.json"
    batch_names+=("${name}")
    index=$((index + 1))
done
started_epoch="$(date +%s)"
echo "[dvcs-swif2] start stage=corpus-merge job=${SLURM_JOB_ID} host=$(hostname) batches=${#batch_archives[@]}"
set +e
"${container[@]}" corpus-merge --consume-sources "${corpus}" \
    "${batch_names[@]}" > corpus-merge.json 2> >(tee corpus-merge.err >&2) &
payload_pid=$!
heartbeat_pid=
if (( heartbeat_seconds > 0 )); then
    (
        while sleep "${heartbeat_seconds}"; do
            kill -0 "${payload_pid}" 2>/dev/null || exit 0
            echo "[dvcs-swif2] heartbeat stage=corpus-merge elapsed_seconds=$(($(date +%s) - started_epoch)) batches=${#batch_archives[@]}"
        done
    ) &
    heartbeat_pid=$!
fi
wait "${payload_pid}"
status=$?
if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null
    wait "${heartbeat_pid}" 2>/dev/null
fi
set -e
(( status == 0 )) || exit "${status}"
"${container[@]}" corpus-verify "${corpus}" --deep > corpus-verify.json
"${container[@]}" corpus-export "${corpus}" staged-final > corpus-export.json
cp workspace/.corpus_exports/staged-final.tar.gz "${archive_output}"
python3 - "${summary_output}" "${started_epoch}" <<'PY'
import json, os, sys, time
from pathlib import Path

verification = json.loads(Path("corpus-verify.json").read_text())
if verification.get("status") != "verified":
    raise SystemExit("merged corpus failed deep verification")
Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": 1,
    "status": "ok",
    "elapsed_seconds": int(time.time()) - int(sys.argv[2]),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "swif_job_id": os.environ.get("SWIF_JOB_ID"),
    "merge": json.loads(Path("corpus-merge.json").read_text()),
    "verification": verification,
}, indent=2, sort_keys=True) + "\n")
PY
echo "[dvcs-swif2] finish stage=corpus-merge elapsed_seconds=$(($(date +%s) - started_epoch))"
