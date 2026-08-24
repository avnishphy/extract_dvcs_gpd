#!/usr/bin/env bash
set -euo pipefail

: "${SWIF_JOB_WORK_DIR:?SWIF2 work directory is required}"
: "${SLURM_JOB_ID:?Slurm job ID is required}"
: "${SLURM_CPUS_PER_TASK:?Slurm CPU allocation is required}"
project="${1:?project}" profile="${2:?profile}" corpus="${3:?corpus}"
image="${4:?image}" database_archive="${5:?database archive}"
experiment="${6:?experiment}" shard_size="${7:?shard size}"
shard_start="${8:?shard start}" shard_count="${9:?shard count}"
archive_output="${10:?archive output}" summary_output="${11:?summary output}"
heartbeat_seconds="${12:-300}"
[[ "${heartbeat_seconds}" =~ ^[0-9]+$ ]] || exit 64
cd "${SWIF_JOB_WORK_DIR}"
for path in "${image}" "${database_archive}" "${experiment}"; do
    [[ -s "${path}" && ! -L "${path}" ]] || exit 66
done
python3 - "${database_archive}" <<'PY'
import sys, tarfile
from pathlib import PurePosixPath

with tarfile.open(sys.argv[1], "r:*") as archive:
    members = archive.getmembers()
    if not members:
        raise SystemExit("database archive is empty")
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise SystemExit(f"unsafe database archive member: {member.name}")
PY
mkdir -p workspace/.corpus_exports results cache database
tar -xf "${database_archive}" -C database
[[ "$(git -C database/gpddatabase rev-parse HEAD)" == \
   "1e9e97fd417ce1d6d44fc73550bdbf32cca4eeb1" ]] || exit 65
sha256sum "${image}" "${database_archive}" "${experiment}" > staged-inputs-sha256.txt
image_sha="$(sha256sum "${image}" | awk '{print $1}')"
container=(
    /usr/bin/apptainer run --cleanenv
    --env DVCS_ACCELERATOR=cpu --env DVCS_PROGRESS=0
    --env "DVCS_IMAGE_DIGEST=sha256:${image_sha}"
    --env DVCS_GPDDATABASE_ROOT=/database/gpddatabase
    --env DVCS_WORKSPACE_ROOT=/workspace
    --env DVCS_NATIVE_WORKERS=all_available
    --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
    --env "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
    --bind "${SWIF_JOB_WORK_DIR}/workspace:/workspace"
    --bind "${SWIF_JOB_WORK_DIR}/results:/results"
    --bind "${SWIF_JOB_WORK_DIR}/cache:/cache"
    --bind "${SWIF_JOB_WORK_DIR}/database:/database:ro"
    "${image}"
)
"${container[@]}" init "${project}" > init.json
cp "${experiment}" "workspace/${project}/experiment.json"
"${container[@]}" corpus-create "${project}" "${corpus}" \
    --profile "${profile}" --shard-size "${shard_size}" > corpus-create.json
started_epoch="$(date +%s)"
echo "[dvcs-swif2] start stage=corpus-worker job=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} shards=${shard_start}+$((shard_count))"
set +e
"${container[@]}" corpus-generate "${project}" "${corpus}" \
    --shard-start "${shard_start}" --max-shards "${shard_count}" \
    --no-progress > corpus-generate.json 2> >(tee corpus-generate.err >&2) &
payload_pid=$!
heartbeat_pid=
if (( heartbeat_seconds > 0 )); then
    (
        while sleep "${heartbeat_seconds}"; do
            kill -0 "${payload_pid}" 2>/dev/null || exit 0
            echo "[dvcs-swif2] heartbeat stage=corpus-worker elapsed_seconds=$(($(date +%s) - started_epoch)) shards=${shard_start}+$((shard_count))"
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
"${container[@]}" corpus-checkpoint-export "${corpus}" staged-batch \
    > corpus-export.json
cp workspace/.corpus_exports/staged-batch.tar.gz "${archive_output}"
python3 - "${summary_output}" "${shard_start}" "${shard_count}" \
  "${started_epoch}" <<'PY'
import json, os, sys, time
from pathlib import Path

output, start, count, started = sys.argv[1:]
generation = json.loads(Path("corpus-generate.json").read_text())
expected = list(range(int(start), int(start) + int(count)))
if generation.get("generated_shard_indices") != expected:
    raise SystemExit("worker shard coverage mismatch")
Path(output).write_text(json.dumps({
    "schema_version": 1,
    "status": "ok",
    "shard_indices": expected,
    "elapsed_seconds": int(time.time()) - int(started),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "swif_job_id": os.environ.get("SWIF_JOB_ID"),
    "generation": generation,
}, indent=2, sort_keys=True) + "\n")
PY
echo "[dvcs-swif2] finish stage=corpus-worker elapsed_seconds=$(($(date +%s) - started_epoch))"
