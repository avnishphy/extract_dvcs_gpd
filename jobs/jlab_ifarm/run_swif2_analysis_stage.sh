#!/usr/bin/env bash
set -euo pipefail

: "${SWIF_JOB_WORK_DIR:?SWIF2 work directory is required}"
: "${SLURM_JOB_ID:?Slurm job ID is required}"
: "${SLURM_CPUS_PER_TASK:?Slurm CPU allocation is required}"

stage="${1:?stage}" project="${2:?project}" profile="${3:?profile}"
corpus="${4:?corpus}" selection="${5:?selection}" image="${6:?image}"
database_archive="${7:?database archive}" corpus_archive="${8:?corpus archive or none}"
state_input="${9:?project input}" state_output="${10:?project output}"
summary_output="${11:?summary output}" accelerator="${12:?accelerator}"
heartbeat_seconds="${13:?heartbeat seconds}"

[[ "${stage}" =~ ^(selection|optimize|train|evaluate|compare|holdout|plot)$ ]] || {
    echo "unsupported analysis stage: ${stage}" >&2
    exit 64
}
[[ "${accelerator}" == cpu || "${accelerator}" == cuda ]] || exit 64
[[ "${heartbeat_seconds}" =~ ^[0-9]+$ ]] || exit 64
cd "${SWIF_JOB_WORK_DIR}"

for path in "${image}" "${database_archive}" "${state_input}"; do
    [[ -s "${path}" && ! -L "${path}" ]] || {
        echo "staged input is missing or unsafe: ${path}" >&2
        exit 66
    }
done
if [[ "${corpus_archive}" != none ]]; then
    [[ -s "${corpus_archive}" && ! -L "${corpus_archive}" ]] || exit 66
fi

safe_archive() {
    python3 - "$1" <<'PY'
import sys, tarfile
from pathlib import PurePosixPath

with tarfile.open(sys.argv[1], "r:*") as archive:
    members = archive.getmembers()
    if not members:
        raise SystemExit("archive is empty")
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise SystemExit(f"unsafe archive member: {member.name}")
PY
}

safe_archive "${database_archive}"
safe_archive "${state_input}"
mkdir -p workspace/.corpus_exports results cache database
tar -xf "${database_archive}" -C database
tar -xf "${state_input}" -C workspace
[[ -f "workspace/${project}/experiment.json" ]] || {
    echo "project archive does not contain ${project}/experiment.json" >&2
    exit 66
}
[[ "$(git -C database/gpddatabase rev-parse HEAD)" == \
   "1e9e97fd417ce1d6d44fc73550bdbf32cca4eeb1" ]] || {
    echo "staged GPD database revision mismatch" >&2
    exit 65
}

image_sha="$(sha256sum "${image}" | awk '{print $1}')"
sha256sum "${image}" "${database_archive}" "${state_input}" > staged-inputs-sha256.txt
if [[ "${corpus_archive}" != none ]]; then
    sha256sum "${corpus_archive}" >> staged-inputs-sha256.txt
fi

container=(
    /usr/bin/apptainer run --cleanenv
    --env "DVCS_ACCELERATOR=${accelerator}"
    --env "DVCS_IMAGE_DIGEST=sha256:${image_sha}"
    --env DVCS_PROGRESS=0
    --env DVCS_GPDDATABASE_ROOT=/database/gpddatabase
    --env DVCS_WORKSPACE_ROOT=/workspace
    --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
    --env "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
    --bind "${SWIF_JOB_WORK_DIR}/workspace:/workspace"
    --bind "${SWIF_JOB_WORK_DIR}/results:/results"
    --bind "${SWIF_JOB_WORK_DIR}/cache:/cache"
    --bind "${SWIF_JOB_WORK_DIR}/database:/database:ro"
)
if [[ "${accelerator}" == cuda ]]; then
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != NoDevFiles ]] || {
        echo "GPU stage has no visible Slurm GPU allocation" >&2
        exit 69
    }
    container+=(--nv --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi
container+=("${image}")

if [[ "${corpus_archive}" != none ]]; then
    cp --reflink=auto "${corpus_archive}" workspace/.corpus_exports/staged-corpus.tar.gz
    "${container[@]}" corpus-import staged-corpus "${corpus}" > corpus-import.json
fi

case "${stage}" in
    selection)
        payload=(selection-create "${project}" "${corpus}" "${selection}" --profile "${profile}") ;;
    optimize)
        payload=(optimize "${project}" --profile "${profile}" --corpus "${corpus}" --selection "${selection}" --no-progress) ;;
    train)
        payload=(train "${project}" --profile "${profile}" --corpus "${corpus}" --selection "${selection}" --no-progress) ;;
    evaluate)
        payload=(evaluate "${project}" --profile "${profile}" --no-progress) ;;
    compare)
        payload=(compare "${project}" --profile "${profile}" --no-progress) ;;
    holdout)
        payload=(holdout "${project}" --profile "${profile}" --no-progress) ;;
    plot)
        payload=(plot "${project}" --profile "${profile}") ;;
esac

progress_snapshot() {
    python3 - "${stage}" "workspace/${project}/results/${profile}" <<'PY' 2>/dev/null || true
import json, sys
from pathlib import Path

stage, root = sys.argv[1], Path(sys.argv[2])
materialization = root / "materialization_progress.json"
if materialization.is_file():
    record = json.loads(materialization.read_text())
    print(f"materialization={record.get('completed', 0)}/{record.get('total', '?')}", end=" ")
if stage == "train":
    print(f"models={len(list((root / 'training').glob('seed_*/training_metrics.json')))}", end="")
elif stage == "optimize":
    print(f"trials={len(list((root / 'optimization/trials').glob('trial_*')))}", end="")
PY
}

started_epoch="$(date +%s)"
echo "[dvcs-swif2] start stage=${stage} job=${SWIF_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} gpus=${CUDA_VISIBLE_DEVICES:-none}"
set +e
"${container[@]}" "${payload[@]}" > payload.json 2> >(tee payload.err >&2) &
payload_pid=$!
heartbeat_pid=
if (( heartbeat_seconds > 0 )); then
    (
        while sleep "${heartbeat_seconds}"; do
            kill -0 "${payload_pid}" 2>/dev/null || exit 0
            echo "[dvcs-swif2] heartbeat stage=${stage} elapsed_seconds=$(($(date +%s) - started_epoch)) $(progress_snapshot)"
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

python3 - "${summary_output}" "${stage}" "${status}" "${started_epoch}" <<'PY'
import json, os, platform, sys, time
from pathlib import Path

output, stage, status, started = sys.argv[1:]
record = {
    "schema_version": 1,
    "status": "ok" if int(status) == 0 else "failed",
    "stage": stage,
    "exit_code": int(status),
    "elapsed_seconds": int(time.time()) - int(started),
    "hostname": platform.node(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "swif_job_id": os.environ.get("SWIF_JOB_ID"),
    "swif_job_attempt_id": os.environ.get("SWIF_JOB_ATTEMPT_ID"),
}
runtime = Path("results/provenance/runtime.json")
if runtime.is_file():
    record["runtime"] = json.loads(runtime.read_text())
Path(output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
(( status == 0 )) || exit "${status}"

if find "workspace/${project}" -type l -print -quit | grep -q .; then
    echo "project output contains a symlink" >&2
    exit 65
fi
tar -cf "${state_output}.partial" -C workspace "${project}"
mv "${state_output}.partial" "${state_output}"
echo "[dvcs-swif2] finish stage=${stage} elapsed_seconds=$(($(date +%s) - started_epoch))"
