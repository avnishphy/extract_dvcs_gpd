#!/usr/bin/env bash
set -euo pipefail

: "${SWIF_JOB_WORK_DIR:?SWIF2 work directory is required}"
: "${SLURM_JOB_ID:?Slurm job ID is required}"
: "${SLURM_CPUS_PER_TASK:?Slurm CPU allocation is required}"
batch="${1:?usage: run_staged_partons_validation_chunk_v2.sh BATCH}"
[[ "${batch}" =~ ^([1-9]|[12][0-9]|3[0-2])$ ]] || {
    echo "batch must be an integer from 1 through 32" >&2
    exit 64
}
cd "${SWIF_JOB_WORK_DIR}"

image="extract-dvcs-gpd-jlab_ifarm-0.2.0-validation.sif"
database_archive="gpddatabase-v1.1.3.tar.gz"
experiment_input="experiment.json"
project="my_test_bigcorpus_validation_2"
corpus="my_test_bigcorpus_validation_2-partons-validation-16384-batch-${batch}"
export_name="corpus-batch-${batch}"
shard_start=$(( (batch - 1) * 32 ))

for path in "${image}" "${database_archive}" "${experiment_input}"; do
    [[ -s "${path}" ]] || { echo "staged input is missing: ${path}" >&2; exit 66; }
done

mkdir -p workspace/.corpus_exports results cache database
tar -xzf "${database_archive}" -C database
[[ "$(git -C database/gpddatabase rev-parse HEAD)" == \
    "1e9e97fd417ce1d6d44fc73550bdbf32cca4eeb1" ]] || exit 65
sha256sum "${image}" "${database_archive}" "${experiment_input}" > staged-inputs-sha256.txt

container=(
    /usr/bin/apptainer run --cleanenv
    --env DVCS_ACCELERATOR=cpu
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
cp "${experiment_input}" "workspace/${project}/experiment.json"
"${container[@]}" corpus-create "${project}" "${corpus}" \
    --profile validation --shard-size 16 > corpus-create.json

started_epoch="$(date +%s)"
"${container[@]}" corpus-generate "${project}" "${corpus}" \
    --shard-start "${shard_start}" --max-shards 32 --no-progress \
    > corpus-generate.json
"${container[@]}" corpus-checkpoint-export "${corpus}" "${export_name}" \
    > corpus-export.json
finished_epoch="$(date +%s)"

cp "workspace/.corpus_exports/${export_name}.tar.gz" "${export_name}.tar.gz"
python3 - "${batch}" "${shard_start}" "${started_epoch}" "${finished_epoch}" <<'PY'
import json, sys
from pathlib import Path

batch = int(sys.argv[1])
generate = json.loads(Path("corpus-generate.json").read_text())
expected = list(range(int(sys.argv[2]), int(sys.argv[2]) + 32))
if generate.get("generated_shard_indices") != expected:
    raise SystemExit("worker did not generate its declared shard range")
Path(f"corpus-batch-{batch}-summary.json").write_text(json.dumps({
    "status": "ok",
    "batch": batch,
    "shard_indices": expected,
    "elapsed_seconds": int(sys.argv[4]) - int(sys.argv[3]),
    "generation": generate,
    "runtime": json.loads(Path("results/provenance/runtime.json").read_text()),
}, indent=2, sort_keys=True) + "\n")
PY
