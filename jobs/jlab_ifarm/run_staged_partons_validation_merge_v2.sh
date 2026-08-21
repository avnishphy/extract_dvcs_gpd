#!/usr/bin/env bash
set -euo pipefail

: "${SWIF_JOB_WORK_DIR:?SWIF2 work directory is required}"
: "${SLURM_JOB_ID:?Slurm job ID is required}"
: "${SLURM_CPUS_PER_TASK:?Slurm CPU allocation is required}"
cd "${SWIF_JOB_WORK_DIR}"

image="extract-dvcs-gpd-jlab_ifarm-0.2.0-validation.sif"
corpus="my_test_bigcorpus_validation_2-partons-validation-16384"
export_name="partons-corpus-validation-16384"
[[ -s "${image}" ]] || { echo "staged image is missing: ${image}" >&2; exit 66; }
for batch in $(seq 1 32); do
    [[ -s "corpus-batch-${batch}.tar.gz" ]] || {
        echo "staged corpus batch is missing: corpus-batch-${batch}.tar.gz" >&2
        exit 66
    }
done

mkdir -p workspace/.corpus_exports results cache database
sha256sum "${image}" corpus-batch-*.tar.gz > staged-inputs-sha256.txt
container=(
    /usr/bin/apptainer run --cleanenv
    --env DVCS_ACCELERATOR=cpu
    --env DVCS_WORKSPACE_ROOT=/workspace
    --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
    --env "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
    --bind "${SWIF_JOB_WORK_DIR}/workspace:/workspace"
    --bind "${SWIF_JOB_WORK_DIR}/results:/results"
    --bind "${SWIF_JOB_WORK_DIR}/cache:/cache"
    --bind "${SWIF_JOB_WORK_DIR}/database:/database:ro"
    "${image}"
)

started_epoch="$(date +%s)"
batch_names=()
for batch in $(seq 1 32); do
    mv "corpus-batch-${batch}.tar.gz" \
        "workspace/.corpus_exports/input-batch-${batch}.tar.gz"
    "${container[@]}" corpus-checkpoint-import "input-batch-${batch}" \
        "batch-${batch}" > "corpus-batch-${batch}-import.json"
    unlink "workspace/.corpus_exports/input-batch-${batch}.tar.gz"
    batch_names+=("batch-${batch}")
done
"${container[@]}" corpus-merge --consume-sources "${corpus}" \
    "${batch_names[@]}" > corpus-merge.json
"${container[@]}" corpus-verify "${corpus}" --deep > corpus-verify.json
"${container[@]}" corpus-export "${corpus}" "${export_name}" \
    > corpus-export.json
finished_epoch="$(date +%s)"

cp "workspace/.corpus_exports/${export_name}.tar.gz" "${export_name}.tar.gz"
python3 - "${started_epoch}" "${finished_epoch}" <<'PY'
import json, sys
from pathlib import Path

merge = json.loads(Path("corpus-merge.json").read_text())
verification = json.loads(Path("corpus-verify.json").read_text())
if merge.get("shard_count") != 1024 or verification.get("status") != "verified":
    raise SystemExit("merged corpus did not satisfy the complete corpus contract")
Path("partons-corpus-validation-16384-summary.json").write_text(json.dumps({
    "status": "ok",
    "elapsed_seconds": int(sys.argv[2]) - int(sys.argv[1]),
    "merge": merge,
    "verification": verification,
    "runtime": json.loads(Path("results/provenance/runtime.json").read_text()),
}, indent=2, sort_keys=True) + "\n")
PY
