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

heartbeat_seconds="${JLAB_HEARTBEAT_SECONDS:-300}"
[[ "${heartbeat_seconds}" =~ ^[0-9]+$ ]] || {
    echo "JLAB_HEARTBEAT_SECONDS must be a nonnegative integer" >&2
    exit 64
}
started_epoch="$(date +%s)"

progress_snapshot() {
    python3 - "${step}" "${DVCS_WORKSPACE}" "${DVCS_PROJECT}" \
      "${DVCS_CORPUS:-}" <<'PY' 2>/dev/null || true
import json
from pathlib import Path
import sys

step, workspace, project, corpus = sys.argv[1:]
root = Path(workspace)
if step == "generate" and corpus:
    record = json.loads((root / ".corpora" / corpus / "corpus.json").read_text())
    print(f"shards={len(record.get('core_shards', []))}/{record['shard_count']}")
elif step == "train":
    experiment = json.loads((root / project / "experiment.json").read_text())
    total = len(experiment["inference"]["profiles"]["validation"]["ensemble_seeds"])
    completed = len(list((root / project / "results" / "validation" / "training").glob(
        "seed_*/training_metrics.json"
    )))
    print(f"ensemble_members={completed}/{total}")
elif step == "optimize":
    experiment = json.loads((root / project / "experiment.json").read_text())
    total = experiment["inference"]["hyperparameter_optimization"]["trials"]["validation"]
    completed = len(list((root / project / "results" / "validation" / "optimization").glob(
        "trials/trial_*/workspace/training/training_summary.json"
    )))
    print(f"completed_trials={completed}/{total}")
PY
}

log_progress() {
    printf '%s\n' "$1" | tee -a "${provenance}/progress.log"
}

heartbeat_loop() {
    local payload_pid="$1" now elapsed snapshot suffix
    while sleep "${heartbeat_seconds}"; do
        kill -0 "${payload_pid}" 2>/dev/null || return
        now="$(date +%s)"
        elapsed=$((now - started_epoch))
        snapshot="$(progress_snapshot)"
        suffix=
        [[ -z "${snapshot}" ]] || suffix=" progress=${snapshot}"
        log_progress "[dvcs-job] heartbeat utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) job=${SLURM_JOB_ID} step=${step} elapsed_seconds=${elapsed}${suffix}"
    done
}

log_progress "[dvcs-job] start utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) job=${SLURM_JOB_ID} step=${step} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-unknown} gpus=${CUDA_VISIBLE_DEVICES:-none} heartbeat_seconds=${heartbeat_seconds}"
set +e
"${DVCS_REPOSITORY}/dvcs" "$@" > >(tee "${provenance}/stdout.log") 2> >(tee "${provenance}/stderr.log" >&2) &
payload_pid=$!
heartbeat_pid=
if (( heartbeat_seconds > 0 )); then
    heartbeat_loop "${payload_pid}" &
    heartbeat_pid=$!
fi
wait "${payload_pid}"
status=$?
if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null
    wait "${heartbeat_pid}" 2>/dev/null
fi
set -e
printf '%s\n' "${status}" > "${provenance}/exit-code.txt"
log_progress "[dvcs-job] finish utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) job=${SLURM_JOB_ID} step=${step} elapsed_seconds=$(($(date +%s) - started_epoch)) exit_code=${status}"
exit "${status}"
