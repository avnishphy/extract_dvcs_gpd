#!/usr/bin/env bash
set -euo pipefail

job_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
resources="${job_dir}/resources.env"
[[ -f "${resources}" ]] || { echo "copy resources.env.example to resources.env and edit it" >&2; exit 66; }
set -a
# shellcheck disable=SC1090
source "${resources}"
set +a
project="${DVCS_PROJECT:-}" corpus="${DVCS_CORPUS:-}" profile="${DVCS_PROFILE_NAME:-validation}"
workflow= shard_size=16 shards_per_worker=32 dry_run=false import_only=false
while (($#)); do
    case "$1" in
        --project) project="${2:?missing project}"; shift 2 ;;
        --corpus) corpus="${2:?missing corpus}"; shift 2 ;;
        --profile) profile="${2:?missing profile}"; shift 2 ;;
        --workflow) workflow="${2:?missing workflow}"; shift 2 ;;
        --shard-size) shard_size="${2:?missing shard size}"; shift 2 ;;
        --shards-per-worker) shards_per_worker="${2:?missing count}"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --import-only) import_only=true; shift ;;
        -h|--help)
            echo "usage: ./dvcs farm-corpus-submit --project NAME --corpus NAME [--profile NAME] [--shard-size N] [--shards-per-worker N] [--dry-run|--import-only]"
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done
for value in "${project}" "${corpus}" "${profile}"; do
    [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 64
done
[[ "${shard_size}" =~ ^[1-9][0-9]*$ && "${shards_per_worker}" =~ ^[1-9][0-9]*$ ]] || exit 64
[[ "${JLAB_ACCOUNT:-}" != REQUIRED_* && -n "${JLAB_ACCOUNT:-}" ]] || { echo "set JLAB_ACCOUNT in resources.env" >&2; exit 64; }
workflow="${workflow:-dvcs-corpus-${project}-$(date -u +%Y%m%dT%H%M%SZ)}"
assets_env="$(${job_dir}/prepare_swif2_corpus_inputs.sh "${project}")"
# shellcheck disable=SC1090
source "${assets_env}"
result_root="${SWIF_OUTPUT_ROOT:-${DVCS_RESULTS}/swif2}/${workflow}"
log_root="${SWIF_LOG_ROOT:-/farm_out/${USER}/dvcs}/${workflow}"
workflow_json="${job_dir}/swif-${workflow}.json"
python3 "${job_dir}/write_swif2_workflow.py" corpus \
    --workflow "${workflow}" --account "${JLAB_ACCOUNT}" \
    --site "${SWIF_SITE_NAME:-jlab/enp}" --constraint "${SWIF_CONSTRAINT:-el9}" \
    --placement-profile "${SWIF_CORPUS_PLACEMENT_PROFILE:-farm25_strict}" \
    --result-root "${result_root}" --log-root "${log_root}" \
    --image "${SWIF_IMAGE}" --database "${SWIF_DATABASE_ARCHIVE}" \
    --project "${project}" --profile "${profile}" --corpus "${corpus}" \
    --experiment "${SWIF_EXPERIMENT}" \
    --gpd-truth-request "${SWIF_GPD_TRUTH_REQUEST}" \
    --shard-size "${shard_size}" \
    --shards-per-worker "${shards_per_worker}" \
    --heartbeat-seconds "${SWIF_HEARTBEAT_SECONDS:-300}" \
    --performance-interval-seconds "${SWIF_METRICS_INTERVAL_SECONDS:-30}" \
    --max-dispatched "${SWIF_MAX_DISPATCHED:-64}" --output "${workflow_json}" >/dev/null
submit_arguments=(--file "${workflow_json}" --max-concurrent "${SWIF_MAX_CONCURRENT:-64}")
[[ "${dry_run}" == false ]] || submit_arguments+=(--dry-run)
[[ "${import_only}" == false ]] || submit_arguments+=(--import-only)
"${job_dir}/submit_swif2_workflow.sh" "${submit_arguments[@]}"
echo "workflow JSON: ${workflow_json}"
echo "input manifest: ${assets_env}"
