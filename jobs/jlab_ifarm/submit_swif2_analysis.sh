#!/usr/bin/env bash
set -euo pipefail

job_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "${job_dir}/../.." && pwd -P)"
resources="${job_dir}/resources.env"
[[ -f "${resources}" ]] || {
    echo "copy resources.env.example to resources.env and edit it" >&2
    exit 66
}
set -a
# shellcheck disable=SC1090
source "${resources}"
set +a

project="${DVCS_PROJECT:-}"
corpus="${DVCS_CORPUS:-}"
selection="${DVCS_SELECTION:-baseline}"
profile="${DVCS_PROFILE_NAME:-validation}"
corpus_archive="${SWIF_CORPUS_ARCHIVE:-}"
workflow=
from_stage=selection
through_stage=plot
only_stage=
include_optimize=false
dry_run=false
import_only=false
while (($#)); do
    case "$1" in
        --project) project="${2:?missing project}"; shift 2 ;;
        --corpus) corpus="${2:?missing corpus}"; shift 2 ;;
        --selection) selection="${2:?missing selection}"; shift 2 ;;
        --profile) profile="${2:?missing profile}"; shift 2 ;;
        --corpus-archive) corpus_archive="${2:?missing corpus archive}"; shift 2 ;;
        --workflow) workflow="${2:?missing workflow}"; shift 2 ;;
        --from) from_stage="${2:?missing stage}"; shift 2 ;;
        --through) through_stage="${2:?missing stage}"; shift 2 ;;
        --only) only_stage="${2:?missing stage}"; shift 2 ;;
        --include-optimize) include_optimize=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        --import-only) import_only=true; shift ;;
        -h|--help)
            cat <<EOF
usage: ./dvcs farm-submit --project NAME --corpus NAME --corpus-archive FILE [options]
  --selection NAME          default: baseline
  --profile NAME            default: validation
  --workflow NAME           default: unique UTC name
  --from STAGE --through STAGE | --only STAGE
  --include-optimize        insert optimize before train
  --dry-run                 package and validate; do not contact SWIF2
  --import-only             import workflow but do not start it
EOF
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done
for value in "${project}" "${corpus}" "${selection}" "${profile}"; do
    [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 64
done
[[ -n "${corpus_archive}" ]] || { echo "set --corpus-archive" >&2; exit 64; }
[[ "${JLAB_ACCOUNT:-}" != REQUIRED_* && -n "${JLAB_ACCOUNT:-}" ]] || {
    echo "set JLAB_ACCOUNT in resources.env" >&2
    exit 64
}
workflow="${workflow:-dvcs-${project}-$(date -u +%Y%m%dT%H%M%SZ)}"

assets_env="$(${job_dir}/prepare_swif2_inputs.sh \
    --project "${project}" --corpus-archive "${corpus_archive}")"
# shellcheck disable=SC1090
source "${assets_env}"
result_root="${SWIF_OUTPUT_ROOT:-${DVCS_RESULTS}/swif2}/${workflow}"
log_root="${SWIF_LOG_ROOT:-/farm_out/${USER}/dvcs}/${workflow}"
workflow_json="${job_dir}/swif-${workflow}.json"
arguments=(
    analysis --workflow "${workflow}" --account "${JLAB_ACCOUNT}"
    --site "${SWIF_SITE_NAME:-jlab/enp}"
    --constraint "${SWIF_CONSTRAINT:-el9}"
    --result-root "${result_root}" --log-root "${log_root}"
    --image "${SWIF_IMAGE}" --database "${SWIF_DATABASE_ARCHIVE}"
    --project "${project}" --profile "${profile}" --corpus "${corpus}"
    --selection "${selection}" --corpus-archive "${SWIF_CORPUS_ARCHIVE}"
    --project-archive "${SWIF_PROJECT_ARCHIVE}"
    --from "${from_stage}" --through "${through_stage}"
    --heartbeat-seconds "${SWIF_HEARTBEAT_SECONDS:-300}"
    --performance-interval-seconds "${SWIF_METRICS_INTERVAL_SECONDS:-30}"
    --max-dispatched "${SWIF_MAX_DISPATCHED:-64}"
    --output "${workflow_json}"
)
[[ -z "${only_stage}" ]] || arguments+=(--only "${only_stage}")
[[ "${include_optimize}" == false ]] || arguments+=(--include-optimize)
python3 "${job_dir}/write_swif2_workflow.py" "${arguments[@]}" >/dev/null

submit_arguments=(--file "${workflow_json}" --max-concurrent "${SWIF_MAX_CONCURRENT:-64}")
[[ "${dry_run}" == false ]] || submit_arguments+=(--dry-run)
[[ "${import_only}" == false ]] || submit_arguments+=(--import-only)
"${job_dir}/submit_swif2_workflow.sh" "${submit_arguments[@]}"
echo "workflow JSON: ${workflow_json}"
echo "input manifest: ${assets_env}"
