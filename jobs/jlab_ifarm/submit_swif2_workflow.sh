#!/usr/bin/env bash
set -euo pipefail

job_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${job_dir}/resources.env"

workflow=""
max_concurrent="${SWIF_MAX_CONCURRENT:-1}"
while (($#)); do
    case "$1" in
        --workflow) workflow="${2:?missing workflow name}"; shift 2 ;;
        --max-concurrent) max_concurrent="${2:?missing job limit}"; shift 2 ;;
        -h|--help)
            echo "usage: $0 [--workflow NAME] [--max-concurrent N]"
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done

[[ "${JLAB_ACCOUNT}" != REQUIRED_* ]] || { echo "set JLAB_ACCOUNT in resources.env" >&2; exit 64; }
[[ -n "${DVCS_PROJECT:-}" && -n "${DVCS_CORPUS:-}" && -n "${DVCS_SELECTION:-}" ]] || {
    echo "set DVCS_PROJECT, DVCS_CORPUS, and DVCS_SELECTION in resources.env" >&2
    exit 64
}
[[ "${max_concurrent}" =~ ^[1-9][0-9]*$ ]] || { echo "max-concurrent must be positive" >&2; exit 64; }
command -v swif2 >/dev/null 2>&1 || { echo "swif2 is not available" >&2; exit 69; }

workflow="${workflow:-dvcs-${DVCS_PROJECT}-$(date -u +%Y%m%dT%H%M%SZ)}"
site_name="${SWIF_SITE_NAME:-jlab/enp}"
scratch="${SWIF_DISK_SCRATCH:-10G}"

resource() {
    local step="$1" field="$2" fallback="$3" key
    key="SWIF_${step^^}_${field}"
    printf '%s\n' "${!key:-${fallback}}"
}

add_step() {
    local step="$1" phase="$2" partition="$3" gpu="$4" cores ram time
    shift 4
    cores="$(resource "${step}" CORES "$1")"; shift
    ram="$(resource "${step}" RAM "$1")"; shift
    time="$(resource "${step}" TIME "$1")"; shift
    local -a antecedents=() command=("${job_dir}/run_swif2_step.sh" "${step}") sbatch_flags=(--nodes=1 --ntasks=1 "--cpus-per-task=${cores}")
    while [[ "$1" != -- ]]; do antecedents+=(-antecedent "dvcs-$1"); shift; done
    shift
    [[ "${gpu}" == yes ]] && sbatch_flags+=(--gres=gpu:1)
    swif2 add-job "${workflow}" -name "dvcs-${step}" -account "${JLAB_ACCOUNT}" \
        -partition "${partition}" -phase "${phase}" -cores "${cores}" -ram "${ram}" \
        -time "${time}" -disk-scratch "${scratch}" "${antecedents[@]}" \
        -sbatch "${sbatch_flags[@]}" :: "${command[@]}"
}

swif2 create "${workflow}" -site-name "${site_name}" -max-concurrent "${max_concurrent}" -max-problems 1
add_step generate 10 "${JLAB_CPU_PARTITION}" no 16 64G 24h -- corpus-generate "${DVCS_PROJECT}" "${DVCS_CORPUS}" --no-progress
add_step selection 20 "${JLAB_CPU_PARTITION}" no 1 4G 30min generate -- selection-create "${DVCS_PROJECT}" "${DVCS_CORPUS}" "${DVCS_SELECTION}" --profile validation
add_step optimize 30 "${JLAB_GPU_PARTITION}" yes 8 64G 12h selection -- optimize "${DVCS_PROJECT}" --profile validation --corpus "${DVCS_CORPUS}" --selection "${DVCS_SELECTION}"
add_step train 40 "${JLAB_GPU_PARTITION}" yes 8 64G 12h optimize -- train "${DVCS_PROJECT}" --profile validation --corpus "${DVCS_CORPUS}" --selection "${DVCS_SELECTION}"
add_step evaluate 50 "${JLAB_GPU_PARTITION}" yes 8 64G 12h train -- evaluate "${DVCS_PROJECT}" --profile validation
add_step compare 60 "${JLAB_CPU_PARTITION}" no 4 32G 4h evaluate -- compare "${DVCS_PROJECT}" --profile validation
add_step holdout 70 "${JLAB_CPU_PARTITION}" no 16 64G 12h compare -- holdout "${DVCS_PROJECT}" --profile validation
add_step plot 80 "${JLAB_CPU_PARTITION}" no 2 8G 1h compare holdout -- plot "${DVCS_PROJECT}" --profile validation
swif2 run "${workflow}" -maxconcurrent "${max_concurrent}" -errorlimit 1
printf 'SWIF2 workflow started: %s\n' "${workflow}"
