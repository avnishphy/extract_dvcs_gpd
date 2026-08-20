#!/usr/bin/env bash
set -euo pipefail
job_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${job_dir}/resources.env"
[[ "${JLAB_ACCOUNT}" != REQUIRED_* ]] || { echo "set JLAB_ACCOUNT in resources.env" >&2; exit 64; }
[[ -n "${DVCS_PROJECT:-}" && -n "${DVCS_CORPUS:-}" && -n "${DVCS_SELECTION:-}" ]] || {
  echo "set DVCS_PROJECT, DVCS_CORPUS, and DVCS_SELECTION in resources.env" >&2
  exit 64
}
mkdir -p "${job_dir}/logs"

stages=(generate selection train evaluate compare holdout plot)
from_stage=generate
through_stage=plot
from_set=false
through_set=false
only_stage=
optimization_only=false
job_tag="${JLAB_JOB_TAG:-${DVCS_PROJECT}}"
slurm_log_root="${JLAB_SLURM_LOG_ROOT:-/farm_out/%u}"
while (( $# )); do
  case "$1" in
    --from)
      (( $# >= 2 )) || { echo "--from requires a stage" >&2; exit 64; }
      from_stage="$2"; from_set=true; shift 2
      ;;
    --through)
      (( $# >= 2 )) || { echo "--through requires a stage" >&2; exit 64; }
      through_stage="$2"; through_set=true; shift 2
      ;;
    --only)
      (( $# >= 2 )) || { echo "--only requires a stage" >&2; exit 64; }
      only_stage="$2"; shift 2
      ;;
    --tag)
      (( $# >= 2 )) || { echo "--tag requires a value" >&2; exit 64; }
      job_tag="$2"; shift 2
      ;;
    --optimization-only) optimization_only=true; shift ;;
    -h|--help)
      cat <<EOF
usage: ${0##*/} [--from STAGE] [--through STAGE] [--only STAGE] [--tag TAG]
       ${0##*/} --optimization-only [--tag TAG]
stages: ${stages[*]}
EOF
      exit 0
      ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done
if [[ -n "${only_stage}" ]]; then
  [[ "${from_set}" == false && "${through_set}" == false ]] || {
    echo "--only cannot be combined with --from or --through" >&2
    exit 64
  }
  from_stage="${only_stage}"
  through_stage="${only_stage}"
fi
if [[ "${optimization_only}" == true ]] && \
   [[ "${from_set}" == true || "${through_set}" == true || -n "${only_stage}" ]]; then
  echo "--optimization-only cannot be combined with workflow stage selectors" >&2
  exit 64
fi
[[ "${job_tag}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "job tag must contain only letters, digits, dot, underscore, or hyphen" >&2
  exit 64
}

positive_integer() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || {
    echo "$1 must be a positive integer, got: $2" >&2
    exit 64
  }
}

resolve_train_gpus() {
  local requested="${JLAB_TRAIN_GPUS:-auto}" maximum="${JLAB_MAX_GPUS_PER_JOB:-4}"
  positive_integer JLAB_MAX_GPUS_PER_JOB "${maximum}"
  if [[ "${requested}" != auto ]]; then
    positive_integer JLAB_TRAIN_GPUS "${requested}"
    (( 10#${requested} <= 10#${maximum} )) || {
      echo "JLAB_TRAIN_GPUS exceeds JLAB_MAX_GPUS_PER_JOB" >&2
      exit 64
    }
    printf '%s\n' "${requested}"
    return
  fi
  python3 - "${DVCS_WORKSPACE:?set DVCS_WORKSPACE}/${DVCS_PROJECT}/experiment.json" \
    "${maximum}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
try:
    experiment = json.loads(path.read_text(encoding="utf-8"))
    seeds = experiment["inference"]["profiles"]["validation"]["ensemble_seeds"]
except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
    raise SystemExit(f"cannot resolve training GPU count from {path}: {exc}")
if not isinstance(seeds, list) or not seeds:
    raise SystemExit(f"validation ensemble_seeds is empty or invalid in {path}")
print(min(len(seeds), int(sys.argv[2])))
PY
}

submit_step() {
  local partition="$1" dependency="$2" script="$3" cpus="$4" memory="$5"
  local walltime="$6" gpus="${7:-0}" comment_stage="${script%.sbatch}" args=()
  comment_stage="${comment_stage%_gpu}"
  positive_integer "${script}: CPUs" "${cpus}"
  args=(
    --parsable --account="${JLAB_ACCOUNT}" --partition="${partition}"
    --nodes=1 --ntasks=1 --cpus-per-task="${cpus}"
  )
  if [[ "${memory}" == per-cpu:* ]]; then
    [[ -n "${memory#per-cpu:}" ]] || { echo "${script}: empty per-CPU memory" >&2; exit 64; }
    args+=(--mem-per-cpu="${memory#per-cpu:}")
  else
    args+=(--mem="${memory}")
  fi
  args+=(
    --time="${walltime}" --comment="dvcs:${job_tag}:${comment_stage}"
    --export="ALL,DVCS_JOB_DIR=${job_dir}"
    --output="${slurm_log_root}/%x-%j-%N.out"
    --error="${slurm_log_root}/%x-%j-%N.err"
  )
  if [[ "${gpus}" != 0 ]]; then
    positive_integer "${script}: GPUs" "${gpus}"
    positive_integer JLAB_MAX_GPUS_PER_JOB "${JLAB_MAX_GPUS_PER_JOB:-4}"
    (( 10#${gpus} <= 10#${JLAB_MAX_GPUS_PER_JOB:-4} )) || {
      echo "${script}: GPUs exceed JLAB_MAX_GPUS_PER_JOB" >&2
      exit 64
    }
    if [[ -n "${JLAB_GPU_TYPE:-}" ]]; then
      args+=(--gres="gpu:${JLAB_GPU_TYPE}:${gpus}")
    else
      args+=(--gres="gpu:${gpus}")
    fi
  fi
  [[ -n "${dependency}" ]] && args+=(--dependency="${dependency}")
  sbatch "${args[@]}" "${job_dir}/${script}"
}

submit_named_stage() {
  local stage="$1" dependency="$2" train_gpus
  case "${stage}" in
    generate)
      submit_step "${JLAB_CPU_PARTITION}" "${dependency}" generate.sbatch \
        "${JLAB_GENERATE_CPUS:-128}" "${JLAB_GENERATE_MEM:-per-cpu:160M}" "${JLAB_GENERATE_TIME:-24:00:00}"
      ;;
    selection)
      submit_step "${JLAB_CPU_PARTITION}" "${dependency}" selection.sbatch \
        "${JLAB_SELECTION_CPUS:-1}" "${JLAB_SELECTION_MEM:-4G}" "${JLAB_SELECTION_TIME:-00:30:00}"
      ;;
    train)
      train_gpus="$(resolve_train_gpus)"
      submit_step "${JLAB_GPU_PARTITION}" "${dependency}" train_gpu.sbatch \
        "${JLAB_TRAIN_CPUS:-8}" "${JLAB_TRAIN_MEM:-64G}" "${JLAB_TRAIN_TIME:-12:00:00}" "${train_gpus}"
      ;;
    evaluate)
      submit_step "${JLAB_GPU_PARTITION}" "${dependency}" evaluate.sbatch \
        "${JLAB_EVALUATE_CPUS:-32}" "${JLAB_EVALUATE_MEM:-64G}" "${JLAB_EVALUATE_TIME:-12:00:00}" "${JLAB_EVALUATE_GPUS:-1}"
      ;;
    compare)
      submit_step "${JLAB_CPU_PARTITION}" "${dependency}" compare.sbatch \
        "${JLAB_COMPARE_CPUS:-1}" "${JLAB_COMPARE_MEM:-32G}" "${JLAB_COMPARE_TIME:-04:00:00}"
      ;;
    holdout)
      submit_step "${JLAB_CPU_PARTITION}" "${dependency}" holdout.sbatch \
        "${JLAB_HOLDOUT_CPUS:-1}" "${JLAB_HOLDOUT_MEM:-32G}" "${JLAB_HOLDOUT_TIME:-12:00:00}"
      ;;
    plot)
      submit_step "${JLAB_CPU_PARTITION}" "${dependency}" plot.sbatch \
        "${JLAB_PLOT_CPUS:-1}" "${JLAB_PLOT_MEM:-8G}" "${JLAB_PLOT_TIME:-01:00:00}"
      ;;
  esac
}

stage_index() {
  local wanted="$1" index
  for index in "${!stages[@]}"; do
    [[ "${stages[index]}" != "${wanted}" ]] || { printf '%s\n' "${index}"; return; }
  done
  echo "unknown workflow stage: ${wanted}; choose one of: ${stages[*]}" >&2
  return 64
}

if [[ "${optimization_only}" == true ]]; then
  optimize_id="$(submit_step "${JLAB_GPU_PARTITION}" "" optimize_gpu.sbatch \
    "${JLAB_OPTIMIZE_CPUS:-8}" "${JLAB_OPTIMIZE_MEM:-64G}" \
    "${JLAB_OPTIMIZE_TIME:-12:00:00}" "${JLAB_OPTIMIZE_GPUS:-4}")"
  printf 'optimize=%s\n' "${optimize_id}" | tee \
    "${job_dir}/logs/optimization-$(date -u +%Y%m%dT%H%M%SZ).jobs"
  exit 0
fi

start_index="$(stage_index "${from_stage}")"
stop_index="$(stage_index "${through_stage}")"
(( start_index <= stop_index )) || {
  echo "--from ${from_stage} occurs after --through ${through_stage}" >&2
  exit 64
}
previous_id=
job_records=()
for (( index=start_index; index<=stop_index; index++ )); do
  stage="${stages[index]}"
  dependency=
  [[ -z "${previous_id}" ]] || dependency="afterok:${previous_id}"
  job_id="$(submit_named_stage "${stage}" "${dependency}")"
  previous_id="${job_id}"
  job_records+=("${stage}=${job_id}")
done
printf '%s\n' "${job_records[@]}" | tee \
  "${job_dir}/logs/workflow-$(date -u +%Y%m%dT%H%M%SZ).jobs"
