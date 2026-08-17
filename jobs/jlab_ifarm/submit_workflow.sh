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
submit_step() {
  local partition="$1" dependency="$2" script="$3" args=()
  args=(
    --parsable --account="${JLAB_ACCOUNT}" --partition="${partition}"
    --export="ALL,DVCS_JOB_DIR=${job_dir}"
    --output="${job_dir}/logs/%x-%j.out"
    --error="${job_dir}/logs/%x-%j.err"
  )
  [[ -n "${dependency}" ]] && args+=(--dependency="${dependency}")
  sbatch "${args[@]}" "${job_dir}/${script}"
}
generate_id="$(submit_step "${JLAB_CPU_PARTITION}" "" generate.sbatch)"
selection_id="$(submit_step "${JLAB_CPU_PARTITION}" "afterok:${generate_id}" selection.sbatch)"
optimize_id="$(submit_step "${JLAB_GPU_PARTITION}" "afterok:${selection_id}" optimize_gpu.sbatch)"
train_id="$(submit_step "${JLAB_GPU_PARTITION}" "afterok:${optimize_id}" train_gpu.sbatch)"
evaluate_id="$(submit_step "${JLAB_GPU_PARTITION}" "afterok:${train_id}" evaluate.sbatch)"
compare_id="$(submit_step "${JLAB_CPU_PARTITION}" "afterok:${evaluate_id}" compare.sbatch)"
holdout_id="$(submit_step "${JLAB_CPU_PARTITION}" "afterok:${compare_id}" holdout.sbatch)"
plot_id="$(submit_step "${JLAB_CPU_PARTITION}" "afterok:${compare_id}:${holdout_id}" plot.sbatch)"
printf 'generate=%s\nselection=%s\noptimize=%s\ntrain=%s\nevaluate=%s\ncompare=%s\nholdout=%s\nplot=%s\n' \
  "${generate_id}" "${selection_id}" "${optimize_id}" "${train_id}" "${evaluate_id}" "${compare_id}" "${holdout_id}" "${plot_id}" | tee "${job_dir}/logs/workflow-$(date -u +%Y%m%dT%H%M%SZ).jobs"
