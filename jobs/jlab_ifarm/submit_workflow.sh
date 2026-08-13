#!/usr/bin/env bash
set -euo pipefail
job_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${job_dir}/resources.env"
[[ "${JLAB_ACCOUNT}" != REQUIRED_* ]] || { echo "set JLAB_ACCOUNT in resources.env" >&2; exit 64; }
mkdir -p "${job_dir}/logs"
generate_id="$(sbatch --parsable --account="${JLAB_ACCOUNT}" --partition="${JLAB_CPU_PARTITION}" "${job_dir}/generate.sbatch")"
optimize_id="$(sbatch --parsable --account="${JLAB_ACCOUNT}" --partition="${JLAB_GPU_PARTITION}" --dependency="afterok:${generate_id}" "${job_dir}/optimize_gpu.sbatch")"
train_id="$(sbatch --parsable --account="${JLAB_ACCOUNT}" --partition="${JLAB_GPU_PARTITION}" --dependency="afterok:${optimize_id}" "${job_dir}/train_gpu.sbatch")"
evaluate_id="$(sbatch --parsable --account="${JLAB_ACCOUNT}" --partition="${JLAB_CPU_PARTITION}" --dependency="afterok:${train_id}" "${job_dir}/evaluate.sbatch")"
compare_id="$(sbatch --parsable --account="${JLAB_ACCOUNT}" --partition="${JLAB_CPU_PARTITION}" --dependency="afterok:${evaluate_id}" "${job_dir}/compare.sbatch")"
holdout_id="$(sbatch --parsable --account="${JLAB_ACCOUNT}" --partition="${JLAB_CPU_PARTITION}" --dependency="afterok:${compare_id}" "${job_dir}/holdout.sbatch")"
plot_id="$(sbatch --parsable --account="${JLAB_ACCOUNT}" --partition="${JLAB_CPU_PARTITION}" --dependency="afterok:${compare_id}:${holdout_id}" "${job_dir}/plot.sbatch")"
printf 'generate=%s\noptimize=%s\ntrain=%s\nevaluate=%s\ncompare=%s\nholdout=%s\nplot=%s\n' \
  "${generate_id}" "${optimize_id}" "${train_id}" "${evaluate_id}" "${compare_id}" "${holdout_id}" "${plot_id}" | tee "${job_dir}/logs/workflow-$(date -u +%Y%m%dT%H%M%SZ).jobs"
