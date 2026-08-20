#!/usr/bin/env bash
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT
#SBATCH --partition=production
#SBATCH --job-name=dvcs-holdout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/farm_out/%u/%x-%j-%N.out
#SBATCH --error=/farm_out/%u/%x-%j-%N.err
set -euo pipefail
job_dir="${DVCS_JOB_DIR:-${SLURM_SUBMIT_DIR:-}/jobs/jlab_ifarm}"
if [[ ! -f "${job_dir}/resources.env" ]] && \
   [[ -f "${SLURM_SUBMIT_DIR:-}/resources.env" ]]; then
    job_dir="${SLURM_SUBMIT_DIR}/resources.env"; job_dir="${job_dir%/*}"
fi
[[ -f "${job_dir}/resources.env" ]] || {
    echo "cannot locate resources.env; submit through submit_workflow.sh or set DVCS_JOB_DIR" >&2
    exit 66
}
job_dir="$(cd -- "${job_dir}" && pwd -P)"
source "${job_dir}/resources.env"
export DVCS_ACCELERATOR=cpu DVCS_NATIVE_WORKERS=all_available
exec "${job_dir}/run_step.sh" holdout holdout "${DVCS_PROJECT}" --profile validation
