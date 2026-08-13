#!/usr/bin/env bash
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT
#SBATCH --partition=production
#SBATCH --job-name=dvcs-plot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
job_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${job_dir}/resources.env"
export DVCS_ACCELERATOR=cpu
exec "${job_dir}/run_step.sh" plot plot "${DVCS_PROJECT}" --profile validation
