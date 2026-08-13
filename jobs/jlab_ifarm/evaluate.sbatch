#!/usr/bin/env bash
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT
#SBATCH --partition=production
#SBATCH --job-name=dvcs-evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
job_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${job_dir}/resources.env"
export DVCS_ACCELERATOR=auto DVCS_NATIVE_WORKERS=all_available
exec "${job_dir}/run_step.sh" evaluate evaluate "${DVCS_PROJECT}" --profile validation
