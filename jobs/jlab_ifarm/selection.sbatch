#!/usr/bin/env bash
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT
#SBATCH --partition=production
#SBATCH --job-name=dvcs-selection
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
job_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${job_dir}/resources.env"
export DVCS_ACCELERATOR=cpu DVCS_NATIVE_WORKERS=1
exec "${job_dir}/run_step.sh" selection selection-create \
  "${DVCS_PROJECT}" "${DVCS_CORPUS}" "${DVCS_SELECTION}" --profile validation
