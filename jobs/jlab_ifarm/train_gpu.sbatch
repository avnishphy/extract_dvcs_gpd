#!/usr/bin/env bash
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT
#SBATCH --partition=gpu
#SBATCH --job-name=dvcs-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
job_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "${job_dir}/resources.env"
export DVCS_ACCELERATOR=cuda
exec "${job_dir}/run_step.sh" train train "${DVCS_PROJECT}" --profile validation
