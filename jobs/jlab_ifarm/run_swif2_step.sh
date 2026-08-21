#!/usr/bin/env bash
set -euo pipefail

step="${1:?step required}"
job_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${job_dir}/resources.env"

case "${step}" in
    generate)
        export DVCS_ACCELERATOR=cpu DVCS_NATIVE_WORKERS=all_available
        set -- corpus-generate "${DVCS_PROJECT}" "${DVCS_CORPUS}" --no-progress
        ;;
    selection)
        export DVCS_ACCELERATOR=cpu DVCS_NATIVE_WORKERS=1
        set -- selection-create "${DVCS_PROJECT}" "${DVCS_CORPUS}" "${DVCS_SELECTION}" --profile validation
        ;;
    optimize)
        export DVCS_ACCELERATOR=cuda
        set -- optimize "${DVCS_PROJECT}" --profile validation --corpus "${DVCS_CORPUS}" --selection "${DVCS_SELECTION}"
        ;;
    train)
        export DVCS_ACCELERATOR=cuda
        set -- train "${DVCS_PROJECT}" --profile validation --corpus "${DVCS_CORPUS}" --selection "${DVCS_SELECTION}"
        ;;
    evaluate)
        export DVCS_ACCELERATOR=cuda DVCS_NATIVE_WORKERS=all_available
        set -- evaluate "${DVCS_PROJECT}" --profile validation
        ;;
    compare)
        export DVCS_ACCELERATOR=cpu
        set -- compare "${DVCS_PROJECT}" --profile validation
        ;;
    holdout)
        export DVCS_ACCELERATOR=cpu DVCS_NATIVE_WORKERS=all_available
        set -- holdout "${DVCS_PROJECT}" --profile validation
        ;;
    plot)
        export DVCS_ACCELERATOR=cpu
        set -- plot "${DVCS_PROJECT}" --profile validation
        ;;
    *) echo "unsupported SWIF2 step: ${step}" >&2; exit 64 ;;
esac

exec "${job_dir}/run_step.sh" "${step}" "$@"
