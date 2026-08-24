#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
mode="${1:-static}"
case "${mode}" in
  static)
    static_cache="$(mktemp -d)"
    trap 'rm -rf -- "${static_cache}"' EXIT
    export MPLCONFIGDIR="${static_cache}/matplotlib"
    python3 "${root}/tests/verify_distribution.py"
    python3 "${root}/tests/verify_documentation.py"
    PYTHONPATH="${root}/src" python3 -m unittest \
      tests.test_reusable_corpus tests.test_result_contract
    "${root}/tests/verify_jlab_gpu_launcher.sh"
    "${root}/tests/verify_jlab_batch_submission.sh"
    "${root}/tests/verify_jlab_progress_logging.sh"
    "${root}/tests/verify_jlab_swif2_workflows.sh"
    "${root}/tests/verify_jlab_swif2_submission.sh"
    grep -Fx 'log.folder.path = /cache/partons-logs' \
      "${root}/cpp/partons_bridge/config/logger.properties.in" >/dev/null
    grep -F '/cache/partons-logs' \
      "${root}/scripts/container-entrypoint.sh" >/dev/null
    while IFS= read -r script; do bash -n "${script}"; done < <(find "${root}" -type f \( -name '*.sh' -o -name '*.sbatch' -o -name install.sh -o -name dvcs \) -not -path '*/.git/*' | sort)
    install_state_before="$(if [[ -f "${root}/.dvcs/install.env" ]]; then sha256sum "${root}/.dvcs/install.env"; else echo absent; fi)"
    "${root}/install.sh" --profile local --accelerator cpu --dry-run >/dev/null
    "${root}/install.sh" --profile jlab_ifarm --accelerator cpu --dry-run >/dev/null
    install_state_after="$(if [[ -f "${root}/.dvcs/install.env" ]]; then sha256sum "${root}/.dvcs/install.env"; else echo absent; fi)"
    [[ "${install_state_before}" == "${install_state_after}" ]] || { echo "dry-run changed installer state" >&2; exit 1; }
    if DVCS_ENGINE=invalid "${root}/install.sh" --dry-run >/dev/null 2>&1; then
        echo "installer accepted invalid DVCS_ENGINE" >&2
        exit 1
    fi
    echo "shell and installer static verification: PASS"
    ;;
  native)
    bridge="${DVCS_BRIDGE:-/opt/dvcs/bin/partons_bridge}"
    [[ -x "${bridge}" ]] || { echo "UNVERIFIED: set DVCS_BRIDGE" >&2; exit 77; }
    "${bridge}" --capabilities
    "${bridge}" --self-test
    ;;
  quick)
    bridge="${DVCS_BRIDGE:-/opt/dvcs/bin/partons_bridge}"
    python_bin="${DVCS_PYTHON:-python3}"
    [[ -x "${bridge}" ]] || { echo "UNVERIFIED: set DVCS_BRIDGE" >&2; exit 77; }
    temp="$(mktemp -d)"; trap 'rm -rf -- "${temp}"' EXIT
    mkdir -p "${temp}/workspace"
    export PYTHONPATH="${root}/src" DVCS_INFER_REPOSITORY_ROOT="${root}" DVCS_WORKSPACE_ROOT="${temp}/workspace" DVCS_GPDDATABASE_ROOT="${temp}/no-database" DVCS_ACCELERATOR=cpu DVCS_CPU_THREADS=1 DVCS_NATIVE_WORKERS=1 MPLCONFIGDIR="${temp}/matplotlib"
    "${python_bin}" -m extract_dvcs_cff.cli.user --bridge "${bridge}" init acceptance
    "${python_bin}" -m extract_dvcs_cff.cli.user --bridge "${bridge}" doctor acceptance
    "${python_bin}" -m extract_dvcs_cff.cli.user --bridge "${bridge}" corpus-create acceptance acceptance-corpus --profile quick
    "${python_bin}" -m extract_dvcs_cff.cli.user --bridge "${bridge}" corpus-generate acceptance acceptance-corpus --no-progress
    "${python_bin}" -m extract_dvcs_cff.cli.user --bridge "${bridge}" corpus-verify acceptance-corpus --deep
    ;;
  offline)
    [[ -f "${root}/.dvcs/install.env" ]] || { echo "install first" >&2; exit 77; }
    "${root}/dvcs" --help
    ;;
  *) echo "usage: tests/run.sh static|native|quick|offline" >&2; exit 64 ;;
esac
