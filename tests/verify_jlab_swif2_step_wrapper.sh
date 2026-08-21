#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
temp="$(mktemp -d)"
trap 'rm -rf -- "${temp}"' EXIT
job_dir="${temp}/checkout/jobs/jlab_ifarm"
mkdir -p "${job_dir}"
cp "${root}/jobs/jlab_ifarm/run_swif2_step.sh" "${job_dir}/"
cat > "${job_dir}/resources.env" <<EOF
DVCS_PROJECT=test-project
DVCS_CORPUS=test-corpus
DVCS_SELECTION=baseline
EOF
cat > "${job_dir}/run_step.sh" <<'EOF'
#!/usr/bin/env bash
printf 'accelerator=%s\n' "${DVCS_ACCELERATOR}" > "${DVCS_TEST_STEP_OUTPUT}"
printf 'workers=%s\n' "${DVCS_NATIVE_WORKERS:-}" >> "${DVCS_TEST_STEP_OUTPUT}"
printf '%s\n' "$@" >> "${DVCS_TEST_STEP_OUTPUT}"
EOF
chmod +x "${job_dir}/run_swif2_step.sh" "${job_dir}/run_step.sh"

check_step() {
    local step="$1" accelerator="$2" workers="$3" expected="$4" output
    output="${temp}/${step}.out"
    DVCS_TEST_STEP_OUTPUT="${output}" "${job_dir}/run_swif2_step.sh" "${step}"
    grep -Fx -- "accelerator=${accelerator}" "${output}" >/dev/null
    grep -Fx -- "workers=${workers}" "${output}" >/dev/null
    grep -Fx -- "${expected}" "${output}" >/dev/null
}

check_step generate cpu all_available corpus-generate
check_step selection cpu 1 selection-create
check_step optimize cuda '' optimize
check_step train cuda '' train
check_step evaluate cuda all_available evaluate
check_step compare cpu '' compare
check_step holdout cpu all_available holdout
check_step plot cpu '' plot
if DVCS_TEST_STEP_OUTPUT="${temp}/invalid.out" "${job_dir}/run_swif2_step.sh" invalid >/dev/null 2>&1; then
    echo "invalid SWIF2 step unexpectedly succeeded" >&2
    exit 1
fi

echo "JLab SWIF2 step wrapper verification: PASS"
