#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
temp="$(mktemp -d)"
trap 'rm -rf -- "${temp}"' EXIT
job_dir="${temp}/checkout/jobs/jlab_ifarm"
mkdir -p "${job_dir}" "${temp}/bin" "${temp}/spool"
cp "${root}/jobs/jlab_ifarm/"*.sbatch \
  "${root}/jobs/jlab_ifarm/submit_workflow.sh" "${job_dir}/"
cat > "${job_dir}/resources.env" <<EOF
JLAB_ACCOUNT=hallc
DVCS_PROJECT=test-project
DVCS_CORPUS=test-corpus
DVCS_SELECTION=baseline
JLAB_CPU_PARTITION=production
JLAB_GPU_PARTITION=gpu
EOF
cat > "${temp}/bin/sbatch" <<'EOF'
#!/usr/bin/env bash
count_file="${DVCS_TEST_SBATCH_CALLS}.count"
count=0
[[ ! -f "${count_file}" ]] || read -r count < "${count_file}"
count=$((count + 1))
printf '%s\n' "${count}" > "${count_file}"
printf '%s\n' "$*" >> "${DVCS_TEST_SBATCH_CALLS}"
printf '%s\n' "$((9000 + count))"
EOF
chmod +x "${temp}/bin/sbatch" "${job_dir}/submit_workflow.sh"

calls="${temp}/sbatch-calls"
PATH="${temp}/bin:${PATH}" DVCS_TEST_SBATCH_CALLS="${calls}" \
  "${job_dir}/submit_workflow.sh" >/dev/null
[[ "$(wc -l < "${calls}")" -eq 8 ]]
grep -F -- "--export=ALL,DVCS_JOB_DIR=${job_dir}" "${calls}" >/dev/null
grep -F -- "--output=${job_dir}/logs/%x-%j.out" "${calls}" >/dev/null
grep -F -- "--error=${job_dir}/logs/%x-%j.err" "${calls}" >/dev/null

cat >> "${job_dir}/resources.env" <<EOF
DVCS_REPOSITORY=${temp}/checkout
DVCS_WORKSPACE=${temp}/workspace
DVCS_RESULTS=${temp}/results
DVCS_CACHE=${temp}/cache
DVCS_DATABASE=${temp}/database
EOF
cat > "${job_dir}/run_step.sh" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" > "${DVCS_TEST_RUN_STEP_ARGS}"
EOF
chmod +x "${job_dir}/run_step.sh"
cp "${job_dir}/generate.sbatch" "${temp}/spool/slurm_script"
run_step_args="${temp}/run-step-args"
SLURM_SUBMIT_DIR="${temp}/checkout" SLURM_JOB_ID=9001 \
  DVCS_TEST_RUN_STEP_ARGS="${run_step_args}" \
  bash "${temp}/spool/slurm_script"
grep -Fx -- generate "${run_step_args}" >/dev/null
grep -Fx -- corpus-generate "${run_step_args}" >/dev/null
grep -Fx -- test-project "${run_step_args}" >/dev/null
grep -Fx -- test-corpus "${run_step_args}" >/dev/null

echo "JLab batch submission verification: PASS"
