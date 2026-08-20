#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
temp="$(mktemp -d)"
trap 'rm -rf -- "${temp}"' EXIT
job_dir="${temp}/checkout/jobs/jlab_ifarm"
mkdir -p "${job_dir}" "${temp}/bin" "${temp}/spool" \
  "${temp}/workspace/test-project"
cp "${root}/jobs/jlab_ifarm/"*.sbatch \
  "${root}/jobs/jlab_ifarm/submit_workflow.sh" "${job_dir}/"
cat > "${job_dir}/resources.env" <<EOF
JLAB_ACCOUNT=hallc
DVCS_PROJECT=test-project
DVCS_CORPUS=test-corpus
DVCS_SELECTION=baseline
DVCS_WORKSPACE=${temp}/workspace
JLAB_CPU_PARTITION=production
JLAB_GPU_PARTITION=gpu
EOF
cat > "${temp}/workspace/test-project/experiment.json" <<'EOF'
{"inference":{"profiles":{"validation":{"ensemble_seeds":[11,22,33]}}}}
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
[[ "$(wc -l < "${calls}")" -eq 7 ]]
grep -F -- "--export=ALL,DVCS_JOB_DIR=${job_dir}" "${calls}" >/dev/null
grep -F -- "--output=/farm_out/%u/%x-%j-%N.out" "${calls}" >/dev/null
grep -F -- "--error=/farm_out/%u/%x-%j-%N.err" "${calls}" >/dev/null
grep -F -- "--comment=dvcs:test-project:generate" "${calls}" >/dev/null
grep -F -- "--comment=dvcs:test-project:train" "${calls}" >/dev/null
grep -F -- "--cpus-per-task=128 --mem-per-cpu=160M --time=24:00:00" "${calls}" >/dev/null
grep -F -- "--gres=gpu:3" "${calls}" >/dev/null
grep -F -- "--cpus-per-task=32 --mem=64G --time=12:00:00" "${calls}" >/dev/null
! grep -F -- "optimize_gpu.sbatch" "${calls}" >/dev/null

PATH="${temp}/bin:${PATH}" DVCS_TEST_SBATCH_CALLS="${calls}" \
  "${job_dir}/submit_workflow.sh" --optimization-only >/dev/null
[[ "$(wc -l < "${calls}")" -eq 8 ]]
tail -n 1 "${calls}" | grep -F -- "--gres=gpu:4" >/dev/null
tail -n 1 "${calls}" | grep -F -- "optimize_gpu.sbatch" >/dev/null

PATH="${temp}/bin:${PATH}" DVCS_TEST_SBATCH_CALLS="${calls}" \
  "${job_dir}/submit_workflow.sh" --only evaluate --tag closure-rerun >/dev/null
[[ "$(wc -l < "${calls}")" -eq 9 ]]
tail -n 1 "${calls}" | grep -F -- "--comment=dvcs:closure-rerun:evaluate" >/dev/null
tail -n 1 "${calls}" | grep -F -- "evaluate.sbatch" >/dev/null
! tail -n 1 "${calls}" | grep -F -- "--dependency=" >/dev/null

PATH="${temp}/bin:${PATH}" DVCS_TEST_SBATCH_CALLS="${calls}" \
  "${job_dir}/submit_workflow.sh" --from compare --through plot >/dev/null
[[ "$(wc -l < "${calls}")" -eq 12 ]]
sed -n '10p' "${calls}" | grep -F -- "compare.sbatch" >/dev/null
! sed -n '10p' "${calls}" | grep -F -- "--dependency=" >/dev/null
sed -n '11p' "${calls}" | grep -F -- "--dependency=afterok:9010" >/dev/null
sed -n '12p' "${calls}" | grep -F -- "--dependency=afterok:9011" >/dev/null

if PATH="${temp}/bin:${PATH}" DVCS_TEST_SBATCH_CALLS="${calls}" \
  "${job_dir}/submit_workflow.sh" --from plot --through train >/dev/null 2>&1; then
  echo "workflow accepted a reversed stage range" >&2
  exit 1
fi

cat >> "${job_dir}/resources.env" <<EOF
DVCS_REPOSITORY=${temp}/checkout
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
