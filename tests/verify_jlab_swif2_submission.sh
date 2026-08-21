#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
temp="$(mktemp -d)"
trap 'rm -rf -- "${temp}"' EXIT
job_dir="${temp}/checkout/jobs/jlab_ifarm"
mkdir -p "${job_dir}" "${temp}/bin"
cp "${root}/jobs/jlab_ifarm/submit_swif2_workflow.sh" "${job_dir}/"
cat > "${job_dir}/resources.env" <<EOF
JLAB_ACCOUNT=hallc
DVCS_PROJECT=test-project
DVCS_CORPUS=test-corpus
DVCS_SELECTION=baseline
JLAB_CPU_PARTITION=production
JLAB_GPU_PARTITION=gpu
EOF
cat > "${temp}/bin/swif2" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${DVCS_TEST_SWIF2_CALLS}"
EOF
chmod +x "${temp}/bin/swif2" "${job_dir}/submit_swif2_workflow.sh"

calls="${temp}/swif2-calls"
PATH="${temp}/bin:${PATH}" DVCS_TEST_SWIF2_CALLS="${calls}" SWIF_TRAIN_RAM=48G \
  "${job_dir}/submit_swif2_workflow.sh" --workflow test-workflow >/dev/null
[[ "$(wc -l < "${calls}")" -eq 10 ]]
grep -Fx -- 'create test-workflow -site-name jlab/enp -max-concurrent 1 -max-problems 1' "${calls}" >/dev/null
grep -F -- 'add-job test-workflow -name dvcs-generate -account hallc -partition production -phase 10 -cores 16 -ram 64G -time 24h -disk-scratch 10G -sbatch --nodes=1 --ntasks=1 --cpus-per-task=16 ::' "${calls}" >/dev/null
grep -F -- '-name dvcs-train' "${calls}" | grep -F -- '-antecedent dvcs-optimize' | grep -F -- '-ram 48G' | grep -F -- '--gres=gpu:1' >/dev/null
grep -F -- '-name dvcs-plot' "${calls}" | grep -F -- '-antecedent dvcs-compare -antecedent dvcs-holdout' >/dev/null
grep -Fx -- 'run test-workflow -maxconcurrent 1 -errorlimit 1' "${calls}" >/dev/null

echo "JLab SWIF2 submission verification: PASS"
