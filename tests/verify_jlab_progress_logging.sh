#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
temp="$(mktemp -d)"
trap 'rm -rf -- "${temp}"' EXIT
job_dir="${temp}/jobs/jlab_ifarm"
repository="${temp}/repository"
workspace="${temp}/workspace"
results="${temp}/results"
mkdir -p "${job_dir}" "${repository}/provenance" "${temp}/bin" \
  "${workspace}/.corpora/test-corpus" "${workspace}/test-project"
cp "${root}/jobs/jlab_ifarm/run_step.sh" "${job_dir}/run_step.sh"
printf '{}\n' > "${repository}/provenance/images.lock.json"
cat > "${workspace}/.corpora/test-corpus/corpus.json" <<'EOF'
{"core_shards":[{}],"shard_count":4}
EOF
cat > "${job_dir}/resources.env" <<EOF
JLAB_ACCOUNT=hallc
JLAB_HEARTBEAT_SECONDS=1
DVCS_PROJECT=test-project
DVCS_CORPUS=test-corpus
DVCS_REPOSITORY=${repository}
DVCS_WORKSPACE=${workspace}
DVCS_RESULTS=${results}
DVCS_CACHE=${temp}/cache
DVCS_DATABASE=${temp}/database
EOF
cat > "${repository}/dvcs" <<'EOF'
#!/usr/bin/env bash
sleep 3
printf '{"status":"ok"}\n'
EOF
cat > "${temp}/bin/scontrol" <<'EOF'
#!/usr/bin/env bash
echo 'mock Slurm job'
EOF
cat > "${temp}/bin/git" <<'EOF'
#!/usr/bin/env bash
echo 0123456789abcdef
EOF
chmod +x "${repository}/dvcs" "${temp}/bin/scontrol" "${temp}/bin/git"

stdout="${temp}/stdout"
stderr="${temp}/stderr"
PATH="${temp}/bin:${PATH}" SLURM_JOB_ID=77 SLURM_CPUS_PER_TASK=8 \
  bash "${job_dir}/run_step.sh" generate corpus-generate \
    test-project test-corpus >"${stdout}" 2>"${stderr}"

grep -F '[dvcs-job] start ' "${stdout}" >/dev/null
grep -F '[dvcs-job] heartbeat ' "${stdout}" >/dev/null
grep -F 'progress=shards=1/4' "${stdout}" >/dev/null
grep -F '[dvcs-job] finish ' "${stdout}" >/dev/null
grep -F 'exit_code=0' "${stdout}" >/dev/null
grep -F '"status":"ok"' "${results}/slurm/77/generate/stdout.log" >/dev/null
grep -F 'progress=shards=1/4' "${results}/slurm/77/generate/progress.log" >/dev/null

echo "JLab progress logging verification: PASS"
