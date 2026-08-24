#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
temp="$(mktemp -d)"
trap 'rm -rf -- "${temp}"' EXIT
mkdir -p "${temp}/bin"
touch "${temp}/input"
cat > "${temp}/workflow.json" <<EOF
{"name":"test-workflow","site_name":"jlab/enp","max_problems":1,"max_dispatched":1,"jobs":[{"name":"test-job","inputs":[{"local":"input","remote":"${temp}/input"}],"outputs":[]}]}
EOF
cat > "${temp}/bin/swif2" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${DVCS_TEST_SWIF2_CALLS}"
EOF
chmod +x "${temp}/bin/swif2"
script="${root}/jobs/jlab_ifarm/submit_swif2_workflow.sh"
"${script}" --file "${temp}/workflow.json" --dry-run >/dev/null
calls="${temp}/calls"
PATH="${temp}/bin:${PATH}" DVCS_TEST_SWIF2_CALLS="${calls}" \
  "${script}" --file "${temp}/workflow.json" --max-concurrent 7 >/dev/null
grep -Fx -- 'list -display json' "${calls}" >/dev/null
grep -Fx -- "import -file ${temp}/workflow.json" "${calls}" >/dev/null
grep -Fx -- 'run test-workflow -maxconcurrent 7' "${calls}" >/dev/null
echo "JLab SWIF2 submission verification: PASS"
