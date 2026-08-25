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

python3 - "${temp}/workflow.json" <<'PY'
import json, sys
path = sys.argv[1]
workflow = json.load(open(path, encoding="utf-8"))
job = workflow["jobs"][0]
job["tags"] = [{"name": "dvcs-stage", "value": "train"}]
job["batch_flags"] = ["--gpus=1"]
json.dump(workflow, open(path, "w", encoding="utf-8"))
PY
"${script}" --file "${temp}/workflow.json" --dry-run >/dev/null
python3 - "${temp}/workflow.json" <<'PY'
import json, sys
path = sys.argv[1]
workflow = json.load(open(path, encoding="utf-8"))
workflow["jobs"][0]["batch_flags"] = ["--gres=gpu:1"]
json.dump(workflow, open(path, "w", encoding="utf-8"))
PY
if "${script}" --file "${temp}/workflow.json" --dry-run >/dev/null 2>&1; then
    echo "submission accepted conflicting SWIF2 GPU GRES" >&2
    exit 1
fi
echo "JLab SWIF2 submission verification: PASS"
