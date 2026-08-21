#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
temp="$(mktemp -d)"
trap 'rm -rf -- "${temp}"' EXIT
job_dir="${temp}/checkout/jobs/jlab_ifarm"
mkdir -p "${job_dir}" "${temp}/checkout"
cp "${root}/jobs/jlab_ifarm/write_swif2_smoke_json.sh" "${job_dir}/"
cat > "${job_dir}/resources.env" <<EOF
JLAB_ACCOUNT=hallc
DVCS_PROJECT=swif-smoke
DVCS_REPOSITORY=${temp}/checkout
JLAB_CPU_PARTITION=production
SWIF_SITE_NAME=jlab/enp
EOF
chmod +x "${job_dir}/write_swif2_smoke_json.sh"
"${job_dir}/write_swif2_smoke_json.sh" "${temp}/swif-smoke.json" test-workflow >/dev/null
python3 - "${temp}/swif-smoke.json" "${job_dir}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["name"] == "test-workflow"
assert payload["max_problems"] == 1
assert payload["max_dispatched"] == 1
assert payload["site"] == "jlab/enp"
job, = payload["jobs"]
assert job["name"] == "dvcs-doctor"
assert job["command"] == [f"{sys.argv[2]}/run_step.sh", "doctor", "doctor", "swif-smoke"]
assert job["account"] == "hallc"
assert job["partition"] == "production"
assert job["cpu_cores"] == 1
assert job["time_secs"] == 1800
assert job["ram_bytes"] == 4294967296
assert job["disk_bytes"] == 2147483648
assert job["disk_bytes_type"] == "scratch"
assert job["batch_flags"] == ["--nodes=1", "--ntasks=1", "--cpus-per-task=1"]
PY

echo "JLab SWIF2 JSON verification: PASS"
