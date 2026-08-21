#!/usr/bin/env bash
set -euo pipefail

job_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck disable=SC1091
source "${job_dir}/resources.env"

output="${1:-${job_dir}/swif-smoke.json}"
workflow="${2:-dvcs-${DVCS_PROJECT}-smoke}"
[[ "${JLAB_ACCOUNT}" != REQUIRED_* ]] || { echo "set JLAB_ACCOUNT in resources.env" >&2; exit 64; }
[[ -n "${DVCS_PROJECT:-}" ]] || { echo "set DVCS_PROJECT in resources.env" >&2; exit 64; }
[[ -d "${DVCS_REPOSITORY}" ]] || { echo "invalid DVCS_REPOSITORY" >&2; exit 66; }

python3 - "${output}" "${workflow}" "${job_dir}" "${JLAB_ACCOUNT}" "${JLAB_CPU_PARTITION}" "${DVCS_PROJECT}" "${SWIF_SITE_NAME:-jlab/enp}" <<'PY'
import json
import pathlib
import sys

output, workflow, job_dir, account, partition, project, site = sys.argv[1:]
payload = {
    "name": workflow,
    "max_problems": 1,
    "max_dispatched": 1,
    "site": site,
    "jobs": [{
        "name": "dvcs-doctor",
        "command": [f"{job_dir}/run_step.sh", "doctor", "doctor", project],
        "batch_flags": ["--nodes=1", "--ntasks=1", "--cpus-per-task=1"],
        "account": account,
        "partition": partition,
        "cpu_cores": 1,
        "time_secs": 1800,
        "ram_bytes": 4294967296,
        "disk_bytes": 2147483648,
        "disk_bytes_type": "scratch",
    }],
}
path = pathlib.Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

echo "wrote ${output}"
