#!/usr/bin/env bash
set -euo pipefail

workflow_json=
dry_run=false
import_only=false
max_concurrent=64
while (($#)); do
    case "$1" in
        --file) workflow_json="${2:?missing JSON path}"; shift 2 ;;
        --dry-run) dry_run=true; shift ;;
        --import-only) import_only=true; shift ;;
        --max-concurrent) max_concurrent="${2:?missing limit}"; shift 2 ;;
        -h|--help)
            echo "usage: $0 --file WORKFLOW.json [--dry-run|--import-only] [--max-concurrent N]"
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done
[[ -n "${workflow_json}" && -f "${workflow_json}" && ! -L "${workflow_json}" ]] || {
    echo "--file must name a real workflow JSON" >&2
    exit 66
}
[[ "${max_concurrent}" =~ ^[1-9][0-9]*$ ]] || exit 64

workflow="$(python3 - "${workflow_json}" <<'PY'
import json, re, sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if set(payload) - {"name", "site_name", "max_problems", "max_dispatched", "jobs"}:
    raise SystemExit("workflow JSON contains unsupported top-level fields")
name = payload.get("name")
jobs = payload.get("jobs")
if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
    raise SystemExit("workflow name is invalid")
if not isinstance(jobs, list) or not jobs:
    raise SystemExit("workflow has no jobs")
names = [job.get("name") for job in jobs if isinstance(job, dict)]
if len(names) != len(jobs) or len(names) != len(set(names)):
    raise SystemExit("workflow job names are missing or duplicated")
known = set(names)
for job in jobs:
    if not set(job.get("antecedents", [])) <= known:
        raise SystemExit(f"job {job['name']} has an unknown antecedent")
    tags = {item.get("name"): item.get("value") for item in job.get("tags", [])}
    stage = tags.get("dvcs-stage")
    flags = job.get("batch_flags", [])
    if not isinstance(flags, list) or not all(isinstance(item, str) for item in flags):
        raise SystemExit(f"job {job['name']} has invalid batch flags")
    if any(item.startswith("--gres=gpu") for item in flags):
        raise SystemExit(
            f"job {job['name']} uses conflicting --gres GPU request; use --gpus=N"
        )
    gpu_flags = [item for item in flags if re.fullmatch(r"--gpus=[1-9][0-9]*", item)]
    needs_gpu = stage in {"optimize", "train", "evaluate"}
    if needs_gpu != (len(gpu_flags) == 1):
        raise SystemExit(f"job {job['name']} has inconsistent GPU request")
    for record in job.get("inputs", []):
        remote = Path(str(record.get("remote", "")))
        if not remote.is_file():
            producers = {
                item["remote"]
                for candidate in jobs
                for item in candidate.get("outputs", [])
            }
            if str(remote) not in producers:
                raise SystemExit(f"missing input with no workflow producer: {remote}")
print(name)
PY
)"

if [[ "${dry_run}" == true ]]; then
    echo "SWIF2 workflow validated: ${workflow} (${workflow_json})"
    exit 0
fi
command -v swif2 >/dev/null 2>&1 || { echo "swif2 is not available" >&2; exit 69; }
swif2 list -display json >/dev/null
swif2 import -file "${workflow_json}"
if [[ "${import_only}" == true ]]; then
    echo "SWIF2 workflow imported but not started: ${workflow}"
    exit 0
fi
swif2 run "${workflow}" -maxconcurrent "${max_concurrent}"
echo "SWIF2 workflow started: ${workflow}"
echo "monitor: swif2 status ${workflow} -display json"
