#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
temp="$(mktemp -d)"
trap 'rm -rf -- "${temp}"' EXIT
workflow="${temp}/validation-corpus.json"
python3 "${root}/jobs/jlab_ifarm/write_swif2_validation_corpus_json.py" \
  "${workflow}" >/dev/null
python3 - "${workflow}" <<'PY'
import json, sys

workflow = json.load(open(sys.argv[1], encoding="utf-8"))
assert workflow["name"] == "extract-dvcs-gpd-partons-validation-16384-parallel-v2"
assert len(workflow["jobs"]) == 33
workers = workflow["jobs"][:32]
for batch, job in enumerate(workers, 1):
    name = f"partons-validation-v2-batch-{batch:02d}"
    assert job["name"] == name
    assert job.get("antecedents", []) == []
    assert job["command"] == [f"/bin/bash run_staged_partons_validation_chunk_v2.sh {batch}"]
    assert (job["cpu_cores"], job["ram_bytes"], job["disk_bytes"], job["time_secs"]) == (
        16, 32000000000, 32000000000, 43200
    )
    inputs = {item["local"] for item in job["inputs"]}
    assert "extract-dvcs-gpd-jlab_ifarm-0.2.0-validation.sif" in inputs
    assert not any(item.startswith("corpus-batch-") for item in inputs)
    outputs = {item["local"] for item in job["outputs"]}
    assert outputs == {
        f"corpus-batch-{batch}.tar.gz",
        f"corpus-batch-{batch}-summary.json",
    }

merge = workflow["jobs"][32]
assert merge["name"] == "partons-validation-v2-merge"
assert set(merge["antecedents"]) == {job["name"] for job in workers}
assert merge["command"] == ["/bin/bash run_staged_partons_validation_merge_v2.sh"]
assert (merge["cpu_cores"], merge["ram_bytes"], merge["disk_bytes"], merge["time_secs"]) == (
    4, 16000000000, 32000000000, 43200
)
assert merge["batch_flags"] == ["--nodes=1", "--ntasks=1", "--cpus-per-task=4"]
merge_inputs = {item["local"] for item in merge["inputs"]}
assert {f"corpus-batch-{batch}.tar.gz" for batch in range(1, 33)} <= merge_inputs
assert {item["local"] for item in merge["outputs"]} == {
    "partons-corpus-validation-16384.tar.gz",
    "partons-corpus-validation-16384-summary.json",
}
PY

worker="${root}/jobs/jlab_ifarm/run_staged_partons_validation_chunk_v2.sh"
merge="${root}/jobs/jlab_ifarm/run_staged_partons_validation_merge_v2.sh"
grep -F -- '--shard-size 16' "${worker}" >/dev/null
grep -F -- '--profile validation' "${worker}" >/dev/null
grep -F -- '--shard-start "${shard_start}" --max-shards 32' "${worker}" >/dev/null
grep -F -- 'shard_start=$(( (batch - 1) * 32 ))' "${worker}" >/dev/null
grep -F -- 'corpus-checkpoint-export' "${worker}" >/dev/null
grep -F -- 'corpus-checkpoint-import' "${merge}" >/dev/null
grep -F -- 'corpus-merge --consume-sources "${corpus}"' "${merge}" >/dev/null
grep -F -- 'corpus-verify "${corpus}" --deep' "${merge}" >/dev/null
grep -F -- 'corpus-export "${corpus}" "${export_name}"' "${merge}" >/dev/null

echo "JLab SWIF2 parallel PARTONS corpus verification: PASS"
