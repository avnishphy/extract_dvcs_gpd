#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
temp="$(mktemp -d)"
trap 'rm -rf -- "${temp}"' EXIT
touch "${temp}/image.sif" "${temp}/database.tar.gz" \
  "${temp}/corpus.tar.gz" "${temp}/project.tar"
printf image > "${temp}/image.sif"
printf database > "${temp}/database.tar.gz"
printf corpus > "${temp}/corpus.tar.gz"
printf project > "${temp}/project.tar"

analysis="${temp}/analysis.json"
python3 "${root}/jobs/jlab_ifarm/write_swif2_workflow.py" analysis \
  --workflow test-analysis --account hallc \
  --result-root "${temp}/outputs" --log-root "${temp}/logs" \
  --image "${temp}/image.sif" --database "${temp}/database.tar.gz" \
  --project test-project --profile validation --corpus test-corpus \
  --selection baseline --corpus-archive "${temp}/corpus.tar.gz" \
  --project-archive "${temp}/project.tar" --output "${analysis}" >/dev/null

python3 - "${analysis}" <<'PY'
import json, sys

workflow = json.load(open(sys.argv[1], encoding="utf-8"))
assert workflow["name"] == "test-analysis"
assert workflow["site_name"] == "jlab/enp"
assert workflow["max_problems"] == 1
jobs = workflow["jobs"]
assert [job["name"] for job in jobs] == [
    "dvcs-selection", "dvcs-train", "dvcs-evaluate",
    "dvcs-compare", "dvcs-holdout", "dvcs-plot",
]
for index, job in enumerate(jobs):
    assert job.get("antecedents", []) == ([] if index == 0 else [jobs[index - 1]["name"]])
    assert {tag["name"] for tag in job["tags"]} == {"dvcs-workflow", "dvcs-stage"}
    assert all("-" in item["local"] for item in job["inputs"][:3])
train = jobs[1]
assert train["partition"] == "gpu"
assert "--gres=gpu:1" in train["batch_flags"]
assert train["ram_bytes"] == 24_000_000_000
assert train["disk_bytes"] == 48_000_000_000
assert "run_swif2_analysis_stage" not in train["command"][0]
assert "analysis-runner-" in train["command"][0]
for left, right in zip(jobs, jobs[1:]):
    produced = left["outputs"][0]
    consumed = right["inputs"][3]
    assert produced == consumed
PY

experiment="${temp}/experiment.json"
python3 - "${root}/configs/inference/stage10_pseudodata_workflow.json" \
  "${experiment}" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
payload["inference"] = {"profiles": {"validation": {"native_parameter_count": 64}}}
json.dump(payload, open(sys.argv[2], "w", encoding="utf-8"))
PY
corpus_workflow="${temp}/corpus-workflow.json"
python3 "${root}/jobs/jlab_ifarm/write_swif2_workflow.py" corpus \
  --workflow test-corpus --account hallc \
  --result-root "${temp}/outputs" --log-root "${temp}/logs" \
  --image "${temp}/image.sif" --database "${temp}/database.tar.gz" \
  --project test-project --profile validation --corpus test-corpus \
  --experiment "${experiment}" --shard-size 16 --shards-per-worker 2 \
  --output "${corpus_workflow}" >/dev/null
python3 - "${corpus_workflow}" <<'PY'
import json, sys
workflow = json.load(open(sys.argv[1], encoding="utf-8"))
assert len(workflow["jobs"]) == 3
workers, merge = workflow["jobs"][:2], workflow["jobs"][2]
assert [job["name"] for job in workers] == ["dvcs-corpus-000", "dvcs-corpus-001"]
assert merge["name"] == "dvcs-corpus-merge"
assert set(merge["antecedents"]) == {job["name"] for job in workers}
assert "--shard" not in workers[0]["command"][0]
assert " 0 2 " in workers[0]["command"][0]
assert " 2 2 " in workers[1]["command"][0]
assert len(merge["inputs"]) == 4
PY

grep -F -- 'corpus-generate "${project}" "${corpus}"' \
  "${root}/jobs/jlab_ifarm/run_swif2_corpus_worker.sh" >/dev/null
grep -F -- '--shard-start "${shard_start}" --max-shards "${shard_count}"' \
  "${root}/jobs/jlab_ifarm/run_swif2_corpus_worker.sh" >/dev/null
grep -F -- 'corpus-merge --consume-sources' \
  "${root}/jobs/jlab_ifarm/run_swif2_corpus_merge.sh" >/dev/null
grep -F -- 'CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}' \
  "${root}/jobs/jlab_ifarm/run_swif2_analysis_stage.sh" >/dev/null

echo "JLab SWIF2 workflow verification: PASS"
