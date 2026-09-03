#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
temp="$(mktemp -d)"
trap 'rm -rf -- "${temp}"' EXIT
export DVCS_SWIF_INPUT_ROOT="${temp}/staged-controls"
touch "${temp}/image.sif" "${temp}/database.tar.gz" \
  "${temp}/corpus.tar.gz" "${temp}/project.tar" "${temp}/holdout.json" \
  "${temp}/lhapdf.tar.gz" "${temp}/gpd-truth.json"
printf image > "${temp}/image.sif"
printf database > "${temp}/database.tar.gz"
printf corpus > "${temp}/corpus.tar.gz"
printf project > "${temp}/project.tar"

analysis="${temp}/analysis.json"
python3 "${root}/jobs/jlab_ifarm/write_swif2_workflow.py" analysis \
  --workflow test-analysis --account hallc \
  --result-root "${temp}/outputs" --log-root "${temp}/logs" \
  --image "${temp}/image.sif" --database "${temp}/database.tar.gz" \
  --lhapdf "${temp}/lhapdf.tar.gz" \
  --project test-project --profile validation --corpus test-corpus \
  --selection baseline --corpus-archive "${temp}/corpus.tar.gz" \
  --project-archive "${temp}/project.tar" \
  --holdout-design "${temp}/holdout.json" --output "${analysis}" >/dev/null

python3 - "${analysis}" <<'PY'
import json, shlex, sys

workflow = json.load(open(sys.argv[1], encoding="utf-8"))
assert workflow["name"] == "test-analysis"
assert workflow["site_name"] == "jlab/enp"
assert workflow["max_problems"] == 1
jobs = workflow["jobs"]
assert [job["name"] for job in jobs] == [
    "dvcs-selection", "dvcs-materialize", "dvcs-train", "dvcs-evaluate",
    "dvcs-exact-reevaluate", "dvcs-compare", "dvcs-holdout-gpdgk11", "dvcs-holdout-gpdgk16",
    "dvcs-holdout-gpdgk19", "dvcs-holdout-gpdvgg99",
    "dvcs-holdout-merge", "dvcs-plot",
]
for job in jobs:
    assert {tag["name"] for tag in job["tags"]} == {"dvcs-workflow", "dvcs-stage"}
    assert all("-" in item["local"] for item in job["inputs"][:4])
    assert len(job["outputs"]) == 4
for index in range(6):
    assert jobs[index].get("antecedents", []) == ([] if index == 0 else [jobs[index - 1]["name"]])
for job in jobs[6:10]:
    assert job["antecedents"] == ["dvcs-compare"]
assert set(jobs[10]["antecedents"]) == {job["name"] for job in jobs[6:10]}
assert jobs[11]["antecedents"] == ["dvcs-holdout-merge"]
materialize = jobs[1]
assert materialize["partition"] == "production"
assert materialize["cpu_cores"] == 2
assert materialize["ram_bytes"] == 20_000_000_000
assert not any(flag.startswith("--gpus=") for flag in materialize["batch_flags"])
assert len(materialize["inputs"]) == 6
assert " materialize " in materialize["command"][0]
train = jobs[2]
assert train["partition"] == "gpu"
assert train["cpu_cores"] == 4
assert "--gpus=1" in train["batch_flags"]
assert not any(flag.startswith("--gres=gpu") for flag in train["batch_flags"])
assert train["ram_bytes"] == 32_000_000_000
assert train["disk_bytes"] == 48_000_000_000
assert "run_swif2_analysis_stage" not in train["command"][0]
assert "analysis-runner-" in train["command"][0]
assert "performance-collector-" in train["command"][0]
train_command = shlex.split(train["command"][0])
assert train_command[-6].startswith("performance-collector-")
assert train_command[-5:-2] == [
    "performance-train.json", "performance-train.jsonl", "30",
]
assert train_command[-2:] == ["none", "none"]
assert {item["local"] for item in train["outputs"][2:]} == {
    "performance-train.json", "performance-train.jsonl",
}
for left, right in zip(jobs[:5], jobs[1:6]):
    produced = left["outputs"][0]
    consumed = right["inputs"][4]
    assert produced == consumed
for job in jobs:
    stage = next(tag["value"] for tag in job["tags"] if tag["name"] == "dvcs-stage")
    gpu_flags = [flag for flag in job["batch_flags"] if flag.startswith("--gpus=")]
    assert bool(gpu_flags) == (stage in {"optimize", "train", "evaluate"})
assert len(train["inputs"]) == 5
assert " --corpus " not in train["command"][0]
exact = jobs[4]
assert exact["partition"] == "production"
assert exact["cpu_cores"] == 8
assert not any(flag.startswith("--gpus=") for flag in exact["batch_flags"])
assert " exact-reevaluate " in exact["command"][0]
compare = jobs[5]
assert compare["ram_bytes"] == 12_000_000_000
holdout = jobs[6]
assert holdout["cpu_cores"] == 4
assert holdout["ram_bytes"] == 20_000_000_000
assert holdout["time_secs"] == 43_200
assert len(holdout["inputs"]) == 7
assert holdout["inputs"][-2]["local"].startswith("holdout-design-")
assert holdout["inputs"][-1]["local"].startswith("MSTW2008nlo68cl-")
holdout_command = shlex.split(holdout["command"][0])
assert holdout_command[-3].startswith("holdout-design-")
assert holdout_command[-2].startswith("MSTW2008nlo68cl-")
assert holdout_command[-1] == "GPDGK11"
merge = jobs[10]
assert merge["cpu_cores"] == 2
assert merge["time_secs"] == 7_200
assert len(merge["inputs"]) == 12
assert "holdout-merger-" in merge["command"][0]
assert jobs[11]["inputs"][4] == merge["outputs"][0]
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
legacy_warning="${temp}/legacy-constraint-warning.txt"
python3 "${root}/jobs/jlab_ifarm/write_swif2_workflow.py" corpus \
  --workflow test-corpus --account hallc \
  --constraint el9,farm25 \
  --result-root "${temp}/outputs" --log-root "${temp}/logs" \
  --image "${temp}/image.sif" --database "${temp}/database.tar.gz" \
  --project test-project --profile validation --corpus test-corpus \
  --experiment "${experiment}" --gpd-truth-request "${temp}/gpd-truth.json" \
  --shard-size 16 --shards-per-worker 2 \
  --output "${corpus_workflow}" >/dev/null 2>"${legacy_warning}"
grep -F -- 'deprecated compatibility input' "${legacy_warning}" >/dev/null
python3 - "${corpus_workflow}" <<'PY'
import json, shlex, sys
workflow = json.load(open(sys.argv[1], encoding="utf-8"))
assert len(workflow["jobs"]) == 3
workers, merge = workflow["jobs"][:2], workflow["jobs"][2]
assert [job["name"] for job in workers] == ["dvcs-corpus-000", "dvcs-corpus-001"]
assert merge["name"] == "dvcs-corpus-merge"
assert workflow["placement_policy"]["profile"] == "farm25_strict"
assert workflow["placement_policy"]["farm23_implicit_fallback"] is False
assert all(job["cpu_cores"] == 16 for job in workers)
assert all(job["constraint"] == "el9,farm25" for job in workers)
assert all(job["placement"]["node_family"] == "farm25" for job in workers)
assert all(job["placement"]["farm23_excluded"] for job in workers)
assert all(len(job["inputs"]) == 6 for job in workers)
assert all(any(item["local"].startswith("gpd-truth-request-") for item in job["inputs"]) for job in workers)
assert merge["constraint"] == "el9"
assert merge["placement"]["inherits_corpus_worker_placement"] is False
assert set(merge["antecedents"]) == {job["name"] for job in workers}
assert "--shard" not in workers[0]["command"][0]
assert " 0 2 " in workers[0]["command"][0]
assert " 2 2 " in workers[1]["command"][0]
worker_command = shlex.split(workers[0]["command"][0])
assert worker_command[-4].startswith("performance-collector-")
assert worker_command[-3:] == [
    "performance-corpus-000.json", "performance-corpus-000.jsonl", "30",
]
assert len(merge["inputs"]) == 5
assert all(len(job["outputs"]) == 4 for job in workflow["jobs"])
PY

fallback="${temp}/corpus-fallback.json"
python3 "${root}/jobs/jlab_ifarm/write_swif2_workflow.py" corpus \
  --workflow test-fallback --account hallc --placement-profile farm19_fallback \
  --result-root "${temp}/outputs" --log-root "${temp}/logs" \
  --image "${temp}/image.sif" --database "${temp}/database.tar.gz" \
  --project test-project --profile validation --corpus test-corpus \
  --experiment "${experiment}" --gpd-truth-request "${temp}/gpd-truth.json" \
  --shard-size 16 --shards-per-worker 2 \
  --output "${fallback}" >/dev/null
python3 - "${fallback}" <<'PY'
import json, sys
workflow = json.load(open(sys.argv[1], encoding="utf-8"))
workers = workflow["jobs"][:-1]
assert all(job["constraint"] == "el9,farm19" for job in workers)
assert all(job["placement"]["node_family"] == "farm19" for job in workers)
assert workflow["jobs"][-1]["constraint"] == "el9"
PY

if SWIF_CORPUS_WORKER_CORES=8 python3 \
  "${root}/jobs/jlab_ifarm/write_swif2_workflow.py" corpus \
  --workflow invalid-cpus --account hallc \
  --result-root "${temp}/outputs" --log-root "${temp}/logs" \
  --image "${temp}/image.sif" --database "${temp}/database.tar.gz" \
  --project test-project --profile validation --corpus test-corpus \
  --experiment "${experiment}" --gpd-truth-request "${temp}/gpd-truth.json" \
  --output "${temp}/invalid.json" >/dev/null 2>&1; then
    echo "accepted non-16-CPU corpus worker" >&2
    exit 1
fi

metrics_dir="${temp}/metrics"
mkdir -p "${metrics_dir}"
"${root}/jobs/jlab_ifarm/collect_performance_metrics.py" \
  --samples "${metrics_dir}/samples.jsonl" \
  --summary "${metrics_dir}/summary.json" \
  --stage test --accelerator cpu --once
python3 - "${metrics_dir}/summary.json" "${metrics_dir}/samples.jsonl" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["schema_version"] == 1
assert summary["sample_count"] == 1
assert summary["allocation"]["cpus"] >= 1
assert summary["gpu"]["devices"] == []
assert len(open(sys.argv[2], encoding="utf-8").readlines()) == 1
PY

continuous="${metrics_dir}/continuous"
mkdir -p "${continuous}"
SLURM_CPUS_PER_TASK=2 SLURM_MEM_PER_NODE=1024 \
  "${root}/jobs/jlab_ifarm/collect_performance_metrics.py" \
  --samples "${continuous}/samples.jsonl" \
  --summary "${continuous}/summary.json" \
  --stage test --accelerator cpu --interval 1 &
collector_pid=$!
sleep 1
kill -TERM "${collector_pid}"
wait "${collector_pid}"
python3 - "${continuous}/summary.json" <<'PY'
import json, sys
summary = json.load(open(sys.argv[1], encoding="utf-8"))
assert summary["sample_count"] >= 2
assert summary["allocation"]["cpus"] == 2
assert summary["memory"]["requested_bytes"] == 1024**3
assert summary["wall_seconds_sampled"] > 0
assert summary["cpu"]["efficiency_percent"] is not None
PY

report="${metrics_dir}/report.json"
python3 "${root}/jobs/jlab_ifarm/report_performance.py" \
  "${metrics_dir}/summary.json" --include-supplied-corpus-aggregates \
  --output "${report}"
python3 - "${report}" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["record_count"] == 1
group = report["grouped_by_workflow_stage_family_host_campaign_status"][0]
assert group["queue_wait_seconds_mean"] is None
assert group["missing_queue_wait_count"] == 1
assert report["supplied_corpus_worker_aggregates"]["farm25"]["completed_jobs"] == 8
assert "per-job rows synthesized" in report["supplied_corpus_worker_aggregates"]["provenance"]
PY

grep -F -- 'corpus-generate "${project}" "${corpus}"' \
  "${root}/jobs/jlab_ifarm/run_swif2_corpus_worker.sh" >/dev/null
grep -F -- '--shard-start "${shard_start}" --max-shards "${shard_count}"' \
  "${root}/jobs/jlab_ifarm/run_swif2_corpus_worker.sh" >/dev/null
grep -F -- 'corpus-merge --consume-sources' \
  "${root}/jobs/jlab_ifarm/run_swif2_corpus_merge.sh" >/dev/null
grep -F -- 'payload=(materialize "${project}"' \
  "${root}/jobs/jlab_ifarm/run_swif2_analysis_stage.sh" >/dev/null
grep -F -- 'payload=(train "${project}" --profile "${profile}" --no-progress)' \
  "${root}/jobs/jlab_ifarm/run_swif2_analysis_stage.sh" >/dev/null
grep -F -- 'CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}' \
  "${root}/jobs/jlab_ifarm/run_swif2_analysis_stage.sh" >/dev/null
grep -F -- 'staged SIF digest mismatch' \
  "${root}/jobs/jlab_ifarm/run_swif2_analysis_stage.sh" >/dev/null
grep -F -- 'scientific gate failed and remains recorded' \
  "${root}/jobs/jlab_ifarm/run_swif2_analysis_stage.sh" >/dev/null
grep -F -- 'staged holdout manifest' \
  "${root}/jobs/jlab_ifarm/run_swif2_analysis_stage.sh" >/dev/null

echo "JLab SWIF2 workflow verification: PASS"
