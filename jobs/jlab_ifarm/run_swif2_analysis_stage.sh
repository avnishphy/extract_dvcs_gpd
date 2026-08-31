#!/usr/bin/env bash
set -euo pipefail

: "${SWIF_JOB_WORK_DIR:?SWIF2 work directory is required}"
: "${SLURM_JOB_ID:?Slurm job ID is required}"
: "${SLURM_CPUS_PER_TASK:?Slurm CPU allocation is required}"

stage="${1:?stage}" project="${2:?project}" profile="${3:?profile}"
corpus="${4:?corpus}" selection="${5:?selection}" image="${6:?image}"
database_archive="${7:?database archive}" corpus_archive="${8:?corpus archive or none}"
state_input="${9:?project input}" state_output="${10:?project output}"
summary_output="${11:?summary output}" accelerator="${12:?accelerator}"
heartbeat_seconds="${13:?heartbeat seconds}"
collector="${14:?performance collector}" performance_summary="${15:?performance summary}"
performance_samples="${16:?performance samples}" performance_interval="${17:?performance interval}"
holdout_design="${18:-none}"
lhapdf_archive="${19:-none}"
holdout_model="${20:-none}"

[[ "${stage}" =~ ^(selection|materialize|optimize|train|evaluate|compare|holdout|plot)$ ]] || {
    echo "unsupported analysis stage: ${stage}" >&2
    exit 64
}
[[ "${accelerator}" == cpu || "${accelerator}" == cuda ]] || exit 64
[[ "${holdout_model}" == none || "${holdout_model}" =~ ^GPD(GK11|GK16|GK19|VGG99)$ ]] || exit 64
[[ "${heartbeat_seconds}" =~ ^[0-9]+$ ]] || exit 64
[[ "${performance_interval}" =~ ^[1-9][0-9]*$ ]] || exit 64
cd "${SWIF_JOB_WORK_DIR}"

for path in "${image}" "${database_archive}" "${state_input}" "${collector}"; do
    [[ -s "${path}" && ! -L "${path}" ]] || {
        echo "staged input is missing or unsafe: ${path}" >&2
        exit 66
    }
done
if [[ "${corpus_archive}" != none ]]; then
    [[ -s "${corpus_archive}" && ! -L "${corpus_archive}" ]] || exit 66
fi
if [[ "${stage}" == holdout && "${holdout_design}" != none ]]; then
    [[ -s "${holdout_design}" && ! -L "${holdout_design}" ]] || {
        echo "staged holdout design is missing or unsafe: ${holdout_design}" >&2
        exit 66
    }
    python3 - "${holdout_design}" <<'PY'
import hashlib, json, sys
from pathlib import Path

path = Path(sys.argv[1])
record = json.loads(path.read_text(encoding="utf-8"))
expected = record.pop("manifest_content_sha256", None)
observed = hashlib.sha256(json.dumps(
    record, sort_keys=True, separators=(",", ":"), allow_nan=False,
).encode("utf-8")).hexdigest()
if expected != observed:
    raise SystemExit("holdout design content hash mismatch")
if record.get("protocol") not in {
    "output_blind_fresh_kinematics_v1",
    "native_model_holdout_design_v2",
}:
    raise SystemExit("unsupported holdout design protocol")
points = record.get("npe_dataset_points")
if not isinstance(points, list) or not points:
    raise SystemExit("holdout design has no kinematic points")
expected_flags = {
    "selection_frozen_before_native_outputs": True,
    "training_use": False,
    "architecture_selection_use": False,
    "optuna_objective_use": False,
}
for key, expected_value in expected_flags.items():
    if record.get(key) is not expected_value:
        raise SystemExit(f"holdout design violates {key}")
PY
fi
if [[ "${stage}" == holdout ]]; then
    [[ -s "${lhapdf_archive}" && ! -L "${lhapdf_archive}" ]] || {
        echo "staged MSTW2008nlo68cl archive is missing or unsafe" >&2
        exit 66
    }
fi

safe_archive() {
    python3 - "$1" <<'PY'
import sys, tarfile
from pathlib import PurePosixPath

with tarfile.open(sys.argv[1], "r:*") as archive:
    members = archive.getmembers()
    if not members:
        raise SystemExit("archive is empty")
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise SystemExit(f"unsafe archive member: {member.name}")
PY
}

safe_archive "${database_archive}"
safe_archive "${state_input}"
[[ "${stage}" != holdout ]] || safe_archive "${lhapdf_archive}"
mkdir -p workspace/.corpus_exports results cache database
tar -xf "${database_archive}" -C database
tar -xf "${state_input}" -C workspace
[[ -f "workspace/${project}/experiment.json" ]] || {
    echo "project archive does not contain ${project}/experiment.json" >&2
    exit 66
}
[[ "$(git -C database/gpddatabase rev-parse HEAD)" == \
   "1e9e97fd417ce1d6d44fc73550bdbf32cca4eeb1" ]] || {
    echo "staged GPD database revision mismatch" >&2
    exit 65
}

image_sha="$(sha256sum "${image}" | awk '{print $1}')"
expected_image_prefix="${image#dvcs-image-}"
expected_image_prefix="${expected_image_prefix%%.*}"
[[ "${expected_image_prefix}" =~ ^[0-9a-f]{16}$ && \
   "${image_sha}" == "${expected_image_prefix}"* ]] || {
    echo "staged SIF digest mismatch: expected prefix ${expected_image_prefix}, observed ${image_sha}; SWIF2 input is incomplete or corrupt" >&2
    exit 65
}
sha256sum "${image}" "${database_archive}" "${state_input}" > staged-inputs-sha256.txt
if [[ "${corpus_archive}" != none ]]; then
    sha256sum "${corpus_archive}" >> staged-inputs-sha256.txt
fi
if [[ "${stage}" == holdout && "${holdout_design}" != none ]]; then
    sha256sum "${holdout_design}" >> staged-inputs-sha256.txt
fi
if [[ "${stage}" == holdout ]]; then
    sha256sum "${lhapdf_archive}" >> staged-inputs-sha256.txt
    mkdir -p cache/lhapdf
    tar -xf "${lhapdf_archive}" -C cache/lhapdf
    [[ -f cache/lhapdf/MSTW2008nlo68cl/MSTW2008nlo68cl.info && \
       -f cache/lhapdf/MSTW2008nlo68cl/MSTW2008nlo68cl_0000.dat ]] || {
        echo "staged MSTW2008nlo68cl content is incomplete" >&2
        exit 65
    }
    echo "[dvcs-swif2] staged LHAPDF set MSTW2008nlo68cl"
fi

container=(
    /usr/bin/apptainer run --cleanenv
    --env "DVCS_ACCELERATOR=${accelerator}"
    --env "DVCS_IMAGE_DIGEST=sha256:${image_sha}"
    --env DVCS_PROGRESS=0
    --env DVCS_GPDDATABASE_ROOT=/database/gpddatabase
    --env DVCS_WORKSPACE_ROOT=/workspace
    --env "SLURM_JOB_ID=${SLURM_JOB_ID}"
    --env "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
    --bind "${SWIF_JOB_WORK_DIR}/workspace:/workspace"
    --bind "${SWIF_JOB_WORK_DIR}/results:/results"
    --bind "${SWIF_JOB_WORK_DIR}/cache:/cache"
    --bind "${SWIF_JOB_WORK_DIR}/database:/database:ro"
)
if [[ "${accelerator}" == cuda ]]; then
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != NoDevFiles ]] || {
        echo "GPU stage has no visible Slurm GPU allocation" >&2
        exit 69
    }
    container+=(--nv --env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi
if [[ "${stage}" == holdout ]]; then
    container+=(--env DVCS_HOLDOUT_DESIGN=/workspace/configs/validation/stage11_blind_fresh_kinematics_v1.json)
    [[ "${holdout_model}" == none ]] || container+=(--env "DVCS_HOLDOUT_MODEL=${holdout_model}")
fi
container+=("${image}")

started_epoch="$(date +%s)"
performance_pid=
stop_performance() {
    if [[ -n "${performance_pid}" ]]; then
        kill -TERM "${performance_pid}" 2>/dev/null || true
        wait "${performance_pid}" 2>/dev/null || true
        performance_pid=
    fi
}
trap stop_performance EXIT
/usr/bin/python3 "${collector}" --stage "${stage}" --accelerator "${accelerator}" \
    --interval "${performance_interval}" --samples "${performance_samples}" \
    --summary "${performance_summary}" &
performance_pid=$!
echo "[dvcs-swif2] start stage=${stage} job=${SLURM_JOB_ID} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} gpus=${CUDA_VISIBLE_DEVICES:-none}"

if [[ "${corpus_archive}" != none ]]; then
    cp --reflink=auto "${corpus_archive}" workspace/.corpus_exports/staged-corpus.tar.gz
    "${container[@]}" corpus-import staged-corpus "${corpus}" > corpus-import.json
fi

# Legacy images infer the holdout-manifest root from the staged engine config,
# which resolves to /workspace. Copy the immutable, self-hashed image manifest
# there so those images remain resumable without changing scientific inputs.
if [[ "${stage}" == holdout ]]; then
    mkdir -p workspace/configs/validation
    if [[ "${holdout_design}" == none ]]; then
        "${container[@]}" shell -c \
          'cp /opt/dvcs/app/configs/validation/stage11_blind_fresh_kinematics_v1.json /workspace/configs/validation/'
    else
        cp "${holdout_design}" \
          workspace/configs/validation/stage11_blind_fresh_kinematics_v1.json
    fi
    [[ -s workspace/configs/validation/stage11_blind_fresh_kinematics_v1.json ]] || {
        echo "failed to stage image-embedded holdout manifest" >&2
        exit 66
    }
    echo "[dvcs-swif2] staged holdout manifest source=${holdout_design}"
fi

case "${stage}" in
    selection)
        payload=(selection-create "${project}" "${corpus}" "${selection}" --profile "${profile}") ;;
    materialize)
        payload=(materialize "${project}" --profile "${profile}" --corpus "${corpus}" --selection "${selection}" --workers all_available --no-progress) ;;
    optimize)
        payload=(optimize "${project}" --profile "${profile}" --no-progress) ;;
    train)
        payload=(train "${project}" --profile "${profile}" --no-progress) ;;
    evaluate)
        payload=(evaluate "${project}" --profile "${profile}" --no-progress) ;;
    compare)
        payload=(compare "${project}" --profile "${profile}" --no-progress) ;;
    holdout)
        payload=(holdout "${project}" --profile "${profile}" --no-progress) ;;
    plot)
        payload=(plot "${project}" --profile "${profile}") ;;
esac

progress_snapshot() {
    python3 - "${stage}" "workspace/${project}/results/${profile}" \
      "${performance_summary}" <<'PY' 2>/dev/null || true
import json, sys
from pathlib import Path

stage, root, performance = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
materialization = root / "materialization_progress.json"
if materialization.is_file():
    record = json.loads(materialization.read_text())
    print(f"materialization={record.get('completed', 0)}/{record.get('total', '?')}", end=" ")
if stage == "train":
    print(f"models={len(list((root / 'training').glob('seed_*/training_metrics.json')))}", end="")
elif stage == "optimize":
    print(f"trials={len(list((root / 'optimization/trials').glob('trial_*')))}", end="")
elif stage == "holdout":
    truths = len(list(root.glob('holdouts/*/*/native_truth/response.json')))
    models = len(list(root.glob('holdouts/*/*/metrics.json')))
    print(f"native_truths={truths}/4 models={models}/4", end="")
if performance.is_file():
    metrics = json.loads(performance.read_text())
    efficiency = metrics.get("cpu", {}).get("efficiency_percent")
    memory = metrics.get("memory", {}).get("max_current_bytes")
    if efficiency is not None:
        print(f" cpu_efficiency={efficiency:.1f}%", end="")
    if memory is not None:
        print(f" memory_gib={memory / 2**30:.2f}", end="")
    devices = metrics.get("gpu", {}).get("devices", [])
    if devices and devices[0].get("mean_gpu_utilization_percent") is not None:
        print(f" gpu_mean={devices[0]['mean_gpu_utilization_percent']:.1f}%", end="")
    if devices and devices[0].get("max_memory_used_mib") is not None:
        print(f" gpu_memory_mib={devices[0]['max_memory_used_mib']:.0f}", end="")
PY
}

set +e
"${container[@]}" "${payload[@]}" > payload.json 2> >(tee payload.err >&2) &
payload_pid=$!
heartbeat_pid=
if (( heartbeat_seconds > 0 )); then
    (
        while sleep "${heartbeat_seconds}"; do
            kill -0 "${payload_pid}" 2>/dev/null || exit 0
            echo "[dvcs-swif2] heartbeat stage=${stage} elapsed_seconds=$(($(date +%s) - started_epoch)) $(progress_snapshot)"
        done
    ) &
    heartbeat_pid=$!
fi
wait "${payload_pid}"
status=$?
if [[ -n "${heartbeat_pid}" ]]; then
    kill "${heartbeat_pid}" 2>/dev/null
    wait "${heartbeat_pid}" 2>/dev/null
fi
set -e
scientific_gate_failed=false
if (( status == 1 )) && [[ "${stage}" == compare ]] && \
   python3 - "payload.json" \
     "workspace/${project}/results/${profile}/comparison/comparison_metrics.json" <<'PY'
import json, sys
from pathlib import Path

public = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
metrics = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
status = str(metrics.get("status", ""))
if (
    public.get("command") != "compare"
    or public.get("status") != "fail"
    or metrics.get("passed") is not False
    or not (status == "complete" or status.startswith("complete_with_"))
):
    raise SystemExit(1)
PY
then
    echo "[dvcs-swif2] compare execution complete; scientific gate failed and remains recorded"
    status=0
    scientific_gate_failed=true
fi
if (( status == 0 )); then
    if [[ "${stage}" == holdout && "${holdout_design}" != none ]]; then
        python3 - "${holdout_design}" \
          "workspace/${project}/results/${profile}" \
          payload.json "${holdout_model}" <<'PY'
import json, os, sys
from pathlib import Path

design_path, result_root, payload_path = map(Path, sys.argv[1:4])
holdout_model = sys.argv[4]
design = json.loads(design_path.read_text(encoding="utf-8"))
design_name = design.get("design_name", design_path.stem)
modern_summary = result_root / "holdouts" / design_name / "summary.json"
legacy_summary = result_root / "holdout" / "summary.json"
partial_summary = result_root / "holdouts" / design_name / f"partial-{holdout_model}.json"
summary_path = (
    modern_summary if modern_summary.is_file()
    else partial_summary if partial_summary.is_file()
    else legacy_summary
)
if not summary_path.is_file():
    raise SystemExit("completed holdout did not write a summary")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
provenance = {
    "design_name": design_name,
    "relationship_to_training": design.get("relationship_to_training"),
    "claim_scope": design.get("claim_scope"),
    "kinematic_count": len(design["npe_dataset_points"]),
    "manifest_content_sha256": design["manifest_content_sha256"],
}
summary["holdout_design"] = provenance
external = summary.get("external_validation_kinematics", {})
same_design = provenance["relationship_to_training"] == "same_training_kinematics"
external["uses_same_kinematic_points_as_dd_pseudodata"] = same_design
external["fresh_kinematics_output_blind"] = (
    provenance["relationship_to_training"] == "fresh_unseen_kinematics"
)
external["relationship_to_training"] = provenance["relationship_to_training"]
external["claim_scope"] = provenance["claim_scope"]
summary["external_validation_kinematics"] = external
payload = json.loads(payload_path.read_text(encoding="utf-8"))
payload["holdout_design"] = provenance
for path, value in ((summary_path, summary), (payload_path, payload)):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
PY
    fi
    if find "workspace/${project}" -type l -print -quit | grep -q .; then
        echo "project output contains a symlink" >&2
        exit 65
    fi
    if [[ "${stage}" == holdout && "${holdout_model}" != none ]]; then
        design_name="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["design_name"])' "${holdout_design}")"
        model_root="workspace/${project}/results/${profile}/holdouts/${design_name}/${holdout_model}"
        [[ -f "${model_root}/metrics.json" ]] || exit 65
        python3 - "${model_root}/swif-partial.json" "${holdout_model}" \
          "${design_name}" "${holdout_design}" "${state_input}" "${image_sha}" <<'PY'
import hashlib, json, sys
from pathlib import Path
output, model, design, design_path, state_path, image_sha = sys.argv[1:]
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
Path(output).write_text(json.dumps({
    "schema_version": 1, "model": model, "design_name": design,
    "design_sha256": sha(design_path), "base_state_sha256": sha(state_path),
    "image_sha256": image_sha,
}, indent=2, sort_keys=True) + "\n")
PY
        tar -cf "${state_output}.partial" -C \
          "workspace/${project}/results/${profile}" \
          "holdouts/${design_name}/${holdout_model}" \
          "holdouts/${design_name}/partial-${holdout_model}.json"
    else
        tar -cf "${state_output}.partial" -C workspace "${project}"
    fi
    mv "${state_output}.partial" "${state_output}"
fi
stop_performance
trap - EXIT

python3 - "${summary_output}" "${stage}" "${status}" "${started_epoch}" \
  "${performance_summary}" "${scientific_gate_failed}" <<'PY'
import json, os, platform, sys, time
from pathlib import Path

output, stage, status, started, performance, scientific_gate_failed = sys.argv[1:]
performance_record = json.loads(Path(performance).read_text())
if performance_record.get("status") != "complete":
    raise SystemExit("performance collection did not complete")
record = {
    "schema_version": 1,
    "status": "ok" if int(status) == 0 else "failed",
    "stage": stage,
    "exit_code": int(status),
    "elapsed_seconds": int(time.time()) - int(started),
    "hostname": platform.node(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "swif_job_id": os.environ.get("SWIF_JOB_ID"),
    "swif_job_attempt_id": os.environ.get("SWIF_JOB_ATTEMPT_ID"),
    "performance": performance_record,
    "scientific_gate_failed": scientific_gate_failed == "true",
}
runtime = Path("results/provenance/runtime.json")
if runtime.is_file():
    record["runtime"] = json.loads(runtime.read_text())
Path(output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
PY
(( status == 0 )) || exit "${status}"
echo "[dvcs-swif2] finish stage=${stage} elapsed_seconds=$(($(date +%s) - started_epoch))"
