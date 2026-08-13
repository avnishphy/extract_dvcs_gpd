#!/usr/bin/env bash
set -euo pipefail

python_bin=/opt/dvcs/venv/bin/python
allowed_cpus="$(${python_bin} -c 'import os; print(len(os.sched_getaffinity(0)) if hasattr(os,"sched_getaffinity") else (os.cpu_count() or 1))')"
slurm_cpus="${SLURM_CPUS_PER_TASK:-${allowed_cpus}}"
if ! [[ "${slurm_cpus}" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid SLURM_CPUS_PER_TASK=${slurm_cpus}" >&2
    exit 64
fi
if (( slurm_cpus < allowed_cpus )); then allocated_cpus="${slurm_cpus}"; else allocated_cpus="${allowed_cpus}"; fi
export DVCS_CPU_THREADS="${DVCS_CPU_THREADS:-${allocated_cpus}}"
export DVCS_NATIVE_WORKERS="${DVCS_NATIVE_WORKERS:-all_available}"
export OMP_NUM_THREADS="${DVCS_MATH_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${DVCS_MATH_THREADS:-1}"
export MKL_NUM_THREADS="${DVCS_MATH_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${DVCS_MATH_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${DVCS_MATH_THREADS:-1}"

requested="${DVCS_ACCELERATOR:-auto}"
export DVCS_REQUESTED_ACCELERATOR="${requested}"
case "${requested}" in auto|cpu|cuda) ;; *) echo "invalid DVCS_ACCELERATOR=${requested}" >&2; exit 64 ;; esac
if [[ "${requested}" == auto ]]; then
    if ${python_bin} -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
        export DVCS_ACCELERATOR=cuda
        fallback=null
    else
        export DVCS_ACCELERATOR=cpu
        fallback='"Torch reported no usable CUDA device"'
    fi
elif [[ "${requested}" == cuda ]]; then
    ${python_bin} -c 'import torch; assert torch.cuda.is_available(), "CUDA explicitly requested but unusable"'
fi

mkdir -p /results/provenance "${LHAPDF_DATA_PATH}" "${MPLCONFIGDIR}"
${python_bin} - <<'PY'
import json, os, platform, subprocess
from pathlib import Path
import torch
rank = int(os.environ.get("RANK", "0"))
record = {
  "schema_version": 1,
  "requested_accelerator": os.environ.get("DVCS_REQUESTED_ACCELERATOR", "auto"),
  "resolved_accelerator": "cuda" if torch.cuda.is_available() and os.environ.get("DVCS_ACCELERATOR") != "cpu" else "cpu",
  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
  "visible_gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
  "affinity_cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
  "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
  "used_cpu_threads": int(os.environ["DVCS_CPU_THREADS"]),
  "native_workers_request": os.environ["DVCS_NATIVE_WORKERS"],
  "math_library_threads": int(os.environ["OMP_NUM_THREADS"]),
  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
  "hostname": platform.node(),
  "torch_version": torch.__version__,
  "torch_cuda_runtime": torch.version.cuda,
  "distribution_commit": subprocess.run(["git","-C","/opt/dvcs/app","rev-parse","HEAD"], text=True, capture_output=True).stdout.strip() or None,
  "image_digest": os.environ.get("DVCS_IMAGE_DIGEST")
}
target = Path(f"/results/provenance/runtime-rank-{rank}.json")
target.write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
if rank == 0:
    Path("/results/provenance/runtime.json").write_text(json.dumps(record, indent=2, sort_keys=True)+"\n")
PY

bridge=/opt/dvcs/bin/partons_bridge
if [[ "${1:-}" == partons-bridge ]]; then shift; exec "${bridge}" "$@"; fi
if [[ "${1:-}" == shell ]]; then shift; exec /bin/bash "$@"; fi
gpu_count="$(${python_bin} -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"
if (( gpu_count > 1 )) && [[ "${1:-}" == train ]]; then
    exec "${python_bin}" -m torch.distributed.run --standalone \
      --nproc-per-node="${gpu_count}" -m extract_dvcs_cff.cli.user \
      --bridge "${bridge}" "$@"
fi
if (( gpu_count > 1 )) && [[ "${1:-}" == optimize ]]; then
    exec /opt/dvcs/app/scripts/multi-gpu-optimize.sh "${bridge}" "$@"
fi
exec /opt/dvcs/venv/bin/dvcs-infer --bridge "${bridge}" "$@"
