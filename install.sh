#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
profile=local
accelerator=auto
dry_run=false
source_build=false
while (($#)); do
    case "$1" in
        --profile) profile="${2:?missing profile}"; shift 2 ;;
        --accelerator) accelerator="${2:?missing accelerator}"; shift 2 ;;
        --source-build) source_build=true; shift ;;
        --dry-run) dry_run=true; shift ;;
        -h|--help)
            echo "usage: ./install.sh [--profile local|jlab_ifarm] [--accelerator auto|cpu|cuda] [--source-build] [--dry-run]"
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done
[[ "${profile}" =~ ^(local|jlab_ifarm)$ ]] || { echo "unsupported profile: ${profile}" >&2; exit 64; }
[[ "${accelerator}" =~ ^(auto|cpu|cuda)$ ]] || { echo "unsupported accelerator: ${accelerator}" >&2; exit 64; }

state="${root}/.dvcs"
workspace="${DVCS_WORKSPACE:-${root}/workspace}"
results="${DVCS_RESULTS:-${root}/results}"
cache="${DVCS_CACHE:-${root}/cache}"
database="${DVCS_DATABASE:-${state}/database}"
mkdir -p "${state}" "${workspace}" "${results}" "${cache}/lhapdf" "${database}"

read_image_field() {
    python3 - "${root}/provenance/images.lock.json" "$1" "$2" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))["images"][sys.argv[2]][sys.argv[3]]
print("" if value is None else value)
PY
}

resolved="${accelerator}"
if [[ "${resolved}" == auto ]]; then
    if [[ "${profile}" == jlab_ifarm ]]; then
        # Login nodes need not expose GPUs; build the CUDA-capable SIF for GPU jobs.
        resolved=cuda
    elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then resolved=cuda; else resolved=cpu; fi
fi
runtime_accelerator="${accelerator}"
variant="${resolved}"
[[ "${profile}" == jlab_ifarm ]] && variant=jlab_ifarm
tag="$(read_image_field "${variant}" tag)"
digest="$(read_image_field "${variant}" digest)"
registry="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["registry"])' "${root}/provenance/images.lock.json")"
published="$(python3 -c 'import json,sys; print(str(json.load(open(sys.argv[1]))["published"]).lower())' "${root}/provenance/images.lock.json")"

if [[ "${profile}" == local ]]; then
    engine="${DVCS_ENGINE:-}"
    if [[ -z "${engine}" ]]; then
        for candidate in podman docker; do command -v "${candidate}" >/dev/null 2>&1 && { engine="${candidate}"; break; }; done
    fi
    if [[ -z "${engine}" && "${dry_run}" != true ]]; then
        echo "rootless Podman or Docker is required; no host packages were changed" >&2
        exit 69
    fi
    engine="${engine:-podman-or-docker}"
    if [[ "${published}" == true && -n "${digest}" && "${source_build}" != true ]]; then
        image="${registry}@${digest}"
        [[ "${dry_run}" == true ]] || "${engine}" pull "${image}"
    else
        image="localhost/extract-dvcs-gpd:${tag}"
        torch_index=https://download.pytorch.org/whl/cpu
        [[ "${resolved}" == cuda ]] && torch_index=https://download.pytorch.org/whl/cu126
        [[ "${dry_run}" == true ]] || "${engine}" build \
            --file "${root}/containers/Dockerfile" --target runtime \
            --build-arg "TORCH_INDEX_URL=${torch_index}" --tag "${image}" "${root}"
    fi
    gpu_args=""
    if [[ "${resolved}" == cuda ]]; then
        if [[ "${engine}" == podman ]]; then
            gpu_args="--device nvidia.com/gpu=all"
        else
            gpu_args="--gpus all"
        fi
    fi
    apptainer_nv=""
else
    engine=apptainer
    if ! command -v apptainer >/dev/null 2>&1 && [[ "${dry_run}" != true ]]; then
        echo "Apptainer is required on ifarm; Docker daemon access is never used" >&2
        exit 69
    fi
    image="${state}/extract-dvcs-gpd-${tag}.sif"
    if [[ ! -s "${image}" && "${dry_run}" != true ]]; then
        if [[ "${published}" == true && -n "${digest}" && "${source_build}" != true ]]; then
            apptainer pull "${image}.partial" "docker://${registry}@${digest}"
        else
            (cd "${root}" && apptainer build --fakeroot "${image}.partial" containers/apptainer.def)
        fi
        mv "${image}.partial" "${image}"
    fi
    gpu_args=""
    apptainer_nv=""
    [[ "${resolved}" == cuda ]] && apptainer_nv="--nv"
    if [[ ! -f "${root}/jobs/jlab_ifarm/resources.env" ]]; then
        cp "${root}/jobs/jlab_ifarm/resources.env.example" "${root}/jobs/jlab_ifarm/resources.env"
    fi
fi

# Data are host-owned and checksum verified, not silently embedded in images.
set_dir="${cache}/lhapdf/MSTW2008nlo68cl"
if [[ ! -f "${set_dir}/MSTW2008nlo68cl_0040.dat" && "${dry_run}" != true ]]; then
    temp="$(mktemp -d "${state}/lhapdf.XXXXXX")"
    trap 'rm -rf -- "${temp}"' EXIT
    curl -fL https://lhapdfsets.web.cern.ch/current/MSTW2008nlo68cl.tar.gz -o "${temp}/set.tar.gz"
    echo '98ec0541e80e223785bb6029ebf81e93ca5111da41d6565b3b5c4aa86d59bb5d  '"${temp}/set.tar.gz" | sha256sum -c -
    tar -xzf "${temp}/set.tar.gz" -C "${temp}"
    [[ ! -e "${set_dir}" ]] || { echo "incomplete existing LHAPDF set: ${set_dir}" >&2; exit 65; }
    mv "${temp}/MSTW2008nlo68cl" "${set_dir}"
    rm -rf -- "${temp}"; trap - EXIT
fi
db_dir="${database}/gpddatabase"
if [[ ! -e "${db_dir}" && "${dry_run}" != true ]]; then
    git clone https://github.com/opengpd/gpddatabase.git "${db_dir}.partial"
    git -C "${db_dir}.partial" checkout --detach 1e9e97fd417ce1d6d44fc73550bdbf32cca4eeb1
    mv "${db_dir}.partial" "${db_dir}"
elif [[ -d "${db_dir}/.git" && "${dry_run}" != true ]]; then
    [[ "$(git -C "${db_dir}" rev-parse HEAD)" == 1e9e97fd417ce1d6d44fc73550bdbf32cca4eeb1 ]] || { echo "database revision mismatch" >&2; exit 65; }
    [[ -z "$(git -C "${db_dir}" status --porcelain)" ]] || { echo "database checkout is dirty; refusing to package it" >&2; exit 65; }
fi

temp_env="${state}/install.env.partial"
{
  printf 'DVCS_PROFILE=%q\n' "${profile}"
  printf 'DVCS_ACCELERATOR=%q\n' "${runtime_accelerator}"
  printf 'DVCS_ENGINE=%q\n' "${engine}"
  printf 'DVCS_IMAGE=%q\n' "${image}"
  printf 'DVCS_IMAGE_DIGEST=%q\n' "${digest}"
  printf 'DVCS_WORKSPACE=%q\n' "${workspace}"
  printf 'DVCS_RESULTS=%q\n' "${results}"
  printf 'DVCS_CACHE=%q\n' "${cache}"
  printf 'DVCS_DATABASE=%q\n' "${database}"
  printf 'DVCS_CONTAINER_GPU_ARGS=%q\n' "${gpu_args}"
  printf 'DVCS_APPTAINER_NV=%q\n' "${apptainer_nv}"
} > "${temp_env}"
mv "${temp_env}" "${state}/install.env"
echo "installed profile=${profile} accelerator=${resolved} image=${image}"
echo "next: ./dvcs init my-study"
