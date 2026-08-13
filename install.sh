#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
profile=local
accelerator=auto
dry_run=false
source_build=false

die() { echo "install: $*" >&2; exit 69; }
log() { echo "install: $*"; }
require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required host command not found: $1"
}
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
[[ -z "${DVCS_ENGINE:-}" || "${DVCS_ENGINE}" =~ ^(podman|docker)$ ]] || {
    echo "DVCS_ENGINE must be podman or docker" >&2
    exit 64
}
require_command python3

state="${root}/.dvcs"
workspace="${DVCS_WORKSPACE:-${root}/workspace}"
results="${DVCS_RESULTS:-${root}/results}"
cache="${DVCS_CACHE:-${root}/cache}"
database="${DVCS_DATABASE:-${state}/database}"

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
        for candidate in podman docker; do
            if command -v "${candidate}" >/dev/null 2>&1 && \
               { [[ "${dry_run}" == true ]] || "${candidate}" info >/dev/null 2>&1; }; then
                engine="${candidate}"
                break
            fi
        done
    fi
    if [[ -z "${engine}" && "${dry_run}" != true ]]; then
        die "no usable Podman or Docker engine found; install/start one or set DVCS_ENGINE"
    fi
    engine="${engine:-podman-or-docker}"
    if [[ "${published}" == true && -n "${digest}" && "${source_build}" != true ]]; then
        image="${registry}@${digest}"
    else
        image="localhost/extract-dvcs-gpd:${tag}"
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
    image="${state}/extract-dvcs-gpd-${tag}.sif"
    gpu_args=""
    apptainer_nv=""
    [[ "${resolved}" == cuda ]] && apptainer_nv="--nv"
fi

if [[ "${dry_run}" == true ]]; then
    if [[ "${published}" == true && -n "${digest}" && "${source_build}" != true ]]; then
        action="pull immutable image ${registry}@${digest}"
    else
        action="build locked ${variant} image from source"
    fi
    log "dry-run only; no files, images, or data will be changed"
    log "profile=${profile} accelerator=${resolved} engine=${engine}"
    log "would ${action}"
    log "would verify/install MSTW2008nlo68cl and gpddatabase, then write ${state}/install.env"
    exit 0
fi

for command_name in curl git sha256sum tar mktemp; do require_command "${command_name}"; done
if [[ "${profile}" == local ]]; then
    require_command "${engine}"
    "${engine}" info >/dev/null 2>&1 || die "${engine} is installed but not usable by this user"
else
    require_command apptainer
fi
mkdir -p "${state}" "${workspace}" "${results}" "${cache}/lhapdf" "${database}"

if [[ "${profile}" == local ]]; then
    log "preparing ${variant} container image with ${engine}"
    if [[ "${published}" == true && -n "${digest}" && "${source_build}" != true ]]; then
        "${engine}" pull "${image}"
    else
        log "no approved published digest is available; the first locked source build can take a while"
        torch_index=https://download.pytorch.org/whl/cpu
        [[ "${resolved}" == cuda ]] && torch_index=https://download.pytorch.org/whl/cu126
        distribution_revision="$(git -C "${root}" rev-parse HEAD 2>/dev/null || echo unknown)"
        "${engine}" build \
            --file "${root}/containers/Dockerfile" --target runtime \
            --build-arg "TORCH_INDEX_URL=${torch_index}" \
            --build-arg "IMAGE_REVISION=${distribution_revision}" \
            --tag "${image}" "${root}"
    fi
    verification_gpu_args=()
    if [[ "${resolved}" == cuda ]]; then
        if [[ "${engine}" == podman ]]; then
            verification_gpu_args=(--device nvidia.com/gpu=all)
        else
            verification_gpu_args=(--gpus all)
        fi
    fi
    log "running native and Python installation self-tests"
    "${engine}" run --rm "${verification_gpu_args[@]}" "${image}" partons-bridge --self-test
    "${engine}" run --rm "${verification_gpu_args[@]}" "${image}" shell -lc \
        'python -c "import matplotlib, numpy, optuna, particle, scipy, sbi, torch, yaml, zuko; assert torch.cuda.is_available() == ('"$([[ "${resolved}" == cuda ]] && echo True || echo False)"')" && pip check'
else
    log "preparing JLab Apptainer image"
    candidate_image="${image}"
    built_image=false
    if [[ ! -s "${image}" ]]; then
        candidate_image="${image}.partial"
        built_image=true
        if [[ "${published}" == true && -n "${digest}" && "${source_build}" != true ]]; then
            apptainer pull --force "${candidate_image}" "docker://${registry}@${digest}"
        else
            log "no approved published digest is available; the first locked fakeroot source build can take a while"
            wheelhouse="${DVCS_WHEELHOUSE:-${TMPDIR:-/tmp}/extract-dvcs-gpd-wheels-${UID}/${resolved}}"
            mkdir -p "${wheelhouse}"
            if [[ "${resolved}" == cuda ]]; then
                torch_wheel="torch-2.12.1+cu126-cp312-cp312-manylinux_2_28_x86_64.whl"
                torch_url="https://download.pytorch.org/whl/cu126/torch-2.12.1%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"
                torch_sha256="fcc3cf2026f15afeb69f51fa3fde7b286ffd7e6d9b8e070021ca8abd95bd220a"
            else
                torch_wheel="torch-2.12.1+cpu-cp312-cp312-manylinux_2_28_x86_64.whl"
                torch_url="https://download.pytorch.org/whl/cpu/torch-2.12.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl"
                torch_sha256="ae4bb28409f5370852bd71af221066236c38d647f780d9b0a7240c330a9c12df"
            fi
            if ! echo "${torch_sha256}  ${wheelhouse}/${torch_wheel}" | sha256sum -c - >/dev/null 2>&1; then
                log "downloading and verifying the pinned PyTorch wheel on the host"
                curl --fail --location --http1.1 --retry 10 --retry-all-errors --connect-timeout 20 \
                    --max-time 1800 --continue-at - "${torch_url}" -o "${wheelhouse}/${torch_wheel}.partial"
                echo "${torch_sha256}  ${wheelhouse}/${torch_wheel}.partial" | sha256sum -c -
                mv "${wheelhouse}/${torch_wheel}.partial" "${wheelhouse}/${torch_wheel}"
            fi
            (
                build_context="$(mktemp -d "${TMPDIR:-/tmp}/extract-dvcs-gpd-context.XXXXXX")"
                trap 'rm -rf -- "${build_context}"' EXIT
                tar --exclude-vcs --exclude-from="${root}/.dockerignore" \
                    -C "${root}" -cf "${build_context}/context.tar" .
                tar -xf "${build_context}/context.tar" -C "${build_context}"
                rm -f "${build_context}/context.tar"
                cd "${build_context}"
                apptainer build --force --fakeroot \
                    --bind "${wheelhouse}:/tmp/dvcs-wheelhouse:ro" \
                    "${candidate_image}" containers/apptainer.def
            )
        fi
    fi
    log "running native and Python installation self-tests"
    apptainer exec --cleanenv --bind "${cache}:/cache" \
        "${candidate_image}" /opt/dvcs/bin/partons_bridge --self-test
    apptainer exec --cleanenv --bind "${cache}:/cache" \
        "${candidate_image}" /opt/dvcs/venv/bin/python -c \
        'import matplotlib, numpy, optuna, particle, scipy, sbi, torch, yaml, zuko'
    apptainer exec --cleanenv --bind "${cache}:/cache" \
        "${candidate_image}" /opt/dvcs/venv/bin/pip check
    [[ "${built_image}" == true ]] && mv "${candidate_image}" "${image}"
    if [[ ! -f "${root}/jobs/jlab_ifarm/resources.env" ]]; then
        cp "${root}/jobs/jlab_ifarm/resources.env.example" "${root}/jobs/jlab_ifarm/resources.env"
    fi
fi

# Data are host-owned and checksum verified, not silently embedded in images.
set_dir="${cache}/lhapdf/MSTW2008nlo68cl"
verify_lhapdf_set() {
    local target="$1" members=()
    [[ -f "${target}/MSTW2008nlo68cl.info" ]] || return 1
    [[ -f "${target}/MSTW2008nlo68cl_0000.dat" ]] || return 1
    [[ -f "${target}/MSTW2008nlo68cl_0040.dat" ]] || return 1
    echo '07b1ee816adfd9e438bb996d20116dbcb4a0e3624e5b2075aecb7e699a37b0f3  '"${target}/MSTW2008nlo68cl.info" | sha256sum -c - >/dev/null
    echo '26e94045264006998fa0af3dc5d5c0e7e304820ac5ce43eee43413197eae7e78  '"${target}/MSTW2008nlo68cl_0000.dat" | sha256sum -c - >/dev/null
    shopt -s nullglob
    members=("${target}"/MSTW2008nlo68cl_*.dat)
    shopt -u nullglob
    [[ "${#members[@]}" -eq 41 ]]
}
if [[ -e "${set_dir}" ]]; then
    verify_lhapdf_set "${set_dir}" || { echo "invalid or incomplete LHAPDF set: ${set_dir}" >&2; exit 65; }
    log "verified existing MSTW2008nlo68cl data"
else
    log "downloading and verifying MSTW2008nlo68cl data"
    temp="$(mktemp -d "${state}/lhapdf.XXXXXX")"
    trap 'rm -rf -- "${temp}"' EXIT
    curl --fail --location --retry 3 --retry-all-errors --connect-timeout 20 \
        https://lhapdfsets.web.cern.ch/current/MSTW2008nlo68cl.tar.gz -o "${temp}/set.tar.gz"
    echo '98ec0541e80e223785bb6029ebf81e93ca5111da41d6565b3b5c4aa86d59bb5d  '"${temp}/set.tar.gz" | sha256sum -c -
    tar -xzf "${temp}/set.tar.gz" -C "${temp}"
    verify_lhapdf_set "${temp}/MSTW2008nlo68cl" || { echo "downloaded LHAPDF set failed content verification" >&2; exit 65; }
    mv "${temp}/MSTW2008nlo68cl" "${set_dir}"
    rm -rf -- "${temp}"; trap - EXIT
fi
db_dir="${database}/gpddatabase"
if [[ ! -e "${db_dir}" ]]; then
    log "cloning pinned gpddatabase data"
    db_temp="$(mktemp -d "${database}/gpddatabase.XXXXXX")"
    trap 'rm -rf -- "${db_temp}"' EXIT
    git clone --filter=blob:none https://github.com/opengpd/gpddatabase.git "${db_temp}/checkout"
    git -C "${db_temp}/checkout" checkout --detach 1e9e97fd417ce1d6d44fc73550bdbf32cca4eeb1
    mv "${db_temp}/checkout" "${db_dir}"
    rm -rf -- "${db_temp}"; trap - EXIT
elif [[ -d "${db_dir}/.git" ]]; then
    [[ "$(git -C "${db_dir}" rev-parse HEAD)" == 1e9e97fd417ce1d6d44fc73550bdbf32cca4eeb1 ]] || { echo "database revision mismatch" >&2; exit 65; }
    [[ -z "$(git -C "${db_dir}" status --porcelain)" ]] || { echo "database checkout is dirty; refusing to package it" >&2; exit 65; }
    log "verified existing gpddatabase checkout"
else
    echo "database path exists but is not a Git checkout: ${db_dir}" >&2
    exit 65
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
