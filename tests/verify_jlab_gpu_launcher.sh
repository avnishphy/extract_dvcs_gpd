#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
temp="$(mktemp -d)"
trap 'rm -rf -- "${temp}"' EXIT
mkdir -p "${temp}/checkout/.dvcs" "${temp}/bin"
cp "${root}/dvcs" "${temp}/checkout/dvcs"
cat > "${temp}/checkout/.dvcs/install.env" <<EOF
DVCS_PROFILE=jlab_ifarm
DVCS_ACCELERATOR=auto
DVCS_ENGINE=apptainer
DVCS_IMAGE=${temp}/image.sif
DVCS_IMAGE_DIGEST=test-digest
DVCS_WORKSPACE=${temp}/workspace
DVCS_RESULTS=${temp}/results
DVCS_CACHE=${temp}/cache
DVCS_DATABASE=${temp}/database
DVCS_CONTAINER_GPU_ARGS=''
DVCS_APPTAINER_NV=--nv
EOF
cat > "${temp}/bin/apptainer" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" > "${DVCS_TEST_ARGS}"
EOF
cat > "${temp}/bin/hostname" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' ifarm2402
EOF
cat > "${temp}/bin/srun" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" > "${DVCS_TEST_SRUN_ARGS}"
EOF
chmod +x "${temp}/bin/apptainer"
chmod +x "${temp}/bin/hostname" "${temp}/bin/srun"
args="${temp}/args"
PATH="${temp}/bin:${PATH}" DVCS_TEST_ARGS="${args}" \
  DVCS_ACCELERATOR=auto CUDA_VISIBLE_DEVICES=2 SLURM_JOB_ID=123 \
  SLURM_CPUS_PER_TASK=4 \
  "${temp}/checkout/dvcs" doctor test
grep -Fx -- '--nv' "${args}" >/dev/null
grep -Fx -- 'CUDA_VISIBLE_DEVICES=2' "${args}" >/dev/null
grep -Fx -- 'SLURM_JOB_ID=123' "${args}" >/dev/null
grep -Fx -- 'SLURM_CPUS_PER_TASK=4' "${args}" >/dev/null

srun_args="${temp}/srun-args"
PATH="${temp}/bin:${PATH}" DVCS_TEST_SRUN_ARGS="${srun_args}" \
  DVCS_IFARM_GPU_DOCTOR=yes JLAB_ACCOUNT=hallc \
  "${temp}/checkout/dvcs" doctor test
grep -Fx -- '--partition=gpu' "${srun_args}" >/dev/null
grep -Fx -- '--account=hallc' "${srun_args}" >/dev/null
grep -Fx -- '--gres=gpu:1' "${srun_args}" >/dev/null
grep -Fx -- '--unbuffered' "${srun_args}" >/dev/null
grep -Fx -- 'DVCS_ACCELERATOR=cuda' "${srun_args}" >/dev/null
grep -Fx -- 'DVCS_IFARM_GPU_DOCTOR=no' "${srun_args}" >/dev/null
grep -Fx -- 'doctor' "${srun_args}" >/dev/null
grep -Fx -- 'test' "${srun_args}" >/dev/null

for stage in optimize train evaluate; do
    rm -f "${srun_args}"
    stage_output="${temp}/${stage}-output"
    if PATH="${temp}/bin:${PATH}" DVCS_TEST_SRUN_ARGS="${srun_args}" \
       "${temp}/checkout/dvcs" "${stage}" test --profile quick \
       2>"${stage_output}"; then
        echo "interactive ${stage} unexpectedly succeeded" >&2
        exit 1
    fi
    grep -F -- 'use ./dvcs farm-submit' "${stage_output}" >/dev/null
    [[ ! -e "${srun_args}" ]] || {
        echo "interactive ${stage} unexpectedly invoked srun" >&2
        exit 1
    }
done

rm -f "${srun_args}"
PATH="${temp}/bin:${PATH}" DVCS_TEST_ARGS="${args}" \
  DVCS_TEST_SRUN_ARGS="${srun_args}" DVCS_IFARM_GPU_DOCTOR=yes \
  "${temp}/checkout/dvcs" show doctor
[[ ! -e "${srun_args}" ]] || {
    echo "a project named doctor unexpectedly triggered GPU allocation" >&2
    exit 1
}

rm -f "${srun_args}"
PATH="${temp}/bin:${PATH}" DVCS_TEST_ARGS="${args}" \
  DVCS_TEST_SRUN_ARGS="${srun_args}" DVCS_IFARM_GPU_TRAIN=yes \
  "${temp}/checkout/dvcs" show train
[[ ! -e "${srun_args}" ]] || {
    echo "a project named train unexpectedly triggered GPU allocation" >&2
    exit 1
}

PATH="${temp}/bin:${PATH}" DVCS_TEST_ARGS="${args}" \
  DVCS_ACCELERATOR=cpu CUDA_VISIBLE_DEVICES=2 SLURM_CPUS_PER_TASK=4 \
  "${temp}/checkout/dvcs" doctor test
if grep -Fx -- '--nv' "${args}" >/dev/null; then
    echo "CPU launch unexpectedly enabled Apptainer GPU passthrough" >&2
    exit 1
fi
if grep -Fx -- 'SLURM_CPUS_PER_TASK=4' "${args}" >/dev/null; then
    echo "interactive launch unexpectedly imported a Slurm CPU limit" >&2
    exit 1
fi

if PATH="${temp}/bin:${PATH}" DVCS_TEST_ARGS="${args}" \
   DVCS_ACCELERATOR=cuda CUDA_VISIBLE_DEVICES=NoDevFiles \
   "${temp}/checkout/dvcs" doctor test >/dev/null 2>&1; then
    echo "CUDA launch without a Slurm GPU allocation unexpectedly succeeded" >&2
    exit 1
fi

echo "JLab GPU launcher verification: PASS"
