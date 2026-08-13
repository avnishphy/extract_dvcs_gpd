#!/usr/bin/env bash
set -euo pipefail
bridge="${1:?bridge required}"; shift
[[ "${1:-}" == optimize ]] || { echo "internal multi-GPU optimizer misuse" >&2; exit 64; }
project="${2:?project required}"
profile=quick
requested=""
args=("$@")
for ((index=0; index<${#args[@]}; index++)); do
    case "${args[index]}" in
        --profile) profile="${args[index+1]:?missing profile}" ;;
        --trials) requested="${args[index+1]:?missing trial count}" ;;
    esac
done
if [[ -z "${requested}" ]]; then
    requested="$(/opt/dvcs/venv/bin/python - "${DVCS_WORKSPACE_ROOT}/${project}/experiment.json" "${profile}" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
print(value["inference"]["hyperparameter_optimization"]["trials"][sys.argv[2]])
PY
)"
fi
[[ "${requested}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid Optuna trial count" >&2; exit 64; }
IFS=, read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:?multi-GPU allocation missing CUDA_VISIBLE_DEVICES}"
workers=${#devices[@]}
(( workers > requested )) && workers=${requested}
log_root="/results/provenance/multi-gpu-optuna-${SLURM_JOB_ID:-local}-$$"
mkdir -p "${log_root}"
pids=()
for ((rank=0; rank<workers; rank++)); do
    count=$((requested / workers + (rank < requested % workers ? 1 : 0)))
    child=(optimize "${project}" --profile "${profile}" --trials "${count}" --no-progress)
    CUDA_VISIBLE_DEVICES="${devices[rank]}" LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 \
      /opt/dvcs/venv/bin/python -m extract_dvcs_cff.cli.user \
      --bridge "${bridge}" "${child[@]}" \
      >"${log_root}/rank-${rank}.stdout" 2>"${log_root}/rank-${rank}.stderr" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
printf '%s\n' "${status}" > "${log_root}/exit-code.txt"
(( status == 0 )) || { echo "multi-GPU Optuna worker failed; see ${log_root}" >&2; exit "${status}"; }
cat "${log_root}/rank-0.stdout"
