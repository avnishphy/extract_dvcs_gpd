#!/usr/bin/env bash
set -u

if [[ $# -lt 4 || "$3" != "--" ]]; then
  echo "usage: $0 LOG_DIR COMMAND_NAME -- COMMAND [ARG ...]" >&2
  exit 64
fi

log_dir=$1
command_name=$2
shift 3

if [[ $# -eq 0 ]]; then
  echo "error: COMMAND is required" >&2
  exit 64
fi

mkdir -p "$log_dir"
stem="${log_dir%/}/${command_name}"
start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

{
  printf 'start_utc=%s\n' "$start_utc"
  printf 'host=%s\n' "$(hostname -f 2>/dev/null || hostname)"
  printf 'working_directory=%s\n' "$(pwd -P)"
  printf 'invocation='
  printf '%q ' "$@"
  printf '\n'
  printf 'user=%s\n' "$(id -un)"
  printf 'shell=%s\n' "${SHELL:-unknown}"
  printf 'git_head=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
  printf 'git_branch=%s\n' "$(git branch --show-current 2>/dev/null || printf unknown)"
  env | LC_ALL=C sort | grep -E '^(APPTAINER|CUDA|EXTRACT_DVCS|HOME|HOST|JLAB|LD_LIBRARY_PATH|MODULE|PARTONS|PATH|PYTHONPATH|SIF|SLURM|SWIF|TMPDIR|USER)=' || true
} >"${stem}.meta"

set +e
"$@" > >(tee "${stem}.stdout") 2> >(tee "${stem}.stderr" >&2)
status=$?
set -e

{
  printf 'end_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'exit_status=%d\n' "$status"
} >>"${stem}.meta"

exit "$status"
