#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' '## pwd'
pwd -P
printf '%s\n' '## git status'
git status --short --branch
printf '%s\n' '## branch'
git branch --show-current
printf '%s\n' '## HEAD'
git rev-parse HEAD
printf '%s\n' '## full diff stat'
git diff --stat
printf '%s\n' '## changed tracked paths'
git diff --name-status
printf '%s\n' '## untracked files'
git ls-files --others --exclude-standard
printf '%s\n' '## staged files'
git diff --cached --name-status
printf '%s\n' '## configured image identities'
if [[ -f provenance/images.lock.json ]]; then
  sed -n '1,240p' provenance/images.lock.json
else
  printf '%s\n' 'provenance/images.lock.json: missing'
fi
printf '%s\n' '## SIF candidates in repository'
find . -type f -name '*.sif' -print
printf '%s\n' '## relevant environment'
env | LC_ALL=C sort | grep -E '^(APPTAINER|CUDA|EXTRACT_DVCS|JLAB|LD_LIBRARY_PATH|MODULE|PARTONS|PATH|PYTHONPATH|SIF|SLURM|SWIF|TMPDIR|USER)=' || true
