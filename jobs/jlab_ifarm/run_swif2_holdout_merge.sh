#!/usr/bin/env bash
set -euo pipefail

main_runner="${1:?main runner}" project="${2:?project}" profile="${3:?profile}"
corpus="${4:?corpus}" selection="${5:?selection}" image="${6:?image}"
database="${7:?database}" base_state="${8:?base state}" output="${9:?output}"
summary="${10:?summary}" heartbeat="${11:?heartbeat}" collector="${12:?collector}"
performance="${13:?performance}" samples="${14:?samples}" interval="${15:?interval}"
design="${16:?design}" lhapdf="${17:?lhapdf}"; shift 17
partials=("$@")
[[ "${#partials[@]}" -eq 4 ]] || { echo "holdout merge requires four partials" >&2; exit 64; }

safe_archive() {
    python3 - "$1" <<'PY'
import sys, tarfile
from pathlib import PurePosixPath
with tarfile.open(sys.argv[1], "r:*") as stream:
    for member in stream.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise SystemExit(f"unsafe archive member: {member.name}")
PY
}
safe_archive "${base_state}"
assembly="${SWIF_JOB_WORK_DIR}/holdout-assembly"
mkdir -p "${assembly}"
tar -xf "${base_state}" -C "${assembly}"
for partial in "${partials[@]}"; do
    safe_archive "${partial}"
    tar -xf "${partial}" -C "${assembly}/${project}/results/${profile}"
done
python3 - "${assembly}/${project}/results/${profile}" "${base_state}" \
  "${design}" "${image}" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, base, design, image = map(Path, sys.argv[1:])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
expected_models = {"GPDGK11", "GPDGK16", "GPDGK19", "GPDVGG99"}
manifests = list(root.glob("holdouts/*/*/swif-partial.json"))
if len(manifests) != 4:
    raise SystemExit("holdout merge coverage is not exactly four partials")
records = [json.loads(path.read_text()) for path in manifests]
if {record["model"] for record in records} != expected_models:
    raise SystemExit("holdout merge model coverage mismatch")
if {record["design_sha256"] for record in records} != {sha(design)}:
    raise SystemExit("holdout merge design hash mismatch")
if {record["base_state_sha256"] for record in records} != {sha(base)}:
    raise SystemExit("holdout merge base-state hash mismatch")
if {record["image_sha256"] for record in records} != {sha(image)}:
    raise SystemExit("holdout merge image hash mismatch")
PY
combined="${SWIF_JOB_WORK_DIR}/combined-project.tar"
tar -cf "${combined}" -C "${assembly}" "${project}"
exec /bin/bash "${main_runner}" holdout "${project}" "${profile}" "${corpus}" \
  "${selection}" "${image}" "${database}" none "${combined}" "${output}" \
  "${summary}" cpu "${heartbeat}" "${collector}" "${performance}" \
  "${samples}" "${interval}" "${design}" "${lhapdf}" none
