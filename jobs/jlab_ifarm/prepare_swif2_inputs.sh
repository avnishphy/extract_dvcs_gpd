#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
install_env="${root}/.dvcs/install.env"
[[ -f "${install_env}" ]] || { echo "install framework first" >&2; exit 66; }
# shellcheck disable=SC1090
source "${install_env}"

project=
corpus_archive=
project_archive=
output_dir="${root}/.dvcs/swif2_inputs"
while (($#)); do
    case "$1" in
        --project) project="${2:?missing project}"; shift 2 ;;
        --corpus-archive) corpus_archive="${2:?missing corpus archive}"; shift 2 ;;
        --project-archive) project_archive="${2:?missing project archive}"; shift 2 ;;
        --output-dir) output_dir="${2:?missing output directory}"; shift 2 ;;
        -h|--help)
            echo "usage: $0 --project NAME --corpus-archive ARCHIVE [--project-archive ARCHIVE] [--output-dir DIR]"
            exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 64 ;;
    esac
done
[[ "${project}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 64
[[ -f "${corpus_archive}" && ! -L "${corpus_archive}" ]] || {
    echo "corpus archive does not exist: ${corpus_archive}" >&2
    exit 66
}
[[ -f "${DVCS_IMAGE}" && ! -L "${DVCS_IMAGE}" ]] || exit 66
[[ -d "${DVCS_DATABASE}/gpddatabase" && ! -L "${DVCS_DATABASE}/gpddatabase" ]] || exit 66
lhapdf_set="${DVCS_CACHE}/lhapdf/MSTW2008nlo68cl"
[[ -f "${lhapdf_set}/MSTW2008nlo68cl.info" && \
   -f "${lhapdf_set}/MSTW2008nlo68cl_0000.dat" && \
   -f "${lhapdf_set}/MSTW2008nlo68cl_0040.dat" ]] || {
    echo "verified MSTW2008nlo68cl installation is missing" >&2
    exit 66
}
mkdir -p "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"
temporary="$(mktemp -d "${output_dir}/prepare.XXXXXX")"
trap 'rm -rf -- "${temporary}"' EXIT

if [[ -n "${project_archive}" ]]; then
    [[ -f "${project_archive}" && ! -L "${project_archive}" ]] || {
        echo "project archive does not exist: ${project_archive}" >&2
        exit 66
    }
    project_archive="$(cd -- "$(dirname -- "${project_archive}")" && pwd -P)/$(basename -- "${project_archive}")"
    python3 - "${project_archive}" "${project}" <<'PY'
import pathlib, sys, tarfile

archive, project = pathlib.Path(sys.argv[1]), sys.argv[2]
required = f"{project}/experiment.json"
with tarfile.open(archive, "r:*") as stream:
    members = stream.getmembers()
    names = set()
    for member in members:
        path = pathlib.PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
            raise SystemExit(f"unsafe project archive member: {member.name}")
        names.add(path.as_posix().removeprefix("./"))
if required not in names:
    raise SystemExit(f"project archive lacks {required}")
PY
    project_asset="${project_archive}"
else
    project_dir="${DVCS_WORKSPACE}/${project}"
    [[ -d "${project_dir}" && ! -L "${project_dir}" ]] || {
        echo "project does not exist: ${project_dir}" >&2
        exit 66
    }
    if find "${project_dir}" -type l -print -quit | grep -q .; then
        echo "project contains a symlink; refusing SWIF2 packaging" >&2
        exit 65
    fi
    tar_excludes=(--exclude="${project}/results/*/.materialization.lock")
    while IFS= read -r generated; do
        profile_dir="$(dirname -- "${generated}")"
        if [[ ! -f "${generated}/array_manifest.json" || \
              ! -f "${profile_dir}/workspace_contract.json" ]]; then
            relative="${generated#${DVCS_WORKSPACE}/}"
            tar_excludes+=(--exclude="${relative}")
            echo "excluding incomplete materialization from staged project: ${relative}" >&2
        fi
    done < <(find "${project_dir}/results" -mindepth 2 -maxdepth 2 \
        -type d -name generated -print 2>/dev/null | sort)
    tar --sparse -cf "${temporary}/project.tar" "${tar_excludes[@]}" \
        -C "${DVCS_WORKSPACE}" "${project}"
fi
tar -czf "${temporary}/gpddatabase.tar.gz" -C "${DVCS_DATABASE}" gpddatabase
tar -czf "${temporary}/MSTW2008nlo68cl.tar.gz" \
    -C "${DVCS_CACHE}/lhapdf" MSTW2008nlo68cl

publish() {
    local source="$1" label="$2" suffix="$3" digest destination
    digest="$(sha256sum "${source}" | awk '{print $1}')"
    destination="${output_dir}/${label}-${digest:0:16}${suffix}"
    if [[ -e "${destination}" ]]; then
        [[ -f "${destination}" && "$(sha256sum "${destination}" | awk '{print $1}')" == "${digest}" ]] || {
            echo "content-addressed asset collision: ${destination}" >&2
            exit 65
        }
    else
        mv "${source}" "${destination}"
    fi
    printf '%s\n' "${destination}"
}

if [[ -z "${project_archive}" ]]; then
    project_asset="$(publish "${temporary}/project.tar" "project-${project}" .tar)"
fi
database_asset="$(publish "${temporary}/gpddatabase.tar.gz" gpddatabase .tar.gz)"
lhapdf_asset="$(publish "${temporary}/MSTW2008nlo68cl.tar.gz" MSTW2008nlo68cl .tar.gz)"
corpus_archive="$(cd -- "$(dirname -- "${corpus_archive}")" && pwd -P)/$(basename -- "${corpus_archive}")"
manifest="${output_dir}/${project}-assets.env"
{
    printf 'SWIF_IMAGE=%q\n' "${DVCS_IMAGE}"
    printf 'SWIF_DATABASE_ARCHIVE=%q\n' "${database_asset}"
    printf 'SWIF_LHAPDF_ARCHIVE=%q\n' "${lhapdf_asset}"
    printf 'SWIF_PROJECT_ARCHIVE=%q\n' "${project_asset}"
    printf 'SWIF_CORPUS_ARCHIVE=%q\n' "${corpus_archive}"
} > "${manifest}.partial"
mv "${manifest}.partial" "${manifest}"
sha256sum "${DVCS_IMAGE}" "${database_asset}" "${lhapdf_asset}" "${project_asset}" \
    "${corpus_archive}" > "${output_dir}/${project}-assets.sha256"
echo "${manifest}"
