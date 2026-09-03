#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)"
install_env="${root}/.dvcs/install.env"
[[ -f "${install_env}" ]] || { echo "install framework first" >&2; exit 66; }
# shellcheck disable=SC1090
source "${install_env}"
project="${1:?project}" output_dir="${2:-${root}/.dvcs/swif2_inputs}"
[[ "${project}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || exit 64
experiment="${DVCS_WORKSPACE}/${project}/experiment.json"
gpd_truth="${DVCS_WORKSPACE}/${project}/gpd_truth.json"
[[ -f "${experiment}" && ! -L "${experiment}" ]] || {
    echo "project experiment does not exist: ${experiment}" >&2
    exit 66
}
[[ -f "${gpd_truth}" && ! -L "${gpd_truth}" ]] || {
    echo "project GPD truth request does not exist: ${gpd_truth}" >&2
    exit 66
}
[[ -f "${DVCS_IMAGE}" && ! -L "${DVCS_IMAGE}" ]] || exit 66
[[ -d "${DVCS_DATABASE}/gpddatabase" && ! -L "${DVCS_DATABASE}/gpddatabase" ]] || exit 66
if find "${DVCS_DATABASE}/gpddatabase" -type l -print -quit | grep -q .; then
    echo "database contains a symlink; refusing SWIF2 packaging" >&2
    exit 65
fi
mkdir -p "${output_dir}"
output_dir="$(cd -- "${output_dir}" && pwd -P)"
temporary="$(mktemp -d "${output_dir}/prepare-corpus.XXXXXX")"
trap 'rm -rf -- "${temporary}"' EXIT
tar -czf "${temporary}/gpddatabase.tar.gz" -C "${DVCS_DATABASE}" gpddatabase

publish() {
    local source="$1" label="$2" suffix="$3" digest destination
    digest="$(sha256sum "${source}" | awk '{print $1}')"
    destination="${output_dir}/${label}-${digest:0:16}${suffix}"
    if [[ -e "${destination}" ]]; then
        [[ -f "${destination}" && "$(sha256sum "${destination}" | awk '{print $1}')" == "${digest}" ]] || exit 65
    else
        mv "${source}" "${destination}"
    fi
    printf '%s\n' "${destination}"
}

database_asset="$(publish "${temporary}/gpddatabase.tar.gz" gpddatabase .tar.gz)"
experiment_digest="$(sha256sum "${experiment}" | awk '{print $1}')"
experiment_asset="${output_dir}/experiment-${project}-${experiment_digest:0:16}.json"
[[ -e "${experiment_asset}" ]] || cp "${experiment}" "${experiment_asset}"
gpd_truth_digest="$(sha256sum "${gpd_truth}" | awk '{print $1}')"
gpd_truth_asset="${output_dir}/gpd-truth-${project}-${gpd_truth_digest:0:16}.json"
[[ -e "${gpd_truth_asset}" ]] || cp "${gpd_truth}" "${gpd_truth_asset}"
manifest="${output_dir}/${project}-corpus-assets.env"
{
    printf 'SWIF_IMAGE=%q\n' "${DVCS_IMAGE}"
    printf 'SWIF_DATABASE_ARCHIVE=%q\n' "${database_asset}"
    printf 'SWIF_EXPERIMENT=%q\n' "${experiment_asset}"
    printf 'SWIF_GPD_TRUTH_REQUEST=%q\n' "${gpd_truth_asset}"
} > "${manifest}.partial"
mv "${manifest}.partial" "${manifest}"
sha256sum "${DVCS_IMAGE}" "${database_asset}" "${experiment_asset}" \
    "${gpd_truth_asset}" \
    > "${output_dir}/${project}-corpus-assets.sha256"
echo "${manifest}"
