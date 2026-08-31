#!/usr/bin/env bash
set -euo pipefail
prefix="${1:-/opt/dvcs}"
source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
build_root="$(mktemp -d /tmp/extract-dvcs-gpd-build.XXXXXX)"
trap 'rm -rf -- "${build_root}"' EXIT
jobs="$(nproc)"
curl_download() {
    curl --fail --location --retry 10 --retry-all-errors \
        --connect-timeout 20 --max-time 1800 "$@"
}
locked_archive() {
    local cached="$1" hash="$2" output="$3"; shift 3
    if [[ -f "${cached}" ]] && echo "${hash}  ${cached}" | sha256sum -c - >/dev/null 2>&1; then
        cp "${cached}" "${output}"
        return
    fi
    local url
    for url in "$@"; do
        if curl_download "${url}" -o "${output}.partial" && \
           echo "${hash}  ${output}.partial" | sha256sum -c - >/dev/null 2>&1; then
            mv "${output}.partial" "${output}"
            return
        fi
        rm -f "${output}.partial"
    done
    echo "all locked download mirrors failed for ${output}" >&2
    exit 69
}
cd "${build_root}"
locked_archive /tmp/dvcs-wheelhouse/gsl-2.8.tar.gz \
    6a99eeed15632c6354895b1dd542ed5a855c0f15d9ad1326c6fe2b2c9e423190 gsl.tgz \
    https://ftpmirror.gnu.org/gsl/gsl-2.8.tar.gz \
    https://mirrors.kernel.org/gnu/gsl/gsl-2.8.tar.gz \
    https://ftp.gnu.org/gnu/gsl/gsl-2.8.tar.gz
echo '6a99eeed15632c6354895b1dd542ed5a855c0f15d9ad1326c6fe2b2c9e423190  gsl.tgz' | sha256sum -c -
tar -xzf gsl.tgz; cd gsl-2.8; ./configure --prefix="${prefix}"; make -j"${jobs}"; make install; cd ..
locked_archive /tmp/dvcs-wheelhouse/LHAPDF-6.5.6.tar.gz \
    6b8b7e38dc26a977a24f5a321215b7054c14a4469d04134d70cb93a860eeeea7 lhapdf.tgz \
    'https://lhapdf.hepforge.org/downloads/?f=LHAPDF-6.5.6.tar.gz'
echo '6b8b7e38dc26a977a24f5a321215b7054c14a4469d04134d70cb93a860eeeea7  lhapdf.tgz' | sha256sum -c -
tar -xzf lhapdf.tgz; cd LHAPDF-6.5.6
# LHAPDF 6.5.6 advertises --incdir but only accepts --includedir. PARTONS uses
# the advertised spelling and otherwise captures the help text as an include path.
sed -i 's/--includedir)/--incdir|--includedir)/' bin/lhapdf-config.in
./configure --prefix="${prefix}" --disable-python; make -j"${jobs}"; make install; cd ..
export PATH="${prefix}/bin:${PATH}" LD_LIBRARY_PATH="${prefix}/lib"
clone_build() {
    name="$1" url="$2" commit="$3"; shift 3
    git clone --filter=blob:none "${url}" "${name}"
    git -C "${name}" checkout --detach "${commit}"
    cmake -S "${name}" -B "${name}-build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${prefix}" "$@"
    cmake --build "${name}-build" -j"${jobs}"; cmake --install "${name}-build"
}
clone_build elementary-utils https://github.com/3d-partons/elementary-utils.git 0133fc6e69872270027c37381a273893f7841b42
clone_build numa https://github.com/3d-partons/numa.git f184ee75f1d61b380e522d38ba48fbcbe9d78ae5 -DElementaryUtils_HINT="${prefix}"
clone_build apfelxx https://github.com/vbertone/apfelxx.git 27deaec493d95bad0686b3b1c91fbbc910c891ff
clone_build partons https://github.com/3d-partons/partons.git 1ad0b7d3bf62328f564c4ded06793e5feed00d4f -DElementaryUtils_HINT="${prefix}" '-DNumA++_HINT='"${prefix}" '-DApfel++_HINT='"${prefix}" -DLHAPDF_HINT="${prefix}"
git clone --filter=blob:none https://github.com/3d-partons/partons-example.git
git -C partons-example checkout --detach 7ca59c36634dce411643e0846b495fcf0690e545
install -D -m0644 partons-example/data/xmlSchema.xsd "${prefix}/share/extract-dvcs-gpd/xmlSchema.xsd"
cmake -S "${source_root}" -B application -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="${prefix}" -DCMAKE_PREFIX_PATH="${prefix}"
cmake --build application -j"${jobs}"; cmake --install application
python3 -m venv "${prefix}/venv"
pip_install() {
    local attempt
    for attempt in 1 2 3; do
        "${prefix}/venv/bin/pip" install --retries 10 --timeout 120 "$@" && return 0
        echo "pip install attempt ${attempt}/3 failed" >&2
    done
    return 1
}
pip_install --upgrade pip==26.2.1 setuptools==81.0.0 wheel==0.48.0
torch_index="${DVCS_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
constraints="${source_root}/requirements/runtime-constraints.txt"
shopt -s nullglob
torch_wheels=(/tmp/dvcs-wheelhouse/torch-2.12.1*.whl)
shopt -u nullglob
if [[ "${#torch_wheels[@]}" -eq 1 ]]; then
    pip_install --constraint "${constraints}" "${torch_wheels[0]}"
else
    pip_install --index-url "${torch_index}" --constraint "${constraints}" torch==2.12.1
fi
pip_install --no-build-isolation --constraint "${constraints}" "${source_root}[neural]"
"${prefix}/venv/bin/pip" check
mkdir -p /cache/partons-logs
"${prefix}/bin/partons_bridge" --self-test
"${prefix}/venv/bin/python" -c 'import matplotlib, numpy, optuna, particle, scipy, sbi, torch, yaml, zuko'
