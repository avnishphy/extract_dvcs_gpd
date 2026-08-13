# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=ubuntu:24.04@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90
FROM ${BASE_IMAGE} AS native-build
ARG DEBIAN_FRONTEND=noninteractive
ARG PREFIX=/opt/dvcs
ARG UBUNTU_SNAPSHOT=20260801T000000Z
ARG PARTONS_COMMIT=1ad0b7d3bf62328f564c4ded06793e5feed00d4f
ARG APFELXX_COMMIT=27deaec493d95bad0686b3b1c91fbbc910c891ff
ARG ELEMENTARY_COMMIT=0133fc6e69872270027c37381a273893f7841b42
ARG NUMA_COMMIT=f184ee75f1d61b380e522d38ba48fbcbe9d78ae5
ARG PARTONS_EXAMPLE_COMMIT=7ca59c36634dce411643e0846b495fcf0690e545
RUN echo "APT::Snapshot \"${UBUNTU_SNAPSHOT}\";" > /etc/apt/apt.conf.d/50snapshot && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git build-essential gfortran cmake pkg-config autoconf \
    automake libtool python3 python3-dev python3-venv libboost-json1.83-dev \
    libcln-dev libsfml-dev libeigen3-dev libxml2-dev && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /build
RUN curl -fL 'https://ftpmirror.gnu.org/gsl/gsl-2.8.tar.gz' -o gsl.tgz && \
    echo '6a99eeed15632c6354895b1dd542ed5a855c0f15d9ad1326c6fe2b2c9e423190  gsl.tgz' | sha256sum -c - && \
    tar -xzf gsl.tgz && cd gsl-2.8 && ./configure --prefix=${PREFIX} && \
    make -j"$(nproc)" && make install
RUN curl -fL 'https://lhapdf.hepforge.org/downloads/?f=LHAPDF-6.5.6.tar.gz' -o lhapdf.tgz && \
    echo '6b8b7e38dc26a977a24f5a321215b7054c14a4469d04134d70cb93a860eeeea7  lhapdf.tgz' | sha256sum -c - && \
    tar -xzf lhapdf.tgz && cd LHAPDF-6.5.6 && \
    ./configure --prefix=${PREFIX} --disable-python && make -j"$(nproc)" && make install
RUN git clone --filter=blob:none https://github.com/3d-partons/elementary-utils.git && \
    git -C elementary-utils checkout --detach ${ELEMENTARY_COMMIT} && \
    cmake -S elementary-utils -B elementary-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX} && \
    cmake --build elementary-build -j"$(nproc)" && cmake --install elementary-build
RUN git clone --filter=blob:none https://github.com/3d-partons/numa.git && \
    git -C numa checkout --detach ${NUMA_COMMIT} && \
    cmake -S numa -B numa-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX} \
      -DElementaryUtils_HINT=${PREFIX} && cmake --build numa-build -j"$(nproc)" && cmake --install numa-build
ENV PATH=/opt/dvcs/bin:${PATH}
ENV LD_LIBRARY_PATH=/opt/dvcs/lib
RUN git clone --filter=blob:none https://github.com/vbertone/apfelxx.git && \
    git -C apfelxx checkout --detach ${APFELXX_COMMIT} && \
    cmake -S apfelxx -B apfelxx-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX} && \
    cmake --build apfelxx-build -j"$(nproc)" && cmake --install apfelxx-build
RUN git clone --filter=blob:none https://github.com/3d-partons/partons.git && \
    git -C partons checkout --detach ${PARTONS_COMMIT} && \
    cmake -S partons -B partons-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${PREFIX} \
      -DElementaryUtils_HINT=${PREFIX} '-DNumA++_HINT='${PREFIX} '-DApfel++_HINT='${PREFIX} -DLHAPDF_HINT=${PREFIX} && \
    cmake --build partons-build -j"$(nproc)" && cmake --install partons-build
RUN git clone --filter=blob:none https://github.com/3d-partons/partons-example.git && \
    git -C partons-example checkout --detach ${PARTONS_EXAMPLE_COMMIT} && \
    install -D -m 0644 partons-example/data/xmlSchema.xsd ${PREFIX}/share/extract-dvcs-gpd/xmlSchema.xsd

FROM native-build AS application-build
COPY . /src/extract-dvcs-gpd
RUN cmake -S /src/extract-dvcs-gpd -B /build/application -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${PREFIX} -DCMAKE_PREFIX_PATH=${PREFIX} && \
    cmake --build /build/application -j"$(nproc)" && cmake --install /build/application && \
    python3 -m venv ${PREFIX}/venv && ${PREFIX}/venv/bin/pip install --no-cache-dir --upgrade \
      pip==26.2.1 setuptools==81.0.0 wheel==0.48.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN ${PREFIX}/venv/bin/pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} \
      --constraint /src/extract-dvcs-gpd/requirements/runtime-constraints.txt torch==2.12.1 && \
    ${PREFIX}/venv/bin/pip install --no-cache-dir --no-build-isolation \
      --constraint /src/extract-dvcs-gpd/requirements/runtime-constraints.txt '/src/extract-dvcs-gpd[neural]' && \
    ${PREFIX}/venv/bin/pip check && \
    ${PREFIX}/bin/partons_bridge --self-test && \
    ${PREFIX}/venv/bin/python -c 'import matplotlib, numpy, optuna, particle, scipy, sbi, torch, yaml, zuko'

FROM ubuntu:24.04@sha256:4fbb8e6a8395de5a7550b33509421a2bafbc0aab6c06ba2cef9ebffbc7092d90 AS runtime
ARG IMAGE_VERSION=0.1.0
ARG IMAGE_SOURCE
ARG IMAGE_REVISION=unknown
LABEL org.opencontainers.image.title="extract-dvcs-gpd" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.source="${IMAGE_SOURCE}" \
      org.opencontainers.image.revision="${IMAGE_REVISION}"
ARG DEBIAN_FRONTEND=noninteractive
ARG UBUNTU_SNAPSHOT=20260801T000000Z
RUN echo "APT::Snapshot \"${UBUNTU_SNAPSHOT}\";" > /etc/apt/apt.conf.d/50snapshot && \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libboost-json1.83.0 libcln6 libsfml-system2.6 \
    libxml2 libgfortran5 python3 && rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 1000 dvcs && useradd --uid 1000 --gid 1000 --create-home dvcs
COPY --from=application-build /opt/dvcs /opt/dvcs
COPY --from=application-build /src/extract-dvcs-gpd /opt/dvcs/app
RUN mkdir -p /workspace /results /cache /database /opt/dvcs/share/extract-dvcs-gpd && \
    dpkg-query -W > /opt/dvcs/share/extract-dvcs-gpd/dpkg-manifest.tsv && \
    chown -R dvcs:dvcs /workspace /results /cache /database /opt/dvcs/app/user
ENV PATH=/opt/dvcs/venv/bin:/opt/dvcs/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/dvcs/lib \
    DVCS_INFER_REPOSITORY_ROOT=/opt/dvcs/app \
    DVCS_WORKSPACE_ROOT=/workspace \
    DVCS_GPDDATABASE_ROOT=/database/gpddatabase \
    LHAPDF_DATA_PATH=/cache/lhapdf \
    MPLCONFIGDIR=/cache/matplotlib \
    PYTHONUNBUFFERED=1
USER dvcs
WORKDIR /workspace
ENTRYPOINT ["/opt/dvcs/app/scripts/container-entrypoint.sh"]
CMD ["--help"]
