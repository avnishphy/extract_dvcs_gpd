# Containers

## Canonical image definition

`containers/Dockerfile` is the canonical multi-stage OCI definition. It has:

1. `native-build`: compilers plus pinned GSL, LHAPDF, ElementaryUtils, NumA++,
   APFEL++, PARTONS, and PARTONS schema source builds;
2. `application-build`: the relocatable bridge and Python environment;
3. `runtime`: only runtime libraries, installed application, manifests, and a
   non-root account.

The base is Ubuntu 24.04 pinned by index digest. APT uses snapshot
`20260801T000000Z`; the final image saves `dpkg-manifest.tsv`. Release/source
archive checksums and Git commits are in `provenance/dependencies.lock.json`.

## CPU and CUDA variants

The native stack is identical. The selected official PyTorch wheel index
differs:

| Variant | Torch index | Intended use |
|---|---|---|
| CPU | `https://download.pytorch.org/whl/cpu` | CPU neural work and all native PARTONS work. |
| CUDA 12.6 | `https://download.pytorch.org/whl/cu126` | GPU neural work; native PARTONS remains CPU. |
| JLab | CUDA 12.6-capable runtime | One SIF supports CPU jobs without `--nv` and GPU jobs with `--nv`. |

CUDA images include a user-space CUDA runtime through wheels, not a host
driver. The host driver is injected by Docker/Podman/Apptainer and must be
compatible. `doctor` verifies PyTorch usability inside the real runtime.

## Manual builds

Rootless Podman CPU:

```bash
podman build \
  -f containers/Dockerfile \
  --target runtime \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  -t localhost/extract-dvcs-gpd:0.1.0-cpu .
```

Docker CUDA:

```bash
docker build \
  -f containers/Dockerfile \
  --target runtime \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 \
  -t localhost/extract-dvcs-gpd:0.1.0-cuda12.6 .
```

Apptainer source build:

```bash
apptainer build --fakeroot extract-dvcs-gpd.sif containers/apptainer.def
apptainer test extract-dvcs-gpd.sif
```

Normal users should prefer `install.sh`, which selects exact names and
transactional output paths.

## Runtime identity and privileges

The OCI image declares UID/GID 1000, but the local launcher supplies the
invoking host UID/GID so bind-mounted files remain user-owned. No workflow
requires root. The Podman launcher disables SELinux relabeling for these
explicit user mounts; administrators may need a site-specific labeling policy
instead on enforcing hosts.

Apptainer uses its normal invoking-user model. The database mount is read-only
in both runtimes.

## Mount contract

| Container path | Access | Purpose |
|---|---|---|
| `/workspace` | read/write | Projects, experiments, profile results. |
| `/results` | read/write | Runtime and scheduler provenance. |
| `/cache` | read/write | LHAPDF, Matplotlib, and reusable caches. |
| `/database` | read-only | Parent of the pinned `gpddatabase` checkout. |

Application source and dependencies live under `/opt/dvcs`. User computation
must not depend on the container writable layer.

## Entrypoint behavior

`scripts/container-entrypoint.sh`:

1. uses full affinity interactively, or intersects it with
   `SLURM_CPUS_PER_TASK` inside a Slurm job;
2. configures Torch CPU and native worker requests;
3. defaults OpenMP/BLAS/NumExpr/vecLib to one thread per process;
4. resolves `auto`, `cpu`, or fail-closed `cuda` through PyTorch;
5. writes rank-specific resource/runtime provenance;
6. launches `torchrun` for multi-GPU training or the GPU-sharded Optuna helper;
7. otherwise dispatches the public CLI or bridge.

Special commands are `partons-bridge` and `shell`. All other arguments are
passed to `dvcs-infer`.

## Environment contract

The image sets:

```text
PATH=/opt/dvcs/venv/bin:/opt/dvcs/bin:...
LD_LIBRARY_PATH=/opt/dvcs/lib
DVCS_INFER_REPOSITORY_ROOT=/opt/dvcs/app
DVCS_WORKSPACE_ROOT=/workspace
DVCS_GPDDATABASE_ROOT=/database/gpddatabase
LHAPDF_DATA_PATH=/cache/lhapdf
MPLCONFIGDIR=/cache/matplotlib
```

The launcher additionally passes accelerator and image-digest provenance.

## Image tags, digests, and publication

Reserved version tags are recorded in `provenance/images.lock.json`. Tags are
human labels; the registry digest is the immutable execution identity. The
installer will not pull a lock entry with a null digest.

The GitHub release workflow builds CPU, CUDA 12.6, and JLab tags, enables OCI
provenance/SBOM generation, and creates a registry attestation. Publication is
fail-closed behind repository variable `DVCS_PUBLICATION_APPROVED=true` because
the application license grant remains unresolved. Do not enable publication
merely to make installation faster.

After an approved publication, maintainers must replace the placeholder GHCR
owner, record actual digests, verify pulls, and commit the image-lock update.

## SBOM and manifests

OCI BuildKit requests an SBOM during publication. CPU CI also uses the Anchore
SBOM action to upload SPDX JSON. `tools/generate_sbom.sh IMAGE OUTPUT` supports
local Syft generation when available. Image provenance should include:

- OCI digest and labels;
- BuildKit attestation;
- SPDX/CycloneDX SBOM;
- `dpkg-manifest.tsv`;
- Python installed-package list/check result;
- native bridge capabilities;
- dependency lock and distribution commit.

## Offline verification examples

Docker:

```bash
docker run --rm --network none \
  -v "$DVCS_WORKSPACE:/workspace" \
  -v "$DVCS_RESULTS:/results" \
  -v "$DVCS_CACHE:/cache" \
  -v "$DVCS_DATABASE:/database:ro" \
  IMAGE partons-bridge --self-test
```

Apptainer:

```bash
apptainer exec --cleanenv --network none \
  --bind "$DVCS_WORKSPACE:/workspace" \
  --bind "$DVCS_RESULTS:/results" \
  --bind "$DVCS_CACHE:/cache" \
  --bind "$DVCS_DATABASE:/database:ro" \
  IMAGE.sif /opt/dvcs/bin/partons_bridge --self-test
```

For CUDA add the engine's GPU passthrough option and run `doctor` plus a real
tensor/training smoke. Device visibility alone is not sufficient acceptance.

## Common container failures

- A native library error usually means the wrong/noninstalled bridge was
  invoked or an image build is incomplete; inspect RUNPATH and capabilities.
- Missing PARTONS schema/config means the bridge was separated from its
  installed sibling files.
- A write error under a mount usually indicates host ownership/permissions,
  not a missing container package.
- CUDA unavailable inside the container usually indicates device passthrough,
  CDI/toolkit, driver, or wheel/runtime mismatch.
- A digest mismatch means the lock or registry object changed; do not fall
  back to an unverified tag.
