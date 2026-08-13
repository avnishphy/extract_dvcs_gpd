# Installation

## Installation model

Installation creates or retrieves a complete container image, downloads
checksum-verified user data, and writes launcher state beneath `.dvcs/`. It
does not install host packages, invoke `sudo`, modify system configuration, or
depend on sibling development repositories.

Persistent state remains on the host:

```text
.dvcs/                    installer state and local image/SIF reference
workspace/                isolated user projects (default)
results/                  runtime and scheduler provenance (default)
cache/lhapdf/             required LHAPDF sets and application caches
.dvcs/database/gpddatabase/  clean pinned read-only checkout (default)
```

All defaults can be redirected to user-owned storage.

## Local Linux prerequisites

Required host tools:

- Git to clone the distribution and database;
- Python 3 for reading machine-readable installer locks;
- curl, tar, and SHA-256 tools for the LHAPDF set;
- rootless Podman or Docker for image build/run;
- network access during first source build and data acquisition.

The image build obtains compilers and libraries inside the image. A host C++
compiler, CMake, PARTONS, CUDA toolkit, Python virtual environment, and system
administrator access are not required.

For CUDA:

- Docker requires a working NVIDIA Container Toolkit `--gpus` interface;
- rootless Podman requires an NVIDIA CDI device named
  `nvidia.com/gpu=all`;
- the host NVIDIA driver must support the image's PyTorch CUDA 12.6 runtime;
- `nvidia-smi -L` must succeed for installer auto-detection.

## Local installation

Automatic accelerator policy:

```bash
./install.sh --profile local --accelerator auto
```

CPU-only image/runtime:

```bash
./install.sh --profile local --accelerator cpu
```

CUDA-capable image with required GPU at runtime:

```bash
./install.sh --profile local --accelerator cuda
```

The installer chooses Podman before Docker when both are on `PATH`, unless
`DVCS_ENGINE` is set. `auto` selects a CUDA build only if the host probe works;
the container entrypoint then independently asks PyTorch whether CUDA is
usable. Explicit `cuda` is an assertion and fails rather than falling back.

## JLab installation

On an ifarm login node:

```bash
./install.sh --profile jlab_ifarm --accelerator auto
```

This requires Apptainer, not a Docker daemon. Because login nodes need not
expose GPUs, `auto` builds or retrieves the CUDA-capable JLab SIF while leaving
runtime selection as `auto`. CPU jobs omit `--nv`; GPU jobs export
`DVCS_ACCELERATOR=cuda` and receive `--nv`.

When no published digest exists, the installer runs an unprivileged/fakeroot
Apptainer source build. Site policy must permit that operation. If it does not,
a maintainer must build/publish the approved OCI image elsewhere or provide an
approved SIF; the installer never escalates privilege itself.

## Select persistent storage

Set paths before installation:

```bash
export DVCS_WORKSPACE=/path/to/projects
export DVCS_RESULTS=/path/to/runtime-provenance
export DVCS_CACHE=/path/to/large-cache
export DVCS_DATABASE=/path/to/database-parent
./install.sh --profile local --accelerator cpu
```

The values are host paths. The launcher maps them to fixed container paths.
They may be overridden later per invocation/job; explicit runtime values take
precedence over `.dvcs/install.env`.

Choose storage with enough quota. Native caches and training artifacts grow
with profile size; project home directories are often unsuitable for large
JLab campaigns. Do not point two incompatible experiments at the same project
directory.

## Published image versus source build

`provenance/images.lock.json` controls this decision. The installer pulls only
when all are true:

- `published` is `true`;
- the selected variant has a non-null registry digest;
- `--source-build` was not requested.

The pull reference is `REGISTRY@sha256:...`, never a mutable tag. Otherwise it
builds from `containers/Dockerfile` or `containers/apptainer.def` using the
pinned dependency sources.

Force a source build even after images are published:

```bash
./install.sh --profile local --accelerator cpu --source-build
```

Dry-run validation performs argument/profile/engine resolution and reports the
planned actions without writing launcher state, creating directories, or
building/downloading anything:

```bash
./install.sh --profile local --accelerator cpu --dry-run
./install.sh --profile jlab_ifarm --accelerator cpu --dry-run
```

## Data installed outside the image

### LHAPDF

The installer downloads `MSTW2008nlo68cl.tar.gz`, verifies the locked archive
SHA-256, the `.info` and central-member hashes, and the 41-member inventory,
then atomically moves the extracted set into `DVCS_CACHE/lhapdf/`.
This set is required only for the VGG99 native-model holdout. The explicit
container `LHAPDF_DATA_PATH` points at that directory.

If an incomplete target directory already exists, installation stops instead
of merging files. Preserve it for diagnosis, then select a fresh cache path or
remove it yourself after confirming it is disposable.

### gpddatabase

The installer clones the public upstream repository at the exact locked
commit into `DVCS_DATABASE/gpddatabase`. Existing Git checkouts are accepted
only when HEAD matches and the worktree is clean. The runtime mounts the
database parent read-only.

Local dirty database content is never reset, cleaned, copied into an image, or
silently used as the pinned dependency. See [Dependencies](DEPENDENCIES.md)
for the unresolved redistribution/wording limitation.

## Transaction and interruption behavior

Images/data are created under `.partial` or temporary names and moved into
place only after success. `.dvcs/install.env` is likewise written through
`install.env.partial`. Re-running the same command is the supported recovery
after interruption.

The installer may reuse an already complete SIF or data set. It does not claim
that a partially created third-party engine cache is valid; the container
engine controls its own layer recovery.

## Automatic and manual verification

Installation now finishes only after the native bridge self-test, direct
imports of the scientific Python stack, and `pip check` succeed. CUDA local
installation additionally requires PyTorch to see a usable GPU through the
selected container engine. This catches a missing NVIDIA Container Toolkit or
Podman CDI configuration before the first physics run.

For an additional project-level check:

```bash
./dvcs init installation-check
./dvcs doctor installation-check
./dvcs partons-bridge --capabilities
./dvcs partons-bridge --self-test
```

For a local CPU workflow:

```bash
./dvcs generate installation-check --profile quick
./dvcs train installation-check --profile quick
```

The default quick generation is substantial. For release acceptance use the
full commands in [Acceptance procedures](ACCEPTANCE.md); do not modify the
profile and then describe it as standard quick acceptance.

## Offline execution

Once the image, LHAPDF set, and database checkout are complete, normal runtime
does not need network access. The container does not fetch dependencies during
a workflow action. Test the installed launcher with:

```bash
./tests/run.sh offline
```

For strict engine-level assurance, run with networking disabled using the
commands in [Containers](CONTAINERS.md) and confirm a project uses only bind
mounts and existing caches.

## Upgrading or changing variants

Rerun `install.sh` with the desired profile/accelerator after updating the
distribution. Existing user directories are not deleted. A new image or
bridge hash intentionally prevents old generated results from being consumed
as if they belonged to the new runtime. Preserve the prior image digest and
Git commit when retaining old results.

Do not hand-edit `.dvcs/install.env` except for diagnosis. Prefer environment
overrides or rerun installation so state remains auditable.
