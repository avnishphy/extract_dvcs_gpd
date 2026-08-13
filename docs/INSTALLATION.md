# Installation

## Requirements

Local Linux requires rootless Podman or Docker. JLab requires Apptainer with unprivileged/fakeroot builds only when no published image exists. Git, curl, and Python 3 are installer utilities; dependencies themselves live in the image or user cache.

CUDA with Docker uses the NVIDIA Container Toolkit `--gpus` interface. CUDA
with rootless Podman uses its NVIDIA CDI device (`nvidia.com/gpu=all`); configure
the NVIDIA Container Toolkit CDI specification before selecting `cuda`.

```bash
./install.sh --profile local --accelerator auto
./install.sh --profile local --accelerator cpu
./install.sh --profile local --accelerator cuda
./install.sh --profile jlab_ifarm --accelerator auto
```

`auto` selects CUDA only when the host probe and then container PyTorch can initialize it. Explicit `cuda` fails if unusable. The installer is transactional for images, data, and its environment file, and is safe to rerun. It refuses dirty or wrong-revision database checkouts.

Override user-owned storage with `DVCS_WORKSPACE`, `DVCS_RESULTS`, `DVCS_CACHE`, and `DVCS_DATABASE`. No sibling development checkout is referenced. A fully installed runtime needs no network for execution.

At this release, `images.lock.json` says `published=false`, so source build is expected. After publication, maintainers must write the real GHCR name and digest; the installer will then pull by digest.
