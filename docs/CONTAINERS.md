# Containers

`containers/Dockerfile` is the canonical multi-stage definition. It builds GSL, LHAPDF, ElementaryUtils, NumA++, APFEL++, PARTONS, the bridge, and Python environment from verified sources. The final image runs as UID/GID 1000 by default; the rootless local wrapper maps the invoking UID/GID.

Variants differ only in the official PyTorch wheel index: CPU or CUDA 12.6. PARTONS never runs on a GPU. The tags reserved in `images.lock.json` are immutable release tags, but no registry digest exists until CI publication.

Publication is fail-closed behind the GitHub repository variable
`DVCS_PUBLICATION_APPROVED=true`; do not set it until the missing application
license grant is resolved. A release builds CPU, CUDA 12.6, and
`jlab_ifarm-<version>` tags and emits registry provenance plus an SBOM.

Mounts are `/workspace`, `/results`, `/cache`, and read-only `/database`. Native libraries resolve through `$ORIGIN/../lib`; PARTONS properties and schema sit beside the bridge. `LHAPDF_DATA_PATH` is explicit.

Build manually:

```bash
podman build -f containers/Dockerfile --target runtime \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  -t localhost/extract-dvcs-gpd:0.1.0-cpu .
```
