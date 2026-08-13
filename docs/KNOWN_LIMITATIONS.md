# Known limitations

- No Docker/Podman, Apptainer, Slurm, or usable GPU exists on the packaging machine. Those acceptance tests are unverified.
- No OCI image is published and no distribution-image digest exists. `images.lock.json` intentionally contains null digests.
- Multi-GPU ensemble training and per-GPU Optuna trial sharding are implemented, but no multi-GPU hardware was available to validate NCCL, concurrent study behavior, determinism, or statistical equivalence. Published templates conservatively request one GPU.
- JLab paths, account association, current node features, driver compatibility, and writable filesystem choice require an actual ifarm run.
- The application/bridge upstream has no repository-level license. Public source/image publication needs an explicit compatible license from its copyright holders.
- `gpddatabase` combines a GPL-3.0 license file with additional restrictive README wording, and individual dataset redistribution status has not been resolved. It is fetched separately and mounted read-only.
- Direct Python versions are pinned but wheel hashes are not yet locked, so builds are not claimed bit-for-bit reproducible.
- Real measurements are diagnostic/quarantined, not enabled for likelihood fitting.
