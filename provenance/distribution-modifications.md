# Distribution-owned modifications to imported runtime

The synchronization state preserves upstream hashes. The following intentional packaging changes therefore appear as conflicts on the next import and require review:

- `src/extract_dvcs_cff/cli/user.py`: configurable workspace/database roots,
  recorded scheduler environment overrides, public `./dvcs` command hints, and
  distribution-root documentation references suitable for external workspace
  roots;
- `src/extract_dvcs_cff/inference/device.py` and `workflows/pseudodata.py`: fail-closed NCCL topology and deterministic ensemble-member sharding across allocated GPUs;
- `src/extract_dvcs_cff/native_parallel.py`: bound native workers by process affinity and `SLURM_CPUS_PER_TASK`;
- `src/extract_dvcs_cff/optimization/stage10.py`: enable a shared SQLite-backed Optuna study for concurrent per-GPU trial shards;
- `cpp/partons_bridge/src/main.cpp`: replace machine-local provenance paths with immutable dependency URIs;
- `configs/physics/stage11_native_validation_models_v1.json`: same provenance-only URI rewrite;
- `pyproject.toml`: distribution identity and exact direct Python pins (packaging-owned, not in the import manifest);
- `cpp/partons_bridge/config/partons.properties.in`: relocatable installed schema name (packaging-owned);
- `cpp/partons_bridge/CMakeLists.txt`: prefix-based dependency discovery, install rules, and relative runtime RPATHs (packaging-owned);
- `user/*.md`: replace development-only launcher/workspace/evidence references
  with the public container launcher and distribution documentation, and remove
  packaging-host-specific wording (packaging-owned).

No physics formula, PARTONS call, parameterization, observable convention, fixture value, or validation threshold is changed.
