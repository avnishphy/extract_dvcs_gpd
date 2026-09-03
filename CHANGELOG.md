# Changelog

## 0.3.0 - 2026-08-31

- Split reusable analysis data into versioned master-corpus, selection,
  architecture-neutral pseudodata-realization, model-view, and result-bundle
  contracts. Historical schema-1 corpora remain valid for the DD baseline;
  neural-GPD mode requires complete checksummed canonical GPD-truth shards.
- Added the `dd_deepsets_maf` and `neural_gpd_deepsets_maf` registry, offline
  latent-function autoencoder/prior contracts, train-only normalization,
  truth-leakage guards, typed physics constraints, and common comparison
  metrics. The neural branch is explicitly a DD-induced closure test when its
  source corpus uses a DD generator.
- Made MAF the sole density estimator. Retired spline-only `num_bins` fields
  are removed from canonical configurations and explicitly warned/migrated in
  historical schema-9 projects; NSF/spline configurations fail closed.
- Separated saved-artifact `evaluate` from explicit native
  `exact-reevaluate`; selection, realization, model-view creation, training,
  comparison, plotting, and literature diagnostics no longer resolve or
  launch the native bridge.
- Added stage-specific SWIF2 placement: 16-CPU corpus workers use farm25 by
  default, farm19 requires an explicit fallback profile, and farm23 is
  excluded. Merge/lightweight stages do not inherit farm25 placement.
- Added dependency/container consistency validation, literature benchmark and
  coverage registries, and the compiled one-time 2026-08-31 progress deck.
- Recorded the current native blocker for external neural-GPD exact
  reevaluation: the installed bridge lacks a validated loader from a frozen
  external function artifact into its PARTONS/APFEL++ table path.

## 0.2.0 - 2026-08-14

- Ported the reusable PARTONS corpus workflow from upstream commit `255b8f7`:
  atomic compressed parameter/CFF/observable shards, consolidated native
  evidence, deep verification, safe export/import, append-only admitted
  observables, immutable group selections, and deterministic neural
  realizations.
- Made the corpus-backed create/plan/generate/verify/select/train sequence the
  sole public workflow, retired the legacy project-local generator console,
  and bound checkpoints to verified materialized input arrays.
- Adapted corpus/export storage to the packaged writable workspace and updated
  JLab Slurm templates with an explicit selection job and corpus/selection
  resource variables.
- Expanded the public documentation into complete onboarding, CLI, schema,
  architecture, native-interface, data-contract, physics, result,
  installation, container, JLab, resource, dependency, reproducibility,
  security, troubleshooting, limitation, reference, and maintenance guides.
- Added documentation regression checks for required coverage, local links,
  and every generated schema-8 leaf.
- Corrected the JLab workflow DAG to run the conventional comparison before
  holdout and plotting.
- Updated PyTorch from 2.11.0 to security-fixed 2.12.1, constrained the full
  non-platform Python graph, pinned packaging tools and GitHub Actions, and
  added a dated dependency/security audit plus high-severity image scanning.
- Made installer dry runs non-mutating, added usable-engine preflight,
  download retries, full LHAPDF set checks, atomic database checkout, clear
  progress messages, and automatic native/Python post-install self-tests.

## 0.1.0 - 2026-08-12

- Imported the approved runtime surface from clean `extract_dvcs_cff` commit `7d690f6` with a content manifest; local uncommitted upstream content is excluded.
- Added relocatable native CMake discovery and `$ORIGIN` runtime linking.
- Added locked CPU/CUDA OCI source builds, an Apptainer definition, rootless launchers, persistent mounts, and fail-closed accelerator/resource resolution.
- Added checksum-verified LHAPDF data and clean pinned `gpddatabase` checkout handling.
- Added JLab CPU/GPU Slurm templates, dependency submission, provenance capture, CI, SBOM hooks, acceptance tests, and public user documentation.
