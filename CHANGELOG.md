# Changelog

## Unreleased

- Expanded the public documentation into complete onboarding, CLI, schema,
  architecture, native-interface, data-contract, physics, result,
  installation, container, JLab, resource, dependency, reproducibility,
  security, troubleshooting, limitation, reference, and maintenance guides.
- Added documentation regression checks for required coverage, local links,
  and every generated schema-7 leaf.
- Corrected the JLab workflow DAG to run the conventional comparison before
  holdout and plotting.

## 0.1.0 - 2026-08-12

- Imported the approved runtime surface from clean `extract_dvcs_cff` commit `7d690f6` with a content manifest; local uncommitted upstream content is excluded.
- Added relocatable native CMake discovery and `$ORIGIN` runtime linking.
- Added locked CPU/CUDA OCI source builds, an Apptainer definition, rootless launchers, persistent mounts, and fail-closed accelerator/resource resolution.
- Added checksum-verified LHAPDF data and clean pinned `gpddatabase` checkout handling.
- Added JLab CPU/GPU Slurm templates, dependency submission, provenance capture, CI, SBOM hooks, acceptance tests, and public user documentation.
