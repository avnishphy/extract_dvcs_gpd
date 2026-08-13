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
