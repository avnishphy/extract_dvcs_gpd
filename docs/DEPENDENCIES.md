# Dependency inventory

## Authority and policy

The authoritative machine-readable inventory is
`provenance/dependencies.lock.json`. This page explains why each dependency is
present and how it is obtained. Exact values here must be updated together
with the lock and container definitions.

The dated upstream/security review is recorded in
`provenance/dependency-audit-2026-08-13.json`. A newer upstream release is not
automatically a required update: for a scientific environment, API and
numerical migrations remain pinned until their complete regression evidence is
recorded. Security fixes and build-compatibility fixes take priority.

Native source dependencies are fetched from verified upstream locations at
exact commits or release archives. No image build copies sibling development
binaries. Distribution/runtime tests may use local trees only as audit
evidence.

## Native physics stack

| Component | Locked identity | Role | Build/license |
|---|---|---|---|
| PARTONS | 5.0.0, commit `1ad0b7d3bf62328f564c4ded06793e5feed00d4f` | Module factories/services, GPD evolution integration, DVCS CFFs/process/observables, named models. | CMake C++17 shared Release; GPL-3.0-only. |
| APFEL++ | source reports 4.8.0, commit `27deaec493d95bad0686b3b1c91fbbc910c891ff` | LO fixed-three-flavor evolution tables. | CMake C++17/Fortran shared Release; GPL-3.0-only. |
| ElementaryUtils | 5.0.0, commit `0133fc6e69872270027c37381a273893f7841b42` | PARTONS utility/logging/factory support. | CMake C++11 shared Release; Apache-2.0. |
| NumA++ | 5.0.0, commit `f184ee75f1d61b380e522d38ba48fbcbe9d78ae5` | Numerical integration used by PARTONS/project modules. | CMake C++11 shared Release; GPL-3.0-only. |
| LHAPDF | 6.5.6 release archive with locked SHA-256 | PDF access required by VGG99 holdout. | Autotools shared, Python disabled; GPL-3.0-or-later. |
| GSL | 2.8 release archive with locked SHA-256 | Quadrature/special numerical support in bridge diagnostics. | Source build; GPL-3.0-or-later. |
| PARTONS example schema | commit `7ca59c36634dce411643e0846b495fcf0690e545` | `xmlSchema.xsd` used by installed PARTONS configuration. | One installed data file; GPL-3.0-only upstream. |

The bridge capability response verifies runtime-observed versions and records
source hashes for GK11/GK16/GK19/VGG99 implementation files.

## System ABI dependencies

Ubuntu snapshot packages provide:

- GCC/G++/GFortran and build tools;
- CMake (project minimum 3.20), pkg-config, Autotools, and libtool;
- Boost JSON 1.83 or newer for protocol parsing/serialization;
- CLN, SFML-system, Eigen3, and LibXml2 required by the native stack;
- Python 3.12 and development/venv support;
- runtime shared libraries in the final stage.

The base image is `ubuntu:24.04` at the recorded index digest. APT snapshot
`20260801T000000Z` fixes the archive view. Package versions are captured in
the image's `dpkg-manifest.tsv` rather than duplicated as incomplete manual
pins in prose.

## Python runtime

Direct versions are exact in `pyproject.toml`/the lock:

| Package | Version | Role |
|---|---:|---|
| NumPy | 1.26.4 | Arrays, sampling, deterministic numerical records. |
| PyYAML | 6.0.3 | Read-only database YAML parsing. |
| munch | 4.0.0 | Database object compatibility. |
| particle | 0.26.1 | Database particle metadata compatibility. |
| Matplotlib | 3.10.8 | Saved-result plots. |
| SciPy | 1.17.1 | Statistical/numerical diagnostics. |
| PyTorch | 2.12.1 | Neural computation and distributed runtime; first release after the affected range of GHSA-rrmf-rvhw-rf47. |
| sbi | 0.26.1 | Neural posterior estimation interface. |
| zuko | 1.6.0 | Conditional MAF implementation used by sbi. |
| Optuna | 4.5.0 | Optional persistent hyperparameter study. |
| tqdm | 4.70.0 | Interactive stderr progress. |

CPU wheels come from the official PyTorch CPU index; CUDA variants use the
official CUDA 12.6 index. The rest resolve from the normal Python index during
image build. Packaging tools are fixed at pip 26.2.1, setuptools 81.0.0, and
wheel 0.48.0. `requirements/runtime-constraints.txt` fixes the complete
non-platform transitive graph; PyTorch wheel metadata fixes its platform CUDA
libraries. `pip check`, direct imports, and the native self-test run while the
image is built and again after installation.

Package versions are deterministic, while wheel files are not hash-locked.
The promoted OCI digest, SBOM, and build attestation are therefore the final
immutable installation identity rather than a claim of bit-for-bit local
source-build reproduction.

The resolved PyTorch dependency requires setuptools below 82. The only
remaining Python advisory in the 13 August 2026 audit is a setuptools sdist
Unicode-normalization issue fixed in 83. It applies to sdist creation on macOS
APFS/HFS+, while supported builds/runs are Linux containers and physics
workflows never create sdists. This scoped exception is recorded in the dated
audit. CI still fails the image build on high-or-critical findings and emits an
SBOM; a future PyTorch release that permits a fixed setuptools must remove the
exception.

## LHAPDF set

Only `MSTW2008nlo68cl` is installed. It is required for `GPDVGG99` in the
native-model holdout, not for the production DD pseudodata generator.

The lock records:

- archive URL and SHA-256;
- `.info` hash;
- central member hash;
- 41-member count;
- redistribution-review note.

The set is downloaded to the host cache and mounted through
`LHAPDF_DATA_PATH`; it is not silently embedded in the image.

## gpddatabase

The database identity is version v1.1.3 at commit
`1e9e97fd417ce1d6d44fc73550bdbf32cca4eeb1`, fetched read-only over public
HTTPS. Its purposes in this release are:

- deterministic selection of measurement-free kinematics/provenance for new
  projects and fresh holdouts;
- generation of mapping-readiness metadata;
- a narrow, quarantined real-observable diagnostic.

It is not a source of training measurements or likelihood covariance.

The repository contains a GPL-3.0 license file while its README adds
“strictly for non-profit scientific use.” Dataset-level citation and
redistribution terms are not fully resolved. To avoid silently assuming
compatibility, the database is a separate clean pinned checkout and read-only
mount, never embedded in distribution images.

## Licensing summary

License texts gathered from audited dependency sources are under `licenses/`.
Their inclusion documents dependencies; it does not resolve compatibility for
the imported application itself.

The application/bridge source imported from `extract_dvcs_cff` has no
repository-level license grant. Without permission from its copyright holders,
public source/image publication is blocked. The GHCR workflow enforces an
explicit approval variable in addition to this documentation.

## Updating dependencies

For any dependency change:

1. verify the authoritative upstream and exact release/commit;
2. check license and redistribution changes;
3. update the lock, Dockerfile, Apptainer definition, and relevant CMake flags;
4. verify source/archive hashes and Git reachability;
5. rebuild CPU and CUDA/JLab variants from a clean cache;
6. run bridge capabilities/self-test and all retained native tests;
7. run Python and end-to-end workflow regression;
8. produce new SBOM/package manifests;
9. record new image digests and compatibility/version changes;
10. do not reuse old scientific result contracts under the changed bridge.
