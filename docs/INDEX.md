# Documentation map

This documentation is organized so that a new user can learn the framework
without reading source code. Start with one of the learning paths below, then
use the reference pages while running a study.

## First-day learning path

1. [Project overview](../README.md) — purpose, scope, and first commands.
2. [Installation](INSTALLATION.md) — local Linux and JLab setup.
3. [User guide](USER_GUIDE.md) — create and run an isolated project.
4. JLab users: [ifarm guide](JLAB_IFARM.md), then [SWIF2
   workflows](JLAB_SWIF2.md).
5. [Workflow and physics](WORKFLOW_AND_PHYSICS.md) — what is computed and why.
6. [Results and interpretation](RESULTS_AND_INTERPRETATION.md) — how to read
   posterior, coverage, observable, CFF, and GPD outputs.

Do not read every page before the first run. The installation and user guides
provide the happy path; the command, experiment, and data-contract pages are
references for editing or interpreting a stage.

## Running and operating the software

- [CLI reference](CLI_REFERENCE.md) documents every `./dvcs` action, exit
  behavior, resumability, and command ordering.
- [Experiment JSON reference](EXPERIMENT_JSON_REFERENCE.md) documents every
  editable schema-9 field, support, unit, and edit policy.
- [Corpus and data selection](CORPUS_AND_DATA_SELECTION.md) documents reusable
  sharded PARTONS data, immutable group selections, deterministic
  realizations, observable extension, verification, and transfer.
- [Canonical GPD truth](CANONICAL_GPD_TRUTH.md) defines the mandatory
  per-parameter function values, explicit coordinate request, native route,
  storage layout, masks, and production checks.
- [Pseudodata kinematic sampling](PSEUDODATA_KINEMATIC_SAMPLING.md) documents
  the proposed physical, stratified maximin/Sobol design, its diagnostics,
  scaling, and the gates required before production use.
- [Resource management](RESOURCE_MANAGEMENT.md) explains CPU affinity,
  isolated PARTONS workers, CUDA selection, and multi-GPU behavior.
- [Containers](CONTAINERS.md) explains image construction, mounts, non-root
  execution, tags, digests, and offline operation.
- [JLab ifarm](JLAB_IFARM.md) explains Apptainer, interactive checks, SWIF2,
  storage, monitoring, and the retained direct-Slurm diagnostics.
- [JLab SWIF2](JLAB_SWIF2.md) explains the SWIF2 wrapper for managed JLab
  workflow dispatch, resource overrides, and monitoring.
- [Troubleshooting](TROUBLESHOOTING.md) maps common failures to checks and
  recovery procedures.
- [Image update and recovery runbook](IMAGE_UPDATE_RUNBOOK.md) defines when an
  image rebuild is required, how to preserve the known-good SIF, how to monitor
  the two Apptainer test phases, and how to recover without restarting blindly.

## Understanding the implementation

- [Architecture](ARCHITECTURE.md) describes components, trust boundaries,
  control flow, repository layout, and failure policy.
- [Artifact architecture and 0.3 migration](ARTIFACT_ARCHITECTURE_0_3.md)
  defines the five immutable layers, identity sensitivity, and native stages.
- [Native bridge](NATIVE_BRIDGE.md) describes the C++/PARTONS protocol,
  supported operations, caching, relocation, and provenance.
- [Data contracts](DATA_CONTRACTS.md) describes project, generated corpus,
  training, evaluation, comparison, holdout, plotting, and cache artifacts.
- [Dependencies](DEPENDENCIES.md) explains every native and Python dependency,
  its role, pinning, and licensing status.
- [Reproducibility](REPRODUCIBILITY.md) explains hashes, seeds, deterministic
  behavior, manifests, and what reproducibility does and does not guarantee.
- [References](REFERENCES.md) lists the primary theory, native-software, and
  simulation-based-inference literature relevant to the implementation.
- [Literature benchmarks](LITERATURE_BENCHMARKS.md) records reproducible
  figure-family status, convention gates, and corpus coverage requirements.
- [Literature benchmark audit](literature/BENCHMARK_AUDIT.md) records the
  source-by-source comparison contract and unresolved protocol details.
- [Uncertainty contracts](UNCERTAINTY_CONTRACTS.md) defines the independent
  realization, covariance, exposure, nuisance, noise, jitter, and output axes.
- [Dated progress presentation](presentations/current_progress_2026-08-31/README.md)
  is the verified one-time 2026-08-31 Beamer snapshot.
- [Architecture decision records](adr/0001-partons-sole-physics-backend.md)
  record the backend, artifact, model-scope, and snapshot decisions.

## Maintainer and assurance material

- [Security and data](SECURITY_AND_DATA.md) covers privilege, downloads,
  credentials, database boundaries, and publication controls.
- [Known limitations](KNOWN_LIMITATIONS.md) distinguishes verified,
  conditional, and unavailable capabilities.
- [Claims ledger](CLAIMS_LEDGER.md) maps potential scientific claims to the
  evidence still required.
- [Acceptance procedures](ACCEPTANCE.md) gives executable local and JLab
  verification sequences.
- [Development configuration and testing](DEVELOPMENT_AND_TESTING.md) maps
  implementation-owned configuration, source components, and available test
  entry points.
- [Image update and recovery runbook](IMAGE_UPDATE_RUNBOOK.md) is the required
  checklist for container, installer, dependency, and in-image source changes.
- [Updating from upstream](UPDATING_FROM_EXTRACT_DVCS_CFF.md) describes the
  allowlisted, conflict-detecting import process.
- [Dependency lock](../provenance/dependencies.lock.json), [image lock](../provenance/images.lock.json),
  [dated dependency/security audit](../provenance/dependency-audit-2026-08-13.json),
  [upstream import record](../provenance/upstream-import.json), and
  [latest verification record](../provenance/verification-2026-08-14.json) are the
  machine-readable authorities.

## Important vocabulary

| Term | Meaning in this repository |
|---|---|
| project or study | One isolated, user-owned directory containing an editable `experiment.json` and results. |
| profile | The `quick` or `validation` compute-size block selected for a run. |
| native | Computation performed by the installed C++ bridge and PARTONS stack. |
| neural | PyTorch/sbi conditional-density estimation and posterior sampling. |
| exact | Re-evaluated through the authoritative native bridge, not a Python surrogate. |
| injected truth | Parameters used to create the displayed synthetic pseudodataset. |
| reusable corpus | Immutable, verified, noise-free PARTONS parameter/GPD/CFF/observable shards. |
| selection | Immutable group indices assigning corpus rows to training, internal validation, and locked outer test. |
| realization | Deterministic nuisance/noise tensors materialized from a corpus and selection for one project. |
| conventional posterior | Exact-bank importance-sampling comparison, not a second truth model. |
| outer test | Grouped DD simulations withheld from training and model selection. |
| native holdout | Frozen-NPE predictive evaluation against named PARTONS models. |
| real comparison | Read-only diagnostic overlay; never a likelihood or posterior update. |
| workflow | A named SWIF2 dependency graph; it is orchestration, not a scientific project identity. |
| project-state archive | The immutable handoff returned by one SWIF2 analysis stage and consumed by the next. |
| reaping | SWIF2 transfer of declared outputs from disposable farm scratch to persistent storage. |
