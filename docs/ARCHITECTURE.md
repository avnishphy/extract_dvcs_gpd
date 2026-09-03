# Architecture

Release 0.3 uses five explicit immutable layers. The authoritative summary,
schema migration, and identity rules are in
[Artifact architecture and migration](ARTIFACT_ARCHITECTURE_0_3.md). The only
registered inference paths are DD + DeepSets + MAF and NNGPD-inspired neural
GPD + DeepSets + MAF; both retain PARTONS as their sole physics backend.

## Purpose and design constraints

The distribution turns a validated research implementation into a portable,
user-facing runtime. Its central constraint is that packaging must not create
a second physics implementation. PARTONS and the project-defined C++ modules
remain authoritative for GPDs, evolution, CFFs, and DVCS observables. Python
orchestrates simulation, uncertainty construction, inference, validation, and
presentation.

The other design constraints are:

- run without sibling development repositories;
- preserve exact source and runtime provenance;
- isolate every PARTONS worker process because native thread safety is not
  established;
- prevent experiment output from modifying framework source;
- reject incomplete, stale, or incompatible artifacts rather than guessing;
- make CPU, CUDA, container, and scheduler resource use explicit;
- keep database measurements outside training and likelihood construction.

## System overview

```text
host launcher ./dvcs
        |
        +-- Docker/Podman (local) or Apptainer (JLab)
                |
                +-- container entrypoint
                |     - resolve affinity and accelerator
                |     - constrain numerical-library threads
                |     - record runtime provenance
                |
                +-- Python user CLI
                |     - validate experiment.json
                |     - construct private engine configuration
                |     - coordinate workflow and artifacts
                |
                +-- isolated C++ bridge processes
                |     - validate JSON request
                |     - initialize PARTONS
                |     - evaluate native modules/services
                |     - return deterministic JSON + provenance
                |
                +-- PyTorch / sbi / Optuna
                      - DeepSets context encoding
                      - neural posterior estimation
                      - optional hyperparameter search
```

Persistent host directories are mounted at `/workspace`, `/results`, `/cache`,
and `/database`. Projects, reusable corpora, and portable corpus exports live
under `/workspace`; the database mount is read-only. Images and containers are
replaceable; user state is not stored in the container writable layer.

## Repository layout

| Path | Responsibility |
|---|---|
| `containers/` | Canonical OCI multi-stage build and Apptainer definition. |
| `cpp/partons_bridge/` | C++ bridge, project-defined native modules, schemas/configuration, and retained physics tests. |
| `src/extract_dvcs_cff/` | Public runtime orchestration, data handling, inference, optimization, and workflows. |
| `configs/` | Implementation-owned Stage 10 templates, native protocol schemas, retained validation contracts, and bridge-request examples; users edit only generated project `experiment.json`. |
| `user/` | Imported scientific user explanations retained from the validated runtime. |
| `docs/` | Distribution onboarding, operation, reference, and assurance documentation. |
| `jobs/jlab_ifarm/` | Authoritative SWIF2 workflow generation, node-local execution/reaping wrappers, performance collection, and retained direct-Slurm diagnostics. |
| `scripts/` | Container entrypoint, native source build, and multi-GPU Optuna launcher. |
| `tests/` | Distribution/static/native/quick/offline verification entry points. |
| `tools/` | Upstream synchronization, license collection, and SBOM helper. |
| `provenance/` | Dependency, image, import, fixture, modification, and verification records. |
| `licenses/` | Dependency license texts and unresolved application-license notice. |

Generated `.dvcs/`, `workspace/`, `results/`, `cache/`, `build/`, and local
JLab `resources.env` paths are ignored by Git.

## Launcher and container boundary

`install.sh` resolves the profile and image variant, installs user-owned data,
and atomically writes `.dvcs/install.env`. `dvcs` reads that state and starts
the runtime with the invoking UID/GID on local engines. Environment values set
by a job or user for workspace, results, cache, database, and accelerator take
precedence over installation defaults.

The launcher does not interpret scientific configuration. It is responsible
for mounts, device passthrough, CPU-set propagation, and selecting the image.
Inside the image, `scripts/container-entrypoint.sh` resolves resources and
dispatches the public command.

## User-project boundary

`./dvcs init NAME` creates exactly one directory beneath the configured
workspace root. The name cannot be absolute, contain traversal, or escape via
a symbolic link. A project contains:

```text
NAME/
  README.md
  experiment.json
  real_data_mapping_readiness.json
  .engine/                 generated; do not edit
  results/
    doctor.json
    quick/                 quick profile contract and artifacts
    validation/            validation profile contract and artifacts
```

The public file is `experiment.json`. Before each action, the CLI validates it
and translates it into `.engine/workflow.json` and `.engine/physics.json`.
Those files are derived implementation inputs, not another editable API.

Training/optimization materialization writes `workspace_contract.json`,
binding results to the scientific engine-configuration hash, profile, bridge
executable hash, synthetic-only flag, corpus, selection, and split policy.
Allocation-dependent accelerator, CPU-thread, and native-worker fields are
recorded as execution provenance but excluded from scientific identity, so
the CPU/GPU stage chain remains compatible without weakening physics checks.
It also hashes the generated array manifest. Downstream evaluation, comparison,
holdout, plotting, and real-data diagnostics refuse incompatible results. A
changed experiment therefore requires a new project, although an unchanged
native identity may reuse a verified corpus.

## Native physics boundary

The bridge is an executable instead of an in-process Python extension. Each
request crosses a JSON boundary and returns structured JSON. This provides:

- process isolation for native global state;
- deterministic request/response hashing;
- direct capture of stdout, stderr, and exit status;
- an auditable list of supported operations;
- no dependency on Python ABI details;
- a clean failure boundary for native exceptions or non-finite values.

The bridge initializes relative to its installed location, so
`partons.properties`, `logger.properties`, and `xmlSchema.xsd` remain
discoverable after relocation. Installed native libraries use
`$ORIGIN/../lib`; development build paths are not runtime dependencies.

See [Native bridge](NATIVE_BRIDGE.md) for the protocol and operations.

## Native scheduling and cache

PARTONS thread safety has not been proven. Parallelism therefore launches one
bridge subprocess per task and never shares a PARTONS object. Worker count is
bounded by:

1. Linux process affinity/cgroup visibility;
2. `SLURM_CPUS_PER_TASK`, only when `SLURM_JOB_ID` identifies an allocation;
3. the user `native_workers` or `DVCS_NATIVE_WORKERS` request;
4. the number of ready native tasks.

Requests are evaluated in deterministic two-parameter chunks and restored to
input order. During corpus construction, content-addressed request directories
are transient shard work state. After a shard succeeds, raw request/response,
executable-hash, stream, metadata, and exit evidence are consolidated into one
compressed hashed archive before the manifest advances. Partial files are
never accepted as shards.

## Simulation and uncertainty layer

The workflow draws 80 GPD-shape coordinates from declared uniform supports,
asks the bridge for the selected ordered subset of six audited observables at
every kinematic point, and stores noise-free parameter/GPD/CFF/observable
shards. Canonical GPD values are evaluated at the explicit project request's
common input-scale coordinates and aligned one-for-one with core groups.
Covariance, nuisances, noise, and DeepSets contexts are materialized later
from an immutable group selection. Experiment-constant encoding terms are
computed once; independently seeded groups fill fixed rows across CPU workers.
The physics prediction is never reimplemented in Python.

Five controls are inferred for every combination of four GPD types and four
channels: normalization, small-β exponent, large-β exponent, profile width,
and t slope. Two standard-normal normalization nuisances bring the posterior
dimension to 82. Native parameter groups, rather than noisy replicas, are the
unit of train/validation/test splitting, preventing leakage between replicas.

## Neural inference boundary

Each measurement token contains kinematics, observable identity, normalized
value, covariance-derived features, and nuisance responses. A shared point
network maps tokens to embeddings; permutation-invariant pooling produces one
dataset embedding; a conditional normalizing flow represents the posterior.

Schema-9 masked designs retain a fixed maximum tensor width. `point_mask=0`
removes a padded embedding after the shared point network, including its
biases; pooling divides by the fixed maximum count. Nested 12/30/60/96 designs
therefore encode absence without manufacturing zero-valued observations.

Candidate ensemble members use fixed seeds. Internal grouped-validation NLL
selects the configured number of active members. The untouched outer DD test,
named native holdouts, and real-data diagnostic are excluded from selection.

Realization publication uses a project/profile process lock. This matters for
multi-GPU Optuna, whose independent device workers share one generated tensor
set: one process writes the atomic realization and matching workers reuse its
fully published identity.

On multiple allocated GPUs, `torchrun` starts one NCCL rank per visible device
and deterministically shards independent ensemble seeds. Rank 0 aggregates
ordinary compatible checkpoints and performs the same selection. Optuna uses
one process per visible GPU and a shared persistent SQLite study.

## Validation boundaries

Validation is layered deliberately:

- training/internal-validation metrics diagnose optimization and select
  ensemble members;
- grouped outer-test NLL and empirical coverage test DD generalization;
- the conventional exact-bank posterior checks neural approximation;
- exact posterior reevaluation checks native validity and predictions;
- named GK11/GK16/GK19/VGG99 holdouts test predictive behavior outside the DD
  generator family after synthetic gates pass;
- real-data comparison is a presentation-only diagnostic and cannot alter the
  posterior.

These roles must not be interchanged. In particular, a passing holdout does
not validate a real-data fit, and real measurements cannot tune the frozen
posterior in this release.

## Provenance flow

Every result can be traced through:

```text
distribution Git commit
  -> upstream import commit and approved-file hashes
  -> dependency lock and image digest
  -> bridge executable/source/dependency capabilities
  -> experiment and engine configuration hashes
  -> native request/response/cache hashes
  -> seed lineage and checkpoint hashes
  -> result-array and plot-manifest hashes
  -> host/container/Slurm resource record
```

This chain supports audit and rerun comparison. It does not imply identical
floating-point bytes across different GPU models or library stacks unless the
entire runtime and hardware contract also match.

## Failure policy

The framework fails closed for:

- missing or incompatible native capabilities;
- unknown, missing, out-of-range, or non-finite configuration values;
- physically invalid kinematics;
- invalid covariance or point ordering;
- explicit CUDA requests without usable CUDA;
- backend exception, nonzero exit, malformed JSON, or non-finite result;
- bridge/configuration/profile mismatch with existing results;
- corpus/configuration/bridge mismatch, corrupt/incomplete shard manifests,
  overlapping selections, or changed materialized training arrays;
- dirty or wrong-revision database checkout during installation;
- absent prerequisites for a later workflow step;
- holdout execution before synthetic evaluation passes or comparison completes.

A completed but collapsed conventional reference stays a failed scientific
gate. Diagnostic-only holdout/plot execution may continue, but cannot enable a
robustness claim.

Invalid prior draws are recorded per corpus shard and replaced
deterministically. They are not converted to zero or silently imputed.
