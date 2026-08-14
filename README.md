# extract-dvcs-gpd

`extract-dvcs-gpd` is a containerized, non-root distribution of a validated
DVCS/GPD synthetic-inference framework. It generates reusable, verified,
sharded exact-native PARTONS corpora, derives pseudodata realizations, trains
simulation-based neural posteriors, evaluates calibration
and exact predictions, compares with a conventional exact-bank posterior, and
produces observable, CFF, and GPD diagnostics.

The scientific boundary is strict: C++ PARTONS modules/services are the sole
forward-physics implementation. Python coordinates simulations, covariance,
nuisance sampling, inference, validation, and plots; it does not contain a
fallback GPD/CFF/observable model.

## Five-minute orientation

Local Linux:

```bash
git clone <repository-url> extract_dvcs_gpd
cd extract_dvcs_gpd
./install.sh --profile local --accelerator auto
./dvcs init first-study
./dvcs doctor first-study
./dvcs corpus-create first-study first-corpus --profile quick
./dvcs corpus-plan first-study first-corpus
./dvcs corpus-generate first-study first-corpus
./dvcs corpus-verify first-corpus --deep
./dvcs selection-create first-study first-corpus baseline --profile quick
./dvcs train first-study --profile quick \
  --corpus first-corpus --selection baseline
./dvcs evaluate first-study --profile quick
./dvcs compare first-study --profile quick
./dvcs plot first-study --profile quick
```

JLab ifarm:

```bash
./install.sh --profile jlab_ifarm --accelerator auto
cp -n jobs/jlab_ifarm/resources.env.example jobs/jlab_ifarm/resources.env
vi jobs/jlab_ifarm/resources.env
./dvcs init first-study
./dvcs corpus-create first-study first-corpus --profile validation
./dvcs corpus-plan first-study first-corpus
jobs/jlab_ifarm/submit_workflow.sh
```

`quick` is a real 82-dimensional workflow with 2,048 accepted native prior
vectors by default; it is not a seconds-long mock. Use it to establish the
pipeline, then assess whether a larger campaign supports your intended claim.

## What the framework infers

The production posterior has 82 coordinates:

- five double-distribution controls for each of H, E, Htilde, and Etilde;
- four independently controlled channels per GPD: u, d, s, and gluon;
- two standard-normal normalization nuisance parameters.

The native backend defines input GPDs at `Q0² = 1 GeV²`, evolves them with
APFEL++ to each datum Q², computes H/E/Htilde/Etilde CFFs, and evaluates an
ordered nonempty subset of six audited DVCS observables. Synthetic data use a dense declared covariance with
independent, local-correlated, phi-shape, and normalization contributions.

The neural posterior uses a permutation-invariant DeepSets context encoder and
an sbi conditional normalizing flow. Candidate ensemble members are selected
only with grouped internal validation. Replicates from one native parameter
vector never cross train/internal-validation/outer-test roles.

## End-to-end workflow

```text
editable experiment.json
  -> immutable sharded exact-PARTONS corpus
  -> immutable train/validation/outer-test selection
  -> deterministic correlated pseudodata realization
  -> NPE ensemble training
  -> outer-test coverage and exact reevaluation
  -> conventional exact-bank comparison
  -> saved-result plots
  -> optional output-blind native-model holdout
  -> optional quarantined real-observable diagnostic
```

The real database is never used as a training or likelihood source. New
projects may borrow its kinematics and metadata from a pinned read-only
catalog. `compare-real` is an explicitly non-fitting frozen-posterior overlay.

## Deployment model

- Docker or rootless Podman is supported for ordinary Linux.
- Apptainer is supported for JLab ifarm/farm without Docker daemon access on
  compute nodes.
- CPU and CUDA-capable images share one canonical multi-stage definition.
- PARTONS always runs on CPU in isolated processes.
- CUDA accelerates eligible PyTorch/sbi work only.
- CPU workers use full process affinity interactively and are bounded by
  `SLURM_CPUS_PER_TASK` only inside a Slurm job.
- Multi-GPU training uses one NCCL rank per allocated visible GPU and shards
  independent ensemble seeds deterministically.
- Workspaces, results, caches, and database data are persistent host mounts.

At this source-distribution state no image has been published, so installation
builds from pinned sources. Null image digests are never treated as verified.

## Documentation

Start at the [documentation map](docs/INDEX.md). The primary guides are:

- [Installation](docs/INSTALLATION.md)
- [User guide](docs/USER_GUIDE.md)
- [CLI reference](docs/CLI_REFERENCE.md)
- [Experiment JSON reference](docs/EXPERIMENT_JSON_REFERENCE.md)
- [Corpus and data selection](docs/CORPUS_AND_DATA_SELECTION.md)
- [Workflow and physics](docs/WORKFLOW_AND_PHYSICS.md)
- [Results and interpretation](docs/RESULTS_AND_INTERPRETATION.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Native bridge](docs/NATIVE_BRIDGE.md)
- [Data contracts](docs/DATA_CONTRACTS.md)
- [JLab ifarm](docs/JLAB_IFARM.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Verification status

Verified on the packaging host:

- clean-clone static/install checks;
- relocatable native bridge build/install, capabilities, and self-test;
- retained DD, evolution, basis, conformal, and shadow physics tests;
- targeted Python regressions and CPU resource bounding;
- exact CPU generation plus workflow-level NPE training smoke;
- explicit CUDA fail-closed behavior;
- dependency/source/checksum and repository hygiene checks.

Docker/Podman, Apptainer, Slurm, usable CUDA hardware, and multi-GPU hardware
were unavailable on that host and remain explicitly unverified. See the
[latest verification record](provenance/verification-2026-08-14.json) and [acceptance
procedures](docs/ACCEPTANCE.md).

## Reproducibility and provenance

The distribution records exact upstream file hashes, dependency commits and
archives, image tags/digests, native fixture checksums, configuration/bridge
hashes, native cache records, seeds, checkpoints, resource use, and Slurm
metadata. See [Reproducibility](docs/REPRODUCIBILITY.md) and the
[`provenance/`](provenance/) directory.

## Licensing and publication boundary

The imported application/bridge source currently lacks a repository-level
license grant. Public source and image publication is therefore blocked until
the copyright holders provide one. `gpddatabase` also has unresolved
additional restrictive wording and is installed separately as a clean,
read-only checkout rather than embedded. See [Dependencies](docs/DEPENDENCIES.md)
and [Known limitations](docs/KNOWN_LIMITATIONS.md).

## Upstream synchronization

Runtime updates are imported through an approved-file manifest and
conflict-detecting tool. Development plans, reports, workspaces, generated
results, caches, credentials, and unrelated history are excluded. See
[Updating from extract_dvcs_cff](docs/UPDATING_FROM_EXTRACT_DVCS_CFF.md).
