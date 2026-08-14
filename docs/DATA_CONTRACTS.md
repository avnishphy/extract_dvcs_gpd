# Data and artifact contracts

## Contract principles

The framework treats files as versioned interfaces rather than incidental
outputs. Each persisted artifact has a declared role, ordering, provenance,
and compatibility relationship. The main rules are:

- JSON must be finite, schema/version labeled, and stably serialized;
- numerical arrays use non-pickled `.npy` files;
- arrays are accompanied by dtype, shape, and SHA-256 metadata;
- covariance is never stored without its point-order map;
- cached native output is bound to the exact request and bridge executable;
- later actions verify generation contracts before reading earlier results;
- synthetic, outer-test, holdout, and real-diagnostic roles remain separate.

## Public project contract

`experiment.json` is the only user-editable scientific contract. Schema 7
contains exactly these top-level blocks:

```text
schema_version
experiment
injected_truth
synthetic_dataset
inference
output_diagnostics
validation_gates
```

The validator rejects missing and unknown keys. See the [field-by-field
reference](EXPERIMENT_JSON_REFERENCE.md).

`real_data_mapping_readiness.json` is generated at initialization. It records
whether the database is installed, candidate observable mappings, units,
uncertainty/covariance-field presence, required audits, and
`real_data_fit_enabled: false`. It contains no measurement values used by the
synthetic workflow.

`.engine/workflow.json` and `.engine/physics.json` are regenerated private
translations. They allow the validated upstream workflow to run without
making its internal configuration another public edit surface.

## Workspace contract

Generation creates `results/PROFILE/workspace_contract.json` with:

| Field | Role |
|---|---|
| `schema_version` | Contract format. |
| `configuration` | Canonical private configuration path. |
| `configuration_sha256` | Exact configuration identity. |
| `profile` | `quick` or `validation`. |
| `bridge_sha256` | Exact native executable identity. |
| `real_data` | Must remain `false` for the synthetic corpus. |
| `synthetic_split_policy` | Grouped train/internal-validation/outer-test policy. |

Every downstream action compares the observed object with a freshly computed
expected object. A mismatch instructs the user to create a new project; the
runtime never combines old arrays with a changed experiment.

## Native cache contract

Each native cache entry stores the exact request, exact response, raw streams,
exit code, and metadata. Cache keys derive from canonical request content and
backend identity. Important metadata includes operation, evaluation count,
bridge hash, request hash, response hash, and explicit `surrogate_used: false`.

Only complete successful entries are reused. Force recomputation is available
for audit. Cache entries may be copied between machines only if their full
hash contract and dependency assumptions remain valid.

## Generated corpus

`generated/` contains the native corpus and displayed pseudodataset. Principal
artifacts include:

| Artifact | Meaning |
|---|---|
| `pseudodata.json` | Ordered synthetic measurement record, covariance/nuisance description, injected truth, native/backend provenance, and real-data false flags. |
| `invalid_simulation_map.json` | Every rejected parameter vector, native error, and deterministic replacement relation. |
| `generation_metrics.json` | Attempted/accepted/invalid counts, timings, cache use, and resource resolution. |
| `array_manifest.json` | Shape, dtype, and SHA-256 for persisted arrays. |
| `.npy` arrays | Parameter vectors, contexts, split identifiers, predictions, CFFs, covariance-derived values, and related numerical records. |

Point order is kinematic-major with the six frozen observables for each
kinematic point. A six-point experiment therefore has 36 tokens. A dense
covariance uses that exact flattened order.

The pseudodataset distinguishes:

- exact noiseless prediction;
- two nuisance responses and injected nuisance coordinates;
- correlated and independent covariance components;
- one sampled observed value vector;
- normalized and native values/units;
- generator/configuration/cache hashes.

Measured database values and uncertainties are not members of this contract.

## Split contract

The unit of splitting is one native 80-coordinate parameter vector. Every
noise/nuisance replicate for that vector stays in the same split. The groups
are assigned to:

- training;
- internal validation used for early stopping and ensemble selection;
- untouched outer DD test.

This prevents a network from seeing a noisy twin of an outer-test simulation.
The configuration must leave all three roles nonempty.

## Training contract

`training/` contains one subdirectory/checkpoint record per candidate seed and
`training_summary.json`. Member records include:

- seed and architecture;
- device and deterministic policy;
- train/internal-validation/test losses;
- epoch history and early stopping;
- checkpoint and source-data hashes;
- selection status;
- CUDA peak-memory metrics when applicable.

The aggregate summary declares all candidate seeds, active member seeds,
selection metric, and `outer_test_used_for_ensemble_selection: false`.
Distributed rank output is temporary orchestration; the final rank-0 aggregate
has the same scientific contract as single-device training.

## Optimization contract

`optimization/` contains a persistent SQLite Optuna study, trial metrics, and
summary. Trial objective is grouped internal-validation NLL. The record states
that the outer test and named/real datasets were unavailable to optimization.
Running more trials extends the study; it does not mutate the public
experiment or selected production architecture automatically.

## Evaluation contract

`evaluation/evaluation_metrics.json` combines:

- selected-member outer-test performance;
- coordinate-wise empirical coverage at declared levels;
- posterior-predictive point coverage;
- exact native posterior reevaluation and invalid fraction;
- exact CFF and GPD diagnostic summaries;
- separate parallel-execution records for posterior and GPD phases;
- validation-gate values, observations, and pass/fail decisions.

Parallel records contain chunk size, task count, requested/resolved workers,
affinity CPUs, cache keys, and process-isolation policy. Associated posterior,
CFF, prediction, GPD-x, truth-GPD, and posterior-GPD arrays are non-pickled and
hash inventoried.

## Comparison contract

`comparison/comparison_metrics.json` and saved samples compare the neural
ensemble with the conventional exact-bank posterior. The record contains:

- conventional effective sample size;
- neural and conventional sample counts;
- standardized sliced Wasserstein distance;
- marginal Wasserstein distances normalized by conventional SD;
- neural/conventional interval-width ratios;
- validation thresholds and pass/fail outcomes;
- seeds and hashes of source artifacts.

The conventional sample and neural sample arrays are separate. Neither should
be described as the injected truth.

## Holdout contract

`holdout/summary.json` and `holdout/MODEL/` artifacts are created only after
synthetic closure passes. The summary records:

- allowlisted model and fresh-kinematic manifests;
- frozen training/configuration/bridge hashes;
- predictive metrics and scientific mismatch status;
- preserved grouped outer-test role;
- protected before/after artifact manifests;
- `parameter_recovery_claimed: false` and `real_data_fit_enabled: false`.

Per-model directories contain native truth, frozen-NPE posterior samples,
exact-DD prediction/function intervals, invalid records, metrics, and plots.
A clean execution with predictive disagreement may legitimately report
`complete_with_scientific_mismatch`; execution success and scientific
agreement are distinct.

## Real-observable diagnostic contract

`real_data_comparison/real_observable_comparison.json` is neither a synthetic
dataset nor posterior. It records an allowlisted source row/path/hash,
protected database revision, source and evaluation Q², reported statistical
uncertainty, frozen posterior/bridge hashes, predictive quantiles, mapping
status, invalid records, and skipped-reason details.

It explicitly marks training use, likelihood use, hyperparameter use, and
posterior update as false. No fit statistic or covariance model is implied.

## Plot contract

`plots/plot_manifest.json` records:

- profile and configuration hash;
- hashes of every required generated/training/evaluation/comparison input;
- every PNG filename and SHA-256;
- saved evaluation/comparison gate states;
- `presentation_only: true`;
- `native_backend_called: false`;
- `neural_inference_called: false`.

Plots are reproducible views of saved results. The existence of a complete
manifest does not imply that scientific gates passed.

## Summary contract

`summary.json` is the machine-readable workflow synopsis. `summary.md` is the
human-readable starting point and links the major results. Consumers should
retain and parse the JSON for automation rather than scrape Markdown.

## Slurm/runtime provenance

JLab jobs write outside the project tree under
`DVCS_RESULTS/slurm/JOB_ID/STEP/`:

```text
environment.txt
job.txt
lscpu.txt
nvidia-smi.txt
distribution-commit.txt
images.lock.json
stdout.log
stderr.log
exit-code.txt
```

The container also writes `/results/provenance/runtime.json` and rank-specific
records with requested/resolved accelerator, visible GPU count, affinity,
interactive/Slurm execution context, Slurm CPU request, used CPU/math threads,
Torch/CUDA versions, host, image digest, and distribution commit.

## Safe artifact handling

- Never edit generated JSON or arrays to make a gate pass.
- Never publish an array without its manifest and experiment/bridge identity.
- Copy entire result-profile directories, not selected arrays with lost
  ordering metadata.
- Preserve failures and invalid maps; they are part of the scientific record.
- Treat checkpoints and pickle-capable third-party files as untrusted input
  when obtained externally.
- Use `plot` for regenerated graphics and a new project for changed science.
