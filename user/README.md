# DVCS posterior inference: user entry point

Everything you may safely edit lives under the workspace root configured by
the installer. Framework source, accepted evidence, and the read-only database
are outside each project. A project is content-bound: after changing an
experiment that already has results, create a new project name.

## Setup and first run

From the repository root:

```bash
./install.sh --profile local --accelerator auto
./dvcs init my-study
./dvcs doctor my-study
./dvcs corpus-create my-study my-corpus --profile quick
./dvcs corpus-plan my-study my-corpus
```

Inspect the generated `experiment.json` in the configured workspace and the
plan before the
expensive generation command. The complete required sequence is shown below.

Installation builds or retrieves the complete container runtime. `auto`
uses CUDA whenever PyTorch can initialize it and otherwise records a CPU
fallback. Use `--accelerator cuda` to require CUDA and fail rather than fall
back. PARTONS simulation/evaluation is CPU work; NPE training, scoring, and
sampling use the selected Torch device. `native_workers: "all_available"`
uses affinity-bounded independent PARTONS processes because installed PARTONS
thread safety is not established. The separate `cpu_threads` setting controls
Torch's CPU fallback only; it does not cap PARTONS workers. Run `./dvcs doctor
my-study` to see requested and resolved counts in the current allocation.

`quick` checks plumbing. It is not a calibrated 82-dimensional scientific
campaign. The larger `validation` profile is also only a starting default;
you must judge closure/SBC/coverage results rather than infer adequacy from
the profile name.

The frozen default encoder/flow is: three 128-wide point layers, two 128-wide
dataset layers, a 96-feature embedding, zuko MAF with 96 hidden features, six
transforms and eight bins, batch size 256, learning rate `5e-4`, and 10%
internal-validation fraction. `quick` uses 2,048 native parameter groups × 2
noise replicas and at most 100 epochs; `validation` uses 4,096 × 4 and at most
200 epochs. These are editable defaults in a new project's `experiment.json`,
not validated adequate sample sizes for an 82-dimensional posterior.

## What a project contains

`experiment.json` is readable, four-space-indented JSON. Public schema 8 has:

- `injected_truth.gpd_parameters`: independent H, E, Htilde, and Etilde
  shapes for u, d, s, and gluon;
- five controls per shape: `normalization`, `a`, `c`, `profile_b`, and
  `t_slope_GeV_minus2`;
- fixed shadow coefficients and channel amplitudes (editable simulator
  settings, not posterior coordinates in this release);
- an ordered, nonempty selection from the six audited native observables,
  with explicit units, normalization scales, and user labels;
- six multi-Q2 kinematic sites copied from the read-only database catalog,
  with source Q2 and beam energy retained;
- a declared full synthetic covariance and two explicit normalization
  nuisances;
- quick/validation simulation, epoch, sample, and seed settings;
- the fixed default DeepSets + conditional-flow architecture;
- optional Optuna search settings and immutable validation thresholds.

The exhaustive field-by-field reference is
[`docs/EXPERIMENT_JSON_REFERENCE.md`](../docs/EXPERIMENT_JSON_REFERENCE.md).
It identifies which values are experiment controls, compute controls,
generated provenance, frozen identifiers, or validation policy.

The posterior contains all 80 DD shape controls plus the two nuisance
parameters. Earlier 14-coordinate project schemas are rejected because they
cannot be silently assigned the new meaning.

`real_data_mapping_readiness.json` is generated beside the experiment. It is
a measurement-free catalog/mapping audit and explicitly keeps real fitting
disabled. `README.md` inside the project records the exact commands and links
back to these guides.

If you edit database-derived kinematics, replace `kinematics_source` with:

```json
{
    "mode": "manual"
}
```

Otherwise the provenance validator correctly rejects the mismatch.

## Required workflow

The reusable-corpus sequence is the only supported public workflow. Its exact
data ownership and leakage rules are specified in
[`docs/CORPUS_AND_DATA_SELECTION.md`](../docs/CORPUS_AND_DATA_SELECTION.md).
It computes exact clean PARTONS shards once, freezes group-disjoint
train/validation/outer-test membership separately, and adds nuisance/noise
replicas only when training tensors are materialized.

```bash
./dvcs corpus-create my-study my-corpus --profile quick
./dvcs corpus-plan my-study my-corpus
./dvcs corpus-generate my-study my-corpus
./dvcs corpus-verify my-corpus --deep
./dvcs selection-create my-study my-corpus baseline --profile quick
./dvcs train my-study --profile quick \
  --corpus my-corpus --selection baseline
./dvcs evaluate my-study --profile quick
./dvcs compare my-study --profile quick
./dvcs plot my-study --profile quick
```

`plot` reads compatible saved artifacts and writes figures plus
`plots/plot_manifest.json`; it does not rerun PARTONS, training, posterior
sampling, evaluation, or comparison. Scientific gate failures remain recorded
but do not prevent their diagnostic plots. `corpus-generate` computes missing
exact shards or admitted observable extensions. It never silently replaces an
existing corpus.

Live progress is on stderr; structured JSON is on stdout. Exact generation
uses independent two-parameter native batches and redraws immediately after
each completed batch. The postfix reports `attempted`, `accepted`, and
`rejected`; the main `N/target` counter is accepted simulations. Work is
submitted in bounded 64-parameter waves after an injected-truth preflight, so
a systematic native failure cannot enqueue the whole validation corpus. It
cannot advance inside one atomic PARTONS call.
Redirected runs automatically hide the bar, or use `--no-progress`.

`evaluate` has three visible phases. `Posterior coverage trials` uses the
trained neural ensemble. `Exact posterior reevaluation` sends the selected
posterior physics samples through PARTONS for all kinematics, and `Exact GPD
diagnostics` evaluates the truth plus valid posterior samples on the requested
x-grid. Both exact phases use independent two-parameter bridge batches across
the same affinity-bounded `native_workers` pool. Their postfix reports
completed, valid, invalid, and active-worker counts. GPU activity is expected
for coverage/posterior sampling; native reevaluation remains CPU work.

If an older project reports `unphysical_fixed_target_kinematics`, do not
resume it. Its catalog selection predates the required
$0<y=Q^2/(2M_pEx_B)<1$ filter. Create a new project name; the corrected
selection is part of project provenance and is intentionally not migrated
silently.

## Optuna is optional and separate

The default architecture is used unless you explicitly run:

```bash
./dvcs optimize my-study --profile quick --trials 50 \
  --corpus my-corpus --selection baseline
```

The profile must match the profile used by both `corpus-create` and
`selection-create`. For a scientific campaign, create a separate validation
profile corpus and selection rather than relabeling a quick corpus.

The resumable study is under
`results/validation/optimization/` (`study.sqlite3`, trial table, and
`best_parameters.json`). Optimization minimizes internal-validation negative
log posterior density and may not inspect the outer test or named models. It
does not silently rewrite `experiment.json`. Apply chosen settings to a new
project and run one untouched final outer test.

## Output-blind native-model validation

Only after `evaluate` and `compare` pass:

```bash
./dvcs holdout my-study --profile validation
```

This evaluates GK11, GK16, GK19, and VGG99 at six fresh catalog-derived
multi-Q2 sites frozen before their outputs were generated. None of their
values can enter DD generation, training, internal validation, Optuna, early
stopping, ensemble selection, or the normal 20% DD outer test. Each model gets
observable, real/imaginary CFF, and u/d/s/gluon GPD comparisons. A mismatch is
a method-improvement result, not an execution failure and not permission to
tune on the holdout.

Fresh native outputs are generated only when you run `holdout`; they are not
shipped as fixtures that could leak into model selection.

## Main plots

Under `results/PROFILE/plots/`:

- `training_validation_nll.png`: train/internal-validation NLL by epoch;
- `dd_train_validation_test_split.png`: grouped native-draw split (display
  PCA only; membership comes from saved index arrays);
- `real_catalog_vs_pseudodata_kinematics.png`: measurement-free catalog
  coverage and common pseudodata sites;
- `posterior_comparison.png`: the two nuisance marginals;
- `posterior_marginals_H.png`, `..._E.png`, `..._Htilde.png`,
  `..._Etilde.png`: every one of the 80 physics marginals;
- `posterior_predictive.png`: a separate panel for every native observable;
- `cff_real_imaginary.png`: separate real and imaginary CFF panels;
- `gpd_posterior_predictions.png` and `gpd_{u,d,s,gluon}_components.png`:
  exact-native GPD reevaluations for all fields/channels;
- `posterior_correlations.png`, `coverage_calibration.png`, and
  `posterior_predictive_pulls.png`: joint/statistical diagnostics.

PNG files are views. JSON metrics and NumPy arrays are authoritative.
For custom inspection, consume saved non-pickled arrays using the contracts in
the main [data-contract reference](../docs/DATA_CONTRACTS.md).

## Interpreting success

The neural posterior is the learned conditional density from sbi NPE. The
“conventional” posterior is a self-normalized exact-likelihood importance
calculation on the same DD prior bank. Agreement is useful only when the
conventional ESS passes. Neither method should be forced to peak at every
injected truth: weakly identifiable directions should remain broad. For an
identifiable, calibrated direction, repeated pseudodata coverage is more
important than one run whose mode happens to equal truth.

Do not proceed to named-model interpretation unless synthetic evaluation and
comparison pass. Do not proceed to real fitting until full 82-dimensional
closure, SBC, empirical coverage, predictive checks, exact reevaluation, and
the fresh native-model campaign are reviewed.

## Real data

`compare-real` remains an optional, quarantined 24-point ALU diagnostic from
a frozen synthetic posterior. It retains source Q2 and evolves the input GPD,
not the data. It is not a likelihood, posterior update, fit, or extraction.
No real-fit command is enabled in this release.

## More detail

- [WORKFLOW_AND_PHYSICS.md](WORKFLOW_AND_PHYSICS.md): equations and complete
  data-to-posterior pipeline.
- [PHYSICS_AND_RESULTS_GUIDE.md](PHYSICS_AND_RESULTS_GUIDE.md): parameter and
  plot interpretation, gates, and troubleshooting.
- [../docs/NATIVE_BRIDGE.md](../docs/NATIVE_BRIDGE.md): native operations,
  provenance, cache, and exact limitations.
