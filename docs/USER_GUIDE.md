# User guide

## What this guide assumes

This guide starts after cloning the repository. You do not need a local C++ or
Python environment: the installer builds or retrieves the complete runtime.
You need a rootless Docker/Podman engine on ordinary Linux, or Apptainer on
JLab. Read [Installation](INSTALLATION.md) if the runtime is not installed.

## The safe working model

Framework source and user studies are separate. You interact through the root
`./dvcs` launcher. Each study gets one isolated directory beneath the
configured workspace root, and only its `experiment.json` is meant to control
science and inference.

Do not edit `.engine/`, generated JSON, saved arrays, checkpoints, or caches.
If an experiment with results must change, create a new project. This makes
comparisons explicit and prevents stale artifacts from acquiring a new
scientific meaning.

## Install and initialize

```bash
./install.sh --profile local --accelerator auto
./dvcs init tutorial
./dvcs show tutorial
./dvcs doctor tutorial
```

`init` creates the project and, when the pinned database is installed,
borrows only kinematics and metadata from its read-only catalog. It never uses
measured values or uncertainties to generate pseudodata.

The project tree starts as:

```text
workspace/tutorial/
  README.md
  experiment.json
  real_data_mapping_readiness.json
  results/
```

The exact host path depends on `DVCS_WORKSPACE` selected during installation.
Use `./dvcs show tutorial` instead of assuming a path.

## Read the experiment before running

Open `experiment.json` in a text editor. Its principal sections are:

- `injected_truth`: one synthetic truth for 80 DD controls and two nuisances;
- `synthetic_dataset`: kinematic sites, provenance, and uncertainty model;
- `inference.profiles`: corpus, replica, sample, epoch, and ensemble sizes;
- `inference.neural_posterior`: DeepSets and flow controls;
- `inference.random_seeds`: independent stochastic streams;
- `inference.runtime`: accelerator, CPU threads, and native workers;
- `inference.hyperparameter_optimization`: optional Optuna study;
- `output_diagnostics`: GPD reference kinematics/x grid and interval level;
- `validation_gates`: frozen acceptance thresholds.

The [experiment reference](EXPERIMENT_JSON_REFERENCE.md) gives every field,
unit, range, default, and edit classification. Unknown or missing keys are
errors. JSON does not allow comments, NaN, or infinity.

### Changes appropriate for a first study

Reasonable experiment controls include the injected truth within support,
synthetic uncertainty magnitudes, fixed shadow stress settings, seeds, and
manual physical kinematics. Reasonable compute controls include profile counts,
sample counts, epochs, worker count, and CPU threads.

Changing architecture/search space is an advanced method change. Changing a
validation threshold is a policy change, not a way to repair a failed run.
Generated database provenance should not be edited piecemeal.

### Editing kinematics

Database-backed projects tie every point to catalog provenance. To replace
kinematics manually, replace the complete `kinematics_source` object with:

```json
{
    "mode": "manual"
}
```

Every point must stay within documented ranges and satisfy

```text
0 < Q2 / (2 * proton_mass * beam_energy * x_b) < 1
```

The public validator and native bridge independently check this condition.

## Choose a profile deliberately

`quick` and `validation` are named compute configurations, not scientific
certificates.

| Default | quick | validation |
|---|---:|---:|
| Accepted prior native vectors | 2,048 | 4,096 |
| Noise replicas per vector | 2 | 4 |
| Candidate ensemble members | 2 | 5 |
| Selected active members | 2 | 3 |
| Maximum epochs/member | 100 | 200 |
| Posterior samples | 1,024 | 16,384 |
| Coverage trials | 20 | 80 |

Even `quick` is a substantial native campaign. It is intended to exercise the
real 82-dimensional workflow, not finish in a few seconds. Begin with it, then
evaluate whether simulation density and coverage precision are adequate for
your claim. Never infer adequacy from the word `validation` alone.

## Run step by step

```bash
./dvcs generate tutorial --profile quick
./dvcs train tutorial --profile quick
./dvcs evaluate tutorial --profile quick
./dvcs compare tutorial --profile quick
./dvcs plot tutorial --profile quick
```

Stepwise execution is recommended because native generation/evaluation are
CPU-heavy while neural training benefits from GPUs. It also makes failures and
resumption easier to understand. `./dvcs run tutorial --profile quick` performs
the same main sequence, excluding Optuna, holdout, and real-data comparison.

### Generation

Generation performs an injected-truth native preflight, draws a deterministic
prior bank, and requests exact predictions in bounded waves. The progress bar
counts accepted vectors and separately displays attempts/rejections. Invalid
draws are recorded and deterministically replaced until the requested accepted
count is reached or the fail-fast policy aborts.

The full covariance and noise replicas are created only after native
predictions are available. Replicas from one native parameter vector stay in
one data split. Interrupting generation is safe: complete content-addressed
native calls are reusable on the next run.

### Training

Training builds an 82-coordinate conditional posterior with a shared point
encoder, permutation-invariant pooling, and conditional flow. Each candidate
member has an independent fixed seed. Early stopping and active-member
selection use grouped internal validation only.

On CPU, `cpu_threads` bounds Torch threads. On one GPU, training uses that
device. On multiple allocated GPUs, independent ensemble members are assigned
deterministically across one NCCL process per visible device. PARTONS is not
involved in training and never runs on a GPU.

### Evaluation

Evaluation has three recognizable phases:

1. neural coverage on grouped outer-test contexts;
2. exact native posterior reevaluation for observables and CFFs;
3. exact GPD diagnostics over the requested x grid.

The exact phases return to CPU PARTONS workers even when posterior sampling
uses a GPU. Inspect the invalid fraction and each gate, not only command exit.

### Conventional comparison

The conventional comparison importance-weights the exact prior bank. Its
effective sample size must be adequate before Wasserstein or width comparisons
are meaningful. A low ESS is a limitation of that comparison, not proof that
the neural posterior is wrong.

### Plotting

Plotting reads saved compatible artifacts and invokes no native or neural
computation. PNG files are convenient views; JSON metrics and arrays are
authoritative. `plot_manifest.json` hashes every input and output.

## Understand the output tree

Start at:

```text
workspace/tutorial/results/quick/summary.md
workspace/tutorial/results/quick/summary.json
```

Then use:

```text
generated/                    native corpus, pseudodata, invalid map
training/training_summary.json
evaluation/evaluation_metrics.json
comparison/comparison_metrics.json
plots/plot_manifest.json
plots/*.png
```

Read [Data contracts](DATA_CONTRACTS.md) for artifact semantics and [Results
and interpretation](RESULTS_AND_INTERPRETATION.md) before drawing physics
conclusions.

## Main plots

The standard plot set includes:

- training/internal-validation NLL history;
- grouped train/internal-validation/outer-test split visualization;
- catalog versus pseudodata kinematic coverage;
- nuisance and all H/E/Htilde/Etilde DD parameter marginals;
- posterior correlations and coverage calibration;
- predictive distributions and pulls for all six observables;
- real and imaginary H/E/Htilde/Etilde CFFs;
- combined and u/d/s/gluon-separated exact GPD bands.

PCA in the split plot is display-only; saved indices define membership.
Posterior peaks need not equal injected truth for weakly identifiable
directions. Repeated coverage and predictive calibration matter more than the
location of one mode.

## Optional hyperparameter optimization

```bash
./dvcs optimize tutorial --profile quick --trials 12
```

Optuna writes a resumable study under `results/quick/optimization/`. It does
not alter the experiment. To use selected values, initialize a new project,
edit its neural controls, and run a fresh untouched outer test. Do not select
architecture using outer-test, holdout, or real-data behavior.

## Native-model holdout

After synthetic evaluation and comparison pass:

```bash
./dvcs holdout tutorial --profile validation
```

This evaluates frozen NPE checkpoints against fresh GK11/GK16/GK19/VGG99
native truth. Outputs cannot flow backward into corpus generation, training,
or selection. A predictive mismatch is a scientific result and may be reported
as such without being an execution error.

## Read-only real-data diagnostic

```bash
./dvcs compare-real tutorial --profile validation --samples 64
```

This action is intentionally narrow. It overlays one audited observable with
frozen-posterior predictions. It does not define a likelihood, use full
covariance, retrain, reweight, or update the posterior. Do not call its output
a GPD extraction from data.

## Resumption and changes

Safe resumable units include complete native cache entries, complete
hash-compatible neural members, persistent Optuna trials, and presentation
plots. The workflow always rechecks inputs before reuse.

Create a new project when changing:

- any injected truth, kinematic, uncertainty, prior/support, or seed;
- profile size or split fraction after generation;
- neural architecture or training controls;
- bridge/image/dependency version;
- validation policy.

Do not copy `workspace_contract.json` into a changed project to bypass checks.

## A disciplined study procedure

1. Record the scientific question and intended claim.
2. Initialize a new project and preserve its starting experiment.
3. Change only controls justified by that question.
4. Run `doctor` in the real execution allocation.
5. Complete `quick` and inspect failures, invalid draws, and resource records.
6. Freeze architecture and gate policy before a larger campaign.
7. Run generation, training, evaluation, and comparison.
8. Interpret identifiability and calibration, not only point estimates.
9. Run named-model holdout only after closure passes.
10. Preserve configuration, hashes, manifests, image digest, and commit with
    any result publication.

## Where to go next

- [CLI reference](CLI_REFERENCE.md)
- [Experiment reference](EXPERIMENT_JSON_REFERENCE.md)
- [Workflow and physics](WORKFLOW_AND_PHYSICS.md)
- [Results and interpretation](RESULTS_AND_INTERPRETATION.md)
- [Resource management](RESOURCE_MANAGEMENT.md)
- [Troubleshooting](TROUBLESHOOTING.md)
