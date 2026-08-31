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

Local Linux:

```bash
./install.sh --profile local --accelerator auto
./dvcs init tutorial
./dvcs show tutorial
./dvcs doctor tutorial
```

On JLab use `--profile jlab_ifarm`; initialize and inspect projects on ifarm,
but submit sustained generation, training, and evaluation through SWIF2. The
complete JLab sequence is in [JLab ifarm](JLAB_IFARM.md).

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

Project names are permanent identities. The directory name and
`experiment.name` must match exactly. If you copy another experiment into a new
project, change only its `experiment.name` before `show`; all other changes
must be scientifically intentional.

## Read the experiment before running

Open `experiment.json` in a text editor. Its principal sections are:

- `injected_truth`: one synthetic truth for 80 DD controls and two nuisances;
- `synthetic_dataset`: kinematic sites, observable selection, provenance, and
  uncertainty model;
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

## Create one reusable native corpus

```bash
./dvcs corpus-create tutorial tutorial-corpus --profile quick --shard-size 256
./dvcs corpus-plan tutorial tutorial-corpus
```

Stop and inspect both `experiment.json` and the printed plan. Then generate
and deeply verify the expensive, noise-free PARTONS data:

```bash
./dvcs corpus-generate tutorial tutorial-corpus
./dvcs corpus-verify tutorial-corpus --deep
```

The corpus stores immutable parameter vectors, CFFs, and each selected
observable in independent atomic shards. It does not store neural
architecture, train/test roles, nuisances, or measurement noise. Those can
change later without repeating compatible PARTONS calculations.

Create one immutable group assignment and run inference:

```bash
./dvcs selection-create tutorial tutorial-corpus baseline --profile quick
./dvcs train tutorial --profile quick \
  --corpus tutorial-corpus --selection baseline
./dvcs evaluate tutorial --profile quick
./dvcs compare tutorial --profile quick
./dvcs plot tutorial --profile quick
```

Stepwise execution is recommended because native generation/evaluation are
CPU-heavy while neural training benefits from GPUs. It also makes failures and
resumption easier to understand. The former project-local `generate` and
all-in-one `run` actions are intentionally no longer public: explicit corpus
and selection names prevent accidental recomputation and split drift.

On JLab, replace direct corpus generation with `farm-corpus-submit`, then give
the merged portable archive to `farm-submit`. Do not launch one oversized
multi-node PARTONS process. See [JLab SWIF2](JLAB_SWIF2.md).

### Corpus generation

Generation performs an injected-truth native preflight, draws a deterministic
prior bank per shard, and requests exact predictions in isolated native
batches. Invalid draws are recorded and deterministically replaced. Each
complete shard and its consolidated raw bridge evidence are verified, hashed,
and atomically published before the manifest advances. An interruption leaves
complete shards reusable and cannot convert a partial file into valid data.

The six audited observables are selectable in canonical order. A compatible
new project may append a missing observable; its exact CFF route must match
the stored CFFs, and no existing shard is modified. See
[Corpus and data selection](CORPUS_AND_DATA_SELECTION.md).

### Selection and realization

`selection-create` deterministically partitions native parameter groups into
training, internal validation, and locked outer test. Every noise replica of a
group keeps the same role. `materialize` builds full-covariance nuisances,
noise, and contexts deterministically from the clean corpus without PARTONS or
a GPU. On SWIF2 it uses a separate eight-CPU job before optimize/training.

### Training

Training builds an 82-coordinate conditional posterior with a shared point
encoder, permutation-invariant pooling, and conditional flow. Each candidate
member has an independent fixed seed. Early stopping and active-member
selection use grouped internal validation only.

On CPU, `cpu_threads` bounds Torch threads. On one GPU, training uses that
device. On multiple allocated GPUs, independent ensemble members are assigned
deterministically across one NCCL process per visible device. PARTONS is not
involved in training and never runs on a GPU.

Run `materialize` before training. Experiment-constant covariance and token
terms are computed once; independent parameter groups fill canonical array rows
across affinity-limited CPU workers. `Writing training arrays` publishes named
tensor files and hashes. SWIF2 reaps that CPU-stage state before requesting a
GPU for training, and reports both stages through separate heartbeats.

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
the neural posterior is wrong. Collapse produces a completed, failed
scientific record with null comparison metrics; plotting continues with
neural/evaluation views. Do not add jitter or lower the ESS gate to manufacture
a conventional width.

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
generated/                    corpus-backed realization, pseudodata, invalid map
training/training_summary.json
evaluation/evaluation_metrics.json
comparison/comparison_metrics.json
plots/plot_manifest.json
plots/*.png
```

Read [Data contracts](DATA_CONTRACTS.md) for artifact semantics and [Results
and interpretation](RESULTS_AND_INTERPRETATION.md) before drawing physics
conclusions.

For a SWIF2 campaign, the original workspace is not modified while jobs run.
Each successful stage returns a project-state tar archive beneath
`SWIF_OUTPUT_ROOT/WORKFLOW/state/`; the last archive contains the complete
project. Stage summaries and measured resource data are in `summaries/` and
`performance/`, while scheduler logs are under `SWIF_LOG_ROOT/WORKFLOW/`.

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
./dvcs optimize tutorial --profile quick --trials 12 \
  --corpus tutorial-corpus --selection baseline
```

Optuna writes a resumable study under `results/quick/optimization/`. It does
not alter the experiment. To use selected values, initialize a new project,
edit its neural controls, and run a fresh untouched outer test. Do not select
architecture using outer-test, holdout, or real-data behavior.

## Native-model holdout

After synthetic evaluation passes and comparison completes (comparison may
remain scientifically inconclusive because its conventional reference collapsed):

```bash
./dvcs holdout tutorial --profile validation --design /absolute/design.json
```

This evaluates frozen NPE checkpoints against fresh GK11/GK16/GK19/VGG99
native truth. Outputs cannot flow backward into corpus generation, training,
or selection. A predictive mismatch is a scientific result and may be reported
as such without being an execution error. If conventional comparison collapsed,
holdout remains diagnostic-only and cannot enable a robustness claim.

Holdout design and native model are separate validation axes. A same-design
holdout isolates GK/VGG-versus-DD model-family shift. A fresh design also tests
kinematic generalization. A 12/30/60-point training subset tests information
loss only after the checkpoint was trained with those masked counts. Never pad
an older fixed-design checkpoint and interpret the result as calibrated.

## Read-only real-data diagnostic

```bash
./dvcs compare-real tutorial --profile validation --samples 64
```

This action is intentionally narrow. It overlays one audited observable with
frozen-posterior predictions. It does not define a likelihood, use full
covariance, retrain, reweight, or update the posterior. Do not call its output
a GPD extraction from data.

## Resumption and changes

Safe resumable units include complete corpus shards, verified observable
extensions, immutable selections, complete hash-compatible neural members,
persistent Optuna trials, and presentation plots. The workflow rechecks
manifest and content identities before reuse.

Create a new project when changing:

- any injected truth, uncertainty, noise seed, split fraction, or neural
  control (new project/selection, but a compatible native corpus can remain);
- any kinematic, native parameter count/seed, prior/support, physics setting,
  or bridge binary (new corpus required);
- image/dependency version (revalidate compatibility; a changed bridge hash
  requires a new corpus);
- validation policy.

Do not copy `workspace_contract.json` into a changed project to bypass checks.

An interrupted local `corpus-generate` safely resumes complete shards. In
SWIF2, retry transient problem jobs through SWIF2; dependency gates prevent a
failed stage from releasing its successors. Never treat an unreaped farm
scratch directory as a result.

## A disciplined study procedure

1. Record the scientific question and intended claim.
2. Initialize a new project and preserve its starting experiment.
3. Change only controls justified by that question.
4. Run `doctor` in the real execution allocation.
5. Complete `quick` and inspect failures, invalid draws, and resource records.
6. Freeze architecture and gate policy before a larger campaign.
7. Create, plan, generate, and deep-verify the corpus; freeze a selection.
8. Run training, evaluation, and comparison.
9. Interpret identifiability and calibration, not only point estimates.
10. Run named-model holdout after evaluation passes and comparison completes;
    require full closure before making a robustness claim.
11. Preserve configuration, hashes, manifests, image digest, and commit with
    any result publication.

## Where to go next

- [CLI reference](CLI_REFERENCE.md)
- [Corpus and data selection](CORPUS_AND_DATA_SELECTION.md)
- [Experiment reference](EXPERIMENT_JSON_REFERENCE.md)
- [Workflow and physics](WORKFLOW_AND_PHYSICS.md)
- [Results and interpretation](RESULTS_AND_INTERPRETATION.md)
- [Resource management](RESOURCE_MANAGEMENT.md)
- [Troubleshooting](TROUBLESHOOTING.md)
