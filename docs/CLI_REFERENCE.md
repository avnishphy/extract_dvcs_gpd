# Command-line reference

## Invocation model

Run all public actions from the repository root through:

```bash
./dvcs [global-options] COMMAND [command-options]
```

`./dvcs` starts the configured container and invokes the installed
`dvcs-infer` CLI with the installed native bridge. Advanced direct Python or
bridge arguments are primarily for verification and development; ordinary
users should use the wrapper.

Successful commands print a concise JSON object to stdout. Progress goes to
stderr so scripts can parse stdout safely. User/configuration errors return
exit code 2. Scientific validation actions return exit code 1 when their
result status is `fail`.

## Global options

The internal CLI accepts `--repository-root PATH` and `--bridge PATH` as
advanced overrides. The container entrypoint supplies both. Do not point them
at an unrelated checkout or binary: result contracts include the bridge hash.

The host wrapper accepts no installation flags. Change the persistent defaults
with environment variables or rerun `install.sh`.

## Project-management actions

### `init`

```bash
./dvcs init NAME
```

Creates a new isolated project, editable experiment, empty results directory,
project README, and real-data mapping-readiness record. It never overwrites an
existing project. If the pinned database is present, initialization selects a
deterministic measurement-free set of catalog kinematics; otherwise it records
manual mode and disables real-data diagnostics.

### `list`

```bash
./dvcs list
```

Lists real, non-symlink project directories under the workspace root.

### `show`

```bash
./dvcs show NAME
```

Validates the experiment, regenerates the private engine inputs, and prints
paths, truth controls, available profiles, and the synthetic-only assertion.
It performs no native or neural computation.

### `doctor`

```bash
./dvcs doctor NAME
```

Checks the native bridge, configuration, accelerator policy, CPU affinity,
worker resolution, batch/update policy, and neural dependencies. It writes
`results/doctor.json`. Run it after initialization, after editing an
experiment, and inside a new scheduler allocation.

## Workflow actions

All actions below accept:

```text
NAME --profile quick|validation [--no-progress]
```

The profile defaults to `quick`. `--no-progress` disables interactive stderr
bars and is useful in scripts and logs.

### `generate`

```bash
./dvcs generate NAME --profile quick
./dvcs generate NAME --profile quick --force-native
```

Creates the result contract, evaluates injected truth, draws prior parameters,
runs exact native simulations, replaces invalid draws, makes noise replicas,
builds grouped data splits, and writes the displayed pseudodataset.

Generation is resumable through its content-addressed native cache. Completed
requests are reused only when request and executable hashes match. The
`--force-native` option deliberately recomputes native requests instead of
using exact cache hits; use it for backend auditing, not routine recovery.

The printed `native_parameter_count` includes the separately evaluated truth
in the current public summary, while the profile's configured count refers to
accepted prior vectors. Inspect `generated/generation_metrics.json` for full
counts and scheduling provenance.

### `train`

```bash
./dvcs train NAME --profile quick
```

Requires a matching generated contract. It trains every candidate seeded NPE
member, records histories and metrics, then selects active members using only
grouped internal-validation NLL. Complete hash-valid member checkpoints are
resumed; partial or incompatible checkpoints are not treated as complete.

On multiple visible GPUs the container entrypoint launches one process per
GPU. Independent ensemble members are deterministically sharded. Asking for
more ranks than ensemble seeds fails.

### `optimize`

```bash
./dvcs optimize NAME --profile quick
./dvcs optimize NAME --profile validation --trials 50
```

Runs a separate Optuna study whose objective is grouped internal-validation
negative log density. The default trial count comes from `experiment.json`;
`--trials` specifies new trials to add. Studies persist in SQLite and resume.

Optimization never changes `experiment.json` automatically. Review the study,
create a new project, apply selected controls there, and repeat closure. Outer
test, native holdout, and real-data outputs are unavailable to the objective.

### `evaluate`

```bash
./dvcs evaluate NAME --profile quick
```

Requires trained active members. It computes grouped outer-test metrics,
coverage, posterior-predictive checks, exact posterior reevaluation, and exact
GPD diagnostics. Native phases use the same isolated affinity-bounded worker
pool as generation. The command writes `evaluation/evaluation_metrics.json`
and associated non-pickled arrays.

### `compare`

```bash
./dvcs compare NAME --profile quick
```

Compares the neural posterior with a conventional self-normalized
importance-sampling posterior constructed from the exact simulation bank.
Outputs include effective sample size, sliced and marginal Wasserstein
metrics, width checks, samples, and gate decisions. This comparison diagnoses
the neural approximation; it is not an independent experimental analysis.

### `plot`

```bash
./dvcs plot NAME --profile quick
```

Reads compatible saved generation, training, evaluation, and comparison
artifacts and writes PNGs plus `plots/plot_manifest.json`. It calls neither
PARTONS nor neural inference. Re-running it is the safe way to regenerate
presentation artifacts without repeating expensive computation.

### `holdout`

```bash
./dvcs holdout NAME --profile validation
```

Runs only after synthetic closure gates pass. It generates fresh native truth
from allowlisted `GPDGK11`, `GPDGK16`, `GPDGK19`, and `GPDVGG99`, conditions
the already frozen NPE, exact-reevaluates inferred DD samples, and writes
model-specific metrics/plots beneath `holdout/`.

The command hashes protected pre-holdout artifacts before and after execution
and aborts if they change. Named model outputs cannot select or retrain the
posterior. VGG99 requires the locked `MSTW2008nlo68cl` LHAPDF set.

### `compare-real`

```bash
./dvcs compare-real NAME --profile validation
./dvcs compare-real NAME --profile validation --samples 64
```

Reads an allowlisted database observable and overlays it with predictions
from the frozen posterior. `--samples` controls exact posterior samples.
Training, likelihood, hyperparameter selection, and posterior updates remain
explicitly false in the result. Missing database data, unsupported mappings,
or failed audits do not become a fit.

### `run`

```bash
./dvcs run NAME --profile quick
```

Runs generation, training, evaluation, comparison, and plotting in order. It
does not run Optuna, native-model holdout, or real-data comparison. For long
campaigns prefer stepwise actions so scheduler resources can match each phase.

## Required ordering

```text
init -> doctor -> generate -> train -> evaluate -> compare -> plot
                              |
                              +-- optimize (separate study)

after passed closure ----------+-- holdout
after frozen posterior --------+-- compare-real
```

`plot`, `holdout`, and `compare-real` can be repeated if their source contract
still matches. Any experiment/profile/bridge change requires regeneration in
a new project.

## Environment overrides

| Variable | Meaning |
|---|---|
| `DVCS_WORKSPACE` | Host directory mounted at `/workspace`. |
| `DVCS_RESULTS` | Host directory mounted at `/results` for runtime and Slurm provenance. |
| `DVCS_CACHE` | Host directory mounted at `/cache`; contains LHAPDF and application caches. |
| `DVCS_DATABASE` | Host database parent mounted read-only at `/database`. |
| `DVCS_ACCELERATOR` | `auto`, `cpu`, or fail-closed `cuda`. |
| `DVCS_CPU_THREADS` | Torch CPU-thread request; full affinity by default interactively, bounded by allocation in Slurm. |
| `DVCS_NATIVE_WORKERS` | Positive integer or `all_available`, bounded by affinity. |
| `DVCS_MATH_THREADS` | Threads per native numerical-library process; default 1. |
| `CUDA_VISIBLE_DEVICES` | Authoritative GPU allocation presented by the runtime/scheduler. |

Job-level overrides are preserved when the launcher reads installation state
and are recorded in runtime provenance.

## Automation guidance

- Treat stdout as JSON and stderr as human progress/logging.
- Check the process exit code and the JSON `status` field.
- Do not infer success solely from the presence of an output directory.
- Preserve `experiment.json`, `workspace_contract.json`, summary JSON, image
  digest, and distribution commit with published results.
- Use a new project name for changed science or method controls.
- Use `--no-progress` in noninteractive automation.
