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
experiment, and inside a new scheduler allocation. Output includes the
execution hostname. On a JLab ifarm login host, an interactive invocation asks
whether to rerun automatically in a short Slurm GPU allocation. Set
`DVCS_IFARM_GPU_DOCTOR=yes` or `no` for noninteractive control.

## Corpus actions

### `corpus-create`

```bash
./dvcs corpus-create PROJECT CORPUS --profile validation --shard-size 256
```

Freezes the expensive native-data identity without running PARTONS. The
identity includes parameter order/support/count/seed, physics, exact
kinematics, bridge hash, and requested observables. Shard size must be in
`[1,4096]`. The corpus is stored under `$DVCS_WORKSPACE/.corpora/` in the
packaged runtime and cannot overwrite an existing name.

### `corpus-plan`

```bash
./dvcs corpus-plan PROJECT CORPUS
```

Checks project/corpus compatibility and reports parameter/kinematic counts,
already available observables, missing observables, and the immutable-shard
policy. Inspect both this output and `experiment.json` before allocating
substantial CPU time.

### `corpus-generate`

```bash
./dvcs corpus-generate PROJECT CORPUS
./dvcs corpus-generate PROJECT CORPUS --force-native --no-progress
```

Generates missing exact PARTONS core and observable shards. Each shard is
written to a same-directory partial file, reopened and checked, atomically
published, hashed, and committed to the manifest. Rejected candidates are
retained. Native request evidence is consolidated per shard. Re-running
continues at the first missing shard; a complete compatible corpus is a
no-native-call cache hit. `--force-native` is an audit control.

When a compatible project requests another admitted observable, this command
appends only that observable's shards. Existing core/CFF/observable hashes are
unchanged, and returned CFFs must match the stored route within the declared
tolerance.

### `corpus-verify`

```bash
./dvcs corpus-verify CORPUS
./dvcs corpus-verify CORPUS --deep
```

Fast verification checks manifest structure, completeness, and absence of
unfinished partial files. `--deep` additionally hashes and opens every shard
and evidence archive and proves core group indices are unique and contiguous.

### `corpus-export` and `corpus-import`

```bash
./dvcs corpus-export CORPUS ARCHIVE_NAME
./dvcs corpus-import ARCHIVE_NAME NEW_CORPUS
```

Export deep-verifies and creates a portable `.tar.gz` below
`$DVCS_WORKSPACE/.corpus_exports/`. Import rejects traversal, links, special
members, and existing targets, then rebinds only the corpus location/name and
deep-verifies it.

### `selection-create`

```bash
./dvcs selection-create PROJECT CORPUS SELECTION --profile validation
```

Creates an immutable project-local selection containing disjoint, complete
native-parameter group lists for training, internal validation, and locked
outer test. It stores no physics arrays. Names cannot be overwritten, and the
profile/count must match the corpus.

## Inference and validation actions

All actions below accept:

```text
NAME --profile quick|validation [--no-progress]
```

The profile defaults to `quick`. `--no-progress` disables interactive stderr
bars and is useful in scripts and logs.

### `train`

```bash
./dvcs train PROJECT --profile quick --corpus CORPUS --selection SELECTION
```

Requires the named verified corpus and immutable selection. It first
deterministically materializes nuisance/noise replicas and DeepSets tensors
without calling PARTONS, then trains every candidate seeded NPE member. Active
members are selected using grouped internal-validation NLL only. Complete
hash-valid member checkpoints are resumed; checkpoints are also bound to the
materialized input manifest.

On multiple visible GPUs the container entrypoint launches one process per
GPU. Independent ensemble members are deterministically sharded. Asking for
more ranks than ensemble seeds fails.

### `optimize`

```bash
./dvcs optimize PROJECT --profile quick --corpus CORPUS --selection SELECTION
./dvcs optimize PROJECT --profile validation --trials 50 \
  --corpus CORPUS --selection SELECTION
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

## Required ordering

```text
init -> doctor -> corpus-create -> corpus-plan -> corpus-generate
                                            -> corpus-verify
                                            -> selection-create
                                            -> train -> evaluate -> compare -> plot
                                                 |
                                                 +-- optimize (separate study)

after passed closure ----------+-- holdout
after frozen posterior --------+-- compare-real
```

`corpus-verify`, `plot`, `holdout`, and `compare-real` can be repeated if their
source contracts still match. A neural-only change requires a new project and
selection but not a new compatible corpus. A physics, prior/count/seed,
kinematic, or bridge change requires a new corpus.

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
  digest, corpus/selection manifests, and distribution commit with published
  results.
- Use a new project name for changed science or method controls.
- Use `--no-progress` in noninteractive automation.
