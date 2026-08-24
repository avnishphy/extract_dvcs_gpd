# Reusable PARTONS corpus and neural data selections

This is the user and technical contract for reusing expensive exact PARTONS
calculations. It explains what is stored, what is generated later, how data
leakage is prevented, and how to add an observable without recomputing the
unchanged GPD/CFF route.

## The three layers

The framework deliberately separates three objects:

1. **Corpus** — clean, noise-free native calculations. A parameter group is
   one 80-component DD parameter vector evaluated at every declared
   kinematic point. The corpus stores that vector, the native CFFs, and each
   selected observable in compact atomic shards.
2. **Selection** — immutable lists of parameter-group indices assigned to
   training, internal validation, and the locked outer test. It contains no
   physics arrays and no neural architecture.
3. **Realization** — tensors made immediately before training or Optuna use.
   Declared nuisance coordinates and measurement noise are drawn
   deterministically for each `(group, replica)` and converted to DeepSets
   tokens. These generated tensors remain in the project result directory.

Consequently, batch size, learning rate, flow architecture, epoch limit,
Optuna trial count, and the number of noise replicas are not corpus identity.
They can be studied without rerunning PARTONS. A train/validation/test group
assignment also does not restrict the neural loss: the current conditional
normalizing flow still minimizes negative conditional log density (a
cross-entropy estimate). The split only determines which independent groups
may contribute to optimization, early stopping, and the final untouched
score.

K-fold validation is intentionally absent. It would multiply 82-dimensional
NPE training cost and complicate the untouched-test rule without fixing a
weak simulator design. The default policy is one deterministic grouped split:
72% training, 8% internal validation, and 20% locked outer test when the
configured internal-validation fraction is 10% and held-out fraction is 20%.
Percentages are rounded to whole groups, and every role must remain nonempty.

## Required public workflow

This is the only supported user workflow. The public launcher enforces corpus
identity, review of the generation plan, and immutable group ownership.

From the repository root, after installation, `init`, and `doctor`:

```bash
./dvcs corpus-create my-study my-corpus --profile validation --shard-size 256
./dvcs corpus-plan my-study my-corpus
```

Stop and inspect `$DVCS_WORKSPACE/my-study/experiment.json` (or
`user/workspaces/my-study/experiment.json` before container installation), especially
`synthetic_dataset.kinematics`, `synthetic_dataset.observables`, DD priors and
physics settings, before the expensive command:

```bash
./dvcs corpus-generate my-study my-corpus
./dvcs corpus-verify my-corpus --deep
./dvcs selection-create my-study my-corpus baseline --profile validation
./dvcs train my-study --profile validation --corpus my-corpus --selection baseline
./dvcs evaluate my-study --profile validation
./dvcs compare my-study --profile validation
./dvcs plot my-study --profile validation
```

Optuna uses exactly the same corpus and selection:

```bash
./dvcs optimize my-study --profile validation --trials 50 \
  --corpus my-corpus --selection baseline
```

It may inspect only the training and internal-validation roles. It may not
inspect the outer test or named PARTONS model holdouts. The selection is not
silently regenerated, and creating another file requires another selection
name.

There is no silent migration of earlier generated arrays into a trusted
corpus. Every public campaign must create and verify a corpus explicitly.

## What defines corpus identity

`$DVCS_CORPUS_ROOT/CORPUS/corpus.json` content-hashes the following. When the
variable is unset, the launcher uses `$DVCS_WORKSPACE/.corpora/CORPUS` in an
installed container and `user/corpora/CORPUS` in a source checkout:

- ordered names and uniform supports of all 80 DD parameters;
- native-parameter count and deterministic parameter seed;
- input representation and complete GPD physics configuration;
- input scale, evolution, flavor/mixing and shadow/stress settings;
- exact ordered kinematics and their beam settings;
- CFF/process configuration; and
- SHA-256 of the exact C++ bridge executable.

Changing any item above requires a new corpus. In particular, a changed GPD
parameterization, prior, parameter seed, kinematic point, input scale,
evolution setting, or bridge binary can never be treated as a cache hit.
Neural and noise controls are absent from this identity by design.

The corpus manifest records requested and available observables separately.
Observable names follow the canonical order documented in
[`EXPERIMENT_JSON_REFERENCE.md`](EXPERIMENT_JSON_REFERENCE.md). Six native
observables are currently admitted. The remaining installed PARTONS factory
classes are not enabled merely because they construct; their conventions and
fixtures still require audit.

## Atomic shards and verification

Core shards contain:

```text
group_index              int64 [groups]
native_parameters        float64 [groups, 80]
native_cffs              float64 [groups, kinematics, 4, 2]
```

Each observable has independent shards containing `group_index` and
`values[groups, kinematics]`. Files are written as compressed NPZ to a
same-directory `.partial` file, flushed, reopened to verify keys/shapes/dtypes,
and atomically renamed. The manifest stores every path, byte count, array
contract, and SHA-256. Rejected parameter draws are recorded per core shard.
Raw bridge requests/responses are consolidated into one compressed, hashed
evidence archive per native shard rather than retained as thousands of small
cache directories.

`corpus-verify` checks manifest integrity and completeness. `--deep` also
rehashes every shard/evidence archive, reopens all arrays, and proves core
group indices are unique and contiguous. Training materialization performs a
fast verification and separately proves selection roles are nonempty,
disjoint, complete, in range, and bound to the corpus core identity.

For transfer or backup:

```bash
./dvcs corpus-export my-corpus my-portable-copy
./dvcs corpus-import my-portable-copy imported-corpus
```

Archives live under the Git-ignored `$DVCS_CORPUS_EXPORT_ROOT/` directory. The
default is `$DVCS_WORKSPACE/.corpus_exports` in an installed container and
`user/corpus_exports` in a source checkout. Export
requires deep verification and rejects symlinks. Import rejects absolute,
parent-traversing, linked, or special members, records the archive hash,
rebinds only the corpus name/location, and deep-verifies before success.

Large corpora may be generated concurrently with disjoint zero-based shard
ranges. Export each range with `corpus-checkpoint-export`, import the batches
on one merge node, and run `corpus-merge`. The merge rejects overlapping or
missing ranges and any change in corpus identity, configuration, backend,
truth arrays, or observables. Only its complete, deeply verified result can be
used by selection and training.

## Safely adding an observable

Edit a **new project's** `synthetic_dataset.observables` list while keeping
the corpus-defining physics and kinematics identical, then run:

```bash
./dvcs corpus-plan new-study my-corpus
./dvcs corpus-generate new-study my-corpus
```

The plan identifies already available and missing observables and states that
existing shards will not be modified. Generation submits only each missing
observable module to the native bridge for the immutable parameter/kinematic
groups. It compares the returned native CFFs bit-for-purpose with the stored
CFF route (absolute tolerance `1e-11`) before publishing the extension.
Existing observable and core shard hashes remain unchanged. If physics or
kinematics changed, compatibility fails and instructs the user to create a new
corpus. There is no observable backfill formula in Python and no silent
surrogate.

An observable extension can change the neural input dimension and therefore
requires a new selection/training result contract, but it does not alter the
underlying corpus parameter groups. Existing selection files remain bound to
the unchanged core identity; users should still use a new project/result
directory for the new observable study.

## Determinism, replicas, and leakage policy

Parameter draws use deterministic shard seeds derived from corpus seed,
shard index, and corpus schema. Nuisance/noise replicas use deterministic
seeds derived from training-noise seed, immutable group index, replica index,
and realization schema. Repeating materialization gives byte-identical
contexts for the same inputs.

All replicas of one native parameter group stay in one role. This prevents a
network from seeing a noisy version of a test physics point during training.
The outer group list is frozen before training and remains unavailable to
early stopping, model selection, Optuna, or architecture choice. Named native
models (GK11/GK16/GK19/VGG99) are a separate output-blind external validation
family and never replace the ordinary grouped DD test.

The current sbi API consumes materialized tensors, so replicas are generated
when the corpus is selected for a training/Optuna invocation rather than by a
custom lazy minibatch dataset. This still avoids PARTONS regeneration and
keeps noise outside the corpus. A future streaming loader may reduce disk/RAM
pressure without changing any physics or split contract.

Concurrent multi-GPU Optuna workers serialize realization publication with a
project/profile lock. One worker writes the deterministic arrays; matching
workers reuse the completed corpus/selection/configuration/bridge identity.

## Claims and limitations

A verified corpus proves reproducible native inputs/outputs and leakage-safe
group ownership. It does not prove that the prior covers physical reality,
that the NPE is calibrated, that a named model is recovered precisely, or
that real data can be fitted. Closure, SBC/coverage, posterior predictive,
exact reevaluation, representation, mixing, and shadow tests remain separate
scientific gates. Real-data likelihood fitting remains disabled in Stage 11.
