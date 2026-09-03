# Canonical GPD truth in the master corpus

Every newly created schema-2 master corpus includes the generator's GPD
function values. This is mandatory, not an optional neural-network add-on.
The same accepted native parameter group therefore owns four aligned products:

1. the sampled generator parameters;
2. canonical input-scale GPD values and validity flags;
3. evolved CFFs at the observable kinematics; and
4. the selected DVCS observables.

This design lets a later model learn parameters, functions, CFFs, observables,
or a combination without rerunning the expensive corpus. It also makes
function-space closure possible: a model is compared with the function that
actually generated each simulated example, rather than only with one special
injected-truth curve.

## Coordinate request is explicit

The framework deliberately does not invent a production grid. Put the reviewed
request at:

```text
$DVCS_WORKSPACE/PROJECT/gpd_truth.json
```

`corpus-create` uses that file by default. `--gpd-truth-request FILE` is an
explicit override. Creation fails if neither exists. The bundled
`configs/examples/master_corpus_gpd_truth_smoke_v1.json` is a two-coordinate
runtime example only; copying it is useful for acceptance tests, but it is not
a scientifically adequate production design.

Run a no-PARTONS planning check before corpus creation:

```bash
./dvcs corpus-preflight PROJECT
./dvcs corpus-create PROJECT CORPUS --profile validation --shard-size 256
./dvcs corpus-plan PROJECT CORPUS
```

The preflight report gives the exact coordinate count, group count, shard
count, uncompressed byte count, compression estimate, and declared coverage.
Review that output together with `experiment.json` before submission.

## Request contract

The request is versioned JSON with these required choices:

- coordinate convention:
  `signed_x_xi_t_at_common_input_scale_v1`;
- flavor basis:
  `PARTONS_physical_u_d_s_gluon_and_charge_parity_v1`;
- scale: `input_scale`, whose `Q0_squared_GeV2` must exactly equal the corpus
  generator input scale;
- scheme: `MSbar`;
- generator representation: either the exact corpus representation or
  `declared_by_master_corpus`, which is resolved during creation;
- interpolation policy, storage dtype, and compression; and
- a nonempty, ordered, duplicate-free coordinate table.

Each coordinate is:

```json
{
  "x": -0.2,
  "xi": 0.1,
  "t_GeV2": -0.2,
  "gpd_type": "H",
  "channel": "u",
  "charge_parity": "native"
}
```

The supported GPD types are `H`, `E`, `Htilde`, and `Etilde`. Supported
channels are native `u`, `d`, `s`; their `_plus` and `_minus` combinations;
`gluon`; and `charge_squared_weighted_c_even_quark_sum`. Channel and parity
must agree. The allowed numerical domain is signed `x` in `[-1,1]`, `xi` in
`[0,1)`, and nonpositive `t_GeV2`.

Coordinate order is scientific data. Changing a coordinate, order, scale,
scheme, dtype, compression, or interpolation declaration changes the master
corpus identity and requires a new corpus name. The manifest records a
SHA-256 of the normalized coordinate table.

## Native calculation

The C++ bridge operation `batch_evaluate_canonical_gpd_truth` constructs the
same `PseudodataInputGPD` used by pseudodata production for each sampled
parameter vector. PARTONS `GPDService` evaluates each explicit coordinate at
`mu_f^2 = mu_r^2 = Q0^2`. Python only validates, schedules, stores, and checks
the returned values; it contains no replacement GPD formula.

For a zero-shadow production configuration, the stored function truth is the
zero-shadow generator itself. The request does not independently enable a
shadow term; simulator coefficients remain frozen by `experiment.json` and
the physics configuration.

The bridge reports each coordinate independently. A finite result has
`valid=true` and `status="ok"`. A coordinate-level native exception or
non-finite result is retained as invalid instead of silently dropping the
whole accepted parameter group.

## On-disk representation

For each `core/shards/shard-N.npz`, schema 2 writes the aligned file:

```text
gpd_truth/shards/shard-N.npz
  group_index  int64                [groups]
  values       float32 or float64   [groups, coordinates]
  status_mask  bool                 [groups, coordinates]
```

The coordinate table lives once in `corpus.json`. Invalid entries use a
finite storage placeholder of zero with `status_mask=false`; every consumer
must mask them. The manifest records array shapes and dtypes, file size,
SHA-256, valid/invalid counts, native-worker resolution, and a compressed
request/response evidence archive.

A schema-2 corpus reaches `status="complete"` only when every core shard has
exactly one aligned GPD-truth shard. Deep verification checks hashes, array
contracts, coordinate width, finite storage, Boolean masks, group alignment,
and evidence archives. Checkpoint export/import and disjoint SWIF2 merge carry
and verify the GPD shards as first-class corpus data.

## Scale and cost

For `N` native parameter groups and `M` requested coordinates, raw GPD storage
is approximately:

```text
N * M * (dtype_bytes + 1 mask byte)
```

For example, float32 values require about `5*N*M` bytes before NPZ/container
overhead. Native execution cost also scales approximately with `N*M`, so a
dense Cartesian grid can dominate a campaign. Prefer an intentional,
physics-motivated coordinate table and use preflight before submitting.

## Historical corpora and injected truth

Schema-1 corpora remain readable by the DD baseline; they are not silently
upgraded or relabeled. They lack canonical per-group function truth and cannot
support a function-target model view without a separately defined migration.

The project's `injected_truth` has a different role. It is one fixed benchmark
used to synthesize and assess a particular closure data set. Corpus GPD truth
is evaluated for every sampled native parameter group and is the supervised
function target available to future architectures. Neither object may be fed
into the observation encoder: truth remains sealed from measured-context
features and is used only as a target or evaluation reference.

## Production checklist

- Set and review all shadow coefficients in `experiment.json`.
- Author the complete ordered `gpd_truth.json`; do not ship the smoke table.
- Run `corpus-preflight PROJECT` and archive its JSON output.
- Run `corpus-create`, then inspect `corpus-plan` before native generation.
- On SWIF2, confirm the staged input manifest contains
  `SWIF_GPD_TRUTH_REQUEST` and its SHA-256.
- After generation or merge, require `corpus-verify CORPUS --deep` to pass.
- Record invalid-mask counts; do not train on placeholder zeros.
