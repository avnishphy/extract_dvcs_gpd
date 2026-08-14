# Native PARTONS bridge

## Why an executable boundary exists

The C++ bridge is the sole authoritative physics interface used by the public
workflow. It loads PARTONS and project-defined `GPDModule` implementations,
validates a versioned JSON request, evaluates through native services, and
returns JSON containing numerical values and backend provenance.

An executable boundary was chosen because PARTONS uses process-global state
and has not been proven thread-safe for this workload. It also makes requests
content-addressable, decouples Python from the native ABI, and gives every
failure an explicit exit code/stdout/stderr record.

## Installed layout and relocation

The native installation places these files together:

```text
PREFIX/bin/partons_bridge
PREFIX/bin/partons.properties
PREFIX/bin/logger.properties
PREFIX/bin/xmlSchema.xsd
PREFIX/lib/libPARTONS.so
PREFIX/lib/libapfelxx.so
PREFIX/lib/libNumA++.so
PREFIX/lib/libElementaryUtils.so
PREFIX/lib/libLHAPDF.so
```

At startup the bridge canonicalizes its executable location and uses that
directory for configuration/schema discovery. Its installed ELF RUNPATH is
`$ORIGIN/../lib`; it contains no development-tree RPATH. Calling the installed
binary from any current directory is supported. Moving the executable without
its sibling configuration files is not.

## Basic invocation

Inside a configured container:

```bash
partons_bridge --capabilities
partons_bridge --self-test
partons_bridge < configs/examples/stage01_evaluate_gpd.json
```

Through the public container wrapper:

```bash
./dvcs partons-bridge --capabilities
./dvcs partons-bridge --self-test
```

The workflow itself invokes the bridge directly and records each request. End
users normally do not need to construct protocol JSON.

## Protocol rules

- The top-level `schema_version` is `1`.
- `operation` selects an allowlisted operation.
- Scalar requests include `request_id`; batch requests contain `requests`.
- Dimensional quantities use objects such as
  `{"value": -0.1, "unit": "GeV2"}`.
- Unknown fields, wrong units, non-finite values, unsupported combinations,
  and physically invalid kinematics are rejected.
- Successful responses contain `status: "ok"`, operation/schema identifiers,
  the numerical result, and a backend block.
- A native error is never converted to a successful zero or NaN response.
- Batch operations are atomic at the process boundary; Python schedules small
  batches in separate processes for responsiveness and isolation.

The JSON Schema files under `configs/schemas/` document the broad wire format.
Runtime C++ validation remains authoritative because it also enforces coupled
physics constraints that are awkward to express completely in JSON Schema.

## Capability and self-test operations

`--capabilities` reports:

- bridge version, schema, source SHA-256, compiler and C++ standard;
- PARTONS, APFEL++, ElementaryUtils, NumA++, LHAPDF, and GSL versions;
- dependency Git revisions where exposed;
- native shared-library names;
- supported operations and the exact scientific scope of each;
- modules, representations, orders, scales, flavors, and known omissions;
- the assertion that native execution is single-process/single-worker.

`--self-test` evaluates the immutable GK16 reference point and compares H-up
and H-gluon values to the retained fixture with absolute tolerance `5e-13`.
It is a deterministic installation/ABI smoke test, not validation of every
workflow operation.

## Supported operation families

| Operations | Purpose |
|---|---|
| `evaluate_gpd`, `batch_evaluate_gpd` | Restricted installed `GPDGK16` H reference evaluation. |
| `evaluate_dd_gpd`, `batch_evaluate_dd_gpd` | Fixed-scale project double-distribution diagnostic. |
| `evaluate_evolved_dd_gpd`, `batch_evaluate_evolved_dd_gpd` | LO APFEL++ evolution of the reduced DD representation. |
| `evaluate_coupled_basis_gpd`, `batch_evaluate_coupled_basis_gpd` | Explicit Σ/T3/gluon basis-injection and mixing diagnostic. |
| `evaluate_conformal_moment_gpd`, `batch_evaluate_conformal_moment_gpd` | Zero-skewness conformal-moment reconstruction diagnostic. |
| `evaluate_shadow_dvcs`, `batch_evaluate_shadow_dvcs` | Restricted GK16 plus BDMMS21 shadow/CFF/observable diagnostic. |
| `evaluate_pseudodata_dvcs`, `batch_evaluate_pseudodata_dvcs` | Production 80-control DD, multi-Q² evolution, four-CFF simulator with an ordered nonempty subset of six audited observables. |
| `evaluate_post_training_comparison_dvcs`, batch form | Exact reevaluation used for post-training comparison. |
| `evaluate_native_model_holdout_dvcs`, batch form | Fresh GK11/GK16/GK19/VGG99 external native validation. |

Earlier diagnostic operations remain present because their retained native
tests establish pieces used by the production path. The public CLI does not
expose them as alternative scientific models for fitting.

## Production pseudodata operation

The production representation is
`stage11_lo_multiq2_full_independent_dd_v1`. A request contains:

- five DD controls for every H/E/Htilde/Etilde × u/d/s/gluon channel;
- fixed type-level shadow coefficients and channel amplitudes;
- one or more physical DVCS kinematic points;
- a fixed LO, three-flavor, MSbar APFEL evolution configuration;
- fixed CFF/process identifiers and an ordered nonempty subset of the six
  audited observable module identifiers.

The input GPD is defined at `Q0² = 1 GeV²` and evolved to each datum Q². The
bridge owns one APFEL evolution table per GPD type for a request and exposes it
to PARTONS through `TabulatedPseudodataGPD`; the adapter does not calculate an
evolution kernel, CFF, or observable. PARTONS always evaluates the four CFFs
and computes only the requested observable subset from:

- real and imaginary parts of H, E, Htilde, and Etilde CFFs;
- `DVCSCrossSectionUUMinus`;
- `DVCSCrossSectionDifferenceLUMinus`;
- `DVCSAc`, `DVCSAluMinus`, `DVCSAulMinus`, and `DVCSAllMinus`;
- optional flavor-separated GPD diagnostic grids.

Every point must satisfy individual ranges and
`0 < Q²/(2 M_p E x_B) < 1`. Data are never evolved.

## Native-model holdout operation

The holdout representation accepts only `GPDGK11`, `GPDGK16`, `GPDGK19`, or
`GPDVGG99`, the six frozen observables, the declared kinematic envelope, and
an optional strictly increasing GPD x grid. It returns CFFs, observables,
flavor-separated GPD diagnostics, model source hashes, and a no-learning
contract. VGG99 also records the LHAPDF set/member and grid/archive hashes.

Named-model internal parameters are not interpreted as DD posterior truth.
The operation supports predictive validation only.

## Cache record

For a request hash `HASH`, the workflow stores:

```text
cache/HASH/
  request.json
  response.json
  metadata.json
  stdout.raw.txt
  stderr.txt
  exit_code.txt
```

`metadata.json` records the executable SHA-256, request/response SHA-256,
operation, evaluation count, cache schema, exit code, and `surrogate_used:
false`. A complete exact hit requires consistent files and hashes. Interrupted
directories are not accepted as successful cache entries.

These directories are transient during reusable-corpus generation. Once a
native shard succeeds, they are consolidated into one compressed, hashed
evidence archive owned by that shard and removed from working state. This
retains the audit trail without creating a permanent many-small-file corpus.

## Parallelism

The bridge is always single-worker. Parallel native execution is implemented
above it by the Python scheduler:

```text
worker 0 -> bridge process -> PARTONS instance
worker 1 -> bridge process -> PARTONS instance
...
```

The scheduler uses affinity-bounded process workers and reconstructs original
input order. OpenMP/BLAS defaults to one thread per worker to avoid multiplying
the allocation. This is intentionally different from claiming that PARTONS is
thread-safe or GPU-enabled.

## Failure diagnosis

When a bridge call fails, inspect the cache request, `stderr.txt`, raw stdout,
and exit code before retrying. Typical categories are configuration/schema
errors, parameter support errors, invalid kinematics, PARTONS exceptions,
missing LHAPDF data, malformed output, or non-finite results. Do not edit a
cached response. Correct the cause and let the workflow create/recompute the
content-addressed entry.
