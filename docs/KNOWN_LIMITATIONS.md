# Known limitations and claim boundary

- Production train/evaluate/compare/holdout/plot dispatch remains DD-only.
  Neural-GPD view construction and a tiny NPE mechanics smoke exist, but the
  CLI rejects neural production stages until the decoder, whitening,
  checkpoint, and result-bundle route is complete.
- The historical SWIF validation-v7 collector rejected every numeric GPU
  sample because of a parser defect. Those utilization/power/VRAM records are
  invalid; only the repaired collector may support future conclusions.
- Evaluation-role, unseal-ledger, strict provenance, covariance-policy,
  uncertainty-realization, DD-projection, and native-failure contracts are
  focused APIs. Existing legacy artifacts are not retroactively relabelled.
- Epoch-level Optuna pruning, full interruption/resume checkpoints, SWIF state
  deltas, SBC/TARP/L-C2ST, simultaneous bands, and inverse-crime-resistant
  closure remain unimplemented.

- Canonical native GPD-truth storage is now mandatory for newly created
  schema-2 corpora, but no reviewed production coordinate table or completed
  production corpus is claimed yet. The bundled two-point table is only a
  runtime smoke test.
- Exact neural posterior reevaluation awaits the native external-function
  loader described in [Native bridge](NATIVE_BRIDGE.md); no Python fallback is
  permitted.
- Stage-09 remains a fixed shadow stress test, not a complete Moffat-style
  shadow-identifiability analysis.

## Packaging and deployment

- No OCI image has been published. `provenance/images.lock.json` intentionally
  has null digests, so source build is the only verified installer decision.
- Apptainer 1.5.2 and the JLab Slurm/SWIF2 tools were available. The dated
  verification record identifies a 0.3.0 SIF with SHA-256 `c850f82c...`; the
  current on-disk 0.3.0 SIF is a different artifact with SHA-256
  `3793116765e64f917f7c464955341735df6c310caecd3568a064a792001effdb` and has
  separately passed CPU/static and native self-tests. No fresh GPU smoke was
  run, so GPU runtime acceptance is not claimed. The historical v7 job saw an
  A800, but its numeric GPU telemetry is invalid. See the dated
  [container verification record](../provenance/container-verification-2026-08-31.json).
  Normal installation uses `install.sh`, whose host-prefetched, checksum-bound
  wheel path avoids the failed manual-build download route.
- The Ubuntu base digest and APT snapshot plus direct/transitive Python
  versions are pinned, but wheel hashes are not; bit-for-bit Python rebuilds
  are not claimed.
- Immutable historical images improve reproducibility but can contain known
  vulnerabilities. Security updates require new reviewed snapshots/images.

## Accelerator and scaling

- PARTONS is CPU-native. This distribution makes no PARTONS-on-GPU claim.
- Multi-GPU ensemble sharding and per-GPU Optuna trial sharding are
  implemented and statically/mocked tested, but NCCL, concurrent SQLite study
  behavior, deterministic equivalence, and posterior statistical equivalence
  have not been validated on real multi-GPU hardware.
- Multi-GPU execution is single-node. No multi-node distributed training or
  distributed native simulation is implemented.
- Native parallelism uses isolated subprocesses because PARTONS thread safety
  is not established. Initialization/memory overhead may limit scaling.
- Corpus generation is atomically sharded and resumable. The JLab SWIF2 layer
  distributes disjoint shard ranges across independent jobs and deep-verifies
  their merge; practical PARTONS scaling, dispatch latency, and filesystem
  throughput remain campaign-dependent. This is not one multi-node PARTONS
  process.
- The current sbi path materializes selected nuisance/noise tensors into named
  array files before training. CPU workers share full in-memory output arrays
  and write disjoint rows; model minibatches consume completed files rather
  than generating noise lazily. Peak host RAM and state-transfer cost therefore
  remain telemetry targets.
- Native evidence is consolidated per shard, reducing inode pressure, but
  storage quota, purge policy, throughput, and practical shard size still need
  measurement on the target iFarm filesystem.
- Multi-process realization publication uses an advisory file lock. Its local
  behavior is regression-tested, but the target iFarm shared filesystem's
  locking semantics and contention under a real multi-GPU allocation remain
  an acceptance item.

## Scientific model scope

- The production path is the declared full-independent DD representation at
  `Q0²=1 GeV²`, fixed-three-flavor LO evolution, LO standard DVCS CFFs, and an
  ordered nonempty subset of six audited observables. It is not a general selector for arbitrary PARTONS modules,
  orders, twists, schemes, thresholds, or processes.
- Five controls per GPD/channel are inferred. Fixed shadow/stress coefficients
  are simulator-family settings, not posterior coordinates. Only the H/u
  shadow direction uses installed `GPDBDMMS21`; other stress directions are
  project test directions and are not claimed CFF-null/native.
- Kinematic support is the validated fixed-target envelope and coupled
  `0<y<1` domain, not the entire formal DVCS phase space.
- The `quick` and `validation` defaults are compute profiles, not guarantees of
  adequate simulation density, calibration precision, or identifiability in
  82 dimensions.

## Inference and validation

- Simulation-based inference is amortized over the declared synthetic model.
  Calibration outside its prior, uncertainty, kinematic, and representation
  envelope is not established.
- The conventional posterior uses self-normalized prior importance sampling
  on the exact bank. In high dimension it can collapse onto one proposal. The
  framework then marks the reference unavailable and emits no distance or
  width claim. A larger bank is not guaranteed to solve this; an adaptive
  exact-likelihood reference sampler remains future work.
- A finite number of coverage trials has Monte Carlo uncertainty. Passing
  configured standard-error gates is not proof of universal calibration.
- Named GK/VGG holdouts test predictive behavior after freezing the NPE. They
  do not establish DD parameter recovery because native model parameters do
  not map to the inferred DD coordinates.
- Masked training supports declared subsets of the training kinematic bank.
  It does not establish calibration at arbitrary new coordinates unless the
  training corpus itself varies kinematic designs across that domain.
- No validation threshold may be interpreted beyond the campaign for which it
  was frozen.
- Corpus integrity and reuse do not establish posterior calibration or
  real-data readiness. Closure, SBC/coverage, predictive checks, exact
  reevaluation, and output-blind named-model validation remain separate.
- Adding an admitted observable can repeat evolution/CFF work inside PARTONS.
  Existing stored shards are not recomputed or rewritten, but no unverified
  internal PARTONS cache is claimed.
- One selection freezes one grouped split; k-fold training is not implemented.

## Real data and database

- Real measurements are quarantined from generation, training, model
  selection, likelihood construction, and posterior updating.
- `compare-real` is a narrow frozen-posterior diagnostic, not a fit or GPD
  extraction from data. It does not implement a complete experimental
  covariance/likelihood.
- Many installed database observables remain pending convention, unit,
  uncertainty, and runtime mapping audits. `TSlope` is derived/unmapped.
- JLab/project data-access policies and individual dataset redistribution terms
  remain the user's responsibility.

## Licensing and access

- The imported application and bridge lack a repository-level license grant.
  Public source/image publication requires permission from copyright holders.
- `gpddatabase` combines a GPL-3.0 file with additional non-profit-use wording;
  compatibility and dataset redistribution require upstream/legal review.
- The database and LHAPDF set are therefore installed separately rather than
  embedded silently.

## Verification record

Exact passed, partial, and unverified checks are recorded in
`provenance/verification-2026-08-14.json`. The earlier acceptance evidence
includes a complete editable
eight-vector native generation and two-member CPU NPE training smoke passed;
the full standard 2,048-vector quick generation was time-bounded after 215
successful native bridge requests and is not marked complete.

Use [Acceptance](ACCEPTANCE.md) to close unavailable-runtime items. Never turn
“unverified” into “passed” based only on code inspection or device detection.
