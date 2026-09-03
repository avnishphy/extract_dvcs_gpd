# Production readiness report — 2026-09-03

## Decision

**NO-GO for full production.** No full SWIF2 workflow was imported, started, or
submitted. The exact-image preflight passed 40 of 47 checks, emitted one
non-blocking quota-query warning, and failed six required checks:

1. the frozen 96-point table reaches `xB=0.4` and `t=-0.7 GeV2`, outside the
   physics metadata ranges `xB<=0.28` and `t>=-0.2 GeV2`;
2. the canonical assumption registry remains draft pending researcher approval;
3. `workspace/experiment_josh/gpd_truth.json` and its approved coordinate-table
   hash are absent;
4. `provenance/images.lock.json` is unpublished, contains `OWNER`, and has no
   registry digest (the local SIF itself is hash-verified);
5. ignored `jobs/jlab_ifarm/resources.env` still names `my_test_big_corpus1`
   rather than this campaign; and
6. the exact production-like vertical rehearsal has not run.

The machine-readable result is
`docs/audits/PRODUCTION_PREFLIGHT_2026-09-03.json`. The gate is deliberately
fail-closed; static-test success cannot override these failures.

## 1. Git preservation and commits

The dirty patch originally found on `main` at
`22f45e94d424e2a73c57d6b2504869a2eebb35f4` was preserved on
`codex/physics-production-readiness` and divided into:

- `9040378` — deterministic validation and telemetry correctness;
- `3f7e765` — uncertainty and validation contracts;
- `a35ddfe` — explicit model-family contracts and tests; and
- `c0ba0f4` — evidence, claims, limitations, and literature scaffolding.

The exact file/function/test/production-connection inventory is in
`PREVIOUS_PATCH_AUDIT_2026-09-03.md`. No push or merge was performed.

## 2. Frozen production identity

The selected project is `experiment_josh`, profile `validation`, selection
`baseline`. The user kinematics were not edited.

| Item | Frozen value |
|---|---|
| Experiment JSON | `22d6dc8d030991f97caef2ea0fb2b168c3635cf598fee3ea77ceb856ac5e7a89` |
| Generated workflow | `3c4f1aeefc86cdeea54a43061b5d868336a7b249209d219a2fc7c1b23727870b` |
| Native physics | `6c2807364b95bc158ed47926f288bd134368a4575e50df1ab541376ac4f79a02` |
| Canonical 96-point table | `701d451a921976459ace744767aca5b0a8d1b4854c6087187785fbb6b77e49be` |
| Selection | `c5de6aa2b01c8dd491fc91ba4ae9e359cfb3ab886556b040b58461c0a6849bf6` |
| Factorized prior | `d85a19777dfdf8209ade87415e1db0fbff324cb510d2a8f9c9519e3aab2af9a5` |
| Assumption registry | `dcbc6623c6f29886a61238d2449502b9d0241dce19331b9f14d4e22ab6e451e4` |
| Local JLab SIF | `3793116765e64f917f7c464955341735df6c310caecd3568a064a792001effdb` |
| Bridge binary in SIF | `2a2d79c2baaa1ce3670a2ca25e2d95b845b9820da78a734ff632781ce814c01d` |
| PARTONS / APFEL++ | 5.0.0 at `1ad0b7d...` / 4.8.0 |

`./dvcs doctor experiment_josh` deterministically removed the retired MAF
`num_bins` field from the generated engine. It did not alter experiment JSON or
kinematics. Both pre- and post-migration evidence is retained in raw logs.

## 3. Native corpus and regeneration impact

The planned architecture-neutral corpus is `experiment-josh-native-v1`:
16,384 accepted parameter groups, shard size 16, 1,024 immutable atomic shards,
32 shards per worker, 32 worker jobs, and one merge job. It requires schema v2
and canonical GPD function grids.

The existing 920 MiB `partons-validation-16384` corpus is not reusable for this
campaign: it is schema v1, was generated with bridge 0.9 (`72a0ed...`), and has
no canonical GPD grids. It remains valid only for its recorded historical DD
scope and must not be relabelled.

The complete 39-choice matrix is
`provenance/regeneration-impact-matrix.json`. Its controlling rules are:

- native physics, kinematics, process definitions, GPD content, or central
  scales require new native generation (usually a new corpus);
- covariance, nuisance, or random-noise changes require new realizations only;
- context normalization and DD/neural architecture changes require model-view
  regeneration/retraining, never PARTONS regeneration;
- generator families and shadow/D-term stress studies are append-only native
  shards/corpora; and
- plotting/metric changes are evaluation-only, while claim text is no-regeneration.

Prior changes are selection-only only when the existing master support covers
the new prior and overlap/ESS is adequate; otherwise new native shards are
required.

## 4. Physics-assumption manifest

The canonical 50-entry registry is
`provenance/production-native-assumptions.json`; the generated human table is
`PRODUCTION_NATIVE_ASSUMPTIONS.md`. Each entry records value, source, scope,
status, enforcement, validation, validity domain, corpus-identity effect, and
claim implication. The submission manifest is
`provenance/production-submission-manifest.json`.

The intended claim is only conditional closure:

> Conditional DD pseudodata closure under the declared simulator, factorized
> prior, fixed covariance policy, nuisance model, and 96-point design.

It does not authorize “model-independent,” neural-GPD, or real-data claims.

## 5. D-term and shadow status

- The production route includes `H`, `E`, `Htilde`, `Etilde` with `u`, `d`,
  `s`, and gluon inputs at `Q0^2=1 GeV2`, LO APFEL evolution, fixed `nf=3`, and
  six ordered observables.
- A D-term is explicitly omitted. This is acceptable only for a labelled
  zero-D-term conditional baseline. It prohibits general D-term/mechanical-
  property claims. Adding an inferred or fixed D-term to the main simulator
  requires new native generation.
- Fixed shadow coefficients are present in the declared truth. `H.u` uses the
  installed `GPDBDMMS21` up-sea direction; other channels use project-defined
  zero-forward-moment DD directions. They are not inferred, and their CFF-null
  property is explicitly unverified.
- A separate immutable shadow/D-term stress corpus is sufficient for
  representation and inference-honesty validation. It is not sufficient to
  claim the main posterior inferred a coordinate absent from main-corpus support.

## 6. Covariance validation

The current SIF/bridge evaluated the exact injected truth at all 96 kinematics.
The report `PRODUCTION_COVARIANCE_VALIDATION_2026-09-03.json` passed:

| Quantity | Result |
|---|---:|
| Shape | `576 x 576` |
| Flattening | kinematic-major, observable-minor |
| Symmetry maximum absolute error | `0` |
| Minimum / maximum eigenvalue | `1.000104958e-4` / `19.13458956` |
| Condition number | `191325.8143` |
| Cholesky | success |
| Diagonal range | `1.000225e-4` to `19.13458885` |
| Correlation range | `-0.02810144` to `1.00000000` |
| Numerical jitter | `0` |
| Masked production submatrix | 576 observations, positive definite |
| Component reconstruction error | `0` |
| Explicit nuisance/covariance overlap | none |

The uncorrelated, local-kernel, and rank-one phi-shape covariance components
were tested separately from the explicit global and LU normalization nuisance
responses. The deterministic 4,096-replica test passed: maximum mean deviation
`3.013 sigma` against a familywise `5 sigma` limit and maximum covariance
deviation `4.522 sigma` against `6 sigma`. Low-rank nuisance variance checks
also passed. The earlier historical-corpus diagnostic is retained only as
corroboration; it is not the authoritative report.

## 7. Production-like rehearsal

No rehearsal was run. `PRODUCTION_REHEARSAL_2026-09-03.json` records the exact
invariants, the only permitted count/shard overrides, and the blocker. Inventing
a function grid or editing the physics metadata without approval would make the
rehearsal scientifically non-equivalent. Synthetic tensor smokes were not
misrepresented as a rehearsal.

## 8. SWIF2 dry-run and pilot

- Corpus dry-run: exit 66 before SWIF2 contact because the approved
  `gpd_truth.json` is absent.
- Analysis dry-run: exit 66 before SWIF2 contact because the verified merged
  corpus archive is absent.
- Farm pilot: not run because the preflight is NO-GO and no pilot authorization
  was given.
- Full production: not imported, started, or submitted.

These failures prove the staging wrappers fail closed; they do not prove the
generated dependency DAG or resource requests, which must be inspected after
the missing immutable inputs exist.

## 9. Exact commands after all six blockers are closed

First rerun the gate without `--allow-dirty`:

```bash
python3 tools/production_preflight.py \
  --output docs/audits/PRODUCTION_PREFLIGHT_2026-09-03.json
```

Corpus dry-run and submission:

```bash
./dvcs farm-corpus-submit --project experiment_josh \
  --corpus experiment-josh-native-v1 --profile validation \
  --shard-size 16 --shards-per-worker 32 \
  --workflow experiment-josh-corpus-baseline-v1 --dry-run

./dvcs farm-corpus-submit --project experiment_josh \
  --corpus experiment-josh-native-v1 --profile validation \
  --shard-size 16 --shards-per-worker 32 \
  --workflow experiment-josh-corpus-baseline-v1
```

Analysis dry-run and submission after corpus completion/reaping:

```bash
./dvcs farm-submit --project experiment_josh \
  --corpus experiment-josh-native-v1 --selection baseline \
  --profile validation \
  --corpus-archive /w/hallc-scshelf2102/nps/singhav/extract_dvcs_gpd/results/swif2/experiment-josh-corpus-baseline-v1/final/experiment-josh-native-v1.tar.gz \
  --from selection --through plot \
  --workflow experiment-josh-analysis-baseline-v1 --dry-run

./dvcs farm-submit --project experiment_josh \
  --corpus experiment-josh-native-v1 --selection baseline \
  --profile validation \
  --corpus-archive /w/hallc-scshelf2102/nps/singhav/extract_dvcs_gpd/results/swif2/experiment-josh-corpus-baseline-v1/final/experiment-josh-native-v1.tar.gz \
  --from selection --through plot \
  --workflow experiment-josh-analysis-baseline-v1
```

The commands without `--dry-run` are documentary only and must not be executed
until the preflight returns GO and explicit authorization is received.

## 10. Jobs, resources, storage, and makespan

Corpus: 32 parallel workers (`16 CPU`, `32G RAM`, `32G scratch`, `12h` each)
plus one merge (`4 CPU`, `16G RAM`, `64G scratch`, `12h`). Analysis: 12 jobs—
selection; materialize; train; evaluate; exact reevaluate; compare; four
parallel named-family holdouts; holdout merge; plot.

| Analysis stage | CPU | RAM | Scratch | Walltime | GPU |
|---|---:|---:|---:|---:|---:|
| selection | 1 | 4G | 8G | 30m | 0 |
| materialize | 2 | 20G | 48G | 4h | 0 |
| train | 4 | 32G | 48G | 12h | 1 |
| evaluate | 8 | 16G | 48G | 12h | 1 |
| exact reevaluate | 8 | 16G | 48G | 12h | 0 |
| compare | 2 | 12G | 32G | 4h | 0 |
| each of four holdouts | 4 | 20G | 48G | 12h | 0 |
| holdout merge | 2 | 20G | 48G | 2h | 0 |
| plot | 2 | 8G | 32G | 1h | 0 |

Historical successful stage telemetry gives materialize 133–214 s, train
1,556 s, evaluate 7,723 s, compare 122 s, the slowest comparable holdout
20,027 s, and plot 90–101 s. This is about 8.2 h for the measured critical-path
pieces, excluding selection, exact reevaluation, corpus production, transfers,
and queue time. There is no successful comparable schema-v2 corpus telemetry,
so a defensible point estimate is unavailable. The allocation ceiling is 24 h
for concurrent corpus workers plus merge and 59.5 h for the serial/parallel
analysis critical path: a low-confidence planning envelope is 32–84 h plus
queue and transfer time. The pilot must replace this envelope with measurements.

The historical schema-v1 corpus is 0.90 GiB and the interrupted materialized
arrays are 2.9 GiB. Plan 80 GiB persistent headroom for schema-v2 grids and
successive reaped project archives; preflight requires at least 100 GiB free.
Observed free space was about 442 GiB on scshelf and 869 GiB on farm_out. The
site `quota -s` command returned no record (non-blocking warning), so quota
must be verified with the storage owner before launch.

## 11. Monitoring, diagnosis, retry, cancellation, and reaping

```bash
swif2 status experiment-josh-corpus-baseline-v1 -jobs -transfers -storage -display json
swif2 diagnose experiment-josh-corpus-baseline-v1
swif2 show-job experiment-josh-corpus-baseline-v1 -name dvcs-corpus-worker-0000 -display json

swif2 status experiment-josh-analysis-baseline-v1 -jobs -transfers -storage -display json
swif2 diagnose experiment-josh-analysis-baseline-v1
swif2 show-job experiment-josh-analysis-baseline-v1 -name dvcs-train -display json

swif2 retry-jobs WORKFLOW -problems PROBLEM_CLASS
swif2 cancel WORKFLOW
```

Do not use `swif2 cancel WORKFLOW -delete` for normal rollback and do not manage
the underlying Slurm jobs separately. Diagnose OOM/timeout from reaped
`summaries/` and `performance/` records before changing resources; retry only
transient site/transfer failures unchanged. Resume a completed stage with a new
workflow name, the same immutable identities, `--project-archive` pointing to
the last reaped state, and the appropriate `--from STAGE`; dry-run it first.

Reaping is automatic and mandatory. Verify/download without overwriting the
source workspace:

```bash
tar -tf results/swif2/experiment-josh-corpus-baseline-v1/final/experiment-josh-native-v1.tar.gz | head
tar -tf results/swif2/experiment-josh-analysis-baseline-v1/state/experiment-josh-analysis-baseline-v1-project-after-plot.tar | head
rsync -a --checksum results/swif2/experiment-josh-analysis-baseline-v1/ /path/to/approved/archive/
```

Final artifacts are the merged corpus archive, last project-state archive,
stage summaries, telemetry JSON/JSONL, and scheduler logs below
`/farm_out/singhav/dvcs/WORKFLOW`. Rollback means cancel the new workflow,
preserve all reaped immutable evidence, and resume from the last verified prior
archive under a new workflow name; the source workspace is never updated in
place by SWIF.

## 12. DD model-dependence status

No new DD science campaign was run. Historical `experiment_josh` artifacts show
successful materialize/train/evaluate jobs and named-family holdout jobs, but
the available compare summary records `scientific_gate_failed: true`. The new
best-DD-projection and uncertainty/provenance APIs are unit-tested but have not
been run on representative GK/VGG holdouts. Therefore there are no new valid
projection floors, paired-noise scorecards, prior-sensitivity results, SBC, or
combined model-dependence conclusions. A poor DD-to-GK/VGG result remains a
scientific diagnostic, not something to tune away.

## 13. Neural-GPD production blockers

The explicit registry correctly blocks neural `train`, `evaluate`, exact
reevaluation, compare, holdout, plot, result assembly, and optimization rather
than falling through to DD. Remaining blockers are:

1. no approved canonical coordinate table/GPD truth request;
2. no schema-v2 multi-function corpus with recorded family balance/splits;
3. no production autoencoder training/reconstruction gate;
4. no frozen decoder checkpoint, latent whitening artifact, or reconstruction
   floor;
5. no production realization-to-latent model view (the code utility is only
   experimentally tested);
6. no neural NPE checkpoint or posterior latent sampling path;
7. no inverse-whitening/decoder/constraint result bundle;
8. no validated external neural-function loader in PARTONS, so exact neural
   reevaluation is unavailable;
9. no multi-family/leave-one-family-out evidence; and
10. no local vertical rehearsal, GPU pilot, or SWIF2 neural workflow.

Until these close, the strongest permitted label is an experimental
“DD-function-manifold neural representation” if trained only on DD functions,
never model-independent inference.

## 14. Remaining scientific gates

- Conditional DD claims: close all six operational blockers; complete the
  exact rehearsal/pilot; run fresh deterministic NLD, coverage/SBC subset,
  contraction, posterior predictive, boundary, seed, and exact-native checks.
- Model-dependence claims: paired-noise fixed-covariance holdouts, separate
  truth-covariance realism studies, best-DD projections, prior/ansatz
  sensitivity, and an uncollapsed scorecard.
- Reduced-parametric-bias neural claims: production neural vertical slice,
  representation gate, multi-family and leave-one-family-out paired seeds,
  prior/constraint sensitivity, exact backend validation, and calibration.
- Real-data extraction: validated real-data covariance/nuisances, conventional
  exact-posterior baselines, full SBC/joint calibration, shadow/D-term honesty,
  external constraints with uncertainties/correlations, and approved physics
  assumptions.

## 15. Tests and command results

- Original full static suite after patch audit: 61 unit tests plus distribution,
  documentation, JLab wrappers, shell, installer, and container-contract tests:
  passed.
- Final pinned-image static suite: 64 unit tests plus all the same wrapper/static
  checks: passed.
- New readiness tests: 3/3 passed (registry rendering, regeneration vocabulary,
  fail-closed preflight).
- Focused scientific contracts: 18/18 passed.
- DD and neural synthetic family smokes: both passed with finite posteriors.
- Current SIF native bridge capabilities and self-test: passed.
- Assumption registry render/check: 50 entries, passed.
- Current native covariance: 96 truth evaluations and all matrix/replica/nuisance
  gates passed.
- `git diff --check`: passed.
- Corpus and analysis SWIF dry-runs: expected fail-closed exits 66; no contact.

Two attempted full-static reruns are intentionally retained as harness failures:
one inherited the host `/scratch/singhav` TMPDIR absent inside the SIF; a second
used `/tmp` but the minimal SIF lacks `rg`. A host rerun used Python 3.9 and
failed because `tomllib` is unavailable. The supported pinned-image rerun with
`/usr/bin/rg` explicitly bound then passed all 64 tests. These are environment
diagnostics, not suppressed failures.

Not run: the production-like vertical rehearsal, SWIF farm pilot, full corpus,
full DD campaign, new GPU training, exact posterior reevaluation campaign,
best-DD projections, neural production stages, shadow inference-honesty/D-term
stress campaign, full calibration/exact baselines, literature reproduction, and
progress presentation. They were blocked, unauthorized, downstream of missing
production results, or deliberately deferred by priority.

## 16. Researcher approvals required

Before GO, approve or correct without changing the kinematic table:

- the xB/t domain metadata contradiction;
- proton target, lepton charge/polarization, target polarization, and phi
  conventions encoded by the selected PARTONS modules;
- twist and overall analysis-order labels (currently null/unknown);
- coefficient/evolution order interpretation, fixed-nf=3 quark–gluon mixing,
  thresholds, alpha_s, and absence of scale variation;
- finite-t/target-mass/kinematic corrections inherited from `DVCSProcessGV08`;
- elastic form-factor source and absence of declared radiative corrections;
- zero D-term, omitted pion-pole/heavy-input/positivity/sum-rule constraints;
- fixed shadow directions/coefficients and the absence of a verified CFF-null
  property;
- factorized uniform DD prior/bounds and fixed-at-truth covariance/nuisance
  semantics;
- the canonical GPD coordinate table for schema-v2 grids; and
- whether the exact local SIF hash is an approved production identity or a
  published registry digest is mandatory.

Approval means updating the canonical registry/manifest and their hashes, not
only signing off in prose.

## 17. Deliberately deferred recommendations

Retain the literature registry, benchmark audit, Moffat scaffold, and
uncertainty-mode support. Defer full Moffat reproduction, direct NNGPD BNN
cloning, literature-specific jobs, central-curve non-inferiority optimization,
speculative archive/state-transfer refactors, architecture searches, and the
one-time progress deck until a successful production result exists. Also defer
resource retuning until the exact rehearsal/pilot provides comparable telemetry.

## 18. Raw evidence

All preservation snapshots, diffs, commits, native checks, tests, covariance
runs, preflights, and both SWIF dry-run failures are under
`docs/audits/logs/20260903T194355Z/`. Each audited command has `.meta`,
`.stdout`, and `.stderr` files containing invocation, UTC start/end, exit status,
host, working directory, Git identity, and environment summary.
