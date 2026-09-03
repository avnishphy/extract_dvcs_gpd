# Current-HEAD scientific audit (2026-09-03)

Audited commit: `22f45e94d424e2a73c57d6b2504869a2eebb35f4` on `main`. The tree was clean before this audit. No submodules were present.

## Environment identity

The host Python is 3.9.25 with NumPy 1.26.4 and does not provide Torch, SBI, Zuko, or Optuna. Scientific tests must therefore run in the pinned Apptainer image. The current environment records Python 3.12, Torch 2.12.1, SBI 0.26.1, Zuko 1.6.0, Optuna 4.5.0, PARTONS 5.0.0 (`1ad0b...`), APFEL++ (`27deaec...`), and LHAPDF 6.5.6. The on-disk `.dvcs/extract-dvcs-gpd-jlab_ifarm-0.3.0.sif` hashes to `3793116765e64f917f7c464955341735df6c310caecd3568a064a792001effdb`; the older 2026-08-31 verification record refers to a different `c850f82c...` SIF. Neither is the historical SWIF-v7 image (`sha256:c6ac0f...`).

## Findings and dispositions

| Priority | Finding | Disposition in this slice |
|---|---|---|
| P0 | SBI 0.26.1 validation divides by nominal batches times batch size; dropping the last group wastes validation evidence. | Validation is sequential, retains the partial batch, and divides by the exact evaluated row count. A 2/2/1 regression test freezes the behavior. |
| P0 | GPU telemetry numeric parsing was unreachable because it was indented after `node_family` returned. | Parser repaired and tested on integer, decimal, exponent, units, whitespace, missing, non-finite, and malformed values. Historical v7 GPU samples remain invalid. |
| P0 | Observation realization, noise model, uncertainty exposure, nuisance treatment, covariance policy, and output semantics were entangled. | Added a validated factorized uncertainty contract, declared presets, paired-noise realization, and strict zero-noise posterior guard. Physical noise and numerical jitter are separately recorded. |
| P0 | Generator identity was too weak for named-family and representation-class holdouts. | Added seven-field generator metadata and explicit holdout classification. New schema-2 corpora record the internal DD family. |
| P0 | Foreign-model tests could accidentally report internal-DD parameter recovery. | Metric guard limits foreign truth to observables, CFFs, common functions/functionals, and a separately labelled best-DD projection floor. |
| P0 | Neural group targets were not explicitly aligned to realization rows. | Neural views now carry `parameter_indices` and a lazy group-to-row alignment contract. |
| P1 | Neural coordinates encoded scientific categories ordinally and diagonal scaling hid correlations/collapse. | Added fixed one-hot typed coordinate tables and training-only shrinkage full-covariance whitening with collapsed-dimension removal. |
| P1 | Production provenance accepted unknown placeholders. | Added fail-closed validation for production/publication profiles; quick/validation profiles receive an explicit missing-field report. |
| P1 | Final evaluation artifacts had no access-control state. | Added role labels plus an explicit append-only unseal ledger gate. |
| P1 | Native rejection evidence existed but had no distortion audit. | Added proposal/acceptance, failure-type, shard, and marginal parameter-region diagnostics; resulting inference is explicitly conditional on simulator validity when rejection occurs. |
| P1 | Literature comparisons mixed unlike objects and omitted unknowns. | Added a strict, machine-readable ten-source comparator registry with null/unresolved fields and comparability grades. |

## Reconciliation with the prior review

| Verified current-HEAD fact | Discrepancy | Implication | Planned stage | Non-goal of this slice |
|---|---|---|---|---|
| Public train/evaluate code loads DD arrays and DD transforms. | Confirmed. | Neural selection must fail before this route. | Stage 3 dispatch gate and smoke now; production route later. | No large neural training. |
| Neural latents are group-level and contexts are row-level; old views omitted `parameter_indices`. | Confirmed. | Pairing could be wrong or require wasteful duplication. | Stage 3 alignment contract and tests now. | No production decoder fit. |
| Validation used random order and `drop_last=True`; SBI's partial-batch denominator is nominal. | Confirmed and the upstream SBI 0.26.1 implementation was inspected. | Missing/misweighted rows can bias early stopping. | Stage 1 exact-count override now. | Training shuffle remains. |
| Holdout covariance was built from the selected truth/model. | Confirmed. | Generator and covariance shifts were confounded. | Stage 1 policy API and Stage 2 variant artifacts now. | Historical holdouts are not reinterpreted. |
| Result provenance allowed `unrecorded`. | Confirmed. | Production reconstruction could be impossible. | Stage 1 fail-closed validator now; workflow-wide migration later. | No fabricated historical commit. |
| Rejected native candidates and replacements were recorded but not summarized spatially. | Confirmed. | Accepted draws can represent a validity-conditioned prior. | Stage 1 audit command now. | No learned validity classifier. |
| Optuna reports pruning after the training call. | Confirmed. | The pruner does not save epoch compute. | Stage 6 later. | No broad sweep. |
| GPU numeric parsing sat after an unconditional return. | Confirmed. | Historical numeric GPU distributions are invalid. | Stage 1 parser/tests now; fresh telemetry later. | No inferred A800 utilization. |
| SWIF stages transfer complete growing archives. | Confirmed. | Staging overhead can dominate and scale poorly. | Stage 6 delta migration later. | No archive rewrite without reconstruction tests. |
| Documentation described both registered families more uniformly than executable maturity justified. | Confirmed. | Users could mistake a scaffold for a production path. | Stage 7 maturity matrix and claims ledger now. | No model-independent claim. |

## Remaining blockers, stated narrowly

- The public CLI still trains the DD target by default. A production neural target requires a published schema-2 corpus with stored canonical GPD truth, a fitted decoder view, and an explicit CLI model-family dispatch. The present neural utilities and tests are a contract-complete slice, not a completed large neural training claim.
- No fresh native generation, full MAF retraining, Optuna campaign, GPU benchmark, or sealed final evaluation was run. This was intentional: the task prohibits expensive production work.
- Optuna pruning is still trial-level after training rather than connected to an epoch callback; it is not reported as functional epoch-wise pruning.
- The SWIF workflow still transfers large growing state archives. Archive-transfer redesign needs a separate workflow migration and storage measurement.
- Existing historical evaluation used `fixed_at_declared_truth`; a fresh generator-independent holdout realization must be materialized before OOD conclusions.

## Verification

Run the focused suite exactly as follows:

```bash
apptainer exec --bind "$PWD":/src .dvcs/extract-dvcs-gpd-jlab_ifarm-0.3.0.sif \
  env PYTHONPATH=/src/src MPLCONFIGDIR=/tmp/matplotlib \
  python3 -m unittest discover -v -s /src/tests -p test_scientific_contracts.py
```

The focused audit run passed 18/18 tests. No native physics call occurs in these tests.

The resolved DD baseline is [recorded separately](../../provenance/dd-baseline-2026-09-03.json); unknown historical source identity and broken historical GPU telemetry are preserved as unknown/invalid rather than guessed.
