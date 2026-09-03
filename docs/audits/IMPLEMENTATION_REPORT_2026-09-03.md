# Implementation report (2026-09-03)

## Repository state

- Authoritative HEAD before and after work: `22f45e94d424e2a73c57d6b2504869a2eebb35f4`, branch `main`.
- Initial state: clean. Final state: uncommitted audit implementation; no commit was created.
- No files were moved or deleted. No production corpus, training campaign, sweep, SWIF workflow, or GPU job was launched.
- Historical DD baseline: [`provenance/dd-baseline-2026-09-03.json`](../../provenance/dd-baseline-2026-09-03.json).

## File ledger

| File | Change and reason | Scientific behavior | Verification |
|---|---|---|---|
| `src/extract_dvcs_cff/inference/stage10.py` | Deterministic sequential validation, retain partial batch, exact row-count denominator. | Changes future early stopping/validation NLD when validation rows do not divide the batch size. Training remains shuffled. | 2/2/1 regression; both CLI smokes; static suite. |
| `jobs/jlab_ifarm/collect_performance_metrics.py` | Repair unreachable finite numeric parser and accept declared units. | Telemetry only; historical broken GPU numbers remain invalid. | Parser table regression; JLab wrapper checks. |
| `src/extract_dvcs_cff/data/uncertainty.py` | New factorized contracts, presets, covariance policies, paired realization and immutable shared-corpus publication. | Adds versioned statistical alternatives; never calls or changes native physics. | Scaling, central, fixed-relative, zero-noise, covariance independence, ancestry/idempotence tests. |
| `src/extract_dvcs_cff/scientific_validation.py` | Generator hierarchy, holdout labels, metric guard, bounded multi-start projection artifact, provenance gate, evaluation-role/unseal ledger, rejection audit. | Adds gates/diagnostics; does not reinterpret legacy artifacts. | Generator, projection, provenance, sealing, native-rejection fixture tests. |
| `src/extract_dvcs_cff/corpus.py` | Add hierarchical generator metadata to new schema-2 corpus identity. | New corpus IDs include stronger generator semantics; stored physics is unchanged. | Static corpus suite. |
| `src/extract_dvcs_cff/neural_gpd.py` | Typed one-hot coordinate identity, lazy row/group alignment, training-only full-covariance whitening, collapsed-dimension removal; neural views carry `parameter_indices` and whitening. | Changes newly published neural model views/targets only; no completed production neural result is claimed. | Alignment, coordinate, whitening and collapse tests; neural CLI smoke. |
| `src/extract_dvcs_cff/model_registry.py` | Stage maturity and fail-closed family/stage resolver. | Prevents neural requests from silently using DD stages. DD remains default. | Dispatch regression and both family smokes. |
| `src/extract_dvcs_cff/inference/family_smoke.py` | Tiny grouped synthetic DD/neural DeepSets+MAF train/sample path. | Software mechanics only; no native or scientific benchmark behavior. | Two direct CLI runs. |
| `src/extract_dvcs_cff/cli/user.py` | Add explicit family options, `model-smoke`, and `corpus-audit-failures`. | DD backward-compatible default; unsupported neural production stages fail early. | CLI smokes, parser/documentation/static suite. |
| `src/extract_dvcs_cff/literature.py` | Strict comparator-registry loader. | Validation only; does not create numerical literature results. | Registry completeness test. |
| `tests/test_scientific_contracts.py` | Add 18 focused scientific-contract tests. | Test-only. | 18/18 pass in pinned image. |
| `tests/run.sh` | Include the new test module in static acceptance. | Test orchestration only. | Complete static run, 61/61 pass. |
| `benchmarks/literature/comparators_v1.json` | Ten-source machine-readable audit with explicit null/unresolved fields and grades. | Benchmark metadata only. | JSON and strict schema validation. |
| `benchmarks/literature/moffat_qgt_2024_partial_v1.json` | Partial reproduction/export scaffold; blocks guessed settings or native generation. | No fit or corpus generated. | JSON validation. |
| `provenance/dd-baseline-2026-09-03.json` | Resolve archived DD-v7 configuration/member hashes, dependencies, architecture, seeds, split, checkpoints, runtime and limitations. | Historical evidence only. | JSON validation and direct archive-member/hash inspection. |
| `docs/audits/CURRENT_HEAD_AUDIT.md` | Current environment, prior-review reconciliation, implications, stages/non-goals, remaining blockers. | Documentation only. | Documentation checker. |
| `docs/audits/IMPLEMENTATION_REPORT_2026-09-03.md` | This final evidence ledger. | Documentation only. | Link/diff checks. |
| `docs/literature/BENCHMARK_AUDIT.md` | Comparison policy, grades, NNGPD/Moffat distinctions and required validation/figures. | Documentation only. | Documentation checker. |
| `docs/UNCERTAINTY_CONTRACTS.md` | Define axes, presets, covariance scaling, jitter and point/posterior semantics. | Documentation only. | Documentation checker. |
| `docs/CLAIMS_LEDGER.md` | Evidence-gated scientific-claim status. | Documentation only. | Documentation checker. |
| `README.md` | Add model maturity matrix and audit links. | Documentation only. | Documentation checker. |
| `docs/ARCHITECTURE.md` | Record dispatch, uncertainty ancestry, neural alignment/coordinate/whitening boundaries. | Documentation only. | Documentation checker. |
| `docs/CLI_REFERENCE.md` | Document new commands and maturity behavior. | Documentation only. | Documentation checker. |
| `docs/DEVELOPMENT_AND_TESTING.md` | Document focused suite coverage. | Documentation only. | Documentation checker. |
| `docs/INDEX.md` | Link new canonical pages. | Documentation only. | Documentation checker. |
| `docs/KNOWN_LIMITATIONS.md` | Correct neural and container maturity, invalid historical telemetry, outstanding work. | Documentation only. | Documentation checker. |

## Executed verification

| Command | Exit | Result/raw-log location |
|---|---:|---|
| `python3 -m json.tool` on all three new JSON authorities | 0 | Passed; output intentionally suppressed. |
| Pinned-image focused unittest discovery | 0 | 18 tests passed in 4.827 s; raw output is retained in the Codex tool transcript, not written to the repository. |
| Pinned-image `model-smoke --model-family dd_deepsets_maf` | 0 | Finite 3-D posterior samples; no native call; raw output in tool transcript. |
| Pinned-image `model-smoke --model-family neural_gpd_deepsets_maf` | 0 | Finite 2-D posterior samples; no native call; raw output in tool transcript. |
| Pinned-image `bash tests/run.sh static` with writable `/tmp` and read-only host `rg` bind | 0 | Distribution/docs checks, 61 tests, all JLab fixture checks, shell/installer checks passed; raw output in tool transcript. |
| Pinned-image `bash tests/run.sh native` with writable temporary `/cache` | 0 | Bridge capabilities and self-test passed exactly; raw output in tool transcript. |
| `git diff --check` | 0 | No whitespace errors. |

Two environmental attempts failed before the final passing commands: static
verification first inherited a nonexistent `/scratch/singhav` TMPDIR and then
found no `rg` in the image; native self-test first lacked a writable PARTONS
log directory. These are not reported as passing attempts. No standalone raw
log files were created; the exact raw stdout/stderr is in the session tool
transcript.

Not run: the quick test because it generates a native corpus; production/DD or
neural training; exact posterior reevaluation; Optuna; SWIF submission; GPU
smoke; calibration/OOD/closure campaigns. They require compute, a suitable
completed corpus/artifact set, or explicit authorization.

## Benchmark evidence

Before: historical DD-v7 mean train NLD 112.76232966997266 and mean test NLD
113.91078092578759, with five checkpoint records. The run predates the
partial-batch fix and has invalid GPU numeric telemetry. After: no comparable
scientific benchmark was run, so there is no before/after accuracy or speed
claim. The only new measured executions are mechanics tests/smokes.

## Schemas and compatibility

- Added uncertainty-realization schema 1, literature-comparator registry schema
  1, projection artifact schema 1, and baseline manifest schema 1.
- New schema-2 corpus identities gain additive `generator_metadata`; this
  intentionally changes IDs for newly created corpora but does not rewrite old
  corpora.
- Neural model-view schema 1 gains `parameter_indices`, alignment, coordinate,
  and whitening semantics. No production neural view is migrated implicitly.
- Existing DD CLI invocations remain valid because DD is the explicit default.
  Neural requests at unsupported stages now fail instead of falling through.
- Uncertainty/architecture variants prove corpus reuse by retaining identical
  `master_corpus_id` and `selection_id`, changing only child realization/view
  identity, and recording `native_physics_executed: false`.

## Literature grades and assumptions

The exact per-source grades and unresolved fields live in the registry. The
2024 Moffat material is `partial`; the inaccessible 2026 deck is
`not_comparable`; different-object neural/global/shadow studies are principally
`conceptual`. No source is claimed as exactly reproduced. Assumptions are
limited to explicit framework defaults (DD backward-compatible CLI default,
positive-noise MAF posterior, exact stored-coordinate decoding); missing paper
settings remain null. The prompt's Bertone arXiv identifier did not resolve to
the relevant shadow paper found by the audit, and the discrepancy is retained.

## Remaining blockers to claims

- DD literature non-inferiority: exact comparator protocol/code, declared
  margin/resampling unit, common benchmark, at least three paired seeds, and an
  untouched final asset.
- Calibrated within-family posterior: fresh post-fix training plus SBC,
  coverage/sharpness, TARP, L-C2ST, PPC, contraction and seed sensitivity.
- Named-family generalization: versioned foreign generator artifacts,
  generator-independent covariance, deterministic DD projection floors,
  multiple seeds, and frozen development/final roles.
- Cross-representation generalization: at least three distinct families under
  one function contract, train-only transforms, leave-one-family-out training,
  and decoder-floor-aware metrics.
- Real-data extraction: audited measurement mappings and covariance/likelihood,
  nuisance/systematic treatment, physics-order scope appropriate to the claim,
  blind validation, and publication traceability.

High-value work deliberately deferred: serious minibatch/resumable
autoencoder, production neural NPE dispatch, full checkpoints, epoch-level
Optuna pruning, calibration/proper scores, inverse-crime closure, simultaneous
bands, SWIF state deltas, and measured GPU/HPC optimization. Implementing these
without the prerequisite corpora, physics-owner decisions, or authorized
compute would create unsupported evidence rather than a defensible result.
