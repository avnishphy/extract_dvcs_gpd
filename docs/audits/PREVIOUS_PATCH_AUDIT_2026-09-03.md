# Previous uncommitted patch audit

The patch found on `main` at `22f45e94d424e2a73c57d6b2504869a2eebb35f4`
was preserved, inspected, tested, and committed on
`codex/physics-production-readiness`. This table distinguishes implemented
contracts from live production coverage. “Standalone” means callable and
tested, but not invoked by the ordinary `selection` through `plot` workflow.

| Reported change | Actual files/functions | Test evidence | Production-path coverage | Remaining risk |
|---|---|---|---|---|
| Deterministic validation | `inference/stage10.py`: seeded validation loader/generator handling | Full static suite; focused scientific-contract tests | Live DD training path | Framework determinism still depends on pinned Torch/CUDA/runtime and deterministic-kernel support. |
| Repaired telemetry parsing | `jobs/jlab_ifarm/collect_performance_metrics.py`: tolerant numeric parsing and retained raw values | JLab progress/SWIF wrapper tests | Live farm telemetry path | Historical records lack queue wait and some GPU fields; they must remain missing, not invented. |
| Factorized uncertainty contracts | `data/uncertainty.py`: `UncertaintyContract`, observation and variant materializers | `test_scientific_contracts.py` | Standalone; the legacy production materializer still uses its established covariance builder | A production migration would need identity/version changes and equivalence tests. |
| Covariance policies | `data/uncertainty.py`; `docs/UNCERTAINTY_CONTRACTS.md` | Unit tests plus current 576-dimensional native covariance audit | Policy vocabulary is standalone; current production covariance semantics are live in `workflows/pseudodata.py` | Do not assume the new policy object governs the current campaign merely because it exists. |
| Generator metadata | `corpus.py`: schema-v2 family and representation metadata | Architecture/scientific contract tests | Live for newly created schema-v2 corpora | Existing `partons-validation-16384` is schema v1 and cannot be relabelled. |
| Best-DD projection artifacts | `scientific_validation.py`: projection record validation | Unit contract tests | Standalone; no GK/VGG projection campaign has run | No representation-floor numbers exist yet. |
| Native-failure diagnostics | `scientific_validation.py:audit_native_failures`; CLI wiring in `cli/user.py` | Unit/CLI coverage in scientific tests | Callable from the user CLI against real corpora | Production corpus does not exist, so no failure map exists for it. |
| Evaluation roles and production provenance | `scientific_validation.py`: role comparison, provenance validation, sealing/unsealing | Unit contract tests | Mostly standalone; ordinary evaluation does not yet call every validator/seal transition | A valid contract can still be bypassed until wired at each publish boundary. |
| Neural row-to-group alignment | `neural_gpd.py`: explicit row/group metadata and validation | Neural contract tests | Standalone neural model-view publisher | No schema-v2 production function corpus/model view exists. |
| Neural latent whitening | `neural_gpd.py`: fitted whitening/inverse transform metadata | Neural contract tests | Standalone autoencoder/model-view utilities | No production autoencoder, whitening artifact, or decoded posterior exists. |
| Explicit model-family dispatch | `model_registry.py`; CLI `--model-family`; stage capability gate | Unit tests and both DD/neural synthetic family smokes | Live fail-closed CLI gate; DD is implemented, neural production stages are rejected | The neural smoke is synthetic and is not a vertical production implementation. |
| Synthetic DD and neural CLI smokes | `inference/family_smoke.py` | Both smokes passed with finite posteriors | Diagnostic only | No native generation, exact covariance, GPU, or SWIF staging is exercised. |
| Literature registry/scaffold | `literature.py`, comparator registry, Moffat partial record, audit docs | Static registry/documentation tests | Supporting asset only | No reproduction was run; benchmark claims remain scoped as metadata/scaffolding. |

## Commit inventory

- `9040378` — deterministic validation and telemetry correctness (2 files).
- `3f7e765` — uncertainty/covariance and scientific-validation contracts (4 files).
- `a35ddfe` — model-family dispatch, neural contracts, smoke paths, and tests (8 files).
- `c0ba0f4` — evidence, claims, limitations, and literature scaffolding (12 files).

No implicit DD fallback was found in model-family dispatch: unsupported neural
production stages fail at the capability gate. No user-approved kinematics or
native physics files were changed by these four commits. The chief audit result
is that several scientifically useful APIs are test-covered but are not yet
production publish-path gates.
