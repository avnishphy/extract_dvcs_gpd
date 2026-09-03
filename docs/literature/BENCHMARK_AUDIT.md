# Literature benchmark audit

The source of truth is [`benchmarks/literature/comparators_v1.json`](../../benchmarks/literature/comparators_v1.json). Every record names the inferred object, generator/model family, representation, D-term status, observable/input level, kinematics/evolution, parameterization, uncertainty/covariance/nuisance assumptions, splits, optimization, metrics, bands, calibration/holdout status, code availability, unresolved facts, and an explicit comparability grade. `null` never means zero or absent; it means the audited source did not establish the value.

## Comparison policy

Only like-for-like quantities may share a numerical table. Raw-DVCS-to-DD posterior inference, raw-DVCS-to-CFF inverse maps, aggregate-CFF/Mellin neural GPD constraints, and global experimental/lattice fits are different inference objects. Conceptual comparators may motivate plots or validation checks but do not constitute reproduced numerical baselines.

The closest accessible parametric closure is Moffat's 2024 presentation: GK pseudodata, 31 fitted controls, the first three Gegenbauer D-term terms, independent 10% Gaussian replicas, Adam, and replica mean/standard-deviation reporting. Replica count, exact bounds, full kinematic/observable tables, correlations, split, and stopping rule remain unresolved; its grade is therefore `partial`, not `exact`. The linked 2026 ECT* deck could not be retrieved during the audit, so all method fields remain unresolved and its grade is `not_comparable`.

NNGPD is a direct neural function representation constrained by aggregate CFF and Mellin/form-factor information from a single UVA2 model family; it is not raw-DVCS amortized posterior inference and does not expose a common internal-DD truth vector. The Dutrieux neural-DD paper, VAIM/C-VAIM inverse maps, GUMP global extraction, Watkins extraction, and shadow-GPD studies likewise receive only the comparison level their published inference object supports.

## Reproducibility statuses

- `exact`: all relevant contracts and numerical data are available and matched.
- `high`: small, declared differences do not change the scientific object.
- `partial`: some numerical slices are reproducible, with material unresolved protocol details.
- `conceptual`: useful scientific comparator, but the inference object or data contract differs.
- `not_comparable`: evidence is unavailable or the required contract cannot be established.

At this HEAD no paper is claimed as an exact reproduced baseline. No plotted pixels, values digitized from figures, guessed priors, or invented uncertainties are permitted. New numerical overlays require convention checks for GPD/flavor combination, normalization, scale, scheme, perturbative order, and sign.

## Validation requirements inherited from the audit

Posterior claims require SBC rank diagnostics, coverage-versus-nominal curves, posterior predictive checks, calibration-error summaries, and at least one local two-sample diagnostic such as L-C2ST or a coverage diagnostic such as TARP. NLL remains useful for optimization but is not sufficient evidence of calibration. Point-estimate comparators must be labelled as such and cannot be promoted to posterior bands by adding numerical jitter.

## Publication figures that remain required

Saved-artifact-only plotting should eventually generate: DD and neural function closure; CFF closure; observable posterior predictive checks; same-family interpolation/extrapolation; named-family holdout; cross-representation holdout; best-DD projection floor with restart spread; coverage/calibration; and uncertainty ablations. Every panel must name the uncertainty contract, generator class/family, evaluation role, model family, transform identity, and whether the target family was seen in training.
