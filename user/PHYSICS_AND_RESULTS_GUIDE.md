# Physics parameters and result interpretation

Read this after [WORKFLOW_AND_PHYSICS.md](WORKFLOW_AND_PHYSICS.md). It explains
what the files mean and what constitutes a trained, calibrated result.

## The 82 posterior coordinates

There are 80 physics coordinates: five shape controls for every combination
of H/E/Htilde/Etilde and u/d/s/gluon. The final two coordinates are global and
LU normalization nuisances. Every physics marginal appears in the matching
`posterior_marginals_<GPD>.png`; nuisance marginals appear in
`posterior_comparison.png`.

`normalization` controls the signed forward moment, `a` the small-x rise, `c`
the large-x falloff, `profile_b` the skewness profile, and `t_slope` the
exponential t dependence. Correlations matter: changing a and c together can
leave an observable nearly invariant, and evolution couples quark-singlet and
gluon directions. Do not read marginal peaks without
`posterior_correlations.png`.

Shadows are fixed simulator settings in this release. No posterior plot
should label them inferred. H/u is the one installed native shadow; the other
directions are project stress tests.

## Should peaks equal injected truth?

For an identifiable direction and a calibrated estimator, the posterior mass
should be centered consistently with the injected truth—but one noisy dataset
does not require its mode to equal truth. The decisive check is repeated
coverage: nominal 90% credible regions should contain truth about 90% of the
time within statistical uncertainty.

A broad marginal containing truth can be the correct answer when the dataset
does not identify that direction. A narrow wrong peak is worse: it indicates
false certainty. With 82 dimensions and only 36 observable tokens, many broad
or correlated directions are physically plausible unless the campaign and
kinematics are highly informative.

## “Neural” and “conventional”

- Neural: samples from the trained conditional normalizing flow
  \(q_\phi(\theta,\eta\mid D)\).
- Conventional: the exact declared Gaussian likelihood evaluated on the
  exact-native DD prior bank, with self-normalized importance weights.

They target the same synthetic posterior but approximate it differently.
Some retained documentation writes the same density as
\(q_\varphi(\theta,\eta\mid D)\); \(\phi\) and \(\varphi\) are only notation
for learned network parameters, not the DVCS azimuth.
Agreement supports the NPE only when conventional effective sample size is
adequate. In 82 dimensions, importance weights may collapse; an ESS failure
means the reference is inconclusive and calls for a stronger MCMC/nested
sampling campaign, not that either curve wins.

## Dataset roles

The same six kinematic sites are used for every DD parameter draw; train,
internal-validation, and outer-test are splits of native parameter groups,
not kinematic splits. `dd_train_validation_test_split.png` visualizes their
membership in display PCA. `real_catalog_vs_pseudodata_kinematics.png` shows
how the six sites sit within measurement-free real-catalog coverage.

The fresh GK/VGG slice is different: its six sites were predeclared from the
catalog before native model outputs existed. It is an external test, never a
replacement for the DD outer test. Do not repeatedly modify the method based
on this fixed holdout and continue calling it blind.

## Training scores

`training_validation_nll.png` plots negative log posterior density. Lower is
better only for comparable data/contracts. A falling train curve with rising
validation curve indicates overfitting. Similar curves do not establish
calibration; they only show optimization behavior. Inspect member-by-member
metrics and device provenance in `training/training_metrics.json`.

The default architecture is deliberately unchanged. `optimize` is a separate
user action. Its validation objective cannot see outer-test or named-model
scores, and `best_parameters.json` is advisory—it does not rewrite the study.

## Observable, CFF, and GPD plots

Run `./dvcs plot PROJECT --profile PROFILE` after the stepwise `compare`
action. `plots/plot_manifest.json` identifies every input and output by hash
and separately retains evaluation/comparison pass states. Plotting success is
only presentation success.

`posterior_predictive.png` has one subplot per native observable. Points are
the noisy pseudodata with marginal error bars; dashed curves are injected
exact-native means; bands/medians are posterior samples reevaluated through
PARTONS. Good appearance is not a gate by itself: consult numerical coverage,
full-covariance residual, and invalid-sample metrics.

`cff_real_imaginary.png` separates real and imaginary parts of all four CFFs.
`gpd_*` plots show exact-native reevaluations at the declared reference
kinematics. The network predicts parameter densities; it is not directly
predicting these curves. u/d/s plots include value, C-even plus, and C-odd
minus components; gluon uses the PARTONS/APFEL input convention.

## Required green checks before external validation

### What a `pass` means

A pass means every predeclared numerical gate for that campaign succeeded; it
does not mean unique physics or real-data readiness. In particular, the
beam-spin-difference token is `DVCSCrossSectionDifferenceLUMinus`. Its passing
predictive panel cannot supply the missing aggregate analysis-order label,
twist label, or experimental convention audit.

- zero native fallback/imputation and an accepted invalid fraction;
- byte-stable cache/restart hashes;
- disjoint grouped roles;
- held-out NLL generalization;
- empirical coverage within its declared standard error;
- posterior-predictive point and full-covariance checks;
- adequate conventional ESS and accepted neural/conventional distances;
- zero invalid exact posterior reevaluations;
- no false-certainty width failure.

After these pass, run `holdout` once for the first accepted fresh campaign.
Report execution integrity separately from predictive precision. A low
GK/VGG coverage is a robustness failure requiring later method development,
not a broken executable.

## Common failure diagnoses

- Neural curves look prior-like: too few independent native draws for 80
  dimensions, weak observables, or genuinely unidentifiable directions.
- Train improves but validation does not: overfitting or insufficient
  distribution coverage.
- Conventional curves are spiky: low ESS/weight collapse.
- Posterior is narrow and misses truth: false certainty; stop.
- Native invalid map is nonzero: inspect retained parameter/error records;
  nothing was imputed.
- CUDA shows no activity during generation: expected, because PARTONS is CPU.
  Check CUDA during `train`, and inspect recorded `neural_device`/peak memory.
- PARTONS CPU use is bounded by both the number of ready native batches and
  the affinity-visible CPUs. With `native_workers: "all_available"`, `doctor`
  reports the resolved count for the current allocation.
- Progress advances after each completed two-parameter native batch and is
  redrawn immediately. The bar count is accepted parameters; its postfix
  distinguishes cumulative attempted, accepted, and rejected draws. It cannot
  advance inside one atomic bridge call, so a slow pair may still cause a
  short pause.
- Evaluation similarly parallelizes both `Exact posterior reevaluation` and
  `Exact GPD diagnostics`. The first validation default processes 64 posterior
  samples at every kinematic point; the second processes injected truth plus
  the valid samples over the diagnostic x-grid. Their counters are sample
  counters, while each native request contains all required kinematics for up
  to two samples.
- `unphysical_fixed_target_kinematics` means the coupled
  $0<Q^2/(2M_pEx_B)<1$ condition failed. Do not resume that older
  project; create a new selector-v2 project.

## What the release does not establish

It does not establish physical positivity, proton form-factor sum rules,
heavy flavors, a D-term or Etilde pion pole, evolved shadow nullity,
representation independence, unique GPD extraction, an aggregate LO/NLO or
twist label, GK/VGG robustness, agreement with literature, or a real-data
posterior. Real fitting remains disabled until the user reviews the complete
82-dimensional synthetic and fresh external-validation evidence.
