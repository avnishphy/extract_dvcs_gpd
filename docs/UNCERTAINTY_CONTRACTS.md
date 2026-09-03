# Uncertainty contracts

An uncertainty realization changes statistical treatment, never native
physics. `materialize_uncertainty_variant` consumes stored central predictions,
a declared covariance source, and paired standard-normal draws. Its immutable
manifest retains corpus/selection ancestry and records that PARTONS was not
called.

The independent axes are `observation_realization` (`central` or `sampled`),
`uncertainty_model` (`full_covariance`, `diagonal`, `fixed_relative`, or
`equal_weight_point_fit`), `uncertainty_exposure`, `nuisance_policy`,
`physical_noise_scale`, `numerical_jitter`, `output_kind`, and
`covariance_policy` (`training_reference`, `external_fixed`, or
`holdout_truth_derived`). Generator identity is not a covariance selector.

| Preset | Meaning |
|---|---|
| `full_uncertainty_posterior` | sampled full covariance, exposed descriptors, inferred nuisances, posterior |
| `central_observation_full_likelihood` | exact central observation while retaining the full likelihood contract |
| `literature_fixed_relative_replica` | configurable independent relative Gaussian replicas; the 10% scaffold is a point/replica comparator |
| `near_noiseless_posterior` | positive reduced full covariance; configure paired scales such as 1, 0.5, 0.25, 0.1 |
| `uncertainty_blind_point_benchmark` | exact central values, no descriptors, equal-weight point estimate, no posterior claim |

The physical covariance is
`physical_noise_scale**2 * reference_covariance`. Numerical jitter is added
only to the separately named stabilized covariance. A zero-noise ordinary
full-dimensional MAF posterior is rejected; the point benchmark is the valid
deterministic alternative. Central evaluation may reuse a compatible full-
uncertainty checkpoint because changing the evaluated observation alone does
not change the training distribution.
