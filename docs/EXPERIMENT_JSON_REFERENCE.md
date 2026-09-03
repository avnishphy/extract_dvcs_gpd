# `experiment.json` reference

This is the complete field reference for the public, user-editable
`experiment.json` **schema 9** created by:

```bash
./dvcs init NAME
```

It describes the current full-independent DD, multi-$Q^2$ synthetic workflow:
80 inferred GPD-shape controls plus two inferred nuisance parameters. It does
not describe the private `.engine/` files or authorize real-data fitting.

## How to use this reference

Paths use these placeholders:

- `{GPD}` is one of `H`, `E`, `Htilde`, or `Etilde`;
- `{channel}` is one of `u`, `d`, `s`, or `gluon`;
- `{profile}` is `quick` or `validation`;
- `[]` means every element of a JSON array.

Edit classifications used below are:

| Classification | Meaning |
|---|---|
| **Experiment control** | Intended to be changed for a new synthetic study. |
| **Compute control** | Intended to be changed to trade cost against statistical precision. |
| **Advanced method control** | Supported by the current implementation, but changing it defines a different training method that needs fresh validation. |
| **Generated provenance** | Written by `init`; do not hand-edit. Use manual provenance when changing kinematics. |
| **Frozen identifier** | Must retain the documented value for this release. |
| **Validation policy** | A gate threshold, not a tuning target. Changing it makes results incomparable with the documented release campaign. |

Any change to `experiment.json` after results exist intentionally breaks the
workspace content contract. Create a new project name and regenerate. Unknown,
missing, or misspelled keys fail closed.

## 1. Document identity

| Path | Type/default | Meaning and edit policy |
|---|---|---|
| `schema_version` | integer, `8` | **Frozen identifier.** Adds explicit selectable-observable metadata to the full-independent multi-$Q^2$ schema. Schema 7 remains readable with the canonical six-observable list; schemas 2–6 are retired. |
| `experiment.name` | string, project name | **Frozen identity.** Must exactly equal the directory name below the configured workspace root. |
| `experiment.description` | string | **Experiment control.** Free-form human description; it does not enter simulation or inference. |

## 2. Injected GPD truth

### 2.1 Repeated DD shape blocks

For every `{GPD}` × `{channel}` combination, the object
`injected_truth.gpd_parameters.{GPD}.{channel}` contains exactly these five
entries:

| Path suffix | Type/unit | Enforced support | Physics meaning |
|---|---|---|---|
| `normalization` | finite number | $-4<N<4$ | Signed forward zeroth-moment control $N_{F,p}$. |
| `a` | finite number | $0.1<a<1$ | Small-$\beta$ exponent in $\beta^a$. |
| `c` | finite number | $2<c<6$ | Large-$\beta$ exponent in $(1-\beta)^c$. |
| `profile_b` | finite number | $1<b<4$ | Double-distribution skewness-profile exponent. |
| `t_slope_GeV_minus2` | finite number, GeV$^{-2}$ | $0<B<2$ | Exponential $t$ slope in $e^{Bt}$. |

Thus the five machine paths are
`injected_truth.gpd_parameters.{GPD}.{channel}.normalization`,
`injected_truth.gpd_parameters.{GPD}.{channel}.a`,
`injected_truth.gpd_parameters.{GPD}.{channel}.c`,
`injected_truth.gpd_parameters.{GPD}.{channel}.profile_b`, and
`injected_truth.gpd_parameters.{GPD}.{channel}.t_slope_GeV_minus2`.

These are **experiment controls**. They define the one injected truth used to
make the displayed pseudodataset. The training corpus still samples the full
declared uniform supports; editing the injection does not narrow the prior.

Generated defaults are:

| GPD/channel | `normalization` | `a` | `c` | `profile_b` | `t_slope_GeV_minus2` |
|---|---:|---:|---:|---:|---:|
| H/u | 2.00 | 0.55 | 3.8 | 1.2 | 0.8 |
| H/d | 1.00 | 0.60 | 4.0 | 1.2 | 0.8 |
| H/s | 0.25 | 0.65 | 5.0 | 1.5 | 0.8 |
| H/gluon | 1.50 | 0.35 | 5.0 | 1.2 | 0.8 |
| E/u | 1.00 | 0.70 | 4.2 | 1.5 | 1.0 |
| E/d | -1.20 | 0.70 | 4.5 | 1.5 | 1.0 |
| E/s | -0.10 | 0.70 | 5.2 | 2.0 | 1.0 |
| E/gluon | 0.30 | 0.40 | 5.0 | 1.5 | 1.0 |
| Htilde/u | 1.27 | 0.45 | 3.2 | 1.2 | 0.7 |
| Htilde/d | -0.40 | 0.50 | 3.8 | 1.2 | 0.7 |
| Htilde/s | -0.10 | 0.55 | 4.5 | 1.5 | 0.7 |
| Htilde/gluon | 0.50 | 0.40 | 4.5 | 1.2 | 0.7 |
| Etilde/u | 0.80 | 0.60 | 4.5 | 2.0 | 1.2 |
| Etilde/d | -0.30 | 0.65 | 4.8 | 2.0 | 1.2 |
| Etilde/s | 0.05 | 0.70 | 5.2 | 2.5 | 1.2 |
| Etilde/gluon | 0.20 | 0.45 | 5.0 | 2.0 | 1.2 |

The parameterization and conventions are derived in
[Workflow and physics](WORKFLOW_AND_PHYSICS.md).

### 2.2 Fixed shadow/stress settings

| Path | Type/default | Enforced range | Meaning |
|---|---|---|---|
| `injected_truth.gpd_parameters.shadow_coefficients.{GPD}` | number; H `0.6`, E `-0.35`, Htilde `0.4`, Etilde `-0.2` | $[-1,1]$ | Type-level multiplier for the fixed shadow/stress direction. |
| `injected_truth.gpd_parameters.shadow_channel_amplitudes.{GPD}.{channel}` | number; u `1`, d `0.5`, s `0.25`, gluon `1` for every GPD | $[-4,4]$ | Distributes the type-level multiplier across parton channels. |

Both are **experiment controls**, but they are fixed simulator inputs rather
than inferred coordinates. Changing either defines a new simulator family.
Only H/u uses the installed `GPDBDMMS21` direction; the other entries are
project stress directions and are not claimed to be native or CFF-null.

### 2.3 Injected nuisance truth

| Path | Type/default | Meaning |
|---|---|---|
| `injected_truth.nuisance_parameters.eta_global_normalization` | finite number, `0.35` | Injected standard-normal coordinate for the multiplicative normalization response applied to all observables. |
| `injected_truth.nuisance_parameters.eta_lu_normalization` | finite number, `-0.25` | Injected standard-normal coordinate applied only to `DVCSCrossSectionDifferenceLUMinus`. |

These are **experiment controls** and are the final two inferred coordinates.
They are measured in prior standard deviations, not fractional shifts. The
fractional responses are specified separately in the uncertainty model.

## 3. Synthetic dataset

### 3.1 Kinematic points

`synthetic_dataset.kinematics` is a nonempty array of user-selected sites. Each
point is evaluated for every selected native observable. The default six
observables therefore turn six kinematic points into 36 DeepSets tokens.

| Path | Type/unit | Enforced range | Meaning |
|---|---|---|---|
| `synthetic_dataset.kinematics[].x_b` | finite number | $(0,1)$ | Bjorken $x_B$ selected by the user in `visualize_gpddatabase_dvcs.ipynb`. |
| `synthetic_dataset.kinematics[].t_GeV2` | finite number, GeV$^2$ | exact coupled DVCS limits | Momentum transfer $t=(p'-p)^2$; no historical fixed envelope is imposed. |
| `synthetic_dataset.kinematics[].Q2_GeV2` | finite number, GeV$^2$ | $[1,\infty)$ | Positive photon virtuality at or above native input scale $Q_0^2=1\,\mathrm{GeV}^2$. |
| `synthetic_dataset.kinematics[].beam_energy_GeV` | finite number, GeV | $(0,\infty)$ | Incident lepton energy used by the native observable. |
| `synthetic_dataset.kinematics[].phi_rad` | finite number, radians | $[0,2\pi]$ | Lepton–hadron plane azimuth. |

Notebook-selected points are authoritative. Public loader and native bridge
reject only non-finite values, native scale violations, or coupled physical
violations. Every point must satisfy

$$
0 < y = \frac{Q^2}{2M_p E x_B} < 1,
\qquad M_p = 0.938272013\ \mathrm{GeV}.
$$

It must also lie between exact finite-$Q^2$ forward and backward DVCS transfer
limits

$$
t_{\rm backward}\le t\le t_{\rm forward},\qquad
t_{\rm forward/backward}=-Q^2
\frac{2(1-x_B)(1\mp\sqrt{1+\epsilon^2})+\epsilon^2}
{4x_B(1-x_B)+\epsilon^2},\quad
\epsilon^2=\frac{4M_p^2x_B^2}{Q^2}.
$$

The public loader and native bridge both reject a point outside this domain
before simulation. This coupled constraint matters when editing $Q^2$, beam
energy, or $x_B$: each field can be inside its individual range while their
combination is impossible.

The generated values come from selected real-catalog kinematics, not from
measured observables. To edit any generated point, replace the entire
`kinematics_source` object with `{"mode": "manual"}` first. Data are never
evolved.

### 3.2 Kinematic provenance

`synthetic_dataset.kinematics_source.mode` accepts exactly:

- `gpddatabase_native_scale_kinematics_v2`: generated, revision-pinned,
  physical-$y$-filtered
  provenance; all fields below are required and revalidated; or
- `manual`: the object must contain only `mode`, and the user owns the
  kinematic choices.

Legacy `v1` projects remain readable only so the validator can give an exact
physical-domain error. Do not resume a rejected `v1` generation; create a new
project to receive the corrected deterministic `v2` selection.

Database-backed mode contains:

| Path | Type | Meaning/edit policy |
|---|---|---|
| `synthetic_dataset.kinematics_source.database_root` | absolute path string | **Generated provenance.** Read-only protected database location. |
| `synthetic_dataset.kinematics_source.database_revision` | Git revision string | **Generated provenance.** Required revision; mismatch aborts. |
| `synthetic_dataset.kinematics_source.catalog_sha256` | SHA-256 string | **Generated provenance.** Hash of the measurement-free catalog; mismatch aborts. |
| `synthetic_dataset.kinematics_source.selection_metric` | string | **Generated provenance.** Deterministic selection algorithm identifier. |
| `synthetic_dataset.kinematics_source.selected_records` | array | **Generated provenance.** One record for each kinematic point, in identical order. |
| `synthetic_dataset.kinematics_source.measurements_quarantined` | Boolean, `true` | **Frozen safety assertion.** Measured values and uncertainties did not enter pseudodata generation. |
| `synthetic_dataset.kinematics_source.data_evolved` | Boolean, `false` | **Frozen physics assertion.** Experimental data are not evolved. |
| `synthetic_dataset.kinematics_source.gpd_evolved_to_each_datum_Q2` | Boolean, `true` | **Frozen physics assertion.** Input GPDs are evolved to datum scales. |

Each selected record contains:

| Path | Meaning |
|---|---|
| `synthetic_dataset.kinematics_source.selected_records[].record_id` | Unique catalog row identifier. |
| `synthetic_dataset.kinematics_source.selected_records[].dataset_id` | Source dataset identifier. |
| `synthetic_dataset.kinematics_source.selected_records[].source_uuid` | Source database UUID. |
| `synthetic_dataset.kinematics_source.selected_records[].source_path` | Relative source YAML path. |
| `synthetic_dataset.kinematics_source.selected_records[].collaboration` | Experimental collaboration label. |
| `synthetic_dataset.kinematics_source.selected_records[].reference` | Source literature/reference string. |
| `synthetic_dataset.kinematics_source.selected_records[].source_Q2_GeV2` | Catalog $Q^2$ in GeV$^2$. |
| `synthetic_dataset.kinematics_source.selected_records[].source_beam_energy_GeV` | Catalog beam energy in GeV. |
| `synthetic_dataset.kinematics_source.selected_records[].generated_Q2_GeV2` | $Q^2$ actually sent to PARTONS; schemas 8/9 retain the source value. |
| `synthetic_dataset.kinematics_source.selected_records[].Q2_projected` | Boolean projection flag; generated schemas 8/9 project no $Q^2$. |
| `synthetic_dataset.kinematics_source.selected_records[].generated_beam_energy_GeV` | Beam energy actually sent to PARTONS; schemas 8/9 retain the source value. |
| `synthetic_dataset.kinematics_source.selected_records[].measurement_values_used` | Must be `false`; any other value aborts. |

### 3.3 Native observable selection

`synthetic_dataset.observables` is a nonempty array in canonical order. It may
contain any ordered subset of the six audited entries below. Unknown,
duplicated, reordered, or metadata-altered entries fail before native work.
The default contains all six.

| Path | Type | Meaning/edit policy |
|---|---|---|
| `synthetic_dataset.observables[].id` | native module ID | **Experiment control.** One of `DVCSCrossSectionUUMinus`, `DVCSCrossSectionDifferenceLUMinus`, `DVCSAc`, `DVCSAluMinus`, `DVCSAulMinus`, or `DVCSAllMinus`. |
| `synthetic_dataset.observables[].native_unit` | `nb/GeV4` or `1` | **Frozen metadata.** Cross sections use nb/GeV$^4$; asymmetries are dimensionless. |
| `synthetic_dataset.observables[].normalization_scale` | positive number | **Frozen metadata.** Native-to-neural scaling: `0.1` for the LU difference and `1.0` otherwise. |
| `synthetic_dataset.observables[].user_label` | string | **Frozen metadata.** Human-readable plot/report label bound to the ID. |

Removing an observable changes the neural input but not the DD parameter
family. A reusable corpus can append a missing admitted observable without
rewriting existing core, CFF, or observable shards; see
[Corpus and data selection](CORPUS_AND_DATA_SELECTION.md). Other installed
PARTONS observable classes remain disabled pending convention and fixture
audits.

### 3.4 Uncertainty and nuisance-response model

All seven values are finite and nonnegative; `local_correlation_length` must
be strictly positive. They are **experiment controls**. The covariance is
fixed at the declared injected truth for one experiment.

| Path | Default | Meaning |
|---|---:|---|
| `synthetic_dataset.uncertainty_model.uncorrelated_relative_sigma` | `0.05` | Relative independent standard-deviation term multiplying the absolute exact prediction. |
| `synthetic_dataset.uncertainty_model.uncorrelated_absolute_floor_normalized` | `0.01` | Independent absolute standard-deviation floor in normalized-observable units. |
| `synthetic_dataset.uncertainty_model.local_correlation_fraction` | `0.015` | Scale of the correlated local-index covariance component. |
| `synthetic_dataset.uncertainty_model.local_correlation_length` | `1.5` | Exponential correlation length in flattened point-index units. |
| `synthetic_dataset.uncertainty_model.phi_shape_correlated_fraction` | `0.01` | Rank-one correlated source proportional to prediction times $\sin\phi$. |
| `synthetic_dataset.uncertainty_model.global_normalization_fraction` | `0.02` | Fractional response $r_g$ multiplying `eta_global_normalization` for every observable. |
| `synthetic_dataset.uncertainty_model.lu_normalization_fraction` | `0.03` | Fractional response multiplying `eta_lu_normalization` for the LU-difference observable only. |

The diagonal standard deviation is
$\sigma_i=f_{\rm floor}+f_{\rm rel}|\mu_i|$. The full covariance also adds
the local kernel and phi-shape source; it is not treated as diagonal.

## 4. Inference profiles

Both `inference.profiles.quick` and `inference.profiles.validation` must exist
and have identical fields. Use `--profile quick` or `--profile validation` to
choose one. All are **compute controls** unless noted.

| Path | Quick default | Validation default | Meaning |
|---|---:|---:|---|
| `inference.profiles.{profile}.native_parameter_count` | 2,048 | 4,096 | Number of accepted prior-drawn 80-control vectors in the DD corpus, excluding the separately evaluated injected truth. |
| `inference.profiles.{profile}.noise_replicates_per_parameter` | 2 | 4 | Independent nuisance/noise contexts per native vector. Replicas stay in the same split. |
| `inference.profiles.{profile}.held_out_fraction` | 0.20 | 0.20 | Fraction of native parameter groups reserved for the untouched outer DD test. |
| `inference.profiles.{profile}.posterior_sample_count` | 1,024 | 16,384 | Samples drawn for the injected pseudodataset and neural/conventional comparison. Coverage trials use at most 512 each. |
| `inference.profiles.{profile}.conventional_nuisance_draws_per_parameter` | 8 | 8 | Standard-normal nuisance proposals paired with every exact native DD vector in prior importance sampling. |
| `inference.profiles.{profile}.coverage_trial_count` | 20 | 80 | Requested held-out contexts used for empirical coverage; capped by available test rows. |
| `inference.profiles.{profile}.exact_reevaluation_sample_count` | 48 | 64 | Posterior physics vectors re-evaluated with exact PARTONS for CFF/GPD/observable checks. |
| `inference.profiles.{profile}.max_num_epochs` | 100 | 200 | Hard maximum training epochs per candidate ensemble member. |
| `inference.profiles.{profile}.stop_after_epochs` | 20 | 25 | Early-stopping patience after no internal-validation improvement. |
| `inference.profiles.{profile}.ensemble_seeds[]` | `[51001,51002]` | `[51001,…,51005]` | Unique candidate-model seeds. At least one is required. |
| `inference.profiles.{profile}.active_ensemble_member_count` | 2 | 3 | Number of candidates retained by lowest grouped internal-validation NLL; integer from 1 through the seed count. |

The outer test and named GK/VGG datasets never select ensemble members.
Counts and fractions must still leave at least one train, internal-validation,
and outer-test parameter group; impossible combinations fail at runtime.

## 5. Neural posterior

The complete `inference.neural_posterior` object is:

| Path | Default | Meaning/edit policy |
|---|---|---|
| `inference.neural_posterior.estimator` | `sbi.NPE` | **Frozen provenance label.** The current training path always constructs grouped-validation NPE; this entry does not dynamically select another inference algorithm. |
| `inference.neural_posterior.flow` | `zuko_maf` | **Advanced method control.** Conditional density family supplied through `sbi`. |
| `inference.neural_posterior.point_hidden` | 128 | Width of hidden point-encoder layers. |
| `inference.neural_posterior.point_layers` | 3 | Number of point-encoder hidden layers. |
| `inference.neural_posterior.dataset_hidden` | 128 | Width of post-pooling dataset layers. |
| `inference.neural_posterior.dataset_layers` | 2 | Number of post-pooling dataset layers. |
| `inference.neural_posterior.embedding_features` | 96 | Dataset embedding width delivered to the flow. |
| `inference.neural_posterior.point_layer_norm` | `false` | Enable/disable layer normalization in the point encoder. |
| `inference.neural_posterior.flow_hidden_features` | 96 | Hidden width in the conditional flow. |
| `inference.neural_posterior.num_transforms` | 6 | Number of autoregressive flow transforms. |
| `inference.neural_posterior.z_score_theta` | `independent` | **Advanced method control.** `sbi` parameter-standardization mode. |
| `inference.neural_posterior.z_score_x` | `none` | **Advanced method control.** Context standardization is disabled because the context has explicit scaling/whitening. |
| `inference.neural_posterior.training_batch_size` | 256 | Training minibatch size. |
| `inference.neural_posterior.learning_rate` | 0.0005 | Optimizer learning rate. |
| `inference.neural_posterior.validation_fraction` | 0.10 | Fraction of the non-test native groups assigned to grouped internal validation. Must yield a nonempty train and validation set. |

Architecture, width, depth, batch-size, and learning-rate changes are
**advanced method controls** and require a new project and fresh closure. The
default architecture is intentionally used unless the user applies separately
reviewed Optuna results.

## 6. Random seeds

Every seed is an integer **experiment/reproducibility control**. Reusing all
settings and seeds reproduces the same stochastic streams on the declared
runtime; changing any seed defines a new experiment.

| Path | Default | Controls |
|---|---:|---|
| `inference.random_seeds.native_parameters` | 51010 | Prior DD-vector draws. |
| `inference.random_seeds.training_noise` | 51011 | Training nuisance draws and covariance noise. |
| `inference.random_seeds.pseudodata` | 51012 | The one displayed injected pseudodataset noise realization. |
| `inference.random_seeds.conventional_nuisance` | 51013 | Conventional-posterior nuisance proposals. |
| `inference.random_seeds.comparison` | 51014 | Posterior resampling, sliced projections, and predictive noise used in comparisons. |
| `inference.random_seeds.coverage` | 51015 | Reserved schema field. The current evenly spaced held-out-trial selector does not consume this seed; changing it changes the configuration hash but not current coverage draws. |
| `inference.random_seeds.optimization` | 51016 | Optuna sampler and single-member trial training seed. |

Candidate network seeds are separately listed in each profile's
`ensemble_seeds`.

## 7. Runtime

| Path | Default | Meaning/edit policy |
|---|---|---|
| `inference.runtime.accelerator` | `auto` | **Compute control.** `auto` uses CUDA when Torch can initialize it and otherwise records CPU fallback; `cuda` requires CUDA; `cpu` forces CPU neural work. PARTONS remains CPU-only. |
| `inference.runtime.cpu_threads` | 4 | **Compute control.** Integer $[1,256]$ used by Torch only when neural work runs on CPU. It does not limit PARTONS. |
| `inference.runtime.deterministic_algorithms` | `true` | **Frozen reproducibility policy.** This release rejects `false`. |
| `inference.runtime.native_workers` | `all_available` | **Compute control.** Either `all_available` or integer $[1,256]$; capped by process CPU affinity and ready batch count. Each worker supervises an isolated PARTONS bridge subprocess during generation, exact posterior reevaluation, GPD diagnostics, and other supported native campaigns. |

Run `./dvcs doctor NAME` to see the resolved accelerator, affinity CPUs,
native worker count, and current native batch size.
Evaluation uses the same two-parameter batch size and reports its two parallel
phase names and update semantics in `doctor` output.

## 8. Hyperparameter optimization

This block configures the optional, separately invoked `optimize` action. It
does nothing during `run` unless you explicitly apply selected parameters to a
new experiment. Trials use internal-validation NLL and cannot inspect the
outer test or GK/VGG data.

| Path | Default | Meaning/edit policy |
|---|---|---|
| `inference.hyperparameter_optimization.package` | `Optuna` | **Frozen identifier.** |
| `inference.hyperparameter_optimization.sampler` | `TPESampler` | **Frozen identifier.** Seeded TPE sampler. |
| `inference.hyperparameter_optimization.objective` | `best_validation_negative_log_density` | **Frozen identifier.** Minimized grouped internal-validation NPE loss. |
| `inference.hyperparameter_optimization.trials.{profile}` | quick 12; validation 40 | **Compute control.** Default new trials when `--trials` is omitted; integer $[1,1000]$. Studies resume in SQLite. |
| `inference.hyperparameter_optimization.pruner` | `MedianPruner` | **Frozen identifier.** |
| `inference.hyperparameter_optimization.pruner_startup_trials` | 5 | Nonnegative integer $\le1000$ before median pruning is considered. |
| `inference.hyperparameter_optimization.pruner_warmup_epochs` | 10 | Nonnegative integer $\le1000$ before pruning is considered. Current `sbi` integration reports history after training, so this is not a mid-epoch compute-saving claim. |

`inference.hyperparameter_optimization.search_space` contains:

| Path | Default candidates | Meaning |
|---|---|---|
| `inference.hyperparameter_optimization.search_space.point_hidden[]` | `[64,96,128,192]` | Categorical point width. |
| `inference.hyperparameter_optimization.search_space.dataset_hidden[]` | `[64,96,128,192]` | Categorical dataset width. |
| `inference.hyperparameter_optimization.search_space.embedding_features[]` | `[64,96,128]` | Categorical embedding width. |
| `inference.hyperparameter_optimization.search_space.flow_hidden_features[]` | `[64,96,128,192]` | Categorical flow width. |
| `inference.hyperparameter_optimization.search_space.num_transforms[]` | `[4,6,8]` | Categorical transform count. |
| `inference.hyperparameter_optimization.search_space.training_batch_size[]` | `[128,256,512]` | Categorical minibatch size. |
| `inference.hyperparameter_optimization.search_space.learning_rate[]` | `[0.0001,0.002]` | Two positive endpoints of a continuous log-uniform search interval, not two categorical choices. |

Categorical lists must be nonempty unique positive integers.

## 8.1 Observation-design training

| Path | Default | Meaning |
|---|---|---|
| `inference.observation_design_training.enabled` | `false` | Enable mask-augmented variable-design contexts; requires rematerialization and retraining. |
| `inference.observation_design_training.active_kinematic_counts[]` | `[6]` | Sorted unique trained counts within the configured maximum. Enabled training must include the full count; use `[12,30,60,96]` for a 96-point campaign. |
| `inference.observation_design_training.selection_seed` | `51017` | Deterministic nested space-filling ordering seed. |
| `inference.observation_design_training.selection_policy` | `nested_deterministic_farthest_point_v1` | Every smaller design is a prefix/subset of its larger partner. |
| `inference.observation_design_training.pooling_policy` | `masked_fixed_maximum_normalization_v1` | Remove padded embeddings and divide by maximum token count. |

Masking represents absent observations, not zero measurements and not hidden
physics parameters. It supports trained subsets; arbitrary new-coordinate
calibration requires varied kinematics in the native training corpus.

## 9. Output diagnostics

These entries change saved plots/reevaluations, not the training corpus.

| Path | Default | Meaning/constraint |
|---|---:|---|
| `output_diagnostics.gpd_reference_kinematics.x_b` | 0.2 | Reference Bjorken $x_B$; same $[0.10,0.50]$ kinematic domain. |
| `output_diagnostics.gpd_reference_kinematics.t_GeV2` | -0.12 | Reference diagnostic $t$ in GeV$^2$. |
| `output_diagnostics.gpd_reference_kinematics.Q2_GeV2` | 2.0 | Reference GPD-evaluation scale in GeV$^2$; same $[1,80]$ domain. |
| `output_diagnostics.gpd_reference_kinematics.beam_energy_GeV` | 12.0 | Reference beam energy in GeV; same $[3,200]$ domain. |
| `output_diagnostics.gpd_reference_kinematics.phi_rad` | 0.0 | Reference azimuth in radians; same $[0,2\pi]$ domain. |
| `output_diagnostics.gpd_x_grid.minimum` | 0.001 | First queried GPD $x$; must satisfy $-1\le x_{\min}<x_{\max}\le1$. |
| `output_diagnostics.gpd_x_grid.maximum` | 0.8 | Last queried GPD $x$. |
| `output_diagnostics.gpd_x_grid.point_count` | 65 | Integer grid size in $[3,129]$. Larger values increase exact reevaluation and plotting cost. |
| `output_diagnostics.credible_interval` | 0.9 | Central posterior interval shown in plots; strictly between 0 and 1. It does not alter training or gate levels. |

## 10. Validation gates

These are **validation policy**, not hyperparameters. Defaults are frozen
before a campaign so a failed result cannot be turned into a pass by moving a
threshold afterward.

| Path | Default | Pass condition and interpretation |
|---|---:|---|
| `validation_gates.minimum_conventional_effective_sample_size` | 80 | Self-normalized exact-bank importance-sampling ESS must be at least this value. Below it, conventional distance/width metrics are unavailable rather than regularized. |
| `validation_gates.maximum_ensemble_sliced_wasserstein` | 0.45 | Standardized multivariate sliced-Wasserstein distance between neural and conventional samples must not exceed this value. |
| `validation_gates.maximum_marginal_wasserstein_over_conventional_sd` | 0.60 | Every marginal Wasserstein distance divided by conventional SD must not exceed this value. |
| `validation_gates.maximum_test_nll_minus_validation_nll` | 0.75 | For every selected member, held-out outer-test NLL minus best internal-validation NLL must not exceed this value. |
| `validation_gates.maximum_coverage_standard_error` | 3.5 | Largest absolute empirical-minus-nominal coverage deviation, expressed in binomial standard errors across all 82 coordinates and 50/80/90% levels, must not exceed this value. |
| `validation_gates.minimum_posterior_predictive_point_90_coverage` | 0.75 | Fraction of pseudodata points inside exact-native posterior-predictive central 90% intervals must be at least this value. |
| `validation_gates.minimum_neural_to_conventional_stress_width_ratio` | 0.85 | For each of the 80 DD controls, neural central-90% width divided by conventional central-90% width must not fall below this false-certainty floor. Shadows themselves are fixed. |
| `validation_gates.maximum_exact_reevaluation_invalid_fraction` | 0.0 | Fraction of selected posterior samples rejected by exact PARTONS reevaluation must not exceed this value. Default requires none. |

A passing gate is evidence only for its declared synthetic campaign. It does
not enable real-data fitting or establish representation/shadow robustness.

## 11. Machine-checked field inventory

The documentation regression normalizes repeated list, GPD, channel, and
profile entries to the following paths and verifies that every leaf generated
by the schema-9 template appears in this document:

```text
schema_version
experiment.name
experiment.description
injected_truth.gpd_parameters.{GPD}.{channel}.normalization
injected_truth.gpd_parameters.{GPD}.{channel}.a
injected_truth.gpd_parameters.{GPD}.{channel}.c
injected_truth.gpd_parameters.{GPD}.{channel}.profile_b
injected_truth.gpd_parameters.{GPD}.{channel}.t_slope_GeV_minus2
injected_truth.gpd_parameters.shadow_coefficients.{GPD}
injected_truth.gpd_parameters.shadow_channel_amplitudes.{GPD}.{channel}
injected_truth.nuisance_parameters.eta_global_normalization
injected_truth.nuisance_parameters.eta_lu_normalization
synthetic_dataset.kinematics[].x_b
synthetic_dataset.kinematics[].t_GeV2
synthetic_dataset.kinematics[].Q2_GeV2
synthetic_dataset.kinematics[].beam_energy_GeV
synthetic_dataset.kinematics[].phi_rad
synthetic_dataset.kinematics_source.mode
synthetic_dataset.kinematics_source.database_root
synthetic_dataset.kinematics_source.database_revision
synthetic_dataset.kinematics_source.catalog_sha256
synthetic_dataset.kinematics_source.selection_metric
synthetic_dataset.kinematics_source.selected_records[].record_id
synthetic_dataset.kinematics_source.selected_records[].dataset_id
synthetic_dataset.kinematics_source.selected_records[].source_uuid
synthetic_dataset.kinematics_source.selected_records[].source_path
synthetic_dataset.kinematics_source.selected_records[].collaboration
synthetic_dataset.kinematics_source.selected_records[].reference
synthetic_dataset.kinematics_source.selected_records[].source_Q2_GeV2
synthetic_dataset.kinematics_source.selected_records[].source_beam_energy_GeV
synthetic_dataset.kinematics_source.selected_records[].generated_Q2_GeV2
synthetic_dataset.kinematics_source.selected_records[].Q2_projected
synthetic_dataset.kinematics_source.selected_records[].generated_beam_energy_GeV
synthetic_dataset.kinematics_source.selected_records[].measurement_values_used
synthetic_dataset.kinematics_source.measurements_quarantined
synthetic_dataset.kinematics_source.data_evolved
synthetic_dataset.kinematics_source.gpd_evolved_to_each_datum_Q2
synthetic_dataset.observables[].id
synthetic_dataset.observables[].native_unit
synthetic_dataset.observables[].normalization_scale
synthetic_dataset.observables[].user_label
synthetic_dataset.uncertainty_model.uncorrelated_relative_sigma
synthetic_dataset.uncertainty_model.uncorrelated_absolute_floor_normalized
synthetic_dataset.uncertainty_model.local_correlation_fraction
synthetic_dataset.uncertainty_model.local_correlation_length
synthetic_dataset.uncertainty_model.phi_shape_correlated_fraction
synthetic_dataset.uncertainty_model.global_normalization_fraction
synthetic_dataset.uncertainty_model.lu_normalization_fraction
inference.profiles.{profile}.native_parameter_count
inference.profiles.{profile}.noise_replicates_per_parameter
inference.profiles.{profile}.held_out_fraction
inference.profiles.{profile}.posterior_sample_count
inference.profiles.{profile}.conventional_nuisance_draws_per_parameter
inference.profiles.{profile}.coverage_trial_count
inference.profiles.{profile}.exact_reevaluation_sample_count
inference.profiles.{profile}.max_num_epochs
inference.profiles.{profile}.stop_after_epochs
inference.profiles.{profile}.ensemble_seeds[]
inference.profiles.{profile}.active_ensemble_member_count
inference.neural_posterior.estimator
inference.neural_posterior.flow
inference.neural_posterior.point_hidden
inference.neural_posterior.point_layers
inference.neural_posterior.dataset_hidden
inference.neural_posterior.dataset_layers
inference.neural_posterior.embedding_features
inference.neural_posterior.point_layer_norm
inference.neural_posterior.flow_hidden_features
inference.neural_posterior.num_transforms
inference.neural_posterior.z_score_theta
inference.neural_posterior.z_score_x
inference.neural_posterior.training_batch_size
inference.neural_posterior.learning_rate
inference.neural_posterior.validation_fraction
inference.random_seeds.native_parameters
inference.random_seeds.training_noise
inference.random_seeds.pseudodata
inference.random_seeds.conventional_nuisance
inference.random_seeds.comparison
inference.random_seeds.coverage
inference.random_seeds.optimization
inference.runtime.accelerator
inference.runtime.cpu_threads
inference.runtime.deterministic_algorithms
inference.runtime.native_workers
inference.hyperparameter_optimization.package
inference.hyperparameter_optimization.sampler
inference.hyperparameter_optimization.objective
inference.hyperparameter_optimization.trials.{profile}
inference.hyperparameter_optimization.pruner
inference.hyperparameter_optimization.pruner_startup_trials
inference.hyperparameter_optimization.pruner_warmup_epochs
inference.hyperparameter_optimization.search_space.point_hidden[]
inference.hyperparameter_optimization.search_space.dataset_hidden[]
inference.hyperparameter_optimization.search_space.embedding_features[]
inference.hyperparameter_optimization.search_space.flow_hidden_features[]
inference.hyperparameter_optimization.search_space.num_transforms[]
inference.hyperparameter_optimization.search_space.training_batch_size[]
inference.hyperparameter_optimization.search_space.learning_rate[]
output_diagnostics.gpd_reference_kinematics.x_b
output_diagnostics.gpd_reference_kinematics.t_GeV2
output_diagnostics.gpd_reference_kinematics.Q2_GeV2
output_diagnostics.gpd_reference_kinematics.beam_energy_GeV
output_diagnostics.gpd_reference_kinematics.phi_rad
output_diagnostics.gpd_x_grid.minimum
output_diagnostics.gpd_x_grid.maximum
output_diagnostics.gpd_x_grid.point_count
output_diagnostics.credible_interval
validation_gates.minimum_conventional_effective_sample_size
validation_gates.maximum_ensemble_sliced_wasserstein
validation_gates.maximum_marginal_wasserstein_over_conventional_sd
validation_gates.maximum_test_nll_minus_validation_nll
validation_gates.maximum_coverage_standard_error
validation_gates.minimum_posterior_predictive_point_90_coverage
validation_gates.minimum_neural_to_conventional_stress_width_ratio
validation_gates.maximum_exact_reevaluation_invalid_fraction
```

## 12. Editing checklist

Before starting a changed experiment:

1. create a fresh project with `./dvcs init NEW_NAME`;
2. edit only that project's `experiment.json`;
3. if changing generated kinematics, switch provenance to manual mode;
4. run `./dvcs doctor NEW_NAME` and resolve every error;
5. begin with `quick` for plumbing, then run an adequately sized campaign;
6. never tune architecture or gate thresholds using the outer test or fresh
   GK/VGG outputs;
7. retain the resulting configuration hash with every interpretation.

The runtime validator and native backend remain authoritative if this prose
and executable behavior ever disagree.
