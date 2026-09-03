# Production native assumptions

This table is generated from `provenance/production-native-assumptions.json`.
Do not edit it by hand. Unknown, omitted, and deferred entries are claim
boundaries, not zero-valued physics effects.

- Campaign: `experiment-josh-validation-conditional-dd-baseline-v1`
- Registry SHA-256: `dcbc6623c6f29886a61238d2449502b9d0241dce19331b9f14d4e22ab6e451e4`
- Registry status: `draft_requires_researcher_approval`
- Intended claim: Conditional pseudodata closure under the declared DD simulator, prior, likelihood, fixed shadow settings, and 96-point kinematic design; no model-independent or real-data claim.

| Assumption | Configured value | Status | Identity effect | Claim implication |
|---|---|---|---|---|
| gpd.types | ["H","E","Htilde","Etilde"] | implemented_in_native_backend | changing the set or order requires a completely new native corpus | claims are conditional on these four represented types; type completeness beyond this route is not established |
| gpd.input_flavors | ["u","d","s","gluon"] | implemented_in_native_backend | changing flavors or channels requires a completely new native corpus | heavy-flavor input is absent and must not be inferred from evolved output labels |
| gpd.quark_antiquark_basis | signed-x physical u,d,s plus/minus fields assembled by PseudodataInputGPD; no separately parameterized antiquark controls | enforced_by_construction | changing the basis or charge-parity convention requires a completely new native corpus | quark/antiquark separation is representation-defined, not independently learned |
| gpd.coordinate_conventions | {"Q2":"GeV2","t":"GeV2 nonpositive","x":"signed dimensionless","xi":"nonnegative dimensionless"} | enforced_by_construction | changing conventions or units requires a completely new native corpus | all exported functions must retain the signed-x and unit metadata |
| gpd.input_scale | {"Q0_squared_GeV2":1.0} | implemented_in_native_backend | changing Q0 requires a completely new native corpus | results cannot be compared at another input scale without declared evolution |
| qcd.gpd_evolution | {"module":"GPDEvolutionApfel","order":"LO","scheme":"MSbar"} | implemented_in_native_backend | changing module, order, or scheme requires a completely new native corpus | the baseline is LO-evolution conditional and does not quantify evolution-order uncertainty |
| qcd.coefficient_functions | {"module":"DVCSCFFStandard","order":"LO"} | implemented_in_native_backend | changing coefficient functions/order requires a completely new native corpus | no NLO accuracy or truncation-uncertainty claim is permitted |
| qcd.quark_gluon_mixing | GPDEvolutionApfel receives u,d,s,gluon input; mixing in the stage11 route is not independently validated | unknown | changing or correcting mixing semantics requires a completely new native corpus | flavor/gluon evolution claims are blocked pending route-specific validation |
| qcd.heavy_flavor_thresholds | {"active_flavors":3,"thresholds_GeV":[0,0,0],"variable_thresholds":false} | omitted | adding threshold matching requires a completely new native corpus | heavy-flavor and threshold-systematic claims are prohibited |
| qcd.alpha_s | {"MZ_GeV":91.1876,"alpha_s_MZ":0.118,"module":"RunningAlphaStrongApfel","order":"LO","scheme":"MSbar"} | implemented_in_native_backend | changing alpha_s settings requires a completely new native corpus | alpha_s uncertainty is not propagated |
| qcd.factorization_renormalization_scales | {"muF_squared":"Q2","muR_squared":"Q2"} | implemented_in_native_backend | changing central scales requires a completely new native corpus | scale variation is absent and perturbative uncertainty is not quantified |
| qcd.scale_variation | not implemented for the campaign | omitted | adding scale variants requires new native shards for each physics setting | theory truncation/scale uncertainty cannot be included in the posterior |
| process.kinematic_table | {"Q2_GeV2_range":[1.4,7.0],"beam_energy_GeV":[10.6],"phi_count":8,"points":96,"t_GeV2_range":[-0.7,-0.12],"xB_range":[0.12,0.4]} | enforced_by_construction | any point/value/order change requires a completely new native corpus | no interpolation or extrapolation claim outside this design |
| process.physics_declared_domain | {"actual_t_GeV2_range":[-0.7,-0.12],"actual_xB_range":[0.12,0.4],"declared_t_GeV2_range":[-0.2,-0.08],"declared_xB_range":[0.12,0.28]} | unknown | metadata-only correction needs no regeneration if proven non-executable; a true native-domain change requires a new corpus | submission and domain claims are blocked while the configuration contradicts itself |
| process.observables | ["DVCSCrossSectionUUMinus","DVCSCrossSectionDifferenceLUMinus","DVCSAc","DVCSAluMinus","DVCSAulMinus","DVCSAllMinus"] | enforced_by_construction | adding a native observable may use append-only observable shards when compatibility is proven; removal/reorder is selection/model-view work | all inference and covariance summaries must preserve kinematic-major, observable-minor ordering |
| process.beam_target_definition | proton target and minus-lepton-charge PARTONS observable modules; beam/target polarization states are encoded only by module names | assumed | changing target, charge, or polarization definitions requires a completely new native corpus | experimental interpretation is blocked until the mapping is explicitly approved |
| process.phi_convention | phi_rad in [0,2pi] passed to PARTONS | assumed | changing phi convention requires a completely new native corpus | cross-framework sign comparisons require an explicit convention adapter |
| process.bh_dvcs_interference | {"cff_module":"DVCSCFFStandard","process_module":"DVCSProcessGV08"} | implemented_in_native_backend | changing process/BH implementation requires a completely new native corpus | component-level accuracy is inherited from the named PARTONS modules, not newly validated here |
| process.twist | None | unknown | a semantic label correction is metadata-only only if proven identical; changing twist content requires a completely new native corpus | full-production physics claims are blocked; only an explicitly provisional software baseline is possible |
| process.kinematic_power_target_mass_finite_t | inherited from DVCSProcessGV08; no separate campaign declaration | unknown | changing correction content requires a completely new native corpus | target-mass, finite-t, and power-correction accuracy must not be claimed |
| process.elastic_form_factors | PARTONS module default; source not recorded in campaign configuration | unknown | changing form-factor source requires a completely new native corpus | production submission is blocked until the form-factor identity is frozen |
| process.radiative_corrections | not declared | omitted | adding radiative corrections requires a completely new native corpus | no detector/radiative realism claim |
| constraints.support_symmetry_charge_parity | DD support plus signed-x quark/plus/minus construction | enforced_by_construction | changing structural conventions requires a completely new native corpus | structural support is conditional on the native implementations |
| constraints.endpoints_continuity_integrability | {"explicit_endpoint_gate":false,"quadrature_order":128} | diagnostic_only | quadrature or functional changes require a completely new native corpus; diagnostic thresholds do not | accepted samples are conditional on simulator validity and unmeasured endpoint errors |
| constraints.forward_limits | independent beta-like DD forward shapes, not an external PDF fit | imposed_as_prior | changing forward input or bounds requires new native shards/corpus | the baseline cannot claim externally constrained PDF forward limits |
| constraints.form_factor_sum_rules | not imposed | omitted | hard enforcement changes the native prior/corpus; an external probabilistic constraint changes realization/likelihood | elastic form-factor sum-rule consistency cannot be claimed |
| constraints.mellin_polynomiality | DD baseline; project shadow differences inherit DD construction, native H.u shadow has its module behavior | enforced_by_construction | changing representation requires a completely new native corpus | polynomiality is representation-conditional and does not validate missing D-term freedom |
| constraints.d_term | omitted | omitted | adding or varying a D-term requires a completely new native corpus; a separate stress corpus does not invalidate this baseline | D-term ambiguity and D-term-sensitive mechanical quantities cannot be extracted |
| constraints.positivity | not imposed | omitted | hard positivity restriction changes selection/prior support; diagnostics require no regeneration | positivity consistency must not be claimed |
| constraints.evolution_consistency | native GPDEvolutionApfel from common Q0 to each datum Q2 | implemented_in_native_backend | changing evolution semantics requires a completely new native corpus | consistency is only established within the selected perturbative configuration |
| dd.profile_forward_t_dependence | independent FixedScaleDDGPD beta-like forward shape, profile_b, and exponential-like t_slope controls per type/channel | enforced_by_construction | changing profile/forward/t functional form requires a completely new native corpus | posterior uncertainty is within this DD ansatz only |
| dd.native_prior_bounds | {"a":[0.1,1.0],"c":[2.0,6.0],"distribution":"factorized uniform","normalization":[-4,4],"profile_b":[1.0,4.0],"t_slope_GeV_minus2":[0.0,2.0]} | imposed_as_prior | changing bounds/distribution requires new native shards or a new corpus unless a broad master corpus supports valid reselection | support exclusions and boundary mass must be reported separately from data information |
| dd.flavor_correlations | none in the proposal prior; all 80 DD coordinates are independent | imposed_as_prior | changing correlations requires reselection only if support/reweighting is adequate, otherwise new native shards | flavor correlation conclusions are prior-conditional |
| dd.shadow_coordinates | {"H.u":"GPDBDMMS21","fixed_coefficients":{"E":-0.35,"Etilde":-0.2,"H":0.6,"Htilde":0.4},"inferred":false,"other_channels":"project DD difference, not verified CFF-null"} | enforced_by_construction | changing coefficients/amplitudes requires a completely new native corpus; an independent stress corpus is append-only | no shadow coordinate is inferred and no general non-identifiability claim follows |
| neural.function_corpus | absent: no reviewed canonical coordinate request and current schema-1 corpus has no GPD truth grids | deferred | adding the first function grid requires new schema-2 native shards/corpus | neural production and reduced-parametric-bias claims are blocked |
| neural.coordinate_table | None | deferred | coordinate changes require new native function-grid shards but need not recompute compatible observable shards if extension rules support it | decoder support and reconstruction precision cannot be assessed |
| neural.autoencoder_decoder_latent | {"decoder":"coordinate-conditioned MLP","default_latent_dim":16,"training":"full-batch experimental utility"} | deferred | architecture/latent changes require model-view regeneration, never new PARTONS generation | neural representation is not production-ready |
| neural.latent_transform_prior | training-group-only shrinkage full-covariance whitening; nominal standard-normal latent prior after whitening | deferred | transform/prior changes require model-view and checkpoint regeneration only | latent-prior sensitivity remains unknown |
| neural.physical_constraints | native-generation structural constraints only; no additional hard/soft neural constraint suite | deferred | diagnostics need no native regeneration; hard representation changes require a new model view | decoded-function physical validity cannot be claimed |
| statistics.observation_distribution | multivariate Gaussian residual around a nuisance-shifted native mean | included_in_likelihood | distribution/covariance changes require realization regeneration, not native shards | posterior calibration is conditional on Gaussian residuals |
| statistics.covariance_components | {"absolute_floor_normalized":0.01,"local_correlation_fraction":0.015,"local_correlation_length_in_flat_index":1.5,"phi_shape_fraction":0.01,"uncorrelated_relative_sigma":0.05} | included_in_likelihood | changes require realization regeneration only | coverage/generalization must distinguish fixed-covariance and truth-derived covariance policies |
| statistics.explicit_nuisances | {"eta_global_normalization":{"applies_to":"all","fractional_response":0.02,"prior":"N(0,1)"},"eta_lu_normalization":{"applies_to":"DVCSCrossSectionDifferenceLUMinus","fractional_response":0.03,"prior":"N(0,1)"}} | included_as_nuisance | response/prior changes require realization and model-view regeneration, not native shards | normalization uncertainty is explicit and must not be added again as covariance |
| statistics.additive_multiplicative_semantics | Gaussian residual is additive in normalized observable space; two normalization nuisances are multiplicative linear fractional shifts | included_in_likelihood | changes require realization regeneration only | large nonlinear normalization effects are outside the baseline |
| statistics.covariance_parameter_dependence | fixed_at_declared_truth | included_in_likelihood | policy changes require realization regeneration only | named-family results must not confound generator shift with covariance shift |
| statistics.numerical_jitter | 0.0 | enforced_by_construction | changing jitter requires realization metadata regeneration only | any future jitter must remain separately reported from physical covariance |
| statistics.masks_missing_data | {"active_kinematic_counts":[96],"masked_design_training":false,"missing_data":false} | enforced_by_construction | mask/design changes require realization/model-view regeneration, not native shards | no missing-data robustness or arbitrary-design inference claim |
| statistics.target_transform | 80 bounded DD controls mapped by analytic probits plus two standard-normal nuisance coordinates | enforced_by_construction | transform changes require model-view/checkpoint regeneration only | boundary behavior must be reported |
| statistics.inference_network | {"dataset_hidden":128,"dataset_layers":2,"density_estimator":"Zuko MAF","embedding_features":96,"encoder":"DeepSets","flow_hidden_features":96,"num_transforms":6,"point_hidden":128,"point_layers":3} | enforced_by_construction | architecture changes require checkpoint/model-view regeneration only and never new PARTONS generation | amortization error must be separated from representation error |
| statistics.ensemble | {"active_members":3,"candidate_seeds":[57439,85743,98465,56486,78546],"selection_metric":"validation negative log density"} | enforced_by_construction | seed/ensemble changes require checkpoint and evaluation regeneration only | seed variation must be reported separately from posterior uncertainty |
| statistics.calibration_status | historical limited coverage diagnostics only; fresh SBC/TARP/local C2ST absent | deferred | evaluation-only regeneration when checkpoints/corpora are unchanged | conditional closure is not yet a calibrated extraction claim |

## Sources and enforcement

### `gpd.types`

- Source: workspace/experiment_josh/.engine/physics.json:gpd_content.types
- Scope: native corpus and all downstream representations
- Enforcement: PseudodataInputGPD exposes four independent type blocks and the bridge request schema fixes their order.
- Validation: Bridge capabilities and native self-test pass; full per-type production response is not separately acceptance-tested.
- Validity domain: twist-2 proton GPD type labels in the stage11 synthetic route

### `gpd.input_flavors`

- Source: workspace/experiment_josh/.engine/physics.json:gpd_content.channels
- Scope: native input-scale GPD
- Enforcement: PseudodataInputGPD has independent DD controls for u, d, s, and gluon in every type.
- Validation: Bridge capabilities report these four input channels; production-route flavor closure remains unrun.
- Validity domain: Q0^2=1 GeV2, fixed three-active-flavor evolution configuration

### `gpd.quark_antiquark_basis`

- Source: cpp/partons_bridge/src/PseudodataInputGPD.cpp:computeDD and canonical_gpd_truth_v1 capability
- Scope: native input and future function-grid export
- Enforcement: quark, plus, and minus distributions are deterministically constructed from values at x and -x.
- Validation: Native bridge schema/capability tests cover accepted channel labels; a production coordinate table is absent.
- Validity domain: signed x in [-1,1] under PARTONS physical flavor conventions

### `gpd.coordinate_conventions`

- Source: bridge canonical_gpd_truth_v1 capability and configs/schemas/master_native_physics_corpus_v2.schema.json
- Scope: native requests and neural function tables
- Enforcement: strict JSON schemas and native kinematic validators reject unsupported units/ranges.
- Validation: Static schema tests and bridge native self-test pass.
- Validity domain: stage11 accepted request domain

### `gpd.input_scale`

- Source: workspace/experiment_josh/.engine/physics.json:input_scale
- Scope: native corpus
- Enforcement: bridge request parser requires the configured input scale and evolves to each datum Q2.
- Validation: Capabilities report Q0^2=1 GeV2 and the self-test passes; multi-Q2 production rehearsal is pending.
- Validity domain: observable Q2 in [1,80] GeV2

### `qcd.gpd_evolution`

- Source: workspace/experiment_josh/.engine/physics.json:evolution_configuration
- Scope: native corpus
- Enforcement: the bridge builds one APFEL++ table per GPD type and evolves from Q0 to each datum Q2.
- Validation: Bridge capabilities and source anchors identify the module; exact production multi-Q2 rehearsal is pending.
- Validity domain: Q2 in [1,80] GeV2, fixed nf=3

### `qcd.coefficient_functions`

- Source: workspace/experiment_josh/.engine/physics.json:theory_configuration
- Scope: native CFFs and observables
- Enforcement: bridge request validation fixes DVCSCFFStandard and LO for the pseudodata route.
- Validation: Capabilities report LO; no production perturbative configuration is listed by the bridge.
- Validity domain: stage11 pseudodata observables

### `qcd.quark_gluon_mixing`

- Source: bridge capabilities pseudodata_dvcs and coupled_quark_gluon sections
- Scope: native evolution
- Enforcement: no production preflight assertion currently proves the separately tested coupled-basis behavior is identical in stage11.
- Validation: A route-specific quark-to-gluon/gluon-to-quark response test is required.
- Validity domain: fixed nf=3 LO MSbar stage11 evolution

### `qcd.heavy_flavor_thresholds`

- Source: workspace/experiment_josh/.engine/physics.json:evolution_configuration.active_flavors and alpha_s.thresholds
- Scope: native evolution
- Enforcement: ActiveFlavorsThresholdsConstant fixes nf=3.
- Validation: Bridge capabilities explicitly report no heavy-flavor thresholds.
- Validity domain: conditional fixed-nf=3 baseline

### `qcd.alpha_s`

- Source: workspace/experiment_josh/.engine/physics.json:evolution_configuration.alpha_s
- Scope: native evolution
- Enforcement: strict native physics request fields.
- Validation: Capabilities echo the exact settings; scale-variation validation is absent.
- Validity domain: fixed-nf=3 LO evolution

### `qcd.factorization_renormalization_scales`

- Source: workspace/experiment_josh/.engine/physics.json:theory_configuration
- Scope: native CFF and observable generation
- Enforcement: DVCSScalesQ2Multiplier is fixed by the native request.
- Validation: Capabilities echo the scale selection.
- Validity domain: selected kinematic Q2 values

### `qcd.scale_variation`

- Source: workspace/experiment_josh/.engine/physics.json and bridge capabilities
- Scope: theory uncertainty
- Enforcement: no variation axis exists in the selected native request.
- Validation: Production preflight checks that no unsupported scale-uncertainty claim is made.
- Validity domain: conditional baseline only

### `process.kinematic_table`

- Source: workspace/experiment_josh/experiment.json:synthetic_dataset.kinematics
- Scope: native corpus and observation design
- Enforcement: the exact ordered table is hashed into corpus identity.
- Validation: All 96 entries pass the repository fixed-target kinematic validator.
- Validity domain: the exact ordered table with SHA-256 2d4130fce5d4fd27ced6ad833360d95c0b3528141ca09f4c903a94a4ee94cff5

### `process.physics_declared_domain`

- Source: workspace/experiment_josh/.engine/physics.json:observable_domain versus experiment kinematics
- Scope: configuration provenance
- Enforcement: no existing check requires declared physics bounds to enclose the actual table.
- Validation: Production preflight must fail until the domain metadata is reconciled without changing kinematics.
- Validity domain: current project only

### `process.observables`

- Source: workspace/experiment_josh/experiment.json:synthetic_dataset.observables
- Scope: native corpus, covariance ordering, and context
- Enforcement: schema requires an ordered unique subset and the ordered set is hashed into corpus identity.
- Validation: Static schema tests and bridge capabilities pass.
- Validity domain: six selected native observable modules at every kinematic point

### `process.beam_target_definition`

- Source: DVCSProcessGV08 and selected observable module identifiers
- Scope: native observables
- Enforcement: module selection is fixed, but a standalone beam/target metadata object is absent.
- Validation: Researcher must approve the module-to-experimental-state mapping before submission.
- Validity domain: synthetic proton DVCS campaign

### `process.phi_convention`

- Source: experiment kinematics and native request validator
- Scope: native observables
- Enforcement: range and units are checked; sign/orientation convention is inherited from PARTONS.
- Validation: A convention test against an independently declared Trento/native definition is absent.
- Validity domain: PARTONS DVCSProcessGV08 conventions

### `process.bh_dvcs_interference`

- Source: workspace/experiment_josh/.engine/physics.json:theory_configuration
- Scope: native observables
- Enforcement: PARTONS process/observable modules calculate the selected cross sections and asymmetries.
- Validation: Native self-test does not separately decompose BH, DVCS, and interference terms.
- Validity domain: selected modules and kinematics

### `process.twist`

- Source: workspace/experiment_josh/.engine/physics.json:theory_configuration.twist
- Scope: native CFFs and observables
- Enforcement: the bridge currently requires a null field rather than an approved twist label.
- Validation: Researcher/native-owner must assign and verify the intended twist scope.
- Validity domain: unresolved

### `process.kinematic_power_target_mass_finite_t`

- Source: selected PARTONS process module; absent from project physics JSON
- Scope: native observables
- Enforcement: no explicit project-level switch or provenance field.
- Validation: Native source/module audit required before a correction claim.
- Validity domain: selected process module

### `process.elastic_form_factors`

- Source: absent from workspace/experiment_josh/.engine/physics.json
- Scope: BH/interference observables
- Enforcement: no fail-closed form-factor identity in the project manifest.
- Validation: Record native module/source/version and compare a fixture before production.
- Validity domain: unresolved

### `process.radiative_corrections`

- Source: absent from project and native capability records
- Scope: native observables
- Enforcement: no radiative-correction module is selected.
- Validation: Preflight records omission.
- Validity domain: synthetic Born-level-like baseline only

### `constraints.support_symmetry_charge_parity`

- Source: FixedScaleDDGPD and PseudodataInputGPD implementations
- Scope: native GPD
- Enforcement: native DD integration and deterministic signed-x field assembly.
- Validation: Native unit/source tests exist; production grid diagnostics are pending.
- Validity domain: represented DD and fixed shadow components

### `constraints.endpoints_continuity_integrability`

- Source: physics parameters and native DD implementation
- Scope: native GPD numerical behavior
- Enforcement: finite native evaluations and proposal rejection; no production-wide endpoint/continuity threshold.
- Validation: Canonical function-grid diagnostics are blocked by the missing coordinate table.
- Validity domain: accepted simulator proposals

### `constraints.forward_limits`

- Source: FixedScaleDDGPD parameters normalization,a,c
- Scope: native GPD prior
- Enforcement: five bounded controls per type/flavor channel define the forward input and t/profile behavior.
- Validation: Parameter bounds are tested; agreement with measured PDFs is not imposed or validated.
- Validity domain: synthetic DD prior

### `constraints.form_factor_sum_rules`

- Source: no project likelihood/prior field
- Scope: GPD constraints
- Enforcement: none
- Validation: Future function-grid moment diagnostic required.
- Validity domain: not applicable

### `constraints.mellin_polynomiality`

- Source: FixedScaleDDGPD, PseudodataInputGPD, GPDBDMMS21
- Scope: native GPD
- Enforcement: double-distribution construction for the baseline and project-defined directions.
- Validation: No campaign-wide numerical Mellin polynomiality scan has been run.
- Validity domain: represented components with D-term omitted

### `constraints.d_term`

- Source: workspace/experiment_josh/.engine/physics.json:gpd_content.omitted
- Scope: native corpus
- Enforcement: the native stage11 parameter vector contains no D-term coordinate.
- Validation: Bridge capabilities explicitly list D_term among physics omissions.
- Validity domain: zero-D-term conditional baseline

### `constraints.positivity`

- Source: no project/native constraint field
- Scope: GPD prior and diagnostics
- Enforcement: none beyond finite native evaluation.
- Validation: No scheme/order/region-qualified positivity diagnostic exists.
- Validity domain: not applicable

### `constraints.evolution_consistency`

- Source: bridge stage11 implementation and capabilities
- Scope: native GPD/CFF/observables
- Enforcement: all observable predictions pass through the same native evolution route.
- Validation: Direct-wrapper anchor checks exist; production multi-Q2 rehearsal pending.
- Validity domain: LO MSbar fixed nf=3

### `dd.profile_forward_t_dependence`

- Source: FixedScaleDDGPD implementation and 80-coordinate parameter list
- Scope: DD representation
- Enforcement: five controls for each of 4 types x 4 channels.
- Validation: Native proposal and schema tests; alternative ansatz sensitivity unrun.
- Validity domain: stage11 full-independent DD family

### `dd.native_prior_bounds`

- Source: workspace/.corpora/partons-validation-16384/corpus.json:core_identity.prior and current corpus code
- Scope: native proposal distribution
- Enforcement: deterministic PCG64 proposal generation and native input validation.
- Validation: Static prior/schema tests; native rejection distortion must be audited on the new corpus.
- Validity domain: 80 DD coordinates

### `dd.flavor_correlations`

- Source: stage11 parameter sampling implementation
- Scope: native prior
- Enforcement: factorized uniform coordinate draws.
- Validation: Prior sampler tests; sensitivity study unrun.
- Validity domain: current DD family

### `dd.shadow_coordinates`

- Source: experiment injected truth, physics JSON, and bridge capability
- Scope: native simulator family
- Enforcement: fixed coefficients are embedded in every generated native proposal but are absent from the inferred target.
- Validation: Capability audit passes; forward-invisibility and inference-honesty tests are unrun.
- Validity domain: one fixed nonzero shadow setting

### `neural.function_corpus`

- Source: workspace/experiment_josh and workspace/.corpora/partons-validation-16384/corpus.json
- Scope: neural-GPD representation
- Enforcement: schema-2 corpus creation fails without a GPD truth request; neural production dispatch is blocked.
- Validation: Preflight requires an approved coordinate file and complete checksummed truth shards.
- Validity domain: none yet

### `neural.coordinate_table`

- Source: workspace/experiment_josh/gpd_truth.json is absent
- Scope: neural autoencoder
- Enforcement: canonical coordinate validator exists but has no production input.
- Validation: Researcher approval of GPDs/flavors/parities/x/xi/t grid and precision is required.
- Validity domain: unresolved

### `neural.autoencoder_decoder_latent`

- Source: src/extract_dvcs_cff/neural_gpd.py
- Scope: neural model view
- Enforcement: production train/evaluate/plot dispatch fails closed.
- Validation: No production corpus, checkpoint, reconstruction gate, seed study, or decoder-floor comparison exists.
- Validity domain: synthetic utility tests only

### `neural.latent_transform_prior`

- Source: src/extract_dvcs_cff/neural_gpd.py
- Scope: neural model view and NPE target
- Enforcement: utility records training group hash and removes collapsed eigen-directions.
- Validation: Round-trip/collapse unit tests pass; no production latent distribution diagnostics exist.
- Validity domain: future fitted decoder corpus

### `neural.physical_constraints`

- Source: src/extract_dvcs_cff/neural_gpd.py and current model registry
- Scope: decoded neural functions
- Enforcement: none beyond training on future native function grids and exact-coordinate refusal.
- Validation: Support, boundary, moments, polynomiality, sum-rule, positivity, and evolution diagnostics are unrun.
- Validity domain: none yet

### `statistics.observation_distribution`

- Source: src/extract_dvcs_cff/workflows/pseudodata.py:_covariance_and_responses and _effective_prediction
- Scope: pseudodata realization and implicit NPE likelihood
- Enforcement: Cholesky draws from a fixed positive-definite covariance.
- Validation: Deterministic replica/covariance validation is required before submission.
- Validity domain: synthetic pseudodata only

### `statistics.covariance_components`

- Source: workspace/experiment_josh/experiment.json:synthetic_dataset.uncertainty_model
- Scope: pseudodata realization
- Enforcement: fixed covariance is built from the declared truth prediction in kinematic-major/observable-minor order.
- Validation: Exact matrix and replica test pending a current-bridge native truth evaluation.
- Validity domain: 576 ordered observation components

### `statistics.explicit_nuisances`

- Source: workspace/experiment_josh/.engine/workflow.json:experimental_model.nuisance_groups
- Scope: pseudodata realization and 82-dimensional DD posterior
- Enforcement: two explicit standard-normal target coordinates multiply the native prediction response.
- Validation: Code audit confirms normalization terms are excluded from covariance; nuisance recovery campaign is unrun.
- Validity domain: linear fractional response model

### `statistics.additive_multiplicative_semantics`

- Source: pseudodata workflow _effective_prediction and materialization
- Scope: likelihood simulator
- Enforcement: separate covariance and response arrays are stored in the realization.
- Validation: Double-count audit passes by source inspection; empirical nuisance/noise tests pending.
- Validity domain: small linear nuisance response approximation

### `statistics.covariance_parameter_dependence`

- Source: workspace/experiment_josh/.engine/workflow.json:experimental_model.covariance_policy
- Scope: training and evaluation likelihood simulator
- Enforcement: one covariance is computed from injected-truth predictions and reused for all DD proposals.
- Validation: Controlled external-fixed and truth-derived holdout comparisons are pending.
- Validity domain: current synthetic truth and observable normalization

### `statistics.numerical_jitter`

- Source: pseudodata covariance builder reports regularization=null
- Scope: production covariance
- Enforcement: the physical covariance must be positive definite without regularization.
- Validation: Eigenvalue/Cholesky report pending current exact truth.
- Validity domain: declared 576x576 covariance

### `statistics.masks_missing_data`

- Source: schema-8 project migration behavior and current experiment
- Scope: model view/context
- Enforcement: all 96 kinematics and six observables are active in every context.
- Validation: Schema migration/static tests pass.
- Validity domain: complete synthetic design

### `statistics.target_transform`

- Source: src/extract_dvcs_cff/inference/stage10.py
- Scope: DD model view and checkpoint
- Enforcement: stage10_physical_to_latent and inverse transformation.
- Validation: Architecture/static tests pass; prior/transform sensitivity unrun.
- Validity domain: 82-dimensional DD target

### `statistics.inference_network`

- Source: workspace/experiment_josh/experiment.json:inference.neural_posterior
- Scope: DD checkpoint
- Enforcement: frozen workflow network configuration and explicit model-family dispatch.
- Validation: Synthetic smoke passes; production-like retraining is unrun.
- Validity domain: DD DeepSets+MAF baseline

### `statistics.ensemble`

- Source: workspace/experiment_josh/experiment.json:inference.profiles.validation
- Scope: DD training and evaluation
- Enforcement: candidate seed list and active count are part of the workflow configuration.
- Validation: Historical v7 evidence exists; fresh post-fix production-like training is unrun.
- Validity domain: current architecture/profile

### `statistics.calibration_status`

- Source: docs/CLAIMS_LEDGER.md and historical result artifacts
- Scope: posterior interpretation
- Enforcement: claims ledger blocks calibrated-posterior wording.
- Validation: Fresh post-fix calibration campaign required.
- Validity domain: none established

