from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from extract_dvcs_cff.contracts import (
    assert_observation_context_inputs,
    corpus_capabilities,
    fit_train_only_standardization,
    master_corpus_identity,
    model_view_identity,
    preflight_master_corpus,
    realization_identity,
    require_model_capability,
    validate_gpd_truth_request,
)
from extract_dvcs_cff.model_registry import (
    assert_architecture_comparable,
    common_function_metrics,
    model_family_registry,
    validate_constraint_registry,
    validate_maf_only,
)
from extract_dvcs_cff.neural_gpd import (
    fitted_diagonal_gaussian_prior,
    require_exact_coordinate_support,
    train_function_autoencoder,
)
from extract_dvcs_cff.literature import (
    assert_conventions_compatible,
    coverage_requirements,
    load_registry,
)
from extract_dvcs_cff.native_policy import (
    assert_native_launch_allowed,
    assert_saved_artifact_stage,
)


ROOT = Path(__file__).resolve().parents[1]


class ArchitectureContractTests(unittest.TestCase):
    def native_identity(self, **updates):
        arguments = {
            "generator_family": "dd",
            "generator_prior": {"id": "test-prior", "a": [-0.2, 0.2]},
            "kinematics": [{"x_b": 0.2, "Q2_GeV2": 2.0}],
            "observables": [{"id": "DVCSAluMinus"}],
            "native_physics": {"evolution": "APFEL++", "order": "LO"},
            "native_bridge_sha256": "a" * 64,
        }
        arguments.update(updates)
        return master_corpus_identity(**arguments)[0]

    def realization(self, seed=3):
        return realization_identity(
            master_corpus_id=self.native_identity(), selection_id="selection-a",
            noise_contract={"family": "normal"},
            covariance_contract={"id": "dense-v1"},
            nuisance_contract={"groups": ["normalization"]}, masks={"all": True},
            observables=["DVCSAluMinus"], seeds={"noise": seed},
        )[0]

    def view(self, *, width=64, decoder=None):
        return model_view_identity(
            realization_id=self.realization(), model_family="dd_deepsets_maf",
            observation_contract={"encoder": "deepsets", "width": width},
            target_transform={"id": "dd_probit_v1"}, decoder_contract=decoder,
            normalization_contract={"fit_scope": "training_groups_only"},
        )[0]

    def test_downstream_architecture_does_not_change_native_identity(self):
        baseline = self.native_identity()
        for downstream in (
            {"deepsets_width": 32}, {"maf_transforms": 9},
            {"decoder_latent_dim": 8}, {"optimizer": "adamw"},
        ):
            self.assertEqual(baseline, self.native_identity())
            self.assertNotIn(str(downstream), baseline)

    def test_physical_support_changes_native_identity(self):
        baseline = self.native_identity()
        self.assertNotEqual(
            baseline,
            self.native_identity(generator_prior={"id": "another-prior"}),
        )
        self.assertNotEqual(
            baseline,
            self.native_identity(kinematics=[{"x_b": 0.3, "Q2_GeV2": 2.0}]),
        )

    def test_identity_sensitivity_is_layered(self):
        self.assertNotEqual(self.realization(3), self.realization(4))
        self.assertEqual(self.native_identity(), self.native_identity())
        self.assertNotEqual(self.view(width=64), self.view(width=128))
        self.assertNotEqual(self.view(), self.view(decoder={"latent_dim": 8}))
        self.assertEqual(self.realization(), self.realization())

    def test_old_corpus_dd_supported_neural_fails_precisely(self):
        old = {"schema_version": 1}
        capabilities = corpus_capabilities(old)
        self.assertTrue(capabilities.dd_deepsets_maf)
        self.assertFalse(capabilities.neural_gpd_deepsets_maf)
        with self.assertRaisesRegex(RuntimeError, "lacks canonical native GPD"):
            require_model_capability(old, "neural_gpd_deepsets_maf")

    def test_planned_truth_is_not_a_neural_capability(self):
        planned = {
            "schema_version": 2,
            "gpd_truth": {"status": "planned", "coordinates": [{"x": 0.1}], "shards": []},
        }
        self.assertFalse(corpus_capabilities(planned).canonical_gpd_truth)

    def test_truth_cannot_enter_observation_context(self):
        assert_observation_context_inputs(
            ["observations_normalized", "covariance_normalized", "masks"]
        )
        for truth in ("theta_latent", "truth_cffs", "gpd_truth_values"):
            with self.assertRaisesRegex(RuntimeError, "truth is sealed"):
                assert_observation_context_inputs(["observations_normalized", truth])

    def test_native_boundary_is_explicit_and_fail_closed(self):
        for stage in ("selection", "realization", "model_view", "train", "compare", "plot"):
            assert_saved_artifact_stage(stage)
            with self.assertRaisesRegex(RuntimeError, "PARTONS launch is forbidden"):
                assert_native_launch_allowed(stage)
        assert_native_launch_allowed("exact_reevaluate")

    def test_train_only_normalization(self):
        values = np.asarray([[0.0], [2.0], [1000.0]])
        fit = fit_train_only_standardization(values, [0, 1])
        self.assertEqual(fit["mean"], [1.0])
        self.assertEqual(fit["scale"], [1.0])

    def test_truth_request_and_preflight_do_not_expand_coordinates(self):
        request = json.loads((
            ROOT / "configs/examples/master_corpus_gpd_truth_smoke_v1.json"
        ).read_text(encoding="utf-8"))
        validated = validate_gpd_truth_request(request)
        self.assertEqual(len(validated["coordinates"]), 2)
        report = preflight_master_corpus(
            native_group_count=4, observable_kinematic_count=3,
            observable_count=2, shard_size=2, gpd_truth_request=request,
            declared_kinematic_coverage={"source": "test-only"},
        )
        self.assertFalse(report["native_physics_executed"])
        self.assertEqual(report["gpd_truth_points_per_group"], 2)
        self.assertEqual(report["observable_evaluation_count"], 24)
        self.assertTrue(
            report["model_family_compatibility"]["neural_gpd_deepsets_maf"]["compatible"]
        )

    def test_only_two_deepsets_maf_families(self):
        registry = model_family_registry()
        self.assertEqual(
            set(registry), {"dd_deepsets_maf", "neural_gpd_deepsets_maf"}
        )
        for family in registry.values():
            self.assertEqual(family["observation_encoder"], "deepsets")
            self.assertEqual(family["density_estimator"], "maf")
            self.assertEqual(family["physics_backend"], "partons")

    def test_maf_rejects_stale_spline_fields(self):
        validate_maf_only({"density_estimator": "maf", "num_transforms": 5})
        for stale in (
            {"density_estimator": "nsf"},
            {"density_estimator": "maf", "num_bins": 8},
        ):
            with self.assertRaisesRegex(ValueError, "MAF is the only"):
                validate_maf_only(stale)

    def test_constraints_validate_and_lattice_is_disabled(self):
        registry = json.loads((
            ROOT / "configs/physics/constraints_v1.json"
        ).read_text(encoding="utf-8"))
        validate_constraint_registry(registry)
        lattice = next(
            item for item in registry["constraints"]
            if item["identifier"].startswith("lattice_")
        )
        self.assertFalse(lattice["enabled"])

    def test_comparison_rejects_source_mismatch_and_uses_common_metrics(self):
        contract = {
            "master_corpus_id": "c", "selection_id": "s", "realization_id": "r",
            "group_split_id": "g", "noise_contract_id": "n",
            "observable_set_id": "o", "kinematic_set_id": "k",
            "locked_holdout_id": "h",
        }
        assert_architecture_comparable(contract, deepcopy(contract))
        other = deepcopy(contract)
        other["realization_id"] = "different"
        with self.assertRaisesRegex(ValueError, "realization_id"):
            assert_architecture_comparable(contract, other)
        metrics = common_function_metrics(
            np.asarray([0.0, 1.0]),
            np.asarray([[-0.1, 0.9], [0.1, 1.1], [0.0, 1.0]]),
        )
        self.assertFalse(metrics["raw_cross_representation_nll_used"])

    def test_latent_prior_is_explicit_training_only(self):
        prior = fitted_diagonal_gaussian_prior(
            np.asarray([[0.0, 1.0], [2.0, 3.0], [1000.0, 1000.0]]), [0, 1]
        )
        self.assertEqual(prior.fit_scope, "training_groups_only")
        self.assertEqual(prior.mean, (1.0, 2.0))

    def test_decoder_refuses_extrapolation(self):
        stored = np.asarray([[-0.5, 0.1], [0.5, 0.1]])
        require_exact_coordinate_support(stored, stored, "exact_coordinates_only")
        with self.assertRaisesRegex(RuntimeError, "refuses extrapolation"):
            require_exact_coordinate_support(
                np.asarray([[-0.6, 0.1], [0.5, 0.1]]), stored,
                "linear_inside_stored_hull",
            )

    def test_neural_decoder_shapes_masks_and_holdout_contract(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("pinned torch profile is not installed")
        coordinates = np.asarray([
            [-0.5, 0.1, -0.2, 0.0],
            [0.0, 0.1, -0.2, 0.0],
            [0.5, 0.1, -0.2, 0.0],
        ], dtype=np.float32)
        values = np.asarray([
            [-0.5, 0.0, 0.5], [-0.4, 0.1, 0.6], [-0.3, 0.2, 0.7],
            [-0.2, 0.3, 0.8], [-0.1, 0.4, 0.9], [0.0, 0.5, 1.0],
        ], dtype=np.float32)
        masks = np.ones_like(values)
        masks[1, 0] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            result = train_function_autoencoder(
                coordinates=coordinates, values=values, masks=masks,
                training_groups=[0, 1, 2], validation_groups=[3],
                locked_holdout_groups=[4, 5],
                output=Path(directory) / "decoder.pt", latent_dim=2,
                hidden_width=8, hidden_depth=1, epochs=2, seed=17,
            )
        self.assertEqual(result["latent_targets"].shape, (6, 2))
        self.assertFalse(result["holdout_used_for_tuning"])
        self.assertEqual(result["latent_prior"]["fit_scope"], "training_groups_only")
        self.assertEqual(len(result["reconstruction_rmse_by_group"]), 6)

    def test_literature_registry_and_coverage_are_fail_closed(self):
        registry = load_registry(ROOT / "configs/literature/benchmarks_v1.json")
        report = coverage_requirements(registry, {"schema_version": 1})
        nngpd = next(
            item for item in report["coverage_requirements"]
            if item["benchmark_id"] == "nngpd_function_closure"
        )
        self.assertFalse(nngpd["compatible"])
        self.assertEqual(nngpd["missing_artifacts"], ["canonical_gpd_truth"])
        self.assertFalse(nngpd["corpus_configuration_modified"])

    def test_literature_overlay_requires_exact_conventions(self):
        convention = {
            "gpd": "H", "flavor_combination": "u_plus",
            "normalization": "PARTONS_native", "scale_GeV2": 4.0,
            "scheme": "MSbar", "perturbative_order": "LO",
            "sign_convention": "PARTONS_native",
        }
        assert_conventions_compatible(convention, deepcopy(convention))
        mismatch = deepcopy(convention)
        mismatch["scale_GeV2"] = 2.0
        with self.assertRaisesRegex(ValueError, "scale_GeV2"):
            assert_conventions_compatible(convention, mismatch)


if __name__ == "__main__":
    unittest.main()
