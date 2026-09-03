"""Focused regressions for the scientific-validity contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from extract_dvcs_cff.data.uncertainty import (
    materialize_observation,
    materialize_uncertainty_variant,
    resolve_covariance,
    uncertainty_preset,
)
from extract_dvcs_cff.literature import load_comparator_registry
from extract_dvcs_cff.neural_gpd import (
    apply_whitening,
    canonical_coordinate_table,
    expand_group_targets,
    fit_full_covariance_whitening,
    invert_whitening,
    require_exact_coordinate_support,
)
from extract_dvcs_cff.scientific_validation import (
    GeneratorMetadata,
    audit_native_failures,
    best_dd_projection,
    classify_holdout,
    publish_best_dd_projection,
    require_evaluation_access,
    require_foreign_model_metrics,
    unseal_evaluation,
    validate_production_provenance,
)


ROOT = Path(__file__).resolve().parents[1]


class UncertaintyContractTests(unittest.TestCase):
    def test_paired_noise_scaling_does_not_reexecute_native_physics(self) -> None:
        central = np.array([2.0, -1.0])
        covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
        z = np.array([0.5, -0.25])
        full = materialize_observation(
            central=central, covariance=covariance,
            contract=uncertainty_preset("full_uncertainty_posterior"),
            standard_normal=z,
        )
        half = materialize_observation(
            central=central, covariance=covariance,
            contract=uncertainty_preset(
                "full_uncertainty_posterior", physical_noise_scale=0.5
            ), standard_normal=z,
        )
        np.testing.assert_allclose(
            half["observed"] - central, 0.5 * (full["observed"] - central)
        )
        np.testing.assert_allclose(half["physical_covariance"], 0.25 * covariance)
        self.assertFalse(half["native_physics_executed"])
        self.assertGreater(half["stabilized_covariance"][0, 0],
                           half["physical_covariance"][0, 0])

    def test_central_observation_and_zero_noise_policy(self) -> None:
        central = np.array([1.0, 2.0])
        result = materialize_observation(
            central=central, covariance=np.eye(2),
            contract=uncertainty_preset("central_observation_full_likelihood"),
            standard_normal=np.ones(2),
        )
        np.testing.assert_array_equal(result["observed"], central)
        with self.assertRaisesRegex(ValueError, "positive physical noise"):
            uncertainty_preset("full_uncertainty_posterior", physical_noise_scale=0.0)

    def test_fixed_relative_replica_is_diagonal_point_estimate(self) -> None:
        contract = uncertainty_preset("literature_fixed_relative_replica")
        result = materialize_observation(
            central=np.array([2.0, -4.0]), covariance=np.eye(2),
            contract=contract, standard_normal=np.zeros(2),
        )
        np.testing.assert_allclose(np.diag(result["reference_covariance"]), [0.04, 0.16])
        self.assertEqual(contract.output_kind, "point_estimate")
        self.assertEqual(contract.uncertainty_exposure, "values_and_kinematics_only")

    def test_covariance_policy_is_independent_of_generator_label(self) -> None:
        covariance = np.array([[2.0, 0.25], [0.25, 1.0]])
        first = resolve_covariance(policy="training_reference", training_reference=covariance)
        second = resolve_covariance(policy="training_reference", training_reference=covariance)
        np.testing.assert_array_equal(first, second)

    def test_variants_reuse_native_ancestry_and_paired_draws(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            arguments = {
                "artifact_root": Path(temporary), "master_corpus_id": "corpus-1",
                "selection_id": "selection-1", "central": np.array([1.0, 2.0]),
                "training_reference_covariance": np.eye(2),
                "standard_normal_draws": np.array([[0.25, -0.5], [-0.2, 0.1]]),
            }
            full = materialize_uncertainty_variant(
                **arguments, contract=uncertainty_preset("full_uncertainty_posterior")
            )
            repeated = materialize_uncertainty_variant(
                **arguments, contract=uncertainty_preset("full_uncertainty_posterior")
            )
            reduced = materialize_uncertainty_variant(
                **arguments, contract=uncertainty_preset(
                    "near_noiseless_posterior", physical_noise_scale=0.25
                )
            )
            self.assertEqual(full["master_corpus_id"], reduced["master_corpus_id"])
            self.assertEqual(full["realization_id"], repeated["realization_id"])
            self.assertNotEqual(full["realization_id"], reduced["realization_id"])
            self.assertTrue(full["native_predictions_reused"])
            self.assertFalse(full["native_physics_executed"])


class NeuralTargetContractTests(unittest.TestCase):
    def test_group_targets_expand_to_replicas(self) -> None:
        targets = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        indices = np.array([2, 0, 2, 1, 0], dtype=np.int64)
        np.testing.assert_array_equal(expand_group_targets(targets, indices), targets[indices])

    def test_full_covariance_whitening_round_trip(self) -> None:
        latent = np.array([
            [1.0, 2.0], [2.0, 0.0], [3.0, 4.0], [4.0, 1.0], [5.0, 3.0]
        ])
        transform = fit_full_covariance_whitening(latent, [0, 1, 2, 3, 4])
        reconstructed = invert_whitening(apply_whitening(latent, transform), transform)
        np.testing.assert_allclose(reconstructed, latent, atol=1e-9)
        self.assertEqual(transform["fit_scope"], "training_groups_only")

    def test_collapsed_latent_dimension_is_removed(self) -> None:
        latent = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0], [4.0, 7.0]])
        transform = fit_full_covariance_whitening(
            latent, [0, 1, 2, 3], shrinkage=0.0
        )
        self.assertEqual(transform["retained_dimensions"], 1)
        np.testing.assert_allclose(
            invert_whitening(apply_whitening(latent, transform), transform), latent
        )

    def test_coordinate_identity_and_support_are_strict(self) -> None:
        coordinate = {
            "x": 0.2, "xi": 0.1, "t_GeV2": -0.2,
            "gpd_type": "H", "channel": "u", "charge_parity": "native",
        }
        table = canonical_coordinate_table([coordinate], Q2_GeV2=2.0)
        self.assertIn("gpd_is_H", table["identity"]["feature_names"])
        require_exact_coordinate_support(table["values"], table["values"],
                                         "exact_coordinates_only")
        with self.assertRaisesRegex(RuntimeError, "refuses extrapolation"):
            require_exact_coordinate_support(
                table["values"] + 1e-12, table["values"], "exact_coordinates_only"
            )


class ScientificValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.training = GeneratorMetadata(
            "double_distribution", "internal_dd", "v1", "full", "physics-a",
            "omitted", "PARTONS",
        )

    def test_generator_holdout_classification_and_metric_guard(self) -> None:
        gk = GeneratorMetadata(
            "double_distribution", "GK", "published", "GK", "physics-b",
            "present", "PARTONS",
        )
        self.assertEqual(
            classify_holdout(self.training, gk),
            "same_representation_named_family_holdout",
        )
        with self.assertRaisesRegex(ValueError, "no internal-DD coordinate truth"):
            require_foreign_model_metrics(
                common_parameter_truth=False, requested_metrics=["parameter_rmse"]
            )
        self.assertEqual(set(self.training.as_dict()), {
            "representation_class", "named_model_family", "implementation_version",
            "parameterization_variant", "physics_configuration", "D_term_policy",
            "source_backend",
        })

    def test_native_failure_audit_uses_recorded_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus = Path(temporary)
            (corpus / "core/shards").mkdir(parents=True)
            (corpus / "rejections").mkdir()
            np.savez(
                corpus / "core/shards/shard-000000.npz",
                native_parameters=np.array([[0.1, 0.2], [0.8, 0.9]]),
            )
            (corpus / "corpus.json").write_text(json.dumps({
                "manifest_sha256": "manifest-1",
                "core_shards": [{"path": "core/shards/shard-000000.npz"}],
            }), encoding="utf-8")
            (corpus / "rejections/shard-000000.json").write_text(json.dumps({
                "shard_index": 0, "attempted": 3, "accepted": 2,
                "rejected": [{
                    "parameters": [0.95, 0.95],
                    "native_error": "domain_error: invalid point",
                }],
            }), encoding="utf-8")
            report = audit_native_failures(corpus, bin_count=2)
            self.assertEqual(report["proposal_count"], 3)
            self.assertEqual(report["failure_types"], {"domain_error": 1})
            self.assertFalse(report["accepted_distribution_is_original_prior"])

    def test_projection_is_reproducible_and_records_restarts(self) -> None:
        answer = best_dd_projection(
            objective=lambda x: float(np.sum((x - np.array([0.25, -0.4])) ** 2)),
            bounds=np.array([[-1.0, 1.0], [-1.0, 1.0]]), restarts=4,
            iterations=80, seed=11,
        )
        self.assertLess(answer["best_objective"], 1e-14)
        self.assertEqual(len(answer["restarts"]), 4)
        self.assertFalse(answer["global_optimum_claimed"])
        with tempfile.TemporaryDirectory() as temporary:
            artifact = publish_best_dd_projection(
                output=Path(temporary) / "projection.json",
                optimization_result=answer,
                foreign_generator=GeneratorMetadata(
                    "double_distribution", "GK", "published", "GK", "physics-b",
                    "present", "PARTONS",
                ),
                covariance_policy="external_fixed",
                discrepancy="fixed_covariance_weighted_noiseless_observables",
                parameter_transform_id="dd_uniform_probit_v1",
            )
            self.assertFalse(artifact["DD_parameter_recovery_metric_permitted"])

    def test_production_provenance_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "checkpoint_hash"):
            validate_production_provenance({}, profile="production")
        self.assertIn("checkpoint_hash", validate_production_provenance({}, profile="quick"))

    def test_family_dispatch_never_falls_through_to_dd(self) -> None:
        from extract_dvcs_cff.model_registry import require_model_stage
        self.assertEqual(
            require_model_stage("dd_deepsets_maf", "train").identifier,
            "dd_deepsets_maf",
        )
        with self.assertRaisesRegex(RuntimeError, "will not silently dispatch to DD"):
            require_model_stage("neural_gpd_deepsets_maf", "train")

    def test_sealed_artifact_requires_explicit_audited_unseal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "ledger.json"
            with self.assertRaises(PermissionError):
                require_evaluation_access(
                    role="sealed_final", artifact_id="final-v1", unseal_ledger=ledger
                )
            record = unseal_evaluation(
                ledger=ledger, artifact_id="final-v1", reason="frozen paper table",
                git_commit="abc123", configuration={"seed": 7},
                artifact_identities={"realization_id": "realization-1"},
            )
            self.assertEqual(record["artifact_id"], "final-v1")
            require_evaluation_access(
                role="sealed_final", artifact_id="final-v1", unseal_ledger=ledger
            )


class TelemetryAndRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "jobs/jlab_ifarm/collect_performance_metrics.py"
        spec = importlib.util.spec_from_file_location("collect_performance_metrics", path)
        assert spec is not None and spec.loader is not None
        cls.telemetry = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.telemetry)

    def test_gpu_number_parser(self) -> None:
        cases = {
            "12": 12.0, " 12.5 ": 12.5, "-1.25e2": -125.0,
            "42 %": 42.0, "1024 MiB": 1024.0, "": None,
            "N/A": None, "nan": None, "12 parsecs": None,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.telemetry.number(text), expected)

    def test_comparator_registry_is_schema_complete(self) -> None:
        registry = load_comparator_registry(
            ROOT / "benchmarks/literature/comparators_v1.json"
        )
        self.assertGreaterEqual(len(registry["comparators"]), 10)


class ValidationBatchTests(unittest.TestCase):
    def test_partial_validation_batch_is_counted_exactly(self) -> None:
        try:
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            from extract_dvcs_cff.inference.stage10 import (
                GroupedValidationNPE, _SubsetSequentialSampler,
            )
        except ImportError as error:
            self.skipTest(str(error))
        estimator = object.__new__(GroupedValidationNPE)
        estimator._grouped_validation_indices = torch.arange(5)
        estimator._get_losses = lambda batch: batch[0]
        loader = DataLoader(
            TensorDataset(torch.arange(5, dtype=torch.float64)), batch_size=2,
            sampler=_SubsetSequentialSampler([0, 1, 2, 3, 4]), drop_last=False,
        )
        observed = GroupedValidationNPE._validate_epoch(estimator, loader, None)
        first_order = [batch[0].tolist() for batch in loader]
        second_order = [batch[0].tolist() for batch in loader]
        self.assertEqual([len(batch) for batch in first_order], [2, 2, 1])
        self.assertEqual(first_order, second_order)
        self.assertEqual([item for batch in first_order for item in batch], [0, 1, 2, 3, 4])
        self.assertAlmostEqual(observed, 2.0)


if __name__ == "__main__":
    unittest.main()
