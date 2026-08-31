from copy import deepcopy
import hashlib
import tempfile
from pathlib import Path
import unittest
from unittest import mock

import numpy as np

from extract_dvcs_cff.cli.user import _public_result, assert_result_contract
from extract_dvcs_cff.corpus import _masked_design_count, _nested_kinematic_order
from extract_dvcs_cff.workflows.pseudodata import (
    _conventional_reference_diagnostics,
    canonical,
    evaluate_native_model_holdouts,
    scientific_configuration_sha256,
    sha256,
    write_json,
)


class ResultContractTest(unittest.TestCase):
    def test_masked_training_designs_are_nested_and_deterministic(self):
        points = [
            {
                "x_b": 0.1 + 0.01 * index,
                "t_GeV2": -0.1 - 0.01 * index,
                "Q2_GeV2": 1.0 + index,
                "beam_energy_GeV": 10.6,
                "phi_rad": 0.2 * index,
            }
            for index in range(8)
        ]
        first = _nested_kinematic_order(points, 51017)
        second = _nested_kinematic_order(points, 51017)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(range(8)))
        self.assertTrue(set(first[:2]) < set(first[:5]))

    def test_masked_training_counts_are_balanced_across_groups(self):
        counts = (12, 30, 60, 96)
        self.assertEqual(
            {_masked_design_count(counts, 7, replica) for replica in range(4)},
            set(counts),
        )
        quick_rows = [
            _masked_design_count(counts, group, replica)
            for group in range(4)
            for replica in range(2)
        ]
        self.assertEqual({value: quick_rows.count(value) for value in counts}, {
            12: 2, 30: 2, 60: 2, 96: 2,
        })

    def test_holdout_uses_explicit_blind_manifest_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = root / "workspace/project/.engine/workflow.json"
            configuration.parent.mkdir(parents=True)
            write_json(configuration, {})
            workspace = root / "workspace/project/results/validation"
            workspace.mkdir(parents=True)
            blind = root / "release/configs/validation/blind.json"
            blind.parent.mkdir(parents=True)
            point = {
                "x_b": 0.2,
                "t_GeV2": -0.1,
                "Q2_GeV2": 2.0,
                "beam_energy_GeV": 10.6,
                "phi_rad": 0.5,
            }
            manifest = {
                "protocol": "output_blind_fresh_kinematics_v1",
                "selection_frozen_before_native_outputs": True,
                "native_outputs_generated": False,
                "training_use": False,
                "architecture_selection_use": False,
                "optuna_objective_use": False,
                "npe_dataset_points": [point],
            }
            manifest["manifest_content_sha256"] = hashlib.sha256(
                canonical(manifest).encode("utf-8")
            ).hexdigest()
            write_json(blind, manifest)

            with mock.patch(
                "extract_dvcs_cff.workflows.pseudodata.load_configuration",
                return_value={"kinematics": [{}], "diagnostics": {}},
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "completed synthetic evaluation and comparison",
                ):
                    evaluate_native_model_holdouts(
                        bridge=root / "bridge",
                        configuration_path=configuration,
                        blind_configuration_path=blind,
                        workspace=workspace,
                        profile="validation",
                    )

    def test_completed_failed_comparison_is_not_execution_failure(self):
        public = _public_result(
            "compare",
            Path("/workspace/test-project"),
            "validation",
            {
                "status": "complete_with_invalid_conventional_reference",
                "passed": False,
                "effective_sample_size": 1.0,
                "ensemble_sliced_wasserstein": None,
                "conventional_reference_available": False,
                "conventional_reference_diagnostics": {
                    "failure_code": (
                        "conventional_prior_importance_sampling_collapsed"
                    )
                },
            },
        )
        self.assertEqual(
            public["status"],
            "complete_with_invalid_conventional_reference",
        )
        self.assertFalse(public["scientific_passed"])

    def test_collapsed_conventional_reference_is_quarantined(self):
        weights = np.zeros(16, dtype=np.float64)
        weights[3] = 1.0
        diagnostics = _conventional_reference_diagnostics(
            weights=weights,
            resampled_indices=np.full(32, 3, dtype=np.int64),
            samples=np.ones((32, 82), dtype=np.float64),
            nuisance_draws=8,
            minimum_effective_sample_size=80.0,
        )
        self.assertFalse(diagnostics["available"])
        self.assertEqual(diagnostics["effective_sample_size"], 1.0)
        self.assertEqual(diagnostics["unique_resampled_proposal_count"], 1)
        self.assertEqual(diagnostics["collapsed_direction_count"], 82)
        self.assertEqual(
            diagnostics["failure_code"],
            "conventional_prior_importance_sampling_collapsed",
        )

    def test_supported_conventional_reference_remains_available(self):
        sample_count = 160
        samples = np.arange(sample_count * 82, dtype=np.float64).reshape(
            sample_count, 82
        )
        diagnostics = _conventional_reference_diagnostics(
            weights=np.full(sample_count, 1.0 / sample_count),
            resampled_indices=np.arange(sample_count, dtype=np.int64),
            samples=samples,
            nuisance_draws=8,
            minimum_effective_sample_size=80.0,
        )
        self.assertTrue(diagnostics["available"])
        self.assertIsNone(diagnostics["failure_code"])

    def test_downstream_runtime_change_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            generated = workspace / "generated"
            generated.mkdir(parents=True)
            configuration = root / "workflow.json"
            bridge = root / "bridge"
            bridge.write_bytes(b"bridge")

            materialization_configuration = {
                "physics": {"model": "fixed"},
                "runtime": {
                    "accelerator": "cpu",
                    "cpu_threads": 2,
                    "deterministic_algorithms": True,
                    "native_workers": "all_available",
                },
            }
            write_json(configuration, materialization_configuration)
            materialization_hash = scientific_configuration_sha256(
                configuration
            )

            realization = {
                "schema_version": 2,
                "corpus_manifest_sha256": "corpus",
                "selection_manifest_sha256": "selection",
                "realization_schema_version": 1,
            }
            realization_path = generated / "array_manifest.json"
            write_json(realization_path, realization)
            write_json(
                workspace / "workspace_contract.json",
                {
                    "schema_version": 1,
                    "configuration": str(configuration.resolve()),
                    "configuration_sha256": materialization_hash,
                    "profile": "quick",
                    "bridge_sha256": sha256(bridge),
                    "real_data": False,
                    "synthetic_split_policy": (
                        "native_parameter_grouped_train_validation_test_v1"
                    ),
                    "corpus_manifest_sha256": "corpus",
                    "selection_manifest_sha256": "selection",
                    "realization_schema_version": 1,
                    "realization_manifest_sha256": sha256(realization_path),
                },
            )

            for accelerator, cpu_threads in (
                ("cuda", 4),   # train
                ("cuda", 8),   # evaluate
                ("cpu", 2),    # compare / plot
                ("cpu", 16),   # holdout
            ):
                downstream_configuration = deepcopy(
                    materialization_configuration
                )
                downstream_configuration["runtime"]["accelerator"] = (
                    accelerator
                )
                downstream_configuration["runtime"]["cpu_threads"] = (
                    cpu_threads
                )
                write_json(configuration, downstream_configuration)
                assert_result_contract(
                    configuration=configuration,
                    workspace=workspace,
                    profile="quick",
                    bridge=bridge,
                )

            downstream_configuration["physics"]["model"] = "changed"
            write_json(configuration, downstream_configuration)
            with self.assertRaisesRegex(RuntimeError, "result contract mismatch"):
                assert_result_contract(
                    configuration=configuration,
                    workspace=workspace,
                    profile="quick",
                    bridge=bridge,
                )


if __name__ == "__main__":
    unittest.main()
