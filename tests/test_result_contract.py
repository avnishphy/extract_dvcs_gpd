from copy import deepcopy
import tempfile
from pathlib import Path
import unittest

from extract_dvcs_cff.cli.user import assert_result_contract
from extract_dvcs_cff.workflows.pseudodata import (
    scientific_configuration_sha256,
    sha256,
    write_json,
)


class ResultContractTest(unittest.TestCase):
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
