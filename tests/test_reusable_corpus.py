import json
import io
import os
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from extract_dvcs_cff.corpus import (
    create_corpus,
    create_selection,
    export_corpus,
    generate_corpus,
    import_corpus,
    materialize_selection,
    merge_corpora,
    verify_corpus,
)
from extract_dvcs_cff.cli.user import (
    _load_experiment,
    _parser,
    _public_experiment,
    assert_result_contract,
    main,
)
from extract_dvcs_cff.workflows.pseudodata import (
    ParallelNativeResult,
    PartitionedNativeResult,
    _load_generated,
    _mean_estimator_log_prob,
    sha256,
)
from extract_dvcs_cff.inference.stage10 import prepare_stage10_context


REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / "configs/inference/stage10_pseudodata_workflow.json"
PHYSICS = REPO / "configs/physics/stage10_pseudodata_systematics_v1.json"


class ReusableCorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bridge = self.root / "partons_bridge"
        self.bridge.write_bytes(b"audited-test-bridge")
        self.bridge.chmod(0o755)
        config = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        config["physics_configuration"] = "physics.json"
        config["profiles"]["quick"]["native_parameter_count"] = 4
        config["profiles"]["quick"]["noise_replicates_per_parameter"] = 2
        config["observables"] = config["observables"][:1]
        self.configuration = self.root / "workflow.json"
        self.configuration.write_text(json.dumps(config), encoding="utf-8")
        (self.root / "physics.json").write_bytes(PHYSICS.read_bytes())
        self.corpus = self.root / "corpus"

    def tearDown(self):
        self.temporary.cleanup()

    def test_selective_generated_loader_memory_maps_only_requested_arrays(self):
        generated = self.root / "workspace/generated"
        generated.mkdir(parents=True)
        contexts = np.arange(24, dtype=np.float32).reshape(3, 8)
        theta = np.arange(6, dtype=np.float32).reshape(3, 2)
        np.save(generated / "contexts.npy", contexts, allow_pickle=False)
        np.save(generated / "theta_latent.npy", theta, allow_pickle=False)
        manifest = {"arrays": {}}
        for name in ("contexts", "theta_latent"):
            path = generated / f"{name}.npy"
            value = np.load(path, allow_pickle=False)
            manifest["arrays"][name] = {
                "sha256": sha256(path),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        (generated / "array_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        loaded = _load_generated(
            self.root / "workspace", names=("contexts",)
        )
        self.assertEqual(set(loaded), {"contexts"})
        self.assertIsInstance(loaded["contexts"], np.memmap)
        np.testing.assert_array_equal(loaded["contexts"], contexts)

    def test_estimator_scoring_respects_device_batch_bound(self):
        import torch

        class Estimator:
            def __init__(self):
                self.batch_sizes = []

            def log_prob(self, theta, context):
                self.batch_sizes.append(len(theta))
                return theta[:, 0] + context[:, 0]

        estimator = Estimator()
        theta = torch.arange(10, dtype=torch.float32).reshape(10, 1)
        context = (2 * theta).clone()
        indices = torch.tensor([9, 1, 7, 2, 5, 3, 8], dtype=torch.long)
        observed = _mean_estimator_log_prob(
            estimator, theta, context, indices,
            device=torch.device("cpu"), batch_size=3,
        )
        expected = float((3 * indices.float()).mean().item())
        self.assertAlmostEqual(observed, expected)
        self.assertEqual(estimator.batch_sizes, [3, 3, 1])

    def test_precomputed_context_is_exactly_legacy_equivalent(self):
        points = [
            {
                "x_b": 0.2, "t_GeV2": -0.1, "Q2_GeV2": 2.0,
                "beam_energy_GeV": 10.6, "phi_rad": 0.3,
                "observable_id": "DVCSCrossSectionUUMinus",
            },
            {
                "x_b": 0.3, "t_GeV2": -0.2, "Q2_GeV2": 3.0,
                "beam_energy_GeV": 11.0, "phi_rad": 1.2,
                "observable_id": "DVCSAluMinus",
            },
        ]
        observed = np.asarray([0.7, -0.2])
        covariance = np.asarray([[1.2, 0.1], [0.1, 0.8]])
        global_response = np.asarray([0.2, 0.3])
        lu_response = np.asarray([0.0, 0.4])
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        inverse_sqrt = (
            eigenvectors * eigenvalues**-0.5
        ) @ eigenvectors.T
        whitened = inverse_sqrt @ observed
        sigma = np.sqrt(np.diag(covariance))
        observable_ids = (
            "DVCSCrossSectionUUMinus",
            "DVCSCrossSectionDifferenceLUMinus",
            "DVCSAc", "DVCSAluMinus", "DVCSAulMinus", "DVCSAllMinus",
        )
        legacy = []
        for index, point in enumerate(points):
            legacy.extend((
                point["x_b"] / 0.3, -point["t_GeV2"] / 0.2,
                point["Q2_GeV2"], point["beam_energy_GeV"] / 12.0,
                np.sin(point["phi_rad"]), np.cos(point["phi_rad"]),
                *(float(point["observable_id"] == item)
                  for item in observable_ids),
                observed[index], sigma[index], whitened[index] / 10.0,
                global_response[index], lu_response[index], 1.0,
            ))
        legacy.extend((1.0,) * 9)
        expected = np.asarray(legacy, dtype=np.float32)
        builder = prepare_stage10_context(
            points=points, covariance=covariance,
            global_normalization_response=global_response,
            lu_normalization_response=lu_response,
        )
        np.testing.assert_array_equal(builder.build(observed), expected)

    @staticmethod
    def _native_result(parameters, config):
        count = len(parameters)
        kinematics = len(config["kinematics"])
        observables = len(config["observables"])
        base = parameters[:, :1]
        predictions = np.concatenate([
            base + 0.01 * (kinematic * observables + observable)
            for kinematic in range(kinematics)
            for observable in range(observables)
        ], axis=1)
        cffs = np.repeat(base[:, None, None, :], kinematics * 8, axis=1)
        cffs = cffs.reshape(count, kinematics, 4, 2)
        return predictions, cffs

    def _create(self, *, corpus=None):
        capabilities = SimpleNamespace(
            returncode=0, stderr="",
            stdout=json.dumps({"status": "ok", "backend": {"test": True}}),
        )
        with patch("extract_dvcs_cff.corpus.subprocess.run", return_value=capabilities):
            create_corpus(
                corpus=corpus or self.corpus, bridge=self.bridge,
                configuration_path=self.configuration, profile="quick",
                shard_size=2,
            )

    def _generate(
        self, names, *, corpus=None, max_shards=None, shard_start=None,
    ):
        def fake_evaluate(_client, *, parameters, config, **_kwargs):
            predictions, cffs = self._native_result(parameters, config)
            return SimpleNamespace(
                predictions=predictions, cffs=cffs,
                request_sha256="1" * 64, response_sha256="2" * 64,
                bridge_sha256=sha256(self.bridge),
            )

        def fake_parallel(*, parameters, config, **_kwargs):
            predictions, cffs = self._native_result(parameters, config)
            result = PartitionedNativeResult(
                predictions=predictions, cffs=cffs,
                gpds=np.empty((len(parameters), len(config["kinematics"]), 0, 4, 11)),
                valid_mask=np.ones(len(parameters), dtype=bool),
                invalid_records=(), successful_batches=(),
            )
            return ParallelNativeResult(
                result=result,
                worker_resolution={"resolved_workers": 1},
                chunk_size=2, task_count=1,
            )

        with patch(
            "extract_dvcs_cff.corpus.NativeCache.evaluate",
            new=fake_evaluate,
        ), patch(
            "extract_dvcs_cff.corpus._parallel_partitioned_native_evaluation",
            new=fake_parallel,
        ):
            return generate_corpus(
                corpus=corpus or self.corpus, bridge=self.bridge,
                requested_observables=names, show_progress=False,
                max_shards=max_shards,
                shard_start=shard_start,
            )

    def test_atomic_shards_extension_selection_and_deterministic_realization(self):
        self._create()
        first = "DVCSCrossSectionUUMinus"
        second = "DVCSCrossSectionDifferenceLUMinus"
        self._generate([first])
        old_shards = {
            path.name: sha256(path)
            for path in (self.corpus / "observables" / first / "shards").glob("*.npz")
        }
        self._generate([first, second])
        self.assertEqual(old_shards, {
            path.name: sha256(path)
            for path in (self.corpus / "observables" / first / "shards").glob("*.npz")
        })
        verified = verify_corpus(corpus=self.corpus, deep=True)
        self.assertEqual(verified["observable_count"], 2)

        archive = self.root / "portable.tar.gz"
        exported = export_corpus(corpus=self.corpus, archive_path=archive)
        self.assertEqual(exported["status"], "exported")
        imported_path = self.root / "imported-corpus"
        imported = import_corpus(archive_path=archive, corpus=imported_path)
        self.assertEqual(imported["verification"]["status"], "verified")
        self.assertEqual(
            verify_corpus(corpus=imported_path, deep=True)["observable_count"],
            2,
        )

        config = json.loads(self.configuration.read_text(encoding="utf-8"))
        canonical = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        config["observables"] = canonical["observables"][:2]
        self.configuration.write_text(json.dumps(config), encoding="utf-8")
        selection = self.root / "baseline.json"
        selected = create_selection(
            corpus=self.corpus, selection_path=selection, profile="quick",
            configuration_path=self.configuration,
        )
        groups = (
            set(selected["training_group_indices"]),
            set(selected["validation_group_indices"]),
            set(selected["outer_test_group_indices"]),
        )
        self.assertFalse(groups[0] & groups[1])
        self.assertFalse(groups[0] & groups[2])
        self.assertFalse(groups[1] & groups[2])

        outputs = []
        for name, workers in (("run-a", 1), ("run-b", 4)):
            workspace = self.root / name
            with patch(
                "extract_dvcs_cff.native_parallel.available_affinity_cpus",
                return_value=(0, 1, 2, 3),
            ):
                materialized = materialize_selection(
                    corpus=self.corpus, selection_path=selection,
                    configuration_path=self.configuration, workspace=workspace,
                    profile="quick", bridge=self.bridge, workers=workers,
                )
            self.assertEqual(materialized["materialization_workers"], workers)
            progress = json.loads(
                (workspace / "materialization_progress.json").read_text()
            )
            self.assertEqual(progress["status"], "materialized")
            assert_result_contract(
                configuration=self.configuration,
                workspace=workspace,
                profile="quick",
                bridge=self.bridge,
            )
            with patch(
                "extract_dvcs_cff.corpus._load_exact_arrays",
                side_effect=AssertionError("identical realization was rebuilt"),
            ):
                reused = materialize_selection(
                    corpus=self.corpus, selection_path=selection,
                    configuration_path=self.configuration,
                    workspace=workspace, profile="quick", bridge=self.bridge,
                )
            self.assertEqual(reused["status"], "reused")
            progress = json.loads(
                (workspace / "materialization_progress.json").read_text()
            )
            self.assertEqual(progress["status"], "reused")
            outputs.append({
                path.name: sha256(path)
                for path in sorted((workspace / "generated").glob("*.npy"))
            })
        self.assertEqual(outputs[0], outputs[1])

    def test_deep_verification_rejects_corrupt_shard(self):
        self._create()
        self._generate(["DVCSCrossSectionUUMinus"])
        shard = next((self.corpus / "core/shards").glob("*.npz"))
        payload = bytearray(shard.read_bytes())
        payload[len(payload) // 2] ^= 1
        shard.write_bytes(payload)
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            verify_corpus(corpus=self.corpus, deep=True)

    def test_partial_checkpoint_export_import_and_resume(self):
        self._create()
        observable = "DVCSCrossSectionUUMinus"
        partial = self._generate([observable], max_shards=1)
        self.assertEqual(partial["status"], "partial")
        self.assertEqual(partial["completed_shards"], 1)
        with self.assertRaisesRegex(RuntimeError, "core is incomplete"):
            verify_corpus(corpus=self.corpus, deep=True)
        with self.assertRaisesRegex(RuntimeError, "core is incomplete"):
            export_corpus(
                corpus=self.corpus,
                archive_path=self.root / "ordinary-export.tar.gz",
            )
        checkpoint = self.root / "checkpoint.tar.gz"
        exported = export_corpus(
            corpus=self.corpus, archive_path=checkpoint,
            allow_partial=True,
        )
        self.assertEqual(exported["status"], "checkpoint_exported")
        resumed = self.root / "resumed"
        imported = import_corpus(
            archive_path=checkpoint, corpus=resumed, allow_partial=True,
        )
        self.assertEqual(imported["verification"]["completed_shards"], 1)
        with self.assertRaisesRegex(RuntimeError, "core is incomplete"):
            import_corpus(
                archive_path=checkpoint,
                corpus=self.root / "ordinary-import",
            )
        self.assertFalse((self.root / "ordinary-import").exists())
        completed = self._generate([observable], corpus=resumed, max_shards=1)
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(
            verify_corpus(corpus=resumed, deep=True)["status"], "verified"
        )

    def test_disjoint_shard_batches_merge_and_reject_overlap(self):
        observable = "DVCSCrossSectionUUMinus"
        first = self.root / "batch-first"
        second = self.root / "batch-second"
        overlap = self.root / "batch-overlap"
        for corpus in (first, second, overlap):
            self._create(corpus=corpus)
        first_result = self._generate(
            [observable], corpus=first, max_shards=1, shard_start=0,
        )
        second_result = self._generate(
            [observable], corpus=second, max_shards=1, shard_start=1,
        )
        self._generate(
            [observable], corpus=overlap, max_shards=1, shard_start=0,
        )
        self.assertEqual(first_result["generated_shard_indices"], [0])
        self.assertEqual(second_result["generated_shard_indices"], [1])

        imported_batches = []
        for index, corpus in enumerate((first, second), 1):
            archive = self.root / f"batch-{index}.tar.gz"
            export_corpus(
                corpus=corpus, archive_path=archive, allow_partial=True,
            )
            imported = self.root / f"imported-batch-{index}"
            import_corpus(
                archive_path=archive, corpus=imported, allow_partial=True,
            )
            imported_batches.append(imported)

        merged = self.root / "merged"
        result = merge_corpora(
            corpora=list(reversed(imported_batches)), destination=merged,
            consume_sources=True,
        )
        self.assertEqual(result["status"], "merged")
        self.assertEqual(result["shard_count"], 2)
        self.assertTrue(result["sources_consumed"])
        self.assertTrue(all(not path.exists() for path in imported_batches))
        self.assertEqual(
            verify_corpus(corpus=merged, deep=True)["status"], "verified"
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate corpus shard"):
            merge_corpora(
                corpora=[first, overlap],
                destination=self.root / "overlapping-merge",
            )


class PublicCorpusInterfaceTests(unittest.TestCase):
    def test_parser_supports_separate_materialize_and_train(self):
        parser = _parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["generate", "study"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["run", "study"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["materialize", "study"])
        downstream = parser.parse_args(["train", "study"])
        self.assertIsNone(downstream.corpus)
        self.assertIsNone(downstream.selection)
        parsed = parser.parse_args([
            "materialize", "study", "--corpus", "shared",
            "--selection", "baseline", "--workers", "4",
        ])
        self.assertEqual((parsed.corpus, parsed.selection), ("shared", "baseline"))
        self.assertEqual(parsed.workers, "4")

    def test_workspace_listing_excludes_internal_corpus_stores(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            for name in ("study", ".corpora", ".corpus_exports"):
                (workspace / name).mkdir()
            arguments = [
                "dvcs-infer", "--repository-root", str(REPO), "list",
            ]
            output = io.StringIO()
            with patch.dict(
                os.environ, {"DVCS_WORKSPACE_ROOT": str(workspace)}, clear=False
            ), patch("sys.argv", arguments), redirect_stdout(output):
                self.assertEqual(main(), 0)
            self.assertEqual(json.loads(output.getvalue())["projects"], ["study"])

    def test_observable_metadata_is_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "study"
            project.mkdir()
            configuration = json.loads(WORKFLOW.read_text(encoding="utf-8"))
            configuration["_physics"] = json.loads(
                PHYSICS.read_text(encoding="utf-8")
            )
            experiment = _public_experiment(
                "study", configuration, {"mode": "manual"}
            )
            experiment["synthetic_dataset"]["observables"] = (
                experiment["synthetic_dataset"]["observables"][:1]
            )
            experiment["synthetic_dataset"]["observables"][0][
                "normalization_scale"
            ] = 2.0
            (project / "experiment.json").write_text(
                json.dumps(experiment), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "metadata does not match"):
                _load_experiment(project)

            legacy = _public_experiment(
                "study", configuration, {"mode": "manual"}
            )
            legacy["schema_version"] = 7
            del legacy["synthetic_dataset"]["observables"]
            (project / "experiment.json").write_text(
                json.dumps(legacy), encoding="utf-8"
            )
            self.assertIsNone(_load_experiment(project)["observables"])

if __name__ == "__main__":
    unittest.main()
