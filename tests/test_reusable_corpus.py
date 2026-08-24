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
    sha256,
)


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
        for name in ("run-a", "run-b"):
            workspace = self.root / name
            materialize_selection(
                corpus=self.corpus, selection_path=selection,
                configuration_path=self.configuration, workspace=workspace,
                profile="quick", bridge=self.bridge,
            )
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
            outputs.append(sha256(workspace / "generated/contexts.npy"))
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
    def test_parser_requires_explicit_corpus_and_selection(self):
        parser = _parser()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["generate", "study"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["run", "study"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["train", "study"])
        parsed = parser.parse_args([
            "train", "study", "--corpus", "shared",
            "--selection", "baseline",
        ])
        self.assertEqual((parsed.corpus, parsed.selection), ("shared", "baseline"))

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
