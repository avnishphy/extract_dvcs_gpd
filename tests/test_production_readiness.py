from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from render_physics_assumptions import load_registry, render  # noqa: E402


class ProductionReadinessTests(unittest.TestCase):
    def test_assumption_registry_and_generated_table_are_current(self) -> None:
        registry = load_registry(ROOT / "provenance/production-native-assumptions.json")
        self.assertEqual(len(registry["assumptions"]), 50)
        expected = render(registry) + "\n"
        observed = (ROOT / "docs/audits/PRODUCTION_NATIVE_ASSUMPTIONS.md").read_text(
            encoding="utf-8"
        )
        self.assertEqual(observed, expected)

    def test_regeneration_matrix_is_complete_and_uses_closed_vocabulary(self) -> None:
        matrix = json.loads(
            (ROOT / "provenance/regeneration-impact-matrix.json").read_text(
                encoding="utf-8"
            )
        )
        allowed = set(matrix["classification_order"])
        self.assertGreaterEqual(len(matrix["choices"]), 39)
        self.assertTrue(all(item["impact"] in allowed for item in matrix["choices"]))
        self.assertFalse(matrix["policy"]["architecture_changes_trigger_partons"])
        self.assertFalse(matrix["policy"]["covariance_noise_changes_trigger_partons"])

    def test_preflight_fails_closed_on_known_unapproved_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "preflight.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/production_preflight.py"),
                    "--skip-native-runtime",
                    "--allow-dirty",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            blocker_ids = {item["id"] for item in report["blocking_failures"]}
            self.assertEqual(report["status"], "NO-GO")
            self.assertIn("corpus.function_grid_request", blocker_ids)
            self.assertIn("assumptions.researcher_approved", blocker_ids)
            self.assertIn("physics.kinematics_inside_declared_domain", blocker_ids)


if __name__ == "__main__":
    unittest.main()
