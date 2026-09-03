import json
from pathlib import Path
import tempfile
import unittest

from extract_dvcs_cff.cli.user import _load_experiment, _public_experiment
from extract_dvcs_cff.data.gpddatabase import (
    exact_dvcs_t_limits_GeV2,
    require_physical_fixed_target_kinematics,
)


class PhysicalKinematicsTests(unittest.TestCase):
    def test_minus_t_above_former_boundary_is_accepted(self):
        self.assertGreater(
            require_physical_fixed_target_kinematics(
                x_b=0.5,
                Q2_GeV2=1.1,
                beam_energy_GeV=10.6,
                t_GeV2=-1.5,
                phi_rad=1.0,
                context="point",
            ),
            0.0,
        )

    def test_exact_transfer_boundaries_are_accepted(self):
        backward, forward = exact_dvcs_t_limits_GeV2(
            x_b=0.5, Q2_GeV2=1.1
        )
        for value in (backward, forward):
            require_physical_fixed_target_kinematics(
                x_b=0.5,
                Q2_GeV2=1.1,
                beam_energy_GeV=10.6,
                t_GeV2=value,
                phi_rad=0.0,
                context="boundary",
            )

    def test_unphysical_y_and_t_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must lie in \\(0,1\\)"):
            require_physical_fixed_target_kinematics(
                x_b=0.1,
                Q2_GeV2=4.0,
                beam_energy_GeV=10.6,
                t_GeV2=-0.5,
                phi_rad=0.0,
                context="bad-y",
            )
        with self.assertRaisesRegex(ValueError, "exact finite-Q2 DVCS limits"):
            require_physical_fixed_target_kinematics(
                x_b=0.5,
                Q2_GeV2=1.1,
                beam_energy_GeV=10.6,
                t_GeV2=-0.1,
                phi_rad=0.0,
                context="bad-t",
            )

    def test_public_loader_accepts_user_point_beyond_former_boundary(self):
        repository = Path(__file__).resolve().parents[1]
        configuration = json.loads((
            repository / "configs/inference/stage10_pseudodata_workflow.json"
        ).read_text(encoding="utf-8"))
        configuration["_physics"] = json.loads((
            repository / "configs/physics/stage10_pseudodata_systematics_v1.json"
        ).read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "study"
            project.mkdir()
            experiment = _public_experiment(
                "study", configuration, {"mode": "manual"}
            )
            experiment["synthetic_dataset"]["kinematics"] = [
                {
                    "x_b": 0.5,
                    "t_GeV2": -1.5,
                    "phi_rad": 1.0,
                    "Q2_GeV2": 1.1,
                    "beam_energy_GeV": 10.6,
                }
            ]
            (project / "experiment.json").write_text(
                json.dumps(experiment), encoding="utf-8"
            )
            loaded = _load_experiment(project)
            self.assertTrue(all(
                point["t_GeV2"] == -1.5 for point in loaded["kinematics"]
            ))


if __name__ == "__main__":
    unittest.main()
