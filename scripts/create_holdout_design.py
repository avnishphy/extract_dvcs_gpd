#!/usr/bin/env python3
"""Create content-hashed native-model holdout designs without using data values."""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import math
from pathlib import Path
import sys
import tarfile
from typing import Any

import numpy as np

# JLab's control-host Python is 3.9 while runtime containers use 3.12.  The
# catalog connector uses Python 3.10's strict zip only after validating array
# lengths explicitly; accept that keyword in this standalone control script.
if sys.version_info < (3, 10):
    _zip = builtins.zip

    def _compatible_zip(*iterables: Any, strict: bool = False):
        del strict
        return _zip(*iterables)

    builtins.zip = _compatible_zip

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extract_dvcs_cff.data.gpddatabase import (  # noqa: E402
    build_kinematic_catalog,
    require_physical_fixed_target_kinematics,
)


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def load_project(
    archive: Path | None, directory: Path | None, project: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if archive is not None:
        with tarfile.open(archive.resolve(strict=True), "r:*") as stream:
            configuration = json.load(
                stream.extractfile(f"{project}/.engine/workflow.json")
            )
            experiment = json.load(
                stream.extractfile(f"{project}/experiment.json")
            )
        return configuration, experiment
    root = directory.resolve(strict=True)
    configuration = json.loads(
        (root / ".engine/workflow.json").read_text(encoding="utf-8")
    )
    experiment = json.loads(
        (root / "experiment.json").read_text(encoding="utf-8")
    )
    return configuration, experiment


def record(point: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": str(provenance.get("record_id", "unrecorded")),
        "dataset_id": str(provenance.get("dataset_id", "project-design")),
        "source_path": str(provenance.get("source_path", "experiment.json")),
        "source_observable": None,
        "native_observable": "all_configured_observables",
        **{key: float(point[key]) for key in (
            "x_b", "t_GeV2", "Q2_GeV2", "beam_energy_GeV", "phi_rad"
        )},
    }


def same_design(
    configuration: dict[str, Any], experiment: dict[str, Any], count: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    points = configuration["kinematics"]
    if count != len(points):
        raise SystemExit(
            f"same-design holdout requires all {len(points)} training kinematics"
        )
    source = experiment.get("synthetic_dataset", {}).get("kinematics_source", {})
    provenance = source.get("selected_records", [])
    if len(provenance) != len(points):
        provenance = [
            {"record_id": f"project-design:{index}", "dataset_id": "project-design"}
            for index in range(len(points))
        ]
    return [record(point, item) for point, item in zip(points, provenance)], {
        "catalog_sha256": source.get("catalog_sha256", "project-design"),
        "database_revision": source.get("database_revision", "project-design"),
        "selection_rule": "exact_frozen_project_kinematics_in_declared_order",
    }


def training_subset_design(
    configuration: dict[str, Any], experiment: dict[str, Any],
    count: int, seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    points = configuration["kinematics"]
    if not 1 <= count < len(points):
        raise SystemExit("training-subset count must be below full design size")
    order = _training_order(points, seed)
    source = experiment.get("synthetic_dataset", {}).get("kinematics_source", {})
    provenance = source.get("selected_records", [])
    if len(provenance) != len(points):
        provenance = [
            {"record_id": f"project-design:{index}", "dataset_id": "project-design"}
            for index in range(len(points))
        ]
    return [record(points[index], provenance[index]) for index in order[:count]], {
        "catalog_sha256": source.get("catalog_sha256", "project-design"),
        "database_revision": source.get("database_revision", "project-design"),
        "selection_rule": "nested_deterministic_farthest_point_v1",
    }


def _training_order(points: list[dict[str, Any]], seed: int) -> list[int]:
    features = np.asarray([
        [
            point["x_b"] / 0.5, -point["t_GeV2"] / 0.9,
            math.log(point["Q2_GeV2"]) / math.log(80.0),
            math.log(point["beam_energy_GeV"] / 3.0) / math.log(200.0 / 3.0),
            math.sin(point["phi_rad"]), math.cos(point["phi_rad"]),
        ]
        for point in points
    ], dtype=np.float64)
    first = min(range(len(points)), key=lambda index: hashlib.sha256(
        f"{seed}:{index}".encode("utf-8")
    ).hexdigest())
    chosen = [first]
    distance = np.sum((features - features[first]) ** 2, axis=1)
    distance[first] = -1.0
    while len(chosen) < len(points):
        next_index = int(np.flatnonzero(np.isclose(
            distance, distance.max(), rtol=0.0, atol=1e-15
        ))[0])
        chosen.append(next_index)
        distance = np.minimum(
            distance, np.sum((features - features[next_index]) ** 2, axis=1)
        )
        distance[chosen] = -1.0
    return chosen


def fresh_design(
    configuration: dict[str, Any], experiment: dict[str, Any],
    database: Path, count: int, seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    catalog = build_kinematic_catalog(database)
    metadata = {item["dataset_id"]: item for item in catalog["datasets"]}
    training_source = experiment.get("synthetic_dataset", {}).get(
        "kinematics_source", {}
    )
    excluded_ids = {
        str(item["record_id"])
        for item in training_source.get("selected_records", [])
    }
    training_coordinates = {
        tuple(round(float(point[key]), 12) for key in (
            "x_b", "t_GeV2", "Q2_GeV2", "beam_energy_GeV", "phi_rad"
        ))
        for point in configuration["kinematics"]
    }
    candidates: dict[tuple[float, ...], dict[str, Any]] = {}
    for item in catalog["kinematic_records"]:
        if str(item["record_id"]) in excluded_ids:
            continue
        values, units = item["kinematics"], item["kinematic_units"]
        dataset = metadata[item["dataset_id"]]
        beam = dataset.get("beam_energy_GeV")
        if beam is None or not {"xB", "t", "Q2", "phi"}.issubset(values):
            continue
        if units.get("xB") != "none" or units.get("t") != "GeV2" or units.get("Q2") != "GeV2":
            continue
        phi = float(values["phi"])
        if units.get("phi") == "deg":
            phi = math.radians(phi)
        elif units.get("phi") != "rad":
            continue
        point = {
            "x_b": float(values["xB"]), "t_GeV2": float(values["t"]),
            "Q2_GeV2": float(values["Q2"]), "beam_energy_GeV": float(beam),
            "phi_rad": phi % (2.0 * math.pi),
        }
        if not (
            0.10 <= point["x_b"] <= 0.50
            and -0.90 <= point["t_GeV2"] <= -0.10
            and 1.0 <= point["Q2_GeV2"] <= 80.0
            and 3.0 <= point["beam_energy_GeV"] <= 200.0
        ):
            continue
        try:
            require_physical_fixed_target_kinematics(
                x_b=point["x_b"], Q2_GeV2=point["Q2_GeV2"],
                beam_energy_GeV=point["beam_energy_GeV"],
                context=str(item["record_id"]),
            )
        except ValueError:
            continue
        coordinate = tuple(round(point[key], 12) for key in (
            "x_b", "t_GeV2", "Q2_GeV2", "beam_energy_GeV", "phi_rad"
        ))
        if coordinate in training_coordinates:
            continue
        candidate = record(point, {
            "record_id": item["record_id"], "dataset_id": item["dataset_id"],
            "source_path": dataset["source_path"],
        })
        retained = candidates.get(coordinate)
        if retained is None or candidate["record_id"] < retained["record_id"]:
            candidates[coordinate] = candidate
    values = sorted(candidates.values(), key=lambda item: item["record_id"])
    if len(values) < count:
        raise SystemExit(f"only {len(values)} unique fresh points satisfy holdout bounds")
    features = np.asarray([
        [
            item["x_b"] / 0.5,
            -item["t_GeV2"] / 0.9,
            math.log(item["Q2_GeV2"]) / math.log(80.0),
            math.log(item["beam_energy_GeV"] / 3.0) / math.log(200.0 / 3.0),
            math.sin(item["phi_rad"]), math.cos(item["phi_rad"]),
        ]
        for item in values
    ], dtype=np.float64)
    first = min(
        range(len(values)),
        key=lambda index: hashlib.sha256(
            f"{seed}:{values[index]['record_id']}".encode()
        ).hexdigest(),
    )
    chosen = [first]
    distance = np.sum((features - features[first]) ** 2, axis=1)
    distance[first] = -1.0
    while len(chosen) < count:
        maximum = float(distance.max())
        tied = np.flatnonzero(np.isclose(distance, maximum, rtol=0.0, atol=1e-15))
        next_index = min(tied, key=lambda index: values[int(index)]["record_id"])
        chosen.append(int(next_index))
        distance = np.minimum(
            distance,
            np.sum((features - features[next_index]) ** 2, axis=1),
        )
        distance[chosen] = -1.0
    return [values[index] for index in chosen], {
        "catalog_sha256": catalog["catalog_sha256"],
        "database_revision": catalog["database_revision"],
        "selection_rule": (
            "measurement_free_deterministic_farthest_point_xB_t_logQ2_logE_phi_v1"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    project_source = parser.add_mutually_exclusive_group(required=True)
    project_source.add_argument("--project-archive", type=Path)
    project_source.add_argument("--project-dir", type=Path)
    parser.add_argument("--project", required=True)
    parser.add_argument("--design-name", required=True)
    parser.add_argument(
        "--source", choices=("same", "training-subset", "fresh"), required=True
    )
    parser.add_argument("--kinematics", type=int, required=True)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kinematics < 1:
        raise SystemExit("--kinematics must be positive")
    configuration, experiment = load_project(
        args.project_archive, args.project_dir, args.project
    )
    if args.source == "same":
        points, provenance = same_design(configuration, experiment, args.kinematics)
        relationship = "same_training_kinematics"
        claim = "native_model_family_shift_only"
    elif args.source == "training-subset":
        points, provenance = training_subset_design(
            configuration, experiment, args.kinematics, args.seed
        )
        relationship = "training_kinematic_subset"
        claim = "masked_design_information_reduction_test"
    else:
        if args.database is None:
            raise SystemExit("--database is required for a fresh design")
        points, provenance = fresh_design(
            configuration, experiment, args.database,
            args.kinematics, args.seed,
        )
        relationship = "fresh_unseen_kinematics"
        claim = "combined_native_model_and_kinematic_design_shift_diagnostic"
    payload = {
        "schema_version": 2,
        # Kept for the verified legacy v3 image; explicit role fields below
        # override the old protocol name in user-facing provenance.
        "protocol": "output_blind_fresh_kinematics_v1",
        "design_name": args.design_name,
        "relationship_to_training": relationship,
        "claim_scope": claim,
        "role": "post_training_external_native_model_validation_only",
        "database_revision": provenance["database_revision"],
        "catalog_sha256": provenance["catalog_sha256"],
        "selection_frozen_before_native_outputs": True,
        "native_outputs_generated": False,
        "measurement_values_included": False,
        "uncertainties_included": False,
        "training_use": False,
        "architecture_selection_use": False,
        "optuna_objective_use": False,
        "ordinary_dd_outer_test_replaced": False,
        "selection_rule": provenance["selection_rule"],
        "admitted_native_models": ["GPDGK11", "GPDGK16", "GPDGK19", "GPDVGG99"],
        "npe_dataset_contract": (
            f"{args.kinematics} kinematics times all configured observables"
        ),
        "npe_dataset_points": points,
    }
    payload["manifest_content_sha256"] = hashlib.sha256(
        canonical(payload).encode("utf-8")
    ).hexdigest()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
