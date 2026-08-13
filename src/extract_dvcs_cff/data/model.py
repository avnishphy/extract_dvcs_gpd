"""Versioned synthetic-only measurement, covariance, and nuisance models."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .covariance import CovarianceDiagnostics, validate_covariance


class DataValidationError(ValueError):
    """Raised when a versioned data contract is invalid."""


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    observed = set(value)
    missing = expected - observed
    unknown = observed - expected
    if missing:
        raise DataValidationError(
            f"{context} is missing fields: {', '.join(sorted(missing))}"
        )
    if unknown:
        raise DataValidationError(
            f"{context} has unknown fields: {', '.join(sorted(unknown))}"
        )


def _finite(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataValidationError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise DataValidationError(f"{context} must be finite")
    return result


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise DataValidationError(f"{context} must be a non-empty string")
    return value


def _quantity(value: Any, unit: str, context: str) -> float:
    if not isinstance(value, Mapping):
        raise DataValidationError(f"{context} must be a quantity object")
    _require_exact_keys(value, {"value", "unit"}, context)
    if value["unit"] != unit:
        raise DataValidationError(f"{context}.unit must be '{unit}'")
    return _finite(value["value"], f"{context}.value")


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True)
class MeasurementPoint:
    point_id: str
    experiment_id: str
    dataset_id: str
    observable_id: str
    x_b: float
    q2_gev2: float
    t_gev2: float
    phi_rad: float
    beam_energy_gev: float
    beam_charge: int
    beam_helicity: int
    target_polarization: float
    value: float
    observable_unit: str = "1"
    kinematic_valid: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "point_id",
            "experiment_id",
            "dataset_id",
            "observable_id",
        ):
            if not isinstance(getattr(self, field_name), str) or not getattr(
                self, field_name
            ):
                raise DataValidationError(f"{field_name} must not be empty")
        for field_name in (
            "x_b",
            "q2_gev2",
            "t_gev2",
            "phi_rad",
            "beam_energy_gev",
            "target_polarization",
            "value",
        ):
            _finite(getattr(self, field_name), field_name)
        if not 0.0 < self.x_b < 1.0:
            raise DataValidationError("x_b must lie strictly between 0 and 1")
        if self.q2_gev2 <= 0.0:
            raise DataValidationError("q2_gev2 must be positive")
        if self.t_gev2 > 0.0:
            raise DataValidationError("t_gev2 must be non-positive")
        if not 0.0 <= self.phi_rad < 2.0 * math.pi:
            raise DataValidationError("phi_rad must lie in [0, 2*pi)")
        if self.beam_energy_gev <= 0.0:
            raise DataValidationError("beam_energy_gev must be positive")
        if isinstance(self.beam_charge, bool) or not isinstance(
            self.beam_charge, int
        ):
            raise DataValidationError("beam_charge must be an integer")
        if self.beam_charge not in (-1, 1):
            raise DataValidationError("beam_charge must be -1 or 1")
        if isinstance(self.beam_helicity, bool) or not isinstance(
            self.beam_helicity, int
        ):
            raise DataValidationError("beam_helicity must be an integer")
        if self.beam_helicity not in (-1, 0, 1):
            raise DataValidationError("beam_helicity must be -1, 0, or 1")
        if not -1.0 <= self.target_polarization <= 1.0:
            raise DataValidationError(
                "target_polarization must lie in [-1, 1]"
            )
        if self.observable_unit != "1":
            raise DataValidationError(
                "Stage 02 supports only dimensionless synthetic observables"
            )
        if not isinstance(self.kinematic_valid, bool):
            raise DataValidationError("kinematic_valid must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "experiment_id": self.experiment_id,
            "dataset_id": self.dataset_id,
            "observable_id": self.observable_id,
            "kinematics": {
                "x_B": {"value": self.x_b, "unit": "1"},
                "Q2": {"value": self.q2_gev2, "unit": "GeV2"},
                "t": {"value": self.t_gev2, "unit": "GeV2"},
                "phi": {"value": self.phi_rad, "unit": "rad"},
                "beam_energy": {
                    "value": self.beam_energy_gev,
                    "unit": "GeV",
                },
            },
            "beam_charge": self.beam_charge,
            "beam_helicity": self.beam_helicity,
            "target_polarization": self.target_polarization,
            "measurement": {
                "value": self.value,
                "unit": self.observable_unit,
            },
            "kinematic_valid": self.kinematic_valid,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MeasurementPoint":
        _require_exact_keys(
            value,
            {
                "point_id",
                "experiment_id",
                "dataset_id",
                "observable_id",
                "kinematics",
                "beam_charge",
                "beam_helicity",
                "target_polarization",
                "measurement",
                "kinematic_valid",
            },
            "point",
        )
        kinematics = value["kinematics"]
        if not isinstance(kinematics, Mapping):
            raise DataValidationError("point.kinematics must be an object")
        _require_exact_keys(
            kinematics, {"x_B", "Q2", "t", "phi", "beam_energy"}, "kinematics"
        )
        measurement = value["measurement"]
        if not isinstance(measurement, Mapping):
            raise DataValidationError("point.measurement must be an object")
        _require_exact_keys(measurement, {"value", "unit"}, "measurement")
        return cls(
            point_id=_string(value["point_id"], "point_id"),
            experiment_id=_string(value["experiment_id"], "experiment_id"),
            dataset_id=_string(value["dataset_id"], "dataset_id"),
            observable_id=_string(value["observable_id"], "observable_id"),
            x_b=_quantity(kinematics["x_B"], "1", "kinematics.x_B"),
            q2_gev2=_quantity(kinematics["Q2"], "GeV2", "kinematics.Q2"),
            t_gev2=_quantity(kinematics["t"], "GeV2", "kinematics.t"),
            phi_rad=_quantity(kinematics["phi"], "rad", "kinematics.phi"),
            beam_energy_gev=_quantity(
                kinematics["beam_energy"], "GeV", "kinematics.beam_energy"
            ),
            beam_charge=value["beam_charge"],
            beam_helicity=value["beam_helicity"],
            target_polarization=_finite(
                value["target_polarization"], "target_polarization"
            ),
            value=_finite(measurement["value"], "measurement.value"),
            observable_unit=_string(measurement["unit"], "measurement.unit"),
            kinematic_valid=value["kinematic_valid"],
        )


@dataclass(frozen=True)
class NuisanceGroup:
    group_id: str
    mode: str
    response: tuple[float, ...]
    response_unit: str
    prior: str = "standard_normal"

    def __post_init__(self) -> None:
        if not self.group_id:
            raise DataValidationError("nuisance group_id must not be empty")
        if self.mode not in ("additive", "multiplicative"):
            raise DataValidationError(
                "nuisance mode must be 'additive' or 'multiplicative'"
            )
        object.__setattr__(
            self,
            "response",
            tuple(_finite(value, "nuisance response") for value in self.response),
        )
        expected_unit = "1" if self.mode == "multiplicative" else "observable"
        if self.response_unit != expected_unit:
            raise DataValidationError(
                f"{self.mode} nuisance response_unit must be '{expected_unit}'"
            )
        if self.prior != "standard_normal":
            raise DataValidationError(
                "Stage 02 nuisance prior must be 'standard_normal'"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "mode": self.mode,
            "response": list(self.response),
            "response_unit": self.response_unit,
            "prior": self.prior,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NuisanceGroup":
        _require_exact_keys(
            value,
            {"group_id", "mode", "response", "response_unit", "prior"},
            "nuisance_group",
        )
        if not isinstance(value["response"], Sequence) or isinstance(
            value["response"], (str, bytes)
        ):
            raise DataValidationError("nuisance response must be an array")
        return cls(
            group_id=_string(value["group_id"], "nuisance_group.group_id"),
            mode=_string(value["mode"], "nuisance_group.mode"),
            response=tuple(value["response"]),
            response_unit=_string(
                value["response_unit"], "nuisance_group.response_unit"
            ),
            prior=_string(value["prior"], "nuisance_group.prior"),
        )


@dataclass(frozen=True)
class Dataset:
    dataset_id: str
    points: tuple[MeasurementPoint, ...]
    covariance: tuple[tuple[float, ...], ...]
    covariance_construction: str
    nuisance_groups: tuple[NuisanceGroup, ...]
    provenance: Mapping[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ) or self.schema_version != 1:
            raise DataValidationError("dataset schema_version must be 1")
        _string(self.dataset_id, "dataset_id")
        object.__setattr__(self, "points", tuple(self.points))
        object.__setattr__(self, "nuisance_groups", tuple(self.nuisance_groups))
        if not self.points:
            raise DataValidationError("dataset must contain at least one point")
        if any(point.dataset_id != self.dataset_id for point in self.points):
            raise DataValidationError(
                "every point.dataset_id must match dataset.dataset_id"
            )
        point_ids = self.point_order
        if len(set(point_ids)) != len(point_ids):
            raise DataValidationError("point_id values must be unique")
        _string(self.covariance_construction, "covariance_construction")
        matrix, _, _ = validate_covariance(self.covariance, len(self.points))
        object.__setattr__(
            self,
            "covariance",
            tuple(tuple(float(item) for item in row) for row in matrix),
        )
        group_ids = [group.group_id for group in self.nuisance_groups]
        if len(set(group_ids)) != len(group_ids):
            raise DataValidationError("nuisance group_id values must be unique")
        for group in self.nuisance_groups:
            if len(group.response) != len(self.points):
                raise DataValidationError(
                    f"nuisance '{group.group_id}' response length must match "
                    "the point count"
                )
        required_provenance = {
            "kind",
            "generator",
            "seed",
            "configuration_sha256",
        }
        if not isinstance(self.provenance, Mapping):
            raise DataValidationError("provenance must be an object")
        _require_exact_keys(
            self.provenance, required_provenance, "provenance"
        )
        if self.provenance["kind"] != "synthetic":
            raise DataValidationError(
                "Stage 02 accepts synthetic provenance only"
            )
        if not isinstance(self.provenance["seed"], int) or isinstance(
            self.provenance["seed"], bool
        ):
            raise DataValidationError("provenance.seed must be an integer")
        if self.provenance["seed"] < 0:
            raise DataValidationError("provenance.seed must be non-negative")
        _string(self.provenance["generator"], "provenance.generator")
        digest = self.provenance["configuration_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise DataValidationError(
                "provenance.configuration_sha256 must be a lowercase SHA-256 "
                "hex string"
            )
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def point_order(self) -> tuple[str, ...]:
        return tuple(point.point_id for point in self.points)

    @property
    def covariance_diagnostics(self) -> CovarianceDiagnostics:
        _, _, diagnostics = validate_covariance(
            self.covariance, len(self.points)
        )
        return diagnostics

    @property
    def values(self) -> np.ndarray:
        result = np.asarray([point.value for point in self.points], dtype=np.float64)
        result.setflags(write=False)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "point_order": list(self.point_order),
            "points": [point.to_dict() for point in self.points],
            "covariance": {
                "matrix": [list(row) for row in self.covariance],
                "point_order": list(self.point_order),
                "observable_unit_squared": "1^2",
                "construction": self.covariance_construction,
                "diagnostics": self.covariance_diagnostics.to_dict(),
                "regularization": None,
            },
            "nuisance_groups": [
                group.to_dict() for group in self.nuisance_groups
            ],
            "provenance": dict(self.provenance),
        }

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Dataset":
        _require_exact_keys(
            value,
            {
                "schema_version",
                "dataset_id",
                "point_order",
                "points",
                "covariance",
                "nuisance_groups",
                "provenance",
            },
            "dataset",
        )
        if not isinstance(value["points"], list):
            raise DataValidationError("dataset.points must be an array")
        points = tuple(MeasurementPoint.from_dict(item) for item in value["points"])
        if not isinstance(value["point_order"], list) or not all(
            isinstance(item, str) for item in value["point_order"]
        ):
            raise DataValidationError("dataset.point_order must be a string array")
        point_order = tuple(value["point_order"])
        observed_order = tuple(point.point_id for point in points)
        if point_order != observed_order:
            raise DataValidationError(
                "dataset.point_order must exactly match point array order"
            )
        covariance = value["covariance"]
        if not isinstance(covariance, Mapping):
            raise DataValidationError("dataset.covariance must be an object")
        _require_exact_keys(
            covariance,
            {
                "matrix",
                "point_order",
                "observable_unit_squared",
                "construction",
                "diagnostics",
                "regularization",
            },
            "covariance",
        )
        if not isinstance(covariance["point_order"], list) or not all(
            isinstance(item, str) for item in covariance["point_order"]
        ):
            raise DataValidationError(
                "covariance.point_order must be a string array"
            )
        if tuple(covariance["point_order"]) != observed_order:
            raise DataValidationError(
                "covariance.point_order must exactly match point array order"
            )
        if covariance["observable_unit_squared"] != "1^2":
            raise DataValidationError(
                "covariance observable_unit_squared must be '1^2'"
            )
        if covariance["regularization"] is not None:
            raise DataValidationError(
                "Stage 02 does not permit covariance regularization"
            )
        if not isinstance(value["nuisance_groups"], list):
            raise DataValidationError("dataset.nuisance_groups must be an array")
        result = cls(
            schema_version=value["schema_version"],
            dataset_id=_string(value["dataset_id"], "dataset.dataset_id"),
            points=points,
            covariance=tuple(
                tuple(row) for row in covariance["matrix"]
            ),
            covariance_construction=_string(
                covariance["construction"], "covariance.construction"
            ),
            nuisance_groups=tuple(
                NuisanceGroup.from_dict(item)
                for item in value["nuisance_groups"]
            ),
            provenance=value["provenance"],
        )
        if covariance["diagnostics"] != result.covariance_diagnostics.to_dict():
            raise DataValidationError(
                "covariance diagnostics do not match the declared matrix"
            )
        return result

    @classmethod
    def from_json(cls, text: str) -> "Dataset":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise DataValidationError(f"invalid dataset JSON: {error}") from error
        if not isinstance(value, Mapping):
            raise DataValidationError("dataset root must be an object")
        return cls.from_dict(value)
