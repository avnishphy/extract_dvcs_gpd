"""Typed registry for the only supported NPE model families and constraints."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ModelFamily:
    identifier: str
    inference_method: str
    gpd_inference_representation: str
    observation_encoder: str
    density_estimator: str
    density_estimator_implementation: str
    physics_backend: str
    requires_canonical_gpd_truth: bool


_FAMILIES = {
    "dd_deepsets_maf": ModelFamily(
        "dd_deepsets_maf", "npe", "double_distribution_coordinates",
        "deepsets", "maf", "zuko_maf", "partons", False,
    ),
    "neural_gpd_deepsets_maf": ModelFamily(
        "neural_gpd_deepsets_maf", "npe", "nngpd_inspired_latent_function",
        "deepsets", "maf", "zuko_maf", "partons", True,
    ),
}


def model_family(identifier: str) -> ModelFamily:
    try:
        return _FAMILIES[identifier]
    except KeyError as error:
        raise ValueError(
            f"unsupported model family {identifier!r}; choose one of {sorted(_FAMILIES)}"
        ) from error


def model_family_registry() -> dict[str, dict[str, Any]]:
    return {name: asdict(value) for name, value in _FAMILIES.items()}


def _walk_keys(value: Any, prefix: str = ""):
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, child
            yield from _walk_keys(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_keys(child, f"{prefix}[{index}]")


def validate_maf_only(configuration: Mapping[str, Any]) -> None:
    stale = []
    for path, child in _walk_keys(configuration):
        key = path.rsplit(".", 1)[-1].split("[", 1)[0].lower()
        if key == "num_bins" or "spline" in key or key in {"nsf", "flow_type"}:
            stale.append(path)
        if key in {"density_estimator", "flow", "flow_identifier"} and isinstance(child, str):
            if child.lower() not in {"maf", "zuko_maf"}:
                stale.append(path)
    if stale:
        raise ValueError(
            "MAF is the only active density estimator; remove or explicitly migrate "
            f"inactive NSF/spline fields: {sorted(set(stale))}"
        )


@dataclass(frozen=True)
class PhysicsConstraint:
    identifier: str
    gpd: str
    channel: str
    mode: str
    enabled: bool
    scale_GeV2: float | None
    scheme: str | None
    order_or_moment: int | None
    weight: float | None
    covariance: tuple[tuple[float, ...], ...] | None
    source: str
    coordinate_region: Mapping[str, Any]
    numerical_method: str
    tolerance: float
    diagnostics: Mapping[str, Any]

    def validate(self) -> None:
        if self.mode not in {"hard", "soft", "diagnostic", "external_likelihood"}:
            raise ValueError(f"constraint {self.identifier}: invalid mode")
        if self.tolerance < 0 or not np.isfinite(self.tolerance):
            raise ValueError(f"constraint {self.identifier}: invalid tolerance")
        if self.mode == "soft" and (self.weight is None or self.weight < 0):
            raise ValueError(f"constraint {self.identifier}: soft mode requires weight")
        if self.mode == "external_likelihood" and self.covariance is None:
            raise ValueError(
                f"constraint {self.identifier}: external likelihood requires covariance"
            )
        if self.identifier.startswith("lattice_") and self.enabled:
            if not self.scheme or self.scale_GeV2 is None or not self.source:
                raise ValueError(
                    f"constraint {self.identifier}: enabled lattice input requires "
                    "source, scale, and scheme"
                )


def validate_constraint_registry(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") != 1 or not isinstance(value.get("constraints"), list):
        raise ValueError("constraint registry must have schema_version 1 and constraints")
    identifiers: set[str] = set()
    normalized = deepcopy(dict(value))
    for raw in normalized["constraints"]:
        constraint = PhysicsConstraint(
            identifier=str(raw["identifier"]), gpd=str(raw["gpd"]),
            channel=str(raw["channel"]), mode=str(raw["mode"]),
            enabled=bool(raw["enabled"]),
            scale_GeV2=(None if raw.get("scale_GeV2") is None else float(raw["scale_GeV2"])),
            scheme=raw.get("scheme"),
            order_or_moment=raw.get("order_or_moment"),
            weight=(None if raw.get("weight") is None else float(raw["weight"])),
            covariance=(None if raw.get("covariance") is None else tuple(
                tuple(float(item) for item in row) for row in raw["covariance"]
            )), source=str(raw.get("source", "")),
            coordinate_region=dict(raw.get("coordinate_region", {})),
            numerical_method=str(raw["numerical_method"]),
            tolerance=float(raw["tolerance"]),
            diagnostics=dict(raw.get("diagnostics", {})),
        )
        constraint.validate()
        if constraint.identifier in identifiers:
            raise ValueError(f"duplicate constraint {constraint.identifier}")
        identifiers.add(constraint.identifier)
    return normalized


def assert_architecture_comparable(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    fields = (
        "master_corpus_id", "selection_id", "realization_id",
        "group_split_id", "noise_contract_id", "observable_set_id",
        "kinematic_set_id", "locked_holdout_id",
    )
    mismatch = [field for field in fields if left.get(field) != right.get(field)]
    if mismatch:
        raise ValueError(
            "architecture runs violate the comparability contract: "
            + ", ".join(mismatch)
        )


def common_function_metrics(
    truth: np.ndarray, samples: np.ndarray, levels: Sequence[float] = (0.5, 0.9)
) -> dict[str, Any]:
    reference = np.asarray(truth, dtype=np.float64)
    draws = np.asarray(samples, dtype=np.float64)
    if draws.ndim != reference.ndim + 1 or draws.shape[1:] != reference.shape:
        raise ValueError("function samples must have shape [sample, *truth.shape]")
    median = np.median(draws, axis=0)
    result: dict[str, Any] = {
        "rmse_median": float(np.sqrt(np.mean((median - reference) ** 2))),
        "mae_median": float(np.mean(np.abs(median - reference))),
        "raw_cross_representation_nll_used": False,
        "coverage": {},
    }
    for level in levels:
        if not 0 < level < 1:
            raise ValueError("coverage levels must be in (0,1)")
        lower = np.quantile(draws, (1 - level) / 2, axis=0)
        upper = np.quantile(draws, 1 - (1 - level) / 2, axis=0)
        result["coverage"][str(level)] = float(
            np.mean((reference >= lower) & (reference <= upper))
        )
    return result
