"""Synthetic-only Stage 02 generator with explicit seed lineage."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from extract_dvcs_cff.data.covariance import validate_covariance
from extract_dvcs_cff.data.model import Dataset, MeasurementPoint, NuisanceGroup
from extract_dvcs_cff.likelihood.gaussian import apply_nuisance_shifts


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


@dataclass(frozen=True)
class SimulationRecord:
    dataset: Dataset
    truth_prediction: tuple[float, ...]
    nuisance_values: Mapping[str, float]
    noise: tuple[float, ...]
    seed: int
    generator: str
    configuration_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "record_type": "synthetic_dataset_generation",
            "generator": self.generator,
            "seed": self.seed,
            "configuration_sha256": self.configuration_sha256,
            "truth_prediction": list(self.truth_prediction),
            "nuisance_values": dict(self.nuisance_values),
            "noise": list(self.noise),
            "dataset_sha256": self.dataset.sha256(),
            "dataset": self.dataset.to_dict(),
        }

    def canonical_json(self) -> str:
        return _canonical(self.to_dict())


def generate_synthetic_dataset(
    *,
    dataset_id: str,
    point_templates: Sequence[MeasurementPoint],
    truth_prediction: npt.ArrayLike,
    covariance: npt.ArrayLike,
    nuisance_groups: Sequence[NuisanceGroup],
    seed: int,
    generator: str = "numpy.PCG64.stage02.v1",
) -> SimulationRecord:
    """Generate one deterministic correlated synthetic dataset."""

    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("dataset_id must be a non-empty string")
    if not isinstance(generator, str) or not generator:
        raise ValueError("generator must be a non-empty string")
    templates = tuple(point_templates)
    if not templates:
        raise ValueError("point_templates must not be empty")
    truth = np.asarray(truth_prediction, dtype=np.float64)
    if truth.shape != (len(templates),) or not np.all(np.isfinite(truth)):
        raise ValueError(
            "truth_prediction must be a finite vector matching point_templates"
        )
    matrix, cholesky, diagnostics = validate_covariance(
        covariance, len(templates)
    )
    groups = tuple(nuisance_groups)
    configuration = {
        "dataset_id": dataset_id,
        "point_templates": [
            replace(point, dataset_id=dataset_id, value=0.0).to_dict()
            for point in templates
        ],
        "truth_prediction": [float(value) for value in truth],
        "covariance": matrix.tolist(),
        "nuisance_groups": [group.to_dict() for group in groups],
        "generator": generator,
        "covariance_policy": {
            "minimum_relative_eigenvalue": 1e-12,
            "regularization": None,
            "diagnostics": diagnostics.to_dict(),
        },
    }
    configuration_sha256 = hashlib.sha256(
        _canonical(configuration).encode("utf-8")
    ).hexdigest()

    rng = np.random.Generator(np.random.PCG64(seed))
    nuisance_values = {
        group.group_id: float(rng.standard_normal()) for group in groups
    }
    shifted_truth, _ = apply_nuisance_shifts(
        truth, groups, nuisance_values
    )
    noise = cholesky @ rng.standard_normal(len(templates))
    observed = shifted_truth + noise
    points = tuple(
        replace(point, dataset_id=dataset_id, value=float(observed[index]))
        for index, point in enumerate(templates)
    )
    dataset = Dataset(
        dataset_id=dataset_id,
        points=points,
        covariance=tuple(tuple(float(item) for item in row) for row in matrix),
        covariance_construction="declared_dense_synthetic_covariance",
        nuisance_groups=groups,
        provenance={
            "kind": "synthetic",
            "generator": generator,
            "seed": seed,
            "configuration_sha256": configuration_sha256,
        },
    )
    return SimulationRecord(
        dataset=dataset,
        truth_prediction=tuple(float(value) for value in truth),
        nuisance_values=nuisance_values,
        noise=tuple(float(value) for value in noise),
        seed=seed,
        generator=generator,
        configuration_sha256=configuration_sha256,
    )
