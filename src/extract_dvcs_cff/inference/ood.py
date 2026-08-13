"""Fail-closed empirical context-envelope diagnostics for Stage 05."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from .context import GLOBAL_FEATURE_NAMES, POINT_FEATURE_NAMES


@dataclass(frozen=True)
class ContextEnvelope:
    """Axis-aligned envelope on declared varying context dimensions."""

    indices: tuple[int, ...]
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]

    @classmethod
    def fit(
        cls,
        contexts: npt.ArrayLike,
        *,
        point_count: int,
        varying_point_features: Sequence[str],
    ) -> "ContextEnvelope":
        matrix = np.asarray(contexts, dtype=np.float64)
        expected_width = (
            point_count * len(POINT_FEATURE_NAMES)
            + len(GLOBAL_FEATURE_NAMES)
        )
        if (
            matrix.ndim != 2
            or matrix.shape[0] < 2
            or matrix.shape[1] != expected_width
            or not np.all(np.isfinite(matrix))
        ):
            raise ValueError(
                f"contexts must be finite with shape (n>=2, {expected_width})"
            )
        names = tuple(varying_point_features)
        if not names or len(set(names)) != len(names):
            raise ValueError("varying_point_features must be non-empty and unique")
        unknown = set(names) - set(POINT_FEATURE_NAMES)
        if unknown:
            raise ValueError(f"unknown varying features: {sorted(unknown)}")
        indices = tuple(
            point * len(POINT_FEATURE_NAMES)
            + POINT_FEATURE_NAMES.index(name)
            for point in range(point_count)
            for name in names
        )
        selected = matrix[:, indices]
        minimum = selected.min(axis=0)
        maximum = selected.max(axis=0)
        if np.any(maximum <= minimum):
            raise ValueError("declared varying dimensions must have positive range")
        return cls(
            indices=indices,
            minimum=tuple(float(value) for value in minimum),
            maximum=tuple(float(value) for value in maximum),
        )

    def assess(
        self,
        contexts: npt.ArrayLike,
        *,
        margin_fraction: float,
    ) -> dict[str, np.ndarray]:
        matrix = np.asarray(contexts, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if (
            matrix.ndim != 2
            or matrix.shape[1] <= max(self.indices)
            or not np.all(np.isfinite(matrix))
            or not np.isfinite(margin_fraction)
            or margin_fraction < 0.0
        ):
            raise ValueError("invalid contexts or margin_fraction")
        minimum = np.asarray(self.minimum)
        maximum = np.asarray(self.maximum)
        width = maximum - minimum
        lower = minimum - margin_fraction * width
        upper = maximum + margin_fraction * width
        selected = matrix[:, self.indices]
        below = np.maximum(lower - selected, 0.0) / width
        above = np.maximum(selected - upper, 0.0) / width
        excess = np.maximum(below, above)
        return {
            "in_envelope": np.all(excess == 0.0, axis=1),
            "maximum_normalized_excess": np.max(excess, axis=1),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "indices": list(self.indices),
            "minimum": list(self.minimum),
            "maximum": list(self.maximum),
        }
