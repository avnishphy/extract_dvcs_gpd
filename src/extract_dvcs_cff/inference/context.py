"""Versioned Stage 05 full-covariance dataset-context construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
import numpy.typing as npt


POINT_FEATURE_NAMES = (
    "gpd_x",
    "skewness_xi_over_0p5",
    "minus_t_GeV2_over_0p5",
    "log_q2_over_q0_squared",
    "observed_h_up_over_5",
    "marginal_sigma_over_0p25",
    "symmetric_whitened_observation_over_25",
    "normalization_response_over_0p1",
    "observable_is_direct_h_up",
    "point_mask",
)

GLOBAL_FEATURE_NAMES = (
    "representation_is_reduced_dd",
    "representation_is_conformal",
    "q0_squared_GeV2_over_4",
    "profile_b_over_2",
    "t_slope_GeV_minus2",
    "coefficient_function_enabled",
    "evolution_enabled",
    "model_mask",
)

_VARYING_POINT_FEATURES = {
    "observed_h_up_over_5",
    "marginal_sigma_over_0p25",
    "symmetric_whitened_observation_over_25",
}


def _finite_array(
    value: npt.ArrayLike, shape: tuple[int, ...], name: str
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


def build_stage05_context(
    *,
    grid: Sequence[Mapping[str, Any]],
    observed: npt.ArrayLike,
    covariance: npt.ArrayLike,
    q0_squared_gev2: float,
    profile_b: float,
    t_slope_gev_minus2: float,
    normalization_response: float,
) -> np.ndarray:
    """Return flattened point tokens followed by declared global context.

    The symmetric inverse-square-root whitening is permutation equivariant,
    unlike a triangular Cholesky whitening. Thus consistently permuting points
    and covariance rows/columns only permutes tokens.
    """

    point_count = len(grid)
    if point_count <= 0:
        raise ValueError("grid must not be empty")
    values = _finite_array(observed, (point_count,), "observed")
    matrix = _finite_array(
        covariance, (point_count, point_count), "covariance"
    )
    if not np.array_equal(matrix, matrix.T):
        raise ValueError("covariance must be exactly symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("covariance must be positive definite")
    inverse_sqrt = (
        eigenvectors * np.power(eigenvalues, -0.5)[None, :]
    ) @ eigenvectors.T
    whitened = inverse_sqrt @ values
    sigma = np.sqrt(np.diag(matrix))

    scalars = {
        "q0_squared_gev2": q0_squared_gev2,
        "profile_b": profile_b,
        "t_slope_gev_minus2": t_slope_gev_minus2,
        "normalization_response": normalization_response,
    }
    if any(not math.isfinite(float(value)) for value in scalars.values()):
        raise ValueError("global context scalars must be finite")
    if q0_squared_gev2 <= 0.0:
        raise ValueError("q0_squared_gev2 must be positive")

    tokens = np.empty((point_count, len(POINT_FEATURE_NAMES)), dtype=np.float64)
    for index, point in enumerate(grid):
        required = {"x", "xi", "t_GeV2", "q2_GeV2"}
        if set(point) != required:
            raise ValueError(
                f"grid[{index}] must contain exactly {sorted(required)}"
            )
        x = float(point["x"])
        xi = float(point["xi"])
        t = float(point["t_GeV2"])
        q2 = float(point["q2_GeV2"])
        if (
            not all(math.isfinite(value) for value in (x, xi, t, q2))
            or not -1.0 <= x <= 1.0
            or not 0.0 <= xi < 1.0
            or t > 0.0
            or q2 <= 0.0
        ):
            raise ValueError(f"grid[{index}] has invalid kinematics")
        tokens[index] = (
            x,
            xi / 0.5,
            -t / 0.5,
            math.log(q2 / q0_squared_gev2),
            values[index] / 5.0,
            sigma[index] / 0.25,
            whitened[index] / 25.0,
            normalization_response / 0.1,
            1.0,
            1.0,
        )

    global_context = np.asarray(
        (
            1.0,
            0.0,
            q0_squared_gev2 / 4.0,
            profile_b / 2.0,
            t_slope_gev_minus2,
            0.0,
            0.0,
            1.0,
        ),
        dtype=np.float64,
    )
    context = np.concatenate((tokens.reshape(-1), global_context))
    if not np.all(np.isfinite(context)):
        raise ValueError("constructed context contains non-finite values")
    return context


def validate_stage05_constant_dimensions(
    contexts: npt.ArrayLike,
    *,
    point_count: int,
    varying_point_features: Sequence[str] | None = None,
) -> dict[str, list[int]]:
    """Disposition sbi's flat-axis constant-feature diagnostic explicitly.

    Fixed kinematics and metadata are constant across simulated datasets at a
    fixed flattened token position but vary across point tokens, where the
    shared point encoder consumes them. Global model features are constant
    because Stage 05 has one declared model. Only the three measurement and
    uncertainty features are expected to vary across simulations.
    """

    matrix = np.asarray(contexts)
    expected_width = (
        point_count * len(POINT_FEATURE_NAMES) + len(GLOBAL_FEATURE_NAMES)
    )
    if matrix.ndim != 2 or matrix.shape[1] != expected_width:
        raise ValueError(
            f"contexts must have shape (simulation, {expected_width})"
        )
    varying = (
        _VARYING_POINT_FEATURES
        if varying_point_features is None
        else set(varying_point_features)
    )
    unknown = varying - set(POINT_FEATURE_NAMES)
    if unknown:
        raise ValueError(
            f"unknown varying point features: {sorted(unknown)}"
        )
    observed = np.flatnonzero(np.ptp(matrix, axis=0) == 0.0).tolist()
    expected = []
    for point_index in range(point_count):
        for feature_index, name in enumerate(POINT_FEATURE_NAMES):
            if name not in varying:
                expected.append(
                    point_index * len(POINT_FEATURE_NAMES) + feature_index
                )
    token_width = point_count * len(POINT_FEATURE_NAMES)
    expected.extend(
        range(token_width, token_width + len(GLOBAL_FEATURE_NAMES))
    )
    if observed != expected:
        raise ValueError(
            "observed constant context dimensions do not match the "
            f"declared structured contract: expected={expected}, "
            f"observed={observed}"
        )
    return {
        "expected_constant_indices": expected,
        "observed_constant_indices": observed,
    }
