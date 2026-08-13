"""Normalized multivariate Gaussian likelihood with explicit nuisances."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from extract_dvcs_cff.data.covariance import validate_covariance
from extract_dvcs_cff.data.model import Dataset, NuisanceGroup


class LikelihoodValidationError(ValueError):
    """Raised when prediction or nuisance inputs violate the contract."""


def _finite_vector(
    values: npt.ArrayLike, dimension: int, context: str
) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (dimension,):
        raise LikelihoodValidationError(
            f"{context} shape must be {(dimension,)}, observed {vector.shape}"
        )
    if not np.all(np.isfinite(vector)):
        raise LikelihoodValidationError(f"{context} must be finite")
    return vector


def _validate_nuisance_values(
    groups: Sequence[NuisanceGroup], values: Mapping[str, float] | None
) -> dict[str, float]:
    supplied = {} if values is None else dict(values)
    expected = {group.group_id for group in groups}
    observed = set(supplied)
    if observed != expected:
        missing = expected - observed
        extra = observed - expected
        details = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"extra={sorted(extra)}")
        raise LikelihoodValidationError(
            "nuisance keys must exactly match declared groups: "
            + ", ".join(details)
        )
    result: dict[str, float] = {}
    for key, value in supplied.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LikelihoodValidationError(
                f"nuisance '{key}' must be numeric"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise LikelihoodValidationError(
                f"nuisance '{key}' must be finite"
            )
        result[key] = numeric
    return result


def apply_nuisance_shifts(
    prediction: npt.ArrayLike,
    groups: Sequence[NuisanceGroup],
    nuisance_values: Mapping[str, float] | None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply linear additive and fractional multiplicative nuisance shifts."""

    base = np.asarray(prediction, dtype=np.float64)
    if base.ndim != 1 or not np.all(np.isfinite(base)):
        raise LikelihoodValidationError(
            "prediction must be a finite one-dimensional vector"
        )
    nuisance = _validate_nuisance_values(groups, nuisance_values)
    shifted = np.array(base, copy=True)
    for group in groups:
        response = _finite_vector(
            group.response, len(base), f"nuisance '{group.group_id}' response"
        )
        eta = nuisance[group.group_id]
        if group.mode == "additive":
            shifted += eta * response
        else:
            shifted += base * eta * response
    if not np.all(np.isfinite(shifted)):
        raise LikelihoodValidationError(
            "nuisance application produced non-finite predictions"
        )
    return shifted, nuisance


@dataclass(frozen=True)
class LikelihoodResult:
    data_log_likelihood: float
    nuisance_log_prior: float
    joint_log_density: float
    chi_square: float
    log_determinant: float
    residual: tuple[float, ...]
    whitened_residual: tuple[float, ...]
    effective_prediction: tuple[float, ...]

    def to_dict(self) -> dict[str, float | list[float]]:
        return {
            "data_log_likelihood": self.data_log_likelihood,
            "nuisance_log_prior": self.nuisance_log_prior,
            "joint_log_density": self.joint_log_density,
            "chi_square": self.chi_square,
            "log_determinant": self.log_determinant,
            "residual": list(self.residual),
            "whitened_residual": list(self.whitened_residual),
            "effective_prediction": list(self.effective_prediction),
        }


class GaussianLikelihood:
    """Normalized full-covariance Gaussian likelihood for one dataset."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self._covariance, self._cholesky, self.diagnostics = validate_covariance(
            dataset.covariance, len(dataset.points)
        )

    def evaluate(
        self,
        prediction: npt.ArrayLike,
        nuisance_values: Mapping[str, float] | None = None,
    ) -> LikelihoodResult:
        base = _finite_vector(
            prediction, len(self.dataset.points), "prediction"
        )
        effective, nuisance = apply_nuisance_shifts(
            base, self.dataset.nuisance_groups, nuisance_values
        )
        residual = self.dataset.values - effective
        whitened = np.linalg.solve(self._cholesky, residual)
        chi_square = float(np.dot(whitened, whitened))
        normalization = len(residual) * math.log(2.0 * math.pi)
        data_log_likelihood = -0.5 * (
            chi_square + self.diagnostics.log_determinant + normalization
        )
        nuisance_log_prior = sum(
            -0.5 * (value * value + math.log(2.0 * math.pi))
            for value in nuisance.values()
        )
        return LikelihoodResult(
            data_log_likelihood=data_log_likelihood,
            nuisance_log_prior=nuisance_log_prior,
            joint_log_density=data_log_likelihood + nuisance_log_prior,
            chi_square=chi_square,
            log_determinant=self.diagnostics.log_determinant,
            residual=tuple(float(value) for value in residual),
            whitened_residual=tuple(float(value) for value in whitened),
            effective_prediction=tuple(float(value) for value in effective),
        )
