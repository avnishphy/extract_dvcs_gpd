"""Strict covariance validation without implicit regularization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class CovarianceValidationError(ValueError):
    """Raised when a covariance cannot be used as declared."""


@dataclass(frozen=True)
class CovarianceDiagnostics:
    dimension: int
    symmetry_max_abs: float
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    relative_minimum_eigenvalue: float
    condition_number: float
    log_determinant: float
    regularized: bool = False

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "dimension": self.dimension,
            "symmetry_max_abs": self.symmetry_max_abs,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "maximum_eigenvalue": self.maximum_eigenvalue,
            "relative_minimum_eigenvalue": self.relative_minimum_eigenvalue,
            "condition_number": self.condition_number,
            "log_determinant": self.log_determinant,
            "regularized": self.regularized,
        }


def validate_covariance(
    covariance: npt.ArrayLike,
    dimension: int,
    *,
    symmetry_relative_tolerance: float = 1e-12,
    minimum_relative_eigenvalue: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, CovarianceDiagnostics]:
    """Validate and factor a covariance, returning matrix, Cholesky, diagnostics."""

    if dimension <= 0:
        raise CovarianceValidationError("covariance dimension must be positive")
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (dimension, dimension):
        raise CovarianceValidationError(
            f"covariance shape must be {(dimension, dimension)}, observed {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise CovarianceValidationError("covariance entries must all be finite")

    scale = float(np.max(np.abs(matrix)))
    symmetry_max_abs = float(np.max(np.abs(matrix - matrix.T)))
    if symmetry_max_abs > symmetry_relative_tolerance * scale:
        raise CovarianceValidationError(
            "covariance must be symmetric within the declared relative tolerance"
        )

    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    if maximum <= 0.0:
        raise CovarianceValidationError(
            "covariance maximum eigenvalue must be positive"
        )
    relative_minimum = minimum / maximum
    if minimum <= 0.0:
        raise CovarianceValidationError("covariance must be positive definite")
    if relative_minimum < minimum_relative_eigenvalue:
        raise CovarianceValidationError(
            "covariance is numerically near-singular under the declared "
            "relative eigenvalue floor"
        )

    try:
        cholesky = np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise CovarianceValidationError(
            "covariance Cholesky factorization failed"
        ) from error

    log_determinant = float(2.0 * np.sum(np.log(np.diag(cholesky))))
    diagnostics = CovarianceDiagnostics(
        dimension=dimension,
        symmetry_max_abs=symmetry_max_abs,
        minimum_eigenvalue=minimum,
        maximum_eigenvalue=maximum,
        relative_minimum_eigenvalue=relative_minimum,
        condition_number=float(maximum / minimum),
        log_determinant=log_determinant,
    )
    matrix_copy = np.array(symmetric, dtype=np.float64, copy=True)
    cholesky_copy = np.array(cholesky, dtype=np.float64, copy=True)
    matrix_copy.setflags(write=False)
    cholesky_copy.setflags(write=False)
    return matrix_copy, cholesky_copy, diagnostics
