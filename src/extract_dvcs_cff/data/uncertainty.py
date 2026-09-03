"""Factorized uncertainty contracts and corpus-reusing realization helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import numpy as np

from .covariance import validate_covariance
from ..contracts import atomic_json, content_id


OBSERVATION_REALIZATIONS = {"central", "sampled"}
UNCERTAINTY_MODELS = {
    "full_covariance", "diagonal", "fixed_relative", "equal_weight_point_fit",
}
UNCERTAINTY_EXPOSURES = {"full_descriptors", "values_and_kinematics_only"}
NUISANCE_POLICIES = {
    "sampled_and_inferred", "nominal_fixed", "covariance_misspecified",
}
OUTPUT_KINDS = {"posterior", "point_estimate"}
COVARIANCE_POLICIES = {
    "training_reference", "external_fixed", "holdout_truth_derived",
}


@dataclass(frozen=True)
class UncertaintyContract:
    name: str
    observation_realization: str
    uncertainty_model: str
    uncertainty_exposure: str
    nuisance_policy: str
    physical_noise_scale: float
    numerical_jitter: float
    output_kind: str
    covariance_policy: str
    fixed_relative_sigma: float | None = None

    def validate(self) -> "UncertaintyContract":
        choices = (
            (self.observation_realization, OBSERVATION_REALIZATIONS, "observation_realization"),
            (self.uncertainty_model, UNCERTAINTY_MODELS, "uncertainty_model"),
            (self.uncertainty_exposure, UNCERTAINTY_EXPOSURES, "uncertainty_exposure"),
            (self.nuisance_policy, NUISANCE_POLICIES, "nuisance_policy"),
            (self.output_kind, OUTPUT_KINDS, "output_kind"),
            (self.covariance_policy, COVARIANCE_POLICIES, "covariance_policy"),
        )
        for value, allowed, label in choices:
            if value not in allowed:
                raise ValueError(f"unknown {label} {value!r}; choose one of {sorted(allowed)}")
        if not np.isfinite(self.physical_noise_scale) or self.physical_noise_scale < 0:
            raise ValueError("physical_noise_scale must be finite and nonnegative")
        if not np.isfinite(self.numerical_jitter) or self.numerical_jitter < 0:
            raise ValueError("numerical_jitter must be finite and nonnegative")
        if self.uncertainty_model == "fixed_relative":
            if self.fixed_relative_sigma is None or not (
                np.isfinite(self.fixed_relative_sigma) and self.fixed_relative_sigma > 0
            ):
                raise ValueError("fixed_relative requires a positive fixed_relative_sigma")
        elif self.fixed_relative_sigma is not None:
            raise ValueError("fixed_relative_sigma is valid only for fixed_relative")
        if self.output_kind == "posterior" and self.physical_noise_scale <= 0:
            raise ValueError(
                "ordinary full-dimensional MAF posterior requires positive physical noise; "
                "use a positive near-noiseless floor or point_estimate"
            )
        if self.uncertainty_model == "equal_weight_point_fit" and self.output_kind != "point_estimate":
            raise ValueError("equal_weight_point_fit cannot make a posterior claim")
        if self.output_kind == "point_estimate" and self.uncertainty_exposure != "values_and_kinematics_only":
            raise ValueError("point benchmarks must not expose uncertainty descriptors")
        return self

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {"schema_version": 1, **asdict(self)}

    @property
    def identifier(self) -> str:
        return content_id("uncertainty-contract", self.as_dict())


_PRESETS = {
    "full_uncertainty_posterior": UncertaintyContract(
        "full_uncertainty_posterior", "sampled", "full_covariance",
        "full_descriptors", "sampled_and_inferred", 1.0, 1e-10,
        "posterior", "training_reference",
    ),
    "central_observation_full_likelihood": UncertaintyContract(
        "central_observation_full_likelihood", "central", "full_covariance",
        "full_descriptors", "sampled_and_inferred", 1.0, 1e-10,
        "posterior", "training_reference",
    ),
    "literature_fixed_relative_replica": UncertaintyContract(
        "literature_fixed_relative_replica", "sampled", "fixed_relative",
        "values_and_kinematics_only", "nominal_fixed", 1.0, 1e-10,
        "point_estimate", "external_fixed", fixed_relative_sigma=0.10,
    ),
    "near_noiseless_posterior": UncertaintyContract(
        "near_noiseless_posterior", "sampled", "full_covariance",
        "full_descriptors", "sampled_and_inferred", 0.1, 1e-10,
        "posterior", "training_reference",
    ),
    "uncertainty_blind_point_benchmark": UncertaintyContract(
        "uncertainty_blind_point_benchmark", "central", "equal_weight_point_fit",
        "values_and_kinematics_only", "nominal_fixed", 0.0, 0.0,
        "point_estimate", "external_fixed",
    ),
}


def uncertainty_preset(name: str, **updates: Any) -> UncertaintyContract:
    try:
        contract = _PRESETS[name]
    except KeyError as error:
        raise ValueError(f"unknown uncertainty preset {name!r}") from error
    return replace(contract, **updates).validate()


def resolve_covariance(
    *, policy: str, training_reference: np.ndarray,
    external_fixed: np.ndarray | None = None,
    holdout_truth: np.ndarray | None = None,
    truth_derived: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Resolve covariance independently of generator identity."""

    if policy not in COVARIANCE_POLICIES:
        raise ValueError(f"unknown covariance policy {policy!r}")
    if policy == "training_reference":
        selected = training_reference
    elif policy == "external_fixed":
        if external_fixed is None:
            raise ValueError("external_fixed covariance policy requires a matrix")
        selected = external_fixed
    else:
        if holdout_truth is None or truth_derived is None:
            raise ValueError("holdout_truth_derived requires truth and a declared builder")
        selected = truth_derived(np.asarray(holdout_truth, dtype=np.float64))
    matrix = np.asarray(selected, dtype=np.float64)
    validated, _, _ = validate_covariance(matrix, len(matrix))
    return np.array(validated, copy=True)


def materialize_observation(
    *, central: np.ndarray, covariance: np.ndarray,
    contract: UncertaintyContract, standard_normal: np.ndarray,
) -> dict[str, Any]:
    """Materialize one paired realization without invoking the native backend."""

    contract.validate()
    mean = np.asarray(central, dtype=np.float64)
    z = np.asarray(standard_normal, dtype=np.float64)
    if mean.ndim != 1 or z.shape != mean.shape or not np.all(np.isfinite(z)):
        raise ValueError("central and standard_normal must be finite equal-length vectors")
    reference = np.asarray(covariance, dtype=np.float64)
    if contract.uncertainty_model == "fixed_relative":
        sigma = contract.fixed_relative_sigma * np.abs(mean)
        reference = np.diag(sigma**2)
    elif contract.uncertainty_model == "diagonal":
        reference = np.diag(np.diag(reference))
    elif contract.uncertainty_model == "equal_weight_point_fit":
        reference = np.eye(len(mean), dtype=np.float64)
    reference, _, _ = validate_covariance(reference, len(mean))
    physical_covariance = contract.physical_noise_scale**2 * reference
    stabilized_covariance = physical_covariance + contract.numerical_jitter * np.eye(len(mean))
    if contract.observation_realization == "central":
        observed = mean.copy()
    else:
        factor = np.linalg.cholesky(reference)
        observed = mean + contract.physical_noise_scale * factor @ z
    return {
        "schema_version": 1,
        "uncertainty_contract": deepcopy(contract.as_dict()),
        "uncertainty_contract_id": contract.identifier,
        "observed": observed,
        "reference_covariance": np.array(reference, copy=True),
        "physical_covariance": physical_covariance,
        "stabilized_covariance": stabilized_covariance,
        "standard_normal": z.copy(),
        "physical_noise_and_numerical_jitter_separate": True,
        "native_physics_executed": False,
    }


def materialize_uncertainty_variant(
    *, artifact_root: Path, master_corpus_id: str, selection_id: str,
    central: np.ndarray, training_reference_covariance: np.ndarray,
    standard_normal_draws: np.ndarray, contract: UncertaintyContract,
    external_fixed_covariance: np.ndarray | None = None,
    holdout_truth: np.ndarray | None = None,
    truth_derived_covariance: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Publish a realization-only uncertainty variant with immutable ancestry."""

    contract.validate()
    mean = np.asarray(central, dtype=np.float64)
    draws = np.asarray(standard_normal_draws, dtype=np.float64)
    if mean.ndim != 1 or draws.ndim != 2 or draws.shape[1] != len(mean):
        raise ValueError("standard_normal_draws must have shape [replica, observation]")
    covariance = resolve_covariance(
        policy=contract.covariance_policy,
        training_reference=training_reference_covariance,
        external_fixed=external_fixed_covariance,
        holdout_truth=holdout_truth,
        truth_derived=truth_derived_covariance,
    )
    realizations = [
        materialize_observation(
            central=mean, covariance=covariance, contract=contract,
            standard_normal=draw,
        )
        for draw in draws
    ]
    identity = {
        "schema_version": 1, "master_corpus_id": master_corpus_id,
        "selection_id": selection_id, "uncertainty_contract": contract.as_dict(),
        "central_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
        "reference_covariance_sha256": hashlib.sha256(covariance.tobytes()).hexdigest(),
        "standard_normal_draws_sha256": hashlib.sha256(draws.tobytes()).hexdigest(),
        "replica_count": len(draws),
        "native_predictions_reused": True, "native_physics_executed": False,
    }
    realization_id = content_id("uncertainty-realization", identity)
    destination = artifact_root / "realizations" / realization_id
    destination.mkdir(parents=True, exist_ok=True)
    arrays_path = destination / "uncertainty_realizations.npz"
    payload = {
        "observed": np.stack([item["observed"] for item in realizations]),
        "standard_normal": draws,
        "reference_covariance": covariance,
        "physical_covariance": realizations[0]["physical_covariance"],
        "stabilized_covariance": realizations[0]["stabilized_covariance"],
    }
    with tempfile.NamedTemporaryFile(dir=destination, delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **payload)
        stream.flush()
        os.fsync(stream.fileno())
    if arrays_path.exists():
        with np.load(arrays_path, allow_pickle=False) as existing, np.load(
            temporary, allow_pickle=False
        ) as candidate:
            same = set(existing.files) == set(candidate.files) and all(
                np.array_equal(existing[name], candidate[name]) for name in existing.files
            )
        temporary.unlink()
        if not same:
            raise RuntimeError("immutable uncertainty realization collision")
    else:
        os.replace(temporary, arrays_path)
    manifest = {
        "schema_version": 1, "realization_id": realization_id,
        "identity": identity, "master_corpus_id": master_corpus_id,
        "selection_id": selection_id, "uncertainty_contract_id": contract.identifier,
        "uncertainty_contract": contract.as_dict(),
        "arrays": {
            "path": arrays_path.name,
            "sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
            "shapes": {name: list(array.shape) for name, array in payload.items()},
        },
        "paired_standard_normal_draws": True,
        "physical_noise_and_numerical_jitter_separate": True,
        "native_predictions_reused": True, "native_physics_executed": False,
    }
    atomic_json(destination / "realization.json", manifest)
    return manifest
