"""Versioned, architecture-neutral artifact contracts.

This module deliberately has no import of the native launcher, torch, sbi, or
the model implementations.  Selection, realization, view construction, and
saved-result inspection must remain usable while native execution is disabled.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MASTER_CORPUS_SCHEMA_VERSION = 2
SELECTION_SCHEMA_VERSION = 1
PSEUDODATA_REALIZATION_SCHEMA_VERSION = 3
MODEL_VIEW_SCHEMA_VERSION = 1
RESULT_BUNDLE_SCHEMA_VERSION = 2
GPD_TRUTH_REQUEST_SCHEMA_VERSION = 1

SUPPORTED_MASTER_CORPUS_SCHEMAS = (1, 2)
SUPPORTED_REALIZATION_SCHEMAS = (2, 3)

GPD_TYPES = ("H", "E", "Htilde", "Etilde")
GPD_CHANNELS = (
    "u", "u_plus", "u_minus", "d", "d_plus", "d_minus",
    "s", "s_plus", "s_minus", "gluon",
    "charge_squared_weighted_c_even_quark_sum",
)

TRUTH_ONLY_ARRAYS = frozenset({
    "native_parameters", "native_predictions_normalized", "native_cffs",
    "theta_physical", "theta_latent", "truth_theta_physical", "truth_cffs",
    "truth_native_predictions_normalized", "gpd_truth_values",
})
OBSERVATION_CONTEXT_INPUTS = frozenset({
    "observations_normalized", "covariance_normalized",
    "global_normalization_response", "lu_normalization_response",
    "parameter_indices", "context_active_kinematic_counts",
    "context_design_variants", "pseudodata_observed_normalized", "point_table",
    "observable_labels", "masks",
})


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_id(kind: str, payload: Mapping[str, Any]) -> str:
    """Return a stable typed identity for deterministic JSON-compatible data."""

    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return f"{kind}-{digest}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], label: str) -> None:
    wanted = set(expected)
    actual = set(value)
    if actual != wanted:
        raise ValueError(
            f"{label} key mismatch: missing={sorted(wanted - actual)}, "
            f"unknown={sorted(actual - wanted)}"
        )


@dataclass(frozen=True)
class ArtifactCapabilities:
    dd_deepsets_maf: bool
    neural_gpd_deepsets_maf: bool
    canonical_gpd_truth: bool
    reason_neural_unavailable: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "dd_deepsets_maf": self.dd_deepsets_maf,
            "neural_gpd_deepsets_maf": self.neural_gpd_deepsets_maf,
            "canonical_gpd_truth": self.canonical_gpd_truth,
            "reason_neural_unavailable": self.reason_neural_unavailable,
        }


def corpus_capabilities(manifest: Mapping[str, Any]) -> ArtifactCapabilities:
    schema = int(manifest.get("schema_version", -1))
    if schema not in SUPPORTED_MASTER_CORPUS_SCHEMAS:
        raise ValueError(
            f"unsupported master corpus schema {schema}; supported: "
            f"{list(SUPPORTED_MASTER_CORPUS_SCHEMAS)}"
        )
    truth = manifest.get("gpd_truth")
    has_truth = bool(
        isinstance(truth, Mapping)
        and truth.get("status") == "complete"
        and truth.get("coordinates")
        and truth.get("shards")
        and all(
            isinstance(record, Mapping)
            and record.get("path")
            and record.get("sha256")
            for record in truth.get("shards", ())
        )
    )
    reason = None if has_truth else (
        "master corpus lacks canonical native GPD function truth; the DD baseline "
        "remains supported, but neural_gpd_deepsets_maf requires a schema-2 corpus "
        "generated or explicitly extended with complete, checksummed GPD-truth "
        "shards; a planned coordinate request is not sufficient"
    )
    return ArtifactCapabilities(True, has_truth, has_truth, reason)


def require_model_capability(manifest: Mapping[str, Any], model_family: str) -> None:
    capabilities = corpus_capabilities(manifest)
    if model_family == "dd_deepsets_maf":
        return
    if model_family == "neural_gpd_deepsets_maf":
        if not capabilities.neural_gpd_deepsets_maf:
            raise RuntimeError(str(capabilities.reason_neural_unavailable))
        return
    raise ValueError(f"unknown model family {model_family!r}")


def validate_gpd_truth_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a user-supplied coordinate table without expanding it."""

    if not isinstance(value, Mapping):
        raise TypeError("gpd truth request must be an object")
    required = {
        "schema_version", "coordinate_convention", "gpd_flavor_basis",
        "scale", "scheme", "generator_representation", "interpolation_policy",
        "dtype", "compression", "coordinates",
    }
    _exact_keys(value, required, "gpd truth request")
    if int(value["schema_version"]) != GPD_TRUTH_REQUEST_SCHEMA_VERSION:
        raise ValueError("gpd truth request schema_version must be 1")
    scale = value["scale"]
    if not isinstance(scale, Mapping):
        raise TypeError("gpd truth scale must be an object")
    _exact_keys(scale, {"kind", "Q0_squared_GeV2"}, "gpd truth scale")
    if scale["kind"] != "input_scale":
        raise ValueError("canonical GPD truth must be requested at input_scale")
    q0 = float(scale["Q0_squared_GeV2"])
    if not math.isfinite(q0) or q0 <= 0:
        raise ValueError("Q0_squared_GeV2 must be finite and positive")
    if value["dtype"] not in {"float32", "float64"}:
        raise ValueError("gpd truth dtype must be float32 or float64")
    if value["compression"] not in {"npz_deflate", "none"}:
        raise ValueError("gpd truth compression must be npz_deflate or none")
    interpolation = value["interpolation_policy"]
    if interpolation not in {"exact_coordinates_only", "linear_inside_stored_hull"}:
        raise ValueError("unsupported GPD truth interpolation policy")
    if value["coordinate_convention"] != "signed_x_xi_t_at_common_input_scale_v1":
        raise ValueError("unsupported GPD truth coordinate_convention")
    if value["gpd_flavor_basis"] != (
        "PARTONS_physical_u_d_s_gluon_and_charge_parity_v1"
    ):
        raise ValueError("unsupported GPD truth flavor basis")
    if value["scheme"] != "MSbar":
        raise ValueError("canonical GPD truth scheme must be MSbar")
    coordinates = value["coordinates"]
    if not isinstance(coordinates, list) or not coordinates:
        raise ValueError("gpd truth coordinates must be a non-empty explicit table")
    normalized = deepcopy(dict(value))
    seen: set[str] = set()
    for index, point in enumerate(coordinates):
        if not isinstance(point, Mapping):
            raise TypeError(f"gpd truth coordinate {index} must be an object")
        _exact_keys(
            point,
            {"x", "xi", "t_GeV2", "gpd_type", "channel", "charge_parity"},
            f"gpd truth coordinate {index}",
        )
        x = float(point["x"])
        xi = float(point["xi"])
        t = float(point["t_GeV2"])
        if not all(math.isfinite(item) for item in (x, xi, t)):
            raise ValueError(f"gpd truth coordinate {index} is non-finite")
        if not -1.0 <= x <= 1.0:
            raise ValueError(f"gpd truth coordinate {index}.x outside [-1,1]")
        if not 0.0 <= xi < 1.0:
            raise ValueError(f"gpd truth coordinate {index}.xi outside [0,1)")
        if t > 0.0:
            raise ValueError(
                f"gpd truth coordinate {index}.t_GeV2 must be nonpositive"
            )
        if point["gpd_type"] not in GPD_TYPES:
            raise ValueError(f"unsupported GPD type at coordinate {index}")
        if point["channel"] not in GPD_CHANNELS:
            raise ValueError(f"unsupported GPD channel at coordinate {index}")
        if point["charge_parity"] not in {"native", "plus", "minus", "not_applicable"}:
            raise ValueError(f"unsupported charge parity at coordinate {index}")
        channel = point["channel"]
        parity = point["charge_parity"]
        expected_parity = (
            "native" if channel in {"u", "d", "s"}
            else "plus" if channel in {
                "u_plus", "d_plus", "s_plus",
                "charge_squared_weighted_c_even_quark_sum",
            }
            else "minus" if channel in {"u_minus", "d_minus", "s_minus"}
            else "not_applicable"
        )
        if parity != expected_parity:
            raise ValueError(
                f"gpd truth coordinate {index} channel {channel!r} requires "
                f"charge_parity={expected_parity!r}"
            )
        key = canonical_json(point)
        if key in seen:
            raise ValueError(f"duplicate gpd truth coordinate {index}")
        seen.add(key)
    normalized["coordinates"] = [dict(point) for point in coordinates]
    normalized["coordinate_table_sha256"] = hashlib.sha256(
        canonical_json(normalized["coordinates"]).encode("utf-8")
    ).hexdigest()
    return normalized


def master_corpus_identity(
    *, generator_family: str, generator_prior: Mapping[str, Any],
    kinematics: Sequence[Mapping[str, Any]], observables: Sequence[Mapping[str, Any]],
    native_physics: Mapping[str, Any], native_bridge_sha256: str,
    gpd_truth_request: Mapping[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Hash only native simulation support; never hash a downstream model."""

    truth = validate_gpd_truth_request(gpd_truth_request) if gpd_truth_request else None
    payload = {
        "identity_schema_version": 1,
        "generator_family": generator_family,
        "generator_prior": deepcopy(dict(generator_prior)),
        "kinematics": deepcopy(list(kinematics)),
        "observables": deepcopy(list(observables)),
        "native_physics": deepcopy(dict(native_physics)),
        "native_bridge_sha256": native_bridge_sha256,
        "gpd_truth_request": truth,
    }
    return content_id("master-corpus", payload), payload


def realization_identity(
    *, master_corpus_id: str, selection_id: str, noise_contract: Mapping[str, Any],
    covariance_contract: Mapping[str, Any], nuisance_contract: Mapping[str, Any],
    masks: Mapping[str, Any], observables: Sequence[str], seeds: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    payload = {
        "identity_schema_version": 1,
        "master_corpus_id": master_corpus_id,
        "selection_id": selection_id,
        "noise_contract": deepcopy(dict(noise_contract)),
        "covariance_contract": deepcopy(dict(covariance_contract)),
        "nuisance_contract": deepcopy(dict(nuisance_contract)),
        "masks": deepcopy(dict(masks)),
        "observables": list(observables),
        "seeds": deepcopy(dict(seeds)),
    }
    return content_id("pseudodata-realization", payload), payload


def model_view_identity(
    *, realization_id: str, model_family: str, observation_contract: Mapping[str, Any],
    target_transform: Mapping[str, Any], decoder_contract: Mapping[str, Any] | None,
    normalization_contract: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    payload = {
        "identity_schema_version": 1,
        "realization_id": realization_id,
        "model_family": model_family,
        "observation_contract": deepcopy(dict(observation_contract)),
        "target_transform": deepcopy(dict(target_transform)),
        "decoder_contract": deepcopy(dict(decoder_contract)) if decoder_contract else None,
        "normalization_contract": deepcopy(dict(normalization_contract)),
    }
    return content_id("model-view", payload), payload


def assert_observation_context_inputs(names: Iterable[str]) -> None:
    supplied = set(names)
    forbidden = supplied & TRUTH_ONLY_ARRAYS
    if forbidden:
        raise RuntimeError(
            "synthetic truth is sealed from the observation encoder; forbidden "
            f"context inputs: {sorted(forbidden)}"
        )
    unknown = supplied - OBSERVATION_CONTEXT_INPUTS
    if unknown:
        raise RuntimeError(f"unregistered observation context inputs: {sorted(unknown)}")


def fit_train_only_standardization(
    values: np.ndarray, train_indices: Sequence[int]
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    indices = np.asarray(train_indices, dtype=np.int64)
    if array.ndim < 2 or len(indices) == 0:
        raise ValueError("train-only standardization requires a nonempty 2-D array")
    if np.any(indices < 0) or np.any(indices >= len(array)):
        raise ValueError("training index outside normalization array")
    selected = array[indices]
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return {
        "fit_scope": "training_groups_only",
        "training_index_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }


def preflight_master_corpus(
    *, native_group_count: int, observable_kinematic_count: int,
    observable_count: int, shard_size: int,
    gpd_truth_request: Mapping[str, Any] | None,
    declared_kinematic_coverage: Mapping[str, Any],
) -> dict[str, Any]:
    """Estimate the declared campaign exactly; do not invent or expand points."""

    if native_group_count < 1 or observable_kinematic_count < 1 or observable_count < 1:
        raise ValueError("preflight counts must be positive")
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    truth = validate_gpd_truth_request(gpd_truth_request) if gpd_truth_request else None
    points = len(truth["coordinates"]) if truth else 0
    dtype_bytes = np.dtype(truth["dtype"]).itemsize if truth else 0
    values_bytes = native_group_count * points * dtype_bytes
    mask_bytes = native_group_count * points
    uncompressed = values_bytes + mask_bytes
    compression = truth["compression"] if truth else None
    compressed_estimate = (
        int(math.ceil(uncompressed * 0.65)) if compression == "npz_deflate" else uncompressed
    )
    shard_count = math.ceil(native_group_count / shard_size)
    model_compatibility = {
        "dd_deepsets_maf": {"compatible": True, "reason": None},
        "neural_gpd_deepsets_maf": {
            "compatible": truth is not None,
            "reason": None if truth else "canonical GPD truth request is absent",
        },
    }
    return {
        "schema_version": 1,
        "native_physics_executed": False,
        "physical_group_count": native_group_count,
        "observable_evaluation_count": (
            native_group_count * observable_kinematic_count * observable_count
        ),
        "gpd_truth_points_per_group": points,
        "estimated_storage": {
            "uncompressed_bytes": uncompressed,
            "compressed_bytes": compressed_estimate,
            "compressed_estimate_assumption": (
                "0.65 ratio planning assumption; actual native values were not generated"
                if compression == "npz_deflate" else None
            ),
        },
        "expected_shard_count": shard_count,
        "expected_file_count": shard_count * (2 + int(points > 0)) + 5,
        "declared_kinematic_coverage": deepcopy(dict(declared_kinematic_coverage)),
        "gpd_truth_coordinate_coverage": deepcopy(truth) if truth else None,
        "missing_required_fields": [] if truth else ["gpd_truth_request"],
        "model_family_compatibility": model_compatibility,
        "literature_benchmark_compatibility": {
            "nngpd_function_closure": {
                "compatible_after_native_generation": truth is not None,
                "reason": (
                    None if truth else "canonical GPD truth request is absent"
                ),
            },
            "moffat_evolution_identifiability": {
                "compatible_after_native_generation": False,
                "reason": (
                    "requires the paper-compatible type-A/type-B shadow basis "
                    "and evolution contract; Stage-09 is only a fixed stress test"
                ),
            },
        },
    }


def validate_result_bundle_manifest(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "master_corpus_id", "selection_id", "realization_id",
        "model_view_id", "model_family", "inference_method",
        "gpd_inference_representation", "observation_encoder",
        "density_estimator", "physics_backend", "target_transforms",
        "decoder", "code_commit", "container_identity", "seeds",
        "hyperparameters", "resource_summary", "validation", "acceptance",
    }
    _exact_keys(value, required, "result bundle manifest")
    if int(value["schema_version"]) != RESULT_BUNDLE_SCHEMA_VERSION:
        raise ValueError("unsupported result bundle schema")
    if value["observation_encoder"] != "deepsets":
        raise ValueError("result bundle observation_encoder must be deepsets")
    if value["density_estimator"] != "maf":
        raise ValueError("result bundle density_estimator must be maf")
    if value["physics_backend"] != "partons":
        raise ValueError("result bundle physics_backend must be partons")


def _stable_manifest_id(kind: str, manifest: Mapping[str, Any]) -> str:
    content = {
        key: value for key, value in manifest.items()
        if key not in {"manifest_sha256", "corpus_root", "selection_name"}
    }
    return content_id(kind, content)


def _link_array(source: Path, destination: Path) -> dict[str, Any]:
    """Publish one immutable array name without duplicating multi-GB data."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_symlink() or sha256(destination) != sha256(source):
            raise RuntimeError(f"immutable layered artifact collision: {destination}")
    else:
        temporary = destination.with_name(destination.name + ".partial")
        if temporary.exists():
            temporary.unlink()
        os.link(source, temporary)
        os.replace(temporary, destination)
    array = np.load(destination, mmap_mode="r", allow_pickle=False)
    return {
        "path": destination.name, "sha256": sha256(destination),
        "shape": list(array.shape), "dtype": str(array.dtype),
    }


def publish_layered_materialization(
    *, workspace: Path, master_corpus_manifest: Mapping[str, Any],
    selection_manifest: Mapping[str, Any], configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Split the compatibility materialization into canonical Layers C and D."""

    generated = workspace / "generated"
    legacy_manifest_path = generated / "array_manifest.json"
    legacy = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
    records = legacy.get("arrays")
    if not isinstance(records, Mapping):
        raise RuntimeError("legacy materialization array manifest is malformed")
    master_corpus_id = str(
        master_corpus_manifest.get("master_corpus_id")
        or content_id("master-corpus-legacy", {
            "core_identity_sha256": master_corpus_manifest["core_identity_sha256"]
        })
    )
    selection_id = str(
        selection_manifest.get("selection_id")
        or _stable_manifest_id("selection", selection_manifest)
    )
    observables = [str(item["id"]) for item in configuration["observables"]]
    design = configuration.get("observation_design_training", {
        "enabled": False,
        "active_kinematic_counts": [len(configuration["kinematics"])],
    })
    realization_id, realization_payload = realization_identity(
        master_corpus_id=master_corpus_id, selection_id=selection_id,
        noise_contract={
            "family": "multivariate_normal",
            "replicates_per_group": {
                name: int(profile["noise_replicates_per_parameter"])
                for name, profile in configuration["profiles"].items()
            },
        }, covariance_contract=configuration["experimental_model"],
        nuisance_contract={
            "groups": configuration["experimental_model"]["nuisance_groups"]
        }, masks={
            "enabled": bool(design["enabled"]),
            "active_kinematic_counts": list(design["active_kinematic_counts"]),
            "selection_seed": design.get("selection_seed"),
        }, observables=observables, seeds={
            "training_noise": int(configuration["seeds"]["training_noise"]),
            "pseudodata": int(configuration["seeds"]["pseudodata"]),
        },
    )
    realization_arrays = (
        "observations_normalized", "covariance_normalized",
        "global_normalization_response", "lu_normalization_response",
        "parameter_indices", "context_active_kinematic_counts",
        "context_design_variants", "pseudodata_observed_normalized",
    )
    truth_arrays = (
        "native_parameters", "native_predictions_normalized", "native_cffs",
        "theta_physical", "truth_theta_physical", "truth_cffs",
        "truth_native_predictions_normalized",
    )
    dd_view_arrays = (
        "contexts", "theta_physical", "theta_latent", "train_indices",
        "validation_indices", "test_indices", "pseudodata_context",
    )
    artifact_root = workspace / "artifacts"
    realization_root = artifact_root / "realizations" / realization_id
    truth_root = artifact_root / "truth_sidecars" / realization_id
    realization_records = {
        name: _link_array(generated / records[name]["path"], realization_root / f"{name}.npy")
        for name in realization_arrays
    }
    truth_records = {
        name: _link_array(generated / records[name]["path"], truth_root / f"{name}.npy")
        for name in truth_arrays
    }
    atomic_json(realization_root / "realization.json", {
        "schema_version": PSEUDODATA_REALIZATION_SCHEMA_VERSION,
        "realization_id": realization_id, "identity": realization_payload,
        "master_corpus_id": master_corpus_id, "selection_id": selection_id,
        "synthetic": True, "architecture_neutral": True,
        "observation_encoder": None, "density_estimator": None,
        "arrays": realization_records,
        "truth_sidecar": {
            "path": str(truth_root.relative_to(artifact_root)),
            "observation_encoder_access": False,
        }, "legacy_source_manifest_sha256": sha256(legacy_manifest_path),
    })
    atomic_json(truth_root / "truth.json", {
        "schema_version": 1, "realization_id": realization_id,
        "sealed_truth_evaluation_sidecar": True,
        "observation_encoder_access": False, "arrays": truth_records,
    })
    view_id, view_payload = model_view_identity(
        realization_id=realization_id, model_family="dd_deepsets_maf",
        observation_contract={
            "encoder": "deepsets", "context_contract": configuration["context_contract"],
            "network": {
                key: configuration["network"][key] for key in (
                    "point_hidden", "point_layers", "dataset_hidden",
                    "dataset_layers", "embedding_features", "point_layer_norm",
                )
            },
        }, target_transform={
            "identifier": "dd_uniform_probit_plus_nuisance_identity_v1"
        }, decoder_contract=None, normalization_contract={
            "fit_scope": "analytic_dd_transform_and_training_groups_only",
            "z_score_theta": configuration["network"]["z_score_theta"],
            "z_score_x": configuration["network"]["z_score_x"],
        },
    )
    view_root = artifact_root / "model_views" / view_id
    view_records = {
        name: _link_array(generated / records[name]["path"], view_root / f"{name}.npy")
        for name in dd_view_arrays
    }
    atomic_json(view_root / "model_view.json", {
        "schema_version": MODEL_VIEW_SCHEMA_VERSION, "model_view_id": view_id,
        "identity": view_payload, "realization_id": realization_id,
        "model_family": "dd_deepsets_maf", "inference_method": "npe",
        "gpd_inference_representation": "double_distribution_coordinates",
        "observation_encoder": "deepsets", "density_estimator": "maf",
        "density_estimator_implementation": "zuko_maf",
        "physics_backend": "partons", "arrays": view_records,
        "truth_fields_in_context": [],
    })
    atomic_json(artifact_root / "current.json", {
        "schema_version": 1, "master_corpus_id": master_corpus_id,
        "selection_id": selection_id, "realization_id": realization_id,
        "model_view_id": view_id, "model_family": "dd_deepsets_maf",
    })
    return {
        "master_corpus_id": master_corpus_id, "selection_id": selection_id,
        "realization_id": realization_id, "model_view_id": view_id,
        "materialize_compatibility": (
            "deprecated wrapper: use realization-create then model-view-create"
        ),
    }
