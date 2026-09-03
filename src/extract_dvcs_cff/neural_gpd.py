"""NNGPD-inspired latent function representation built only from stored truth.

This is a function autoencoder used inside amortized SBI; it is not a complete
reproduction of the NNGPD method and is not described as model-independent.
The module never imports or starts the PARTONS bridge.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


COORDINATE_SCHEMA_VERSION = 1
COORDINATE_GPD_TYPES = ("H", "E", "Htilde", "Etilde")
COORDINATE_CHANNELS = (
    "u", "u_plus", "u_minus", "d", "d_plus", "d_minus",
    "s", "s_plus", "s_minus", "gluon",
    "charge_squared_weighted_c_even_quark_sum",
)
COORDINATE_PARITIES = ("native", "plus", "minus", "not_applicable")


def canonical_coordinate_table(
    coordinates: Sequence[Mapping[str, Any]], *, Q2_GeV2: float,
    coordinate_normalization: Mapping[str, Any] | None = None,
    value_normalization: Mapping[str, Any] | None = None,
    channel_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Encode typed GPD coordinates without imposing accidental ordinality."""

    if not np.isfinite(Q2_GeV2) or Q2_GeV2 <= 0:
        raise ValueError("Q2_GeV2 must be finite and positive")
    feature_names = (
        "x", "xi", "t_GeV2", "Q2_GeV2", "mask",
        *(f"gpd_is_{name}" for name in COORDINATE_GPD_TYPES),
        *(f"channel_is_{name}" for name in COORDINATE_CHANNELS),
        *(f"charge_parity_is_{name}" for name in COORDINATE_PARITIES),
    )
    rows = []
    normalized_records = []
    for index, raw in enumerate(coordinates):
        required = {"x", "xi", "t_GeV2", "gpd_type", "channel", "charge_parity"}
        if set(raw) != required:
            raise ValueError(f"coordinate {index} must contain exactly {sorted(required)}")
        gpd_type = str(raw["gpd_type"]); channel = str(raw["channel"])
        parity = str(raw["charge_parity"])
        if gpd_type not in COORDINATE_GPD_TYPES or channel not in COORDINATE_CHANNELS:
            raise ValueError(f"coordinate {index} has unknown GPD/channel category")
        if parity not in COORDINATE_PARITIES:
            raise ValueError(f"coordinate {index} has unknown charge parity")
        x, xi, t = float(raw["x"]), float(raw["xi"]), float(raw["t_GeV2"])
        if not all(np.isfinite((x, xi, t))) or not -1 <= x <= 1 or not 0 <= xi < 1 or t > 0:
            raise ValueError(f"coordinate {index} has invalid continuous values")
        rows.append((
            x, xi, t, float(Q2_GeV2), 1.0,
            *(float(gpd_type == name) for name in COORDINATE_GPD_TYPES),
            *(float(channel == name) for name in COORDINATE_CHANNELS),
            *(float(parity == name) for name in COORDINATE_PARITIES),
        ))
        normalized_records.append({**dict(raw), "Q2_GeV2": float(Q2_GeV2), "mask": True})
    if not rows:
        raise ValueError("canonical coordinate table must not be empty")
    table = np.asarray(rows, dtype=np.float64)
    coordinate_normalization = dict(coordinate_normalization or {
        "identifier": "native_coordinate_units_v1",
        "x": "dimensionless", "xi": "dimensionless",
        "t_GeV2": "GeV2", "Q2_GeV2": "GeV2",
    })
    value_normalization = dict(value_normalization or {"identifier": "none_v1"})
    weights = dict(channel_weights or {name: 1.0 for name in COORDINATE_CHANNELS})
    if set(weights) != set(COORDINATE_CHANNELS) or not all(
        np.isfinite(value) and value > 0 for value in weights.values()
    ):
        raise ValueError("channel_weights must define one positive value per channel")
    identity = {
        "schema_version": COORDINATE_SCHEMA_VERSION,
        "categorical_encoding": "fixed_one_hot_v1", "feature_names": list(feature_names),
        "coordinate_normalization": coordinate_normalization,
        "value_normalization": value_normalization,
        "channel_weights": weights, "records": normalized_records,
    }
    identifier = "gpd-coordinate-table-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"identifier": identifier, "identity": identity, "values": table}


def expand_group_targets(
    group_targets: np.ndarray, parameter_indices: np.ndarray,
) -> np.ndarray:
    """Lazily align group-level neural targets with realization rows."""

    targets = np.asarray(group_targets)
    indices = np.asarray(parameter_indices)
    if targets.ndim != 2 or indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("targets must be [group,target] and parameter_indices must be integer [row]")
    if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(targets)):
        raise ValueError("parameter_indices contain an invalid group reference")
    return targets[indices]


def fit_full_covariance_whitening(
    latent: np.ndarray, training_groups: Sequence[int], *, shrinkage: float = 1e-6,
    relative_eigenvalue_floor: float = 1e-8,
) -> dict[str, Any]:
    """Fit a stable full-covariance whitening map on training groups only."""

    values = np.asarray(latent, dtype=np.float64)
    groups = np.asarray(training_groups, dtype=np.int64)
    if values.ndim != 2 or not len(groups) or np.any(groups < 0) or np.any(groups >= len(values)):
        raise ValueError("whitening requires [group,latent] values and valid training groups")
    if not 0 <= shrinkage < 1 or not 0 < relative_eigenvalue_floor < 1:
        raise ValueError("invalid whitening shrinkage or eigenvalue floor")
    selected = values[groups]
    mean = selected.mean(axis=0)
    centered = selected - mean
    covariance = centered.T @ centered / max(1, len(selected) - 1)
    trace_scale = float(np.trace(covariance) / len(mean))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = float(eigenvalues[-1]) if len(eigenvalues) else 0.0
    retained = eigenvalues > max(maximum * relative_eigenvalue_floor, np.finfo(float).eps)
    if not np.any(retained):
        raise ValueError("all latent dimensions collapsed on the training split")
    basis = eigenvectors[:, retained]
    kept = (1 - shrinkage) * eigenvalues[retained] + shrinkage * trace_scale
    whiten = basis / np.sqrt(kept)
    inverse = basis * np.sqrt(kept)
    return {
        "schema_version": 1, "identifier": "full_covariance_shrinkage_whitening_v1",
        "fit_scope": "training_groups_only", "mean": mean,
        "whitening_matrix": whiten, "inverse_matrix": inverse,
        "eigenvalues": eigenvalues, "retained_mask": retained,
        "retained_dimensions": int(np.count_nonzero(retained)),
        "condition_number": float(kept[-1] / kept[0]),
        "shrinkage": float(shrinkage), "relative_eigenvalue_floor": float(relative_eigenvalue_floor),
        "training_group_sha256": hashlib.sha256(groups.tobytes()).hexdigest(),
    }


def apply_whitening(latent: np.ndarray, transform: Mapping[str, Any]) -> np.ndarray:
    return (np.asarray(latent, dtype=np.float64) - transform["mean"]) @ transform["whitening_matrix"]


def invert_whitening(whitened: np.ndarray, transform: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(whitened, dtype=np.float64) @ np.asarray(transform["inverse_matrix"]).T + transform["mean"]


def coordinate_table_hash(coordinates: np.ndarray) -> str:
    array = np.ascontiguousarray(coordinates, dtype=np.float64)
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True)
class LatentPrior:
    identifier: str
    family: str
    fit_scope: str
    training_group_sha256: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "identifier": self.identifier,
            "family": self.family, "fit_scope": self.fit_scope,
            "training_group_sha256": self.training_group_sha256,
            "mean": list(self.mean), "scale": list(self.scale),
        }


def fitted_diagonal_gaussian_prior(
    latent: np.ndarray, training_groups: Sequence[int]
) -> LatentPrior:
    values = np.asarray(latent, dtype=np.float64)
    groups = np.asarray(training_groups, dtype=np.int64)
    if values.ndim != 2 or not len(groups):
        raise ValueError("latent prior requires [group,latent] values and training groups")
    if np.any(groups < 0) or np.any(groups >= len(values)):
        raise ValueError("latent-prior training group outside array")
    selected = values[groups]
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    group_hash = hashlib.sha256(groups.tobytes()).hexdigest()
    payload = {
        "family": "training_fitted_diagonal_gaussian",
        "fit_scope": "training_groups_only", "training_group_sha256": group_hash,
        "mean": mean.tolist(), "scale": scale.tolist(),
    }
    identifier = "latent-prior-" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return LatentPrior(
        identifier, payload["family"], payload["fit_scope"], group_hash,
        tuple(float(item) for item in mean), tuple(float(item) for item in scale),
    )


def require_exact_coordinate_support(
    requested: np.ndarray, stored: np.ndarray, interpolation_policy: str
) -> None:
    requested_array = np.asarray(requested, dtype=np.float64)
    stored_array = np.asarray(stored, dtype=np.float64)
    if interpolation_policy == "exact_coordinates_only":
        if requested_array.shape != stored_array.shape or not np.array_equal(
            requested_array, stored_array
        ):
            raise RuntimeError(
                "neural GPD decoder refuses extrapolation: requested coordinates "
                "must exactly match the stored canonical truth table"
            )
        return
    if interpolation_policy != "linear_inside_stored_hull":
        raise ValueError("unknown interpolation policy")
    if requested_array.ndim != 2 or stored_array.ndim != 2:
        raise ValueError("coordinate arrays must be 2-D")
    lower, upper = stored_array.min(axis=0), stored_array.max(axis=0)
    if np.any(requested_array < lower) or np.any(requested_array > upper):
        raise RuntimeError("neural GPD decoder refuses extrapolation outside stored hull")


def torch_components():
    """Create the auditable permutation-invariant encoder/coordinate decoder."""

    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError("neural-GPD decoder requires the pinned torch profile") from error

    class FunctionEncoder(nn.Module):
        def __init__(self, coordinate_width: int, latent_dim: int, hidden_width: int):
            super().__init__()
            self.point = nn.Sequential(
                nn.Linear(coordinate_width + 1, hidden_width), nn.Tanh(),
                nn.Linear(hidden_width, hidden_width), nn.Tanh(),
            )
            self.project = nn.Linear(hidden_width, latent_dim)

        def forward(self, coordinates, values, mask):
            token = torch.cat((coordinates, values.unsqueeze(-1)), dim=-1)
            encoded = self.point(token) * mask.unsqueeze(-1)
            pooled = encoded.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return self.project(pooled)

    class CoordinateDecoder(nn.Module):
        def __init__(self, coordinate_width: int, latent_dim: int,
                     hidden_width: int, hidden_depth: int):
            super().__init__()
            layers: list[Any] = []
            width = coordinate_width + latent_dim
            for _ in range(hidden_depth):
                layers.extend((nn.Linear(width, hidden_width), nn.Tanh()))
                width = hidden_width
            layers.append(nn.Linear(width, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, coordinates, latent):
            expanded = latent.unsqueeze(1).expand(-1, coordinates.shape[1], -1)
            return self.network(torch.cat((coordinates, expanded), dim=-1)).squeeze(-1)

    class FunctionAutoencoder(nn.Module):
        def __init__(self, coordinate_width: int, latent_dim: int,
                     hidden_width: int = 128, hidden_depth: int = 3):
            super().__init__()
            self.encoder = FunctionEncoder(coordinate_width, latent_dim, hidden_width)
            self.decoder = CoordinateDecoder(
                coordinate_width, latent_dim, hidden_width, hidden_depth
            )

        def forward(self, coordinates, values, mask):
            latent = self.encoder(coordinates, values, mask)
            return self.decoder(coordinates, latent), latent

    return torch, FunctionAutoencoder


def train_function_autoencoder(
    *, coordinates: np.ndarray, values: np.ndarray, masks: np.ndarray,
    training_groups: Sequence[int], validation_groups: Sequence[int],
    locked_holdout_groups: Sequence[int], output: Path, latent_dim: int = 16,
    hidden_width: int = 128, hidden_depth: int = 3, epochs: int = 100,
    learning_rate: float = 1e-3, seed: int = 73129,
) -> dict[str, Any]:
    """Fit on training groups, select by validation, never tune on holdout."""

    torch, Autoencoder = torch_components()
    coordinates_np = np.asarray(coordinates, dtype=np.float32)
    values_np = np.asarray(values, dtype=np.float32)
    masks_np = np.asarray(masks, dtype=np.float32)
    if values_np.ndim != 2 or masks_np.shape != values_np.shape:
        raise ValueError("GPD function values/masks must have shape [group,coordinate]")
    if coordinates_np.ndim != 2 or coordinates_np.shape[0] != values_np.shape[1]:
        raise ValueError("coordinate table must have shape [coordinate,feature]")
    train = np.asarray(training_groups, dtype=np.int64)
    validation = np.asarray(validation_groups, dtype=np.int64)
    holdout = np.asarray(locked_holdout_groups, dtype=np.int64)
    if not len(train) or not len(validation) or not len(holdout):
        raise ValueError("decoder requires nonempty train/validation/locked-holdout groups")
    if set(train) & set(validation) or set(train) & set(holdout) or set(validation) & set(holdout):
        raise ValueError("decoder group roles overlap")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    coordinate_tensor = torch.from_numpy(coordinates_np).unsqueeze(0).expand(
        len(values_np), -1, -1
    )
    value_tensor = torch.from_numpy(values_np)
    mask_tensor = torch.from_numpy(masks_np)
    model = Autoencoder(coordinates_np.shape[1], latent_dim, hidden_width, hidden_depth)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float("inf")
    best_state = None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(
            coordinate_tensor[train], value_tensor[train], mask_tensor[train]
        )
        loss = (((prediction - value_tensor[train]) * mask_tensor[train]) ** 2).sum()
        loss = loss / mask_tensor[train].sum().clamp_min(1.0)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction, _ = model(
                coordinate_tensor[validation], value_tensor[validation],
                mask_tensor[validation]
            )
            validation_loss = float((
                (((prediction - value_tensor[validation]) * mask_tensor[validation]) ** 2).sum()
                / mask_tensor[validation].sum().clamp_min(1.0)
            ).item())
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("decoder training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        reconstruction, latent = model(coordinate_tensor, value_tensor, mask_tensor)
    latent_np = latent.numpy()
    prior = fitted_diagonal_gaussian_prior(latent_np, train)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1, "state_dict": best_state,
        "architecture": {
            "identifier": "permutation_invariant_function_autoencoder_v1",
            "decoder": "coordinate_conditioned_mlp_v1", "latent_dim": latent_dim,
            "coordinate_width": coordinates_np.shape[1],
            "hidden_width": hidden_width, "hidden_depth": hidden_depth,
        }, "coordinate_table_sha256": coordinate_table_hash(coordinates_np),
        "seed": seed, "training_groups": train.tolist(),
        "validation_groups": validation.tolist(),
        "locked_holdout_groups_sha256": hashlib.sha256(holdout.tobytes()).hexdigest(),
        "holdout_used_for_tuning": False, "latent_prior": prior.as_dict(),
    }
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        temporary = Path(stream.name)
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)
    errors = (reconstruction.numpy() - values_np) * masks_np
    return {
        "schema_version": 1, "checkpoint": str(output),
        "checkpoint_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "coordinate_table_sha256": checkpoint["coordinate_table_sha256"],
        "latent_targets": latent_np, "latent_prior": prior.as_dict(),
        "validation_mse": best_loss,
        "reconstruction_rmse_by_group": np.sqrt(
            (errors**2).sum(axis=1) / masks_np.sum(axis=1).clip(min=1)
        ), "holdout_used_for_tuning": False,
    }


def publish_neural_model_view(
    *, artifact_root: Path, realization_id: str,
    observation_arrays: Mapping[str, Path], coordinates: np.ndarray,
    truth_values: np.ndarray, truth_masks: np.ndarray,
    training_groups: Sequence[int], validation_groups: Sequence[int],
    locked_holdout_groups: Sequence[int], decoder_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and publish a neural-GPD view entirely from stored artifacts."""

    from extract_dvcs_cff.contracts import (
        assert_observation_context_inputs, atomic_json, model_view_identity,
    )

    required = {
        "contexts", "pseudodata_context", "train_indices",
        "validation_indices", "test_indices", "parameter_indices",
    }
    if set(observation_arrays) != required:
        raise ValueError(
            "neural model-view observation arrays must be exactly "
            + ", ".join(sorted(required))
        )
    assert_observation_context_inputs(["observations_normalized", "masks"])
    configuration = {
        "latent_dim": int(decoder_configuration.get("latent_dim", 16)),
        "hidden_width": int(decoder_configuration.get("hidden_width", 128)),
        "hidden_depth": int(decoder_configuration.get("hidden_depth", 3)),
        "epochs": int(decoder_configuration.get("epochs", 100)),
        "learning_rate": float(decoder_configuration.get("learning_rate", 1e-3)),
        "seed": int(decoder_configuration.get("seed", 73129)),
    }
    candidate = artifact_root / (
        ".neural-decoder-" + hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest() + ".pt"
    )
    result = train_function_autoencoder(
        coordinates=coordinates, values=truth_values, masks=truth_masks,
        training_groups=training_groups, validation_groups=validation_groups,
        locked_holdout_groups=locked_holdout_groups, output=candidate,
        **configuration,
    )
    whitening = fit_full_covariance_whitening(
        result["latent_targets"], training_groups
    )
    whitened_targets = apply_whitening(result["latent_targets"], whitening)
    latent_transform_id = "latent-whitening-" + hashlib.sha256(
        json.dumps({
            "mean": whitening["mean"].tolist(),
            "matrix": whitening["whitening_matrix"].tolist(),
            "retained_mask": whitening["retained_mask"].tolist(),
            "training_group_sha256": whitening["training_group_sha256"],
        }, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    decoder_contract = {
        "identifier": "coordinate_conditioned_mlp_v1",
        "architecture": configuration,
        "checkpoint_sha256": result["checkpoint_sha256"],
        "coordinate_table_sha256": result["coordinate_table_sha256"],
        "extrapolation_policy": "forbidden",
    }
    view_id, identity = model_view_identity(
        realization_id=realization_id,
        model_family="neural_gpd_deepsets_maf",
        observation_contract={"encoder": "deepsets", "arrays": sorted(required)},
        target_transform={
            "identifier": "neural_gpd_full_covariance_whitening_v1",
            "latent_transform_id": latent_transform_id,
            "fit_scope": "training_groups_only",
            "retained_dimensions": whitening["retained_dimensions"],
        },
        decoder_contract=decoder_contract,
        normalization_contract={"fit_scope": "training_groups_only"},
    )
    view_root = artifact_root / "model_views" / view_id
    view_root.mkdir(parents=True, exist_ok=True)
    checkpoint = view_root / "decoder_checkpoint.pt"
    if checkpoint.exists():
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != result["checkpoint_sha256"]:
            raise RuntimeError("immutable neural decoder checkpoint collision")
        candidate.unlink(missing_ok=True)
    else:
        os.replace(candidate, checkpoint)
    latent_path = view_root / "latent_targets.npy"
    with tempfile.NamedTemporaryFile(dir=view_root, delete=False) as stream:
        temporary = Path(stream.name)
        np.save(stream, whitened_targets, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, latent_path)
    arrays: dict[str, Any] = {
        "latent_targets": {
            "path": latent_path.name,
            "sha256": hashlib.sha256(latent_path.read_bytes()).hexdigest(),
            "shape": list(whitened_targets.shape),
            "dtype": str(whitened_targets.dtype),
        }
    }
    for name, source_value in observation_arrays.items():
        source = source_value.resolve(strict=True)
        destination = view_root / f"{name}.npy"
        if not destination.exists():
            os.link(source, destination)
        arrays[name] = {
            "path": destination.name,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema_version": 1, "model_view_id": view_id, "identity": identity,
        "realization_id": realization_id,
        "model_family": "neural_gpd_deepsets_maf", "inference_method": "npe",
        "gpd_inference_representation": "nngpd_inspired_latent_function",
        "observation_encoder": "deepsets", "density_estimator": "maf",
        "density_estimator_implementation": "zuko_maf",
        "physics_backend": "partons", "arrays": arrays,
        "decoder": decoder_contract,
        "latent_prior": {
            "schema_version": 1,
            "identifier": "standard_normal_after_full_whitening_v1",
            "family": "standard_normal",
            "dimension": whitening["retained_dimensions"],
            "fit_scope": "induced_by_training_groups_only_whitening",
        },
        "latent_transform": {
            "identifier": latent_transform_id,
            "method": whitening["identifier"],
            "fit_scope": whitening["fit_scope"],
            "mean": whitening["mean"].tolist(),
            "whitening_matrix": whitening["whitening_matrix"].tolist(),
            "inverse_matrix": whitening["inverse_matrix"].tolist(),
            "eigenvalues": whitening["eigenvalues"].tolist(),
            "retained_mask": whitening["retained_mask"].tolist(),
            "retained_dimensions": whitening["retained_dimensions"],
            "condition_number": whitening["condition_number"],
            "shrinkage": whitening["shrinkage"],
            "relative_eigenvalue_floor": whitening["relative_eigenvalue_floor"],
            "training_group_sha256": whitening["training_group_sha256"],
            "legacy_diagonal_prior": result["latent_prior"],
        },
        "reconstruction": {
            "validation_mse": result["validation_mse"],
            "rmse_by_group": result["reconstruction_rmse_by_group"].tolist(),
            "holdout_used_for_tuning": False,
        },
        "target_alignment": {
            "target_granularity": "native_truth_group",
            "context_granularity": "pseudodata_realization_row",
            "row_to_group_array": "parameter_indices",
            "expansion_policy": "lazy_indexed_v1",
        },
        "truth_fields_in_context": [],
    }
    atomic_json(view_root / "model_view.json", manifest)
    return manifest
