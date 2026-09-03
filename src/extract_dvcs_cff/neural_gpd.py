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
        "validation_indices", "test_indices",
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
            "identifier": "neural_gpd_latent_standardization_v1",
            "latent_prior_id": result["latent_prior"]["identifier"],
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
        np.save(stream, result["latent_targets"], allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, latent_path)
    arrays: dict[str, Any] = {
        "latent_targets": {
            "path": latent_path.name,
            "sha256": hashlib.sha256(latent_path.read_bytes()).hexdigest(),
            "shape": list(result["latent_targets"].shape),
            "dtype": str(result["latent_targets"].dtype),
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
        "decoder": decoder_contract, "latent_prior": result["latent_prior"],
        "reconstruction": {
            "validation_mse": result["validation_mse"],
            "rmse_by_group": result["reconstruction_rmse_by_group"].tolist(),
            "holdout_used_for_tuning": False,
        }, "truth_fields_in_context": [],
    }
    atomic_json(view_root / "model_view.json", manifest)
    return manifest
