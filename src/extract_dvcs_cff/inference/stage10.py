"""Stage 10 DeepSets context, transforms, and neural-posterior factory.

The native simulator predicts six DVCS observables from sixteen independent
GPD-type/parton-channel shape blocks. This module performs only
statistical representation: analytic parameter transforms, full-covariance
context construction, and the conditional density-estimator definition. It
contains no GPD, CFF, or cross-section formula.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from sbi.inference import NPE
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets import posterior_nn
from sbi.utils import MultipleIndependent
from torch import Tensor
from torch.distributions import Normal
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

from .deepsets import DeepSetsDatasetEncoder


STAGE10_POINT_FEATURE_NAMES = (
    "x_b_over_0p3",
    "minus_t_GeV2_over_0p2",
    "q2_over_q0_squared",
    "beam_energy_GeV_over_12",
    "sin_phi",
    "cos_phi",
    "observable_is_uu",
    "observable_is_lu_difference",
    "observable_is_beam_charge_asymmetry",
    "observable_is_beam_spin_asymmetry",
    "observable_is_target_spin_asymmetry",
    "observable_is_double_spin_asymmetry",
    "observed_normalized",
    "marginal_sigma_normalized",
    "symmetric_whitened_observation_over_10",
    "global_normalization_response",
    "lu_normalization_response",
    "point_mask",
)

STAGE10_GLOBAL_FEATURE_NAMES = (
    "representation_is_full_independent_four_gpd_flavor_dd",
    "q0_squared_GeV2_over_1",
    "coefficient_function_is_lo",
    "evolution_is_configured",
    "datum_q2_native_evolution_enabled",
    "full_covariance_enabled",
    "all_four_twist2_gpd_types_enabled",
    "native_six_observable_route_enabled",
    "model_mask",
)

GPD_TYPES = ("H", "E", "Htilde", "Etilde")
PARTON_CHANNELS = ("u", "d", "s", "gluon")
DD_SHAPE_FIELDS = (
    "normalization",
    "a",
    "c",
    "profile_b",
    "t_slope_GeV_minus2",
)
DD_SHAPE_BOUNDS = {
    "normalization": (-4.0, 4.0),
    "a": (0.1, 1.0),
    "c": (2.0, 6.0),
    "profile_b": (1.0, 4.0),
    "t_slope_GeV_minus2": (0.0, 2.0),
}

# Stable wire order: GPD type, channel, then five independent DD controls.
# Shadow settings remain fixed simulator inputs pending a later physics ADR.
STAGE10_PHYSICS_PARAMETER_NAMES = tuple(
    f"{gpd_type}_{channel}_{field}"
    for gpd_type in GPD_TYPES
    for channel in PARTON_CHANNELS
    for field in DD_SHAPE_FIELDS
)
STAGE10_NUISANCE_PARAMETER_NAMES = (
    "eta_global_normalization",
    "eta_lu_normalization",
)
STAGE10_PARAMETER_NAMES = (
    *STAGE10_PHYSICS_PARAMETER_NAMES,
    *STAGE10_NUISANCE_PARAMETER_NAMES,
)
STAGE10_PHYSICS_PARAMETER_COUNT = len(STAGE10_PHYSICS_PARAMETER_NAMES)

_BOUNDED_MINIMUM = torch.tensor(tuple(
    DD_SHAPE_BOUNDS[field][0]
    for _gpd_type in GPD_TYPES
    for _channel in PARTON_CHANNELS
    for field in DD_SHAPE_FIELDS
))
_BOUNDED_WIDTH = torch.tensor(tuple(
    DD_SHAPE_BOUNDS[field][1] - DD_SHAPE_BOUNDS[field][0]
    for _gpd_type in GPD_TYPES
    for _channel in PARTON_CHANNELS
    for field in DD_SHAPE_FIELDS
))


class GroupedValidationNPE(NPE):
    """NPE with a caller-frozen, group-disjoint validation partition.

    SBI 0.26.1 otherwise draws validation rows uniformly. Stage 10 has
    multiple noise replicates per exact native parameter vector, so that row
    split leaks a simulator parameter between training and early stopping.
    This adapter changes only loader membership; SBI still supplies the NPE
    objective, optimizer, estimator, and training loop. Outer-test rows are
    never appended to this object.
    """

    def __init__(
        self,
        *args: Any,
        grouped_train_indices: Tensor,
        grouped_validation_indices: Tensor,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._grouped_train_indices = torch.as_tensor(
            grouped_train_indices, dtype=torch.long, device="cpu"
        ).reshape(-1)
        self._grouped_validation_indices = torch.as_tensor(
            grouped_validation_indices, dtype=torch.long, device="cpu"
        ).reshape(-1)

    def get_dataloaders(
        self,
        starting_round: int = 0,
        training_batch_size: int = 200,
        validation_fraction: float = 0.1,
        resume_training: bool = False,
        dataloader_kwargs: dict[str, Any] | None = None,
    ) -> tuple[data.DataLoader, data.DataLoader]:
        """Return loaders using frozen groups, never a random row split."""

        del validation_fraction
        theta, context, prior_masks = self.get_simulations(starting_round)
        count = int(theta.shape[0])
        train_indices = self._grouped_train_indices
        validation_indices = self._grouped_validation_indices
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise ValueError("grouped train and validation sets must be nonempty")
        combined = torch.cat((train_indices, validation_indices))
        if (
            bool(torch.any(combined < 0))
            or bool(torch.any(combined >= count))
            or len(torch.unique(combined)) != count
            or len(combined) != count
        ):
            raise ValueError(
                "grouped train/validation indices must be disjoint and cover "
                "every appended synthetic row exactly"
            )
        if resume_training and not hasattr(self, "optimizer"):
            raise ValueError(
                "resume_training requires an initialized SBI optimizer"
            )
        self.train_indices = train_indices
        self.val_indices = validation_indices
        dataset = data.TensorDataset(theta, context, prior_masks)
        additions = dict(dataloader_kwargs or {})
        forbidden = {"batch_size", "sampler", "shuffle", "drop_last"}
        overlap = forbidden & set(additions)
        if overlap:
            raise ValueError(
                "dataloader_kwargs cannot override grouped split controls: "
                f"{sorted(overlap)}"
            )
        train_kwargs = {
            "batch_size": min(training_batch_size, len(train_indices)),
            "drop_last": True,
            "sampler": SubsetRandomSampler(train_indices.tolist()),
            **additions,
        }
        validation_kwargs = {
            "batch_size": min(training_batch_size, len(validation_indices)),
            "drop_last": True,
            "sampler": SubsetRandomSampler(validation_indices.tolist()),
            **additions,
        }
        return (
            data.DataLoader(dataset, **train_kwargs),
            data.DataLoader(dataset, **validation_kwargs),
        )


def stage10_physical_to_latent(theta: Tensor) -> Tensor:
    """Map 80 independent DD controls and two nuisances to normal space.

    Uniform physical priors become exact standard-normal probit coordinates.
    The two nuisance parameters already have standard-normal priors and are
    therefore unchanged. Bounds are numerically open because probability-zero
    endpoints cannot be represented by a finite normalizing-flow coordinate.
    """

    if theta.shape[-1] != len(STAGE10_PARAMETER_NAMES):
        raise ValueError(
            f"milestone theta must have final dimension "
            f"{len(STAGE10_PARAMETER_NAMES)}"
        )
    minimum = _BOUNDED_MINIMUM.to(theta)
    width = _BOUNDED_WIDTH.to(theta)
    fraction = (
        theta[..., :STAGE10_PHYSICS_PARAMETER_COUNT] - minimum
    ) / width
    if torch.any((fraction <= 0.0) | (fraction >= 1.0)):
        raise ValueError(
            "all DD shape controls must lie strictly inside their priors"
        )
    return torch.cat(
        (
            torch.special.ndtri(fraction),
            theta[..., STAGE10_PHYSICS_PARAMETER_COUNT:],
        ),
        dim=-1,
    )


def stage10_latent_to_physical(latent: Tensor) -> Tensor:
    """Invert :func:`stage10_physical_to_latent` analytically."""

    if latent.shape[-1] != len(STAGE10_PARAMETER_NAMES):
        raise ValueError(
            f"milestone latent tensor must have final dimension "
            f"{len(STAGE10_PARAMETER_NAMES)}"
        )
    minimum = _BOUNDED_MINIMUM.to(latent)
    width = _BOUNDED_WIDTH.to(latent)
    bounded = minimum + width * torch.special.ndtr(
        latent[..., :STAGE10_PHYSICS_PARAMETER_COUNT]
    )
    return torch.cat(
        (bounded, latent[..., STAGE10_PHYSICS_PARAMETER_COUNT:]), dim=-1
    )


def stage10_prior(device: str | torch.device = "cpu") -> MultipleIndependent:
    """Return the independent standard-normal latent prior used by sbi."""

    resolved = torch.device(device)
    components = [
        Normal(
            torch.tensor([0.0], device=resolved),
            torch.tensor([1.0], device=resolved),
        )
        for _ in STAGE10_PARAMETER_NAMES
    ]
    # MultipleIndependent defaults to CPU and moves its component
    # distributions there unless its own device argument is explicit.
    return MultipleIndependent(components, device=str(resolved))


def _finite(
    value: npt.ArrayLike, shape: tuple[int, ...], name: str
) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result


class Stage10ContextBuilder:
    """Immutable encoder for many observations sharing one experiment."""

    def __init__(
        self,
        *,
        points: Sequence[Mapping[str, Any]],
        covariance: npt.ArrayLike,
        global_normalization_response: npt.ArrayLike,
        lu_normalization_response: npt.ArrayLike,
    ) -> None:
        count = len(points)
        if count == 0:
            raise ValueError("points must not be empty")
        matrix = _finite(covariance, (count, count), "covariance")
        global_response = _finite(
            global_normalization_response, (count,), "global response"
        )
        lu_response = _finite(
            lu_normalization_response, (count,), "LU response"
        )
        if not np.array_equal(matrix, matrix.T):
            raise ValueError("covariance must be exactly symmetric")
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        if np.any(eigenvalues <= 0.0):
            raise ValueError("covariance must be positive definite")
        self._inverse_sqrt = (
            eigenvectors * eigenvalues**-0.5
        ) @ eigenvectors.T
        marginal_sigma = np.sqrt(np.diag(matrix))
        tokens = np.empty(
            (count, len(STAGE10_POINT_FEATURE_NAMES)), dtype=np.float64
        )
        observable_ids = (
            "DVCSCrossSectionUUMinus",
            "DVCSCrossSectionDifferenceLUMinus",
            "DVCSAc",
            "DVCSAluMinus",
            "DVCSAulMinus",
            "DVCSAllMinus",
        )
        for index, point in enumerate(points):
            required = {
                "x_b", "t_GeV2", "Q2_GeV2", "beam_energy_GeV",
                "phi_rad", "observable_id",
            }
            if set(point) != required:
                raise ValueError(
                    f"points[{index}] must contain exactly {sorted(required)}"
                )
            observable = str(point["observable_id"])
            if observable not in observable_ids:
                raise ValueError(f"points[{index}] has unknown observable")
            x_b = float(point["x_b"])
            t = float(point["t_GeV2"])
            q2 = float(point["Q2_GeV2"])
            energy = float(point["beam_energy_GeV"])
            phi = float(point["phi_rad"])
            if (
                not all(np.isfinite((x_b, t, q2, energy, phi)))
                or not 0.0 < x_b < 1.0
                or t > 0.0
                or q2 <= 0.0
                or energy <= 0.0
            ):
                raise ValueError(f"points[{index}] has invalid kinematics")
            one_hot = tuple(float(observable == item) for item in observable_ids)
            tokens[index] = (
                x_b / 0.3,
                -t / 0.2,
                q2,
                energy / 12.0,
                np.sin(phi),
                np.cos(phi),
                *one_hot,
                0.0,
                marginal_sigma[index],
                0.0,
                global_response[index],
                lu_response[index],
                1.0,
            )
        self._tokens = tokens
        self._global_features = np.ones(
            len(STAGE10_GLOBAL_FEATURE_NAMES), dtype=np.float64
        )
        self._count = count
        self._inverse_sqrt.setflags(write=False)
        self._tokens.setflags(write=False)
        self._global_features.setflags(write=False)

    @property
    def context_width(self) -> int:
        return self._tokens.size + len(self._global_features)

    def build(self, observed: npt.ArrayLike) -> np.ndarray:
        values = _finite(observed, (self._count,), "observed")
        tokens = self._tokens.copy()
        tokens[:, 12] = values
        tokens[:, 14] = (self._inverse_sqrt @ values) / 10.0
        context = np.concatenate((tokens.reshape(-1), self._global_features))
        if not np.all(np.isfinite(context)):
            raise ValueError("constructed Stage 10 context is non-finite")
        return context.astype(np.float32)


def prepare_stage10_context(
    *,
    points: Sequence[Mapping[str, Any]],
    covariance: npt.ArrayLike,
    global_normalization_response: npt.ArrayLike,
    lu_normalization_response: npt.ArrayLike,
) -> Stage10ContextBuilder:
    """Precompute experiment-constant context terms once."""

    return Stage10ContextBuilder(
        points=points,
        covariance=covariance,
        global_normalization_response=global_normalization_response,
        lu_normalization_response=lu_normalization_response,
    )


def build_stage10_context(
    *,
    points: Sequence[Mapping[str, Any]],
    observed: npt.ArrayLike,
    covariance: npt.ArrayLike,
    global_normalization_response: npt.ArrayLike,
    lu_normalization_response: npt.ArrayLike,
) -> np.ndarray:
    """Encode one complete six-observable dataset as DeepSets tokens.

    Dense covariance enters through marginal errors and symmetric inverse-
    square-root whitening. Unlike triangular Cholesky whitening, this
    transformation is permutation equivariant: a common point/covariance
    permutation only permutes the resulting point tokens.
    """

    return prepare_stage10_context(
        points=points,
        covariance=covariance,
        global_normalization_response=global_normalization_response,
        lu_normalization_response=lu_normalization_response,
    ).build(observed)


def stage10_density_builder(configuration: Mapping[str, Any]):
    """Construct the configured DeepSets-conditioned Zuko flow factory."""

    context = configuration["context_contract"]
    network = configuration["network"]
    point_count = len(configuration["kinematics"]) * len(
        configuration["observables"]
    )
    if context["point_features"] != list(STAGE10_POINT_FEATURE_NAMES):
        raise ValueError("configuration point-feature order is not frozen")
    if context["global_features"] != list(STAGE10_GLOBAL_FEATURE_NAMES):
        raise ValueError("configuration global-feature order is not frozen")
    encoder = DeepSetsDatasetEncoder(
        token_count=point_count,
        feature_count=len(STAGE10_POINT_FEATURE_NAMES),
        global_feature_count=len(STAGE10_GLOBAL_FEATURE_NAMES),
        point_hidden=int(network["point_hidden"]),
        embedding_features=int(network["embedding_features"]),
        point_layer_norm=bool(network["point_layer_norm"]),
        point_layers=int(network["point_layers"]),
        dataset_hidden=int(network["dataset_hidden"]),
        dataset_layers=int(network["dataset_layers"]),
    )
    return posterior_nn(
        model=network["flow"],
        embedding_net=encoder,
        hidden_features=int(network["flow_hidden_features"]),
        num_transforms=int(network["num_transforms"]),
        num_bins=int(network["num_bins"]),
        z_score_theta=network["z_score_theta"],
        z_score_x=network["z_score_x"],
    )


def load_stage10_posterior(
    configuration: Mapping[str, Any],
    state_path: Path,
    example_theta: Tensor,
    example_context: Tensor,
    *,
    device: str = "cpu",
) -> DirectPosterior:
    """Reconstruct a Stage 10 posterior from a state-only checkpoint."""

    torch_device = torch.device(device)
    estimator = stage10_density_builder(configuration)(
        example_theta.to(torch_device), example_context.to(torch_device)
    )
    estimator.load_state_dict(
        torch.load(state_path, map_location=torch_device, weights_only=True),
        strict=True,
    )
    estimator.to(torch_device)
    estimator.eval()
    return DirectPosterior(
        posterior_estimator=estimator,
        prior=stage10_prior(torch_device),
        device=device,
    )
