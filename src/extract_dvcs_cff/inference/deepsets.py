"""Stage 05 DeepSets context encoder and analytic parameter transforms."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Normal


_A_MIN = 0.1
_A_WIDTH = 0.9
_C_MIN = 2.0
_C_WIDTH = 4.0


def physical_to_latent(theta: Tensor) -> Tensor:
    """Map `(a,c,eta)` to standard-normal coordinates with analytic probits."""

    if theta.shape[-1] != 3:
        raise ValueError("theta must have final dimension 3")
    a_fraction = (theta[..., 0] - _A_MIN) / _A_WIDTH
    c_fraction = (theta[..., 1] - _C_MIN) / _C_WIDTH
    if torch.any((a_fraction <= 0) | (a_fraction >= 1)):
        raise ValueError("a must lie strictly inside (0.1, 1)")
    if torch.any((c_fraction <= 0) | (c_fraction >= 1)):
        raise ValueError("c must lie strictly inside (2, 6)")
    return torch.stack(
        (
            torch.special.ndtri(a_fraction),
            torch.special.ndtri(c_fraction),
            theta[..., 2],
        ),
        dim=-1,
    )


def latent_to_physical(latent: Tensor) -> Tensor:
    """Invert `physical_to_latent` exactly up to floating-point precision."""

    if latent.shape[-1] != 3:
        raise ValueError("latent must have final dimension 3")
    return torch.stack(
        (
            _A_MIN + _A_WIDTH * torch.special.ndtr(latent[..., 0]),
            _C_MIN + _C_WIDTH * torch.special.ndtr(latent[..., 1]),
            latent[..., 2],
        ),
        dim=-1,
    )


def transformed_stage05_prior() -> list[torch.distributions.Distribution]:
    """Return independent latent priors inducing Stage 04 physical priors."""

    return [
        Normal(torch.tensor([0.0]), torch.tensor([1.0])),
        Normal(torch.tensor([0.0]), torch.tensor([1.0])),
        Normal(torch.tensor([0.0]), torch.tensor([1.0])),
    ]


class DeepSetsDatasetEncoder(nn.Module):
    """Permutation-invariant encoder for fixed-feature measurement tokens."""

    def __init__(
        self,
        token_count: int,
        feature_count: int,
        global_feature_count: int = 0,
        point_hidden: int = 48,
        embedding_features: int = 48,
        point_layer_norm: bool = False,
        point_layers: int = 2,
        dataset_hidden: int | None = None,
        dataset_layers: int = 2,
    ) -> None:
        super().__init__()
        if (
            token_count <= 0
            or feature_count <= 0
            or global_feature_count < 0
            or point_layers < 1
            or dataset_layers < 1
        ):
            raise ValueError(
                "token_count and feature_count must be positive and "
                "global_feature_count must be non-negative"
            )
        self.token_count = token_count
        self.feature_count = feature_count
        self.global_feature_count = global_feature_count
        point_modules: list[nn.Module] = [
            nn.Linear(feature_count, point_hidden),
            nn.SiLU(),
            nn.LayerNorm(point_hidden) if point_layer_norm else nn.Identity(),
        ]
        for _ in range(point_layers - 1):
            point_modules.extend(
                (nn.Linear(point_hidden, point_hidden), nn.SiLU())
            )
        self.point_encoder = nn.Sequential(*point_modules)

        dataset_width = (
            embedding_features if dataset_hidden is None else dataset_hidden
        )
        dataset_modules: list[nn.Module] = []
        input_width = point_hidden + global_feature_count
        if dataset_layers == 1:
            dataset_modules.append(nn.Linear(input_width, embedding_features))
        else:
            dataset_modules.extend(
                (nn.Linear(input_width, dataset_width), nn.SiLU())
            )
            for _ in range(dataset_layers - 2):
                dataset_modules.extend(
                    (nn.Linear(dataset_width, dataset_width), nn.SiLU())
                )
            dataset_modules.append(
                nn.Linear(dataset_width, embedding_features)
            )
        self.dataset_encoder = nn.Sequential(*dataset_modules)

    def forward(self, context: Tensor) -> Tensor:
        if context.ndim != 2:
            raise ValueError("context must have shape (batch, flattened_tokens)")
        token_width = self.token_count * self.feature_count
        expected = token_width + self.global_feature_count
        if context.shape[1] != expected:
            raise ValueError(
                f"context width must be {expected}, observed {context.shape[1]}"
            )
        tokens = context[:, :token_width].reshape(
            context.shape[0], self.token_count, self.feature_count
        )
        point_embeddings = self.point_encoder(tokens)
        pooled = point_embeddings.mean(dim=1)
        if self.global_feature_count:
            pooled = torch.cat((pooled, context[:, token_width:]), dim=1)
        return self.dataset_encoder(pooled)
