"""Historical Stage-05 DeepSets + conditional-MAF construction."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets import posterior_nn
from sbi.utils import MultipleIndependent

from .deepsets import DeepSetsDatasetEncoder, transformed_stage05_prior


def processed_stage05_prior() -> torch.distributions.Distribution:
    """Return the sbi-compatible joint latent prior."""

    return MultipleIndependent(transformed_stage05_prior())


def stage05_density_builder(
    configuration: Mapping[str, Any],
):
    """Build the configured MAF factory with the declared DeepSets encoder."""

    context = configuration["context_contract"]
    network = configuration["network"]
    if network["flow"] != "zuko_maf" or "num_bins" in network:
        raise ValueError("Stage-05 migration accepts only zuko_maf without num_bins")
    encoder = DeepSetsDatasetEncoder(
        token_count=int(context["point_count"]),
        feature_count=len(context["point_features"]),
        global_feature_count=len(context["global_features"]),
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
        z_score_theta=network["z_score_theta"],
        z_score_x=network["z_score_x"],
    )


def load_stage05_posterior(
    configuration: Mapping[str, Any],
    state_path: Path,
    example_theta: torch.Tensor,
    example_context: torch.Tensor,
) -> DirectPosterior:
    """Reconstruct a trained estimator from its state-only artifact."""

    builder = stage05_density_builder(configuration)
    estimator = builder(example_theta, example_context)
    state = torch.load(
        state_path, map_location="cpu", weights_only=True
    )
    estimator.load_state_dict(state, strict=True)
    estimator.eval()
    return DirectPosterior(
        posterior_estimator=estimator,
        prior=processed_stage05_prior(),
        device="cpu",
    )
