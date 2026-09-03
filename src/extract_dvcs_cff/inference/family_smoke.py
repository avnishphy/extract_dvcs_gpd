"""Tiny synthetic end-to-end checks for the shared DeepSets+MAF mechanics."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..model_registry import require_model_stage


def run_family_smoke(model_family: str, *, seed: int = 73129) -> dict[str, Any]:
    """Train and sample a tiny family-shaped NPE without native physics calls."""

    family = require_model_stage(model_family, "smoke")
    import torch
    from sbi.utils import MultipleIndependent
    from torch.distributions import Normal

    from .stage10 import (
        GroupedValidationNPE,
        STAGE10_GLOBAL_FEATURE_NAMES,
        STAGE10_POINT_FEATURE_NAMES,
        stage10_density_builder,
    )

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    target_dimension = 3 if model_family == "dd_deepsets_maf" else 2
    group_count, replicates = 6, 2
    group_targets = torch.randn(group_count, target_dimension)
    parameter_indices = torch.arange(group_count).repeat_interleave(replicates)
    targets = group_targets[parameter_indices]
    token_width = len(STAGE10_POINT_FEATURE_NAMES)
    context_width = token_width + len(STAGE10_GLOBAL_FEATURE_NAMES)
    contexts = torch.randn(group_count * replicates, context_width)
    contexts[:, token_width - 1] = 1.0
    contexts[:, token_width:] = 1.0
    train_indices = torch.arange(0, 8)
    validation_indices = torch.arange(8, 10)
    test_indices = torch.arange(10, 12)
    configuration = {
        "kinematics": [{}], "observables": [{}],
        "context_contract": {
            "point_features": list(STAGE10_POINT_FEATURE_NAMES),
            "global_features": list(STAGE10_GLOBAL_FEATURE_NAMES),
        },
        "network": {
            "flow": "zuko_maf", "point_hidden": 8, "point_layers": 1,
            "dataset_hidden": 8, "dataset_layers": 1,
            "embedding_features": 8, "point_layer_norm": False,
            "flow_hidden_features": 8, "num_transforms": 2,
            "z_score_theta": "none", "z_score_x": "none",
        },
    }
    prior = MultipleIndependent([
        Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        for _ in range(target_dimension)
    ], device="cpu")
    inference = GroupedValidationNPE(
        prior=prior, density_estimator=stage10_density_builder(configuration),
        device="cpu", show_progress_bars=False,
        grouped_train_indices=train_indices,
        grouped_validation_indices=validation_indices,
    )
    estimator = inference.append_simulations(
        targets[:10], contexts[:10], data_device="cpu"
    ).train(
        training_batch_size=4, learning_rate=1e-3,
        validation_fraction=0.2, stop_after_epochs=2, max_num_epochs=2,
        show_train_summary=False,
    )
    posterior = inference.build_posterior(estimator)
    torch.manual_seed(seed + 1)
    samples = posterior.sample((4,), x=contexts[test_indices[0]], show_progress_bars=False)
    finite = bool(torch.isfinite(samples).all())
    if samples.shape != (4, target_dimension) or not finite:
        raise RuntimeError("family smoke posterior produced invalid samples")
    return {
        "schema_version": 1, "status": "pass", "model_family": family.identifier,
        "representation_class": family.gpd_inference_representation,
        "observation_encoder": family.observation_encoder,
        "density_estimator": family.density_estimator,
        "target_dimension": target_dimension, "native_physics_called": False,
        "group_count": group_count, "replicates_per_group": replicates,
        "training_group_count": 4, "validation_group_count": 1,
        "test_group_count": 1, "posterior_sample_count": 4,
        "posterior_samples_finite": finite, "seed": seed,
        "maturity": "synthetic_mechanics_smoke_only",
    }
