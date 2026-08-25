"""Neural and conventional posterior-inference components."""

from .deepsets import (
    DeepSetsDatasetEncoder,
    latent_to_physical,
    physical_to_latent,
    transformed_stage05_prior,
)
from .context import (
    GLOBAL_FEATURE_NAMES,
    POINT_FEATURE_NAMES,
    build_stage05_context,
    validate_stage05_constant_dimensions,
)
from .npe import (
    load_stage05_posterior,
    processed_stage05_prior,
    stage05_density_builder,
)
from .ood import ContextEnvelope
from .stage10 import (
    DD_SHAPE_BOUNDS,
    DD_SHAPE_FIELDS,
    GPD_TYPES,
    PARTON_CHANNELS,
    STAGE10_GLOBAL_FEATURE_NAMES,
    STAGE10_NUISANCE_PARAMETER_NAMES,
    STAGE10_PARAMETER_NAMES,
    STAGE10_PHYSICS_PARAMETER_COUNT,
    STAGE10_PHYSICS_PARAMETER_NAMES,
    STAGE10_POINT_FEATURE_NAMES,
    Stage10ContextBuilder,
    build_stage10_context,
    load_stage10_posterior,
    stage10_density_builder,
    stage10_latent_to_physical,
    stage10_physical_to_latent,
    stage10_prior,
    prepare_stage10_context,
)

__all__ = [
    "DeepSetsDatasetEncoder",
    "GLOBAL_FEATURE_NAMES",
    "POINT_FEATURE_NAMES",
    "build_stage05_context",
    "ContextEnvelope",
    "latent_to_physical",
    "load_stage05_posterior",
    "physical_to_latent",
    "processed_stage05_prior",
    "stage05_density_builder",
    "validate_stage05_constant_dimensions",
    "transformed_stage05_prior",
    "STAGE10_GLOBAL_FEATURE_NAMES",
    "GPD_TYPES",
    "PARTON_CHANNELS",
    "DD_SHAPE_FIELDS",
    "DD_SHAPE_BOUNDS",
    "STAGE10_NUISANCE_PARAMETER_NAMES",
    "STAGE10_PARAMETER_NAMES",
    "STAGE10_PHYSICS_PARAMETER_COUNT",
    "STAGE10_PHYSICS_PARAMETER_NAMES",
    "STAGE10_POINT_FEATURE_NAMES",
    "Stage10ContextBuilder",
    "build_stage10_context",
    "load_stage10_posterior",
    "stage10_density_builder",
    "stage10_latent_to_physical",
    "stage10_physical_to_latent",
    "stage10_prior",
    "prepare_stage10_context",
]
