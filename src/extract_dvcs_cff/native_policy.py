"""Fail-closed boundary for launching the authoritative PARTONS backend."""

from __future__ import annotations

import os
from typing import Final


EXPLICIT_NATIVE_STAGES: Final[frozenset[str]] = frozenset({
    "doctor", "corpus_create_capabilities", "corpus_generate",
    "corpus_extend", "exact_reevaluate", "native_holdout",
    "real_data_exact_prediction",
})

SAVED_ARTIFACT_ONLY_STAGES: Final[frozenset[str]] = frozenset({
    "selection", "realization", "model_view", "decoder_train", "optimize",
    "train", "evaluate_saved", "compare", "plot", "literature_plot",
    "presentation",
})


def assert_native_launch_allowed(stage: str) -> None:
    """Authorize only an explicit native stage and honor the test sentinel."""

    if stage not in EXPLICIT_NATIVE_STAGES:
        raise RuntimeError(
            f"PARTONS launch is forbidden during {stage!r}; use the explicit "
            "exact-reevaluate stage when authoritative native physics is required"
        )
    if os.environ.get("DVCS_DISABLE_NATIVE_EXECUTION") == "1":
        raise RuntimeError(
            f"native execution disabled by DVCS_DISABLE_NATIVE_EXECUTION for {stage}"
        )


def assert_saved_artifact_stage(stage: str) -> None:
    if stage not in SAVED_ARTIFACT_ONLY_STAGES:
        raise ValueError(f"unknown saved-artifact stage {stage!r}")
