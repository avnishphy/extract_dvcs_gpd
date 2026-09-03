"""Saved-artifact-only literature benchmark validation and plotting adapters."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json
from .model_registry import common_function_metrics


STATUSES = {
    "reproduced", "partially_reproduced", "awaiting_compatible_corpus",
    "awaiting_external_numerical_data", "not_scientifically_comparable",
}

COMPARATOR_FIELDS = {
    "id", "citation", "sources", "inferred_object", "generator_truth_model",
    "representation_class", "named_model_family", "model_variant",
    "D_term_assumptions", "observables", "input_level", "kinematic_support",
    "perturbative_evolution_settings", "parameter_dimension", "prior_bounds",
    "uncertainty_generation", "covariance_assumptions",
    "uncertainties_as_model_inputs", "nuisance_treatment", "replica_count",
    "data_split", "evaluation_points_seen_during_training", "optimizer_stopping",
    "reported_metrics", "uncertainty_band_definition", "calibration_coverage_tested",
    "family_or_kinematic_holdout", "source_code", "unresolved_details",
    "comparability_grade",
}
COMPARABILITY_GRADES = {"exact", "high", "partial", "conceptual", "not_comparable"}


def load_comparator_registry(path: Path) -> dict[str, Any]:
    """Validate the publication comparator registry without filling unknowns."""

    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("comparators"), list):
        raise ValueError("comparator registry must have schema_version 1 and comparators")
    identifiers = set()
    for index, comparator in enumerate(value["comparators"]):
        if set(comparator) != COMPARATOR_FIELDS:
            raise ValueError(
                f"comparator {index} key mismatch: "
                f"missing={sorted(COMPARATOR_FIELDS - set(comparator))}, "
                f"unknown={sorted(set(comparator) - COMPARATOR_FIELDS)}"
            )
        if comparator["id"] in identifiers:
            raise ValueError(f"duplicate comparator ID {comparator['id']!r}")
        identifiers.add(comparator["id"])
        if comparator["comparability_grade"] not in COMPARABILITY_GRADES:
            raise ValueError(f"invalid comparability grade for {comparator['id']}")
        if not comparator["sources"] or not all(
            str(source).startswith("https://") for source in comparator["sources"]
        ):
            raise ValueError(f"comparator {comparator['id']} needs HTTPS primary sources")
        if not isinstance(comparator["unresolved_details"], list):
            raise ValueError(f"comparator {comparator['id']} unresolved_details must be a list")
    return value


def load_registry(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported literature registry schema")
    source_ids = {source["id"] for source in value["sources"]}
    if len(source_ids) != len(value["sources"]):
        raise ValueError("duplicate literature source ID")
    for source in value["sources"]:
        if not source["authoritative_url"].startswith("https://arxiv.org/abs/"):
            raise ValueError("literature sources must use authoritative arXiv records")
        for family in source["figure_families"]:
            if family["status"] not in STATUSES:
                raise ValueError("invalid literature figure-family status")
    return value


def assert_conventions_compatible(
    prediction: Mapping[str, Any], benchmark: Mapping[str, Any]
) -> None:
    fields = ("gpd", "flavor_combination", "normalization", "scale_GeV2",
              "scheme", "perturbative_order", "sign_convention")
    mismatches = [field for field in fields if prediction.get(field) != benchmark.get(field)]
    if mismatches:
        raise ValueError(
            "literature overlay convention mismatch: " + ", ".join(mismatches)
        )


def coverage_requirements(
    registry: Mapping[str, Any], corpus_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    has_truth = bool(
        isinstance(corpus_manifest.get("gpd_truth"), Mapping)
        and corpus_manifest["gpd_truth"].get("status") == "complete"
    )
    reports = []
    for benchmark in registry["benchmark_contracts"]:
        missing = []
        if "canonical_gpd_truth" in benchmark["required_artifacts"] and not has_truth:
            missing.append("canonical_gpd_truth")
        reports.append({
            "benchmark_id": benchmark["id"],
            "compatible": not missing and bool(benchmark["compatible_model_families"]),
            "missing_artifacts": missing,
            "kinematic_requirements": deepcopy(benchmark["kinematic_requirements"]),
            "corpus_configuration_modified": False,
        })
    return {"schema_version": 1, "coverage_requirements": reports}


def write_coverage_requirements(
    *, registry_path: Path, corpus_manifest_path: Path, output: Path
) -> dict[str, Any]:
    report = coverage_requirements(
        load_registry(registry_path),
        json.loads(corpus_manifest_path.resolve(strict=True).read_text(encoding="utf-8")),
    )
    atomic_json(output, report)
    return report


def plot_function_closure(
    *, coordinates: np.ndarray, truth: np.ndarray, posterior_samples: np.ndarray,
    output: Path, label: str,
) -> dict[str, Any]:
    """Reproduce a closure diagnostic from local arrays, never paper pixels."""

    import matplotlib.pyplot as plt

    x = np.asarray(coordinates, dtype=np.float64)
    reference = np.asarray(truth, dtype=np.float64)
    samples = np.asarray(posterior_samples, dtype=np.float64)
    metrics = common_function_metrics(reference, samples)
    lower, median, upper = np.quantile(samples, (0.05, 0.5, 0.95), axis=0)
    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    axis.fill_between(x, lower, upper, alpha=0.3, label="posterior 90%")
    axis.plot(x, median, label="posterior median")
    axis.plot(x, reference, "--", label="stored native truth")
    axis.set(xlabel="signed x", ylabel=label)
    axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return metrics
