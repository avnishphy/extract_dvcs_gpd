"""Evaluation roles, generator semantics, projection floors, and provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import getpass
import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import atomic_json, canonical_json, content_id


EVALUATION_ROLES = {
    "training", "validation", "development_ood", "benchmark_visible", "sealed_final",
}
COMPARABILITY_GRADES = {"exact", "high", "partial", "conceptual", "not_comparable"}


@dataclass(frozen=True)
class GeneratorMetadata:
    representation_class: str
    named_model_family: str
    implementation_version: str
    parameterization_variant: str
    physics_configuration: str
    D_term_policy: str
    source_backend: str

    def validate(self) -> "GeneratorMetadata":
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"generator metadata {name} must be a nonempty string")
        return self

    def as_dict(self) -> dict[str, str]:
        self.validate()
        return asdict(self)


def classify_holdout(
    training: GeneratorMetadata,
    evaluation: GeneratorMetadata,
    *, parameter_relation: str | None = None,
    numerical_backend_shift: bool = False,
) -> str:
    training.validate()
    evaluation.validate()
    if numerical_backend_shift:
        return "numerical_backend_holdout"
    if training.named_model_family == evaluation.named_model_family:
        if parameter_relation not in {"interpolation", "extrapolation"}:
            raise ValueError("same-family holdout requires an explicit parameter_relation")
        return f"same_family_parameter_{parameter_relation}"
    if training.representation_class == evaluation.representation_class:
        return "same_representation_named_family_holdout"
    return "cross_representation_class_holdout"


def require_foreign_model_metrics(
    *, common_parameter_truth: bool, requested_metrics: Sequence[str]
) -> None:
    forbidden = {"parameter_recovery", "parameter_rmse", "parameter_coverage"}
    if not common_parameter_truth and forbidden.intersection(requested_metrics):
        raise ValueError(
            "foreign model has no internal-DD coordinate truth; report observables, "
            "CFFs, common functionals/functions, and a DD-compatible projection floor"
        )


def best_dd_projection(
    *, objective: Callable[[np.ndarray], float], bounds: np.ndarray,
    restarts: int = 8, seed: int = 73129, iterations: int = 100,
) -> dict[str, Any]:
    """Deterministic bounded multi-start coordinate search; not a global proof."""

    limits = np.asarray(bounds, dtype=np.float64)
    if limits.ndim != 2 or limits.shape[1] != 2 or np.any(limits[:, 0] >= limits[:, 1]):
        raise ValueError("bounds must have shape [parameter,2] with lower < upper")
    if restarts < 1 or iterations < 1:
        raise ValueError("restarts and iterations must be positive")
    rng = np.random.Generator(np.random.PCG64(seed))
    starts = rng.uniform(limits[:, 0], limits[:, 1], size=(restarts, len(limits)))
    records = []
    best: tuple[float, np.ndarray] | None = None
    initial_step = 0.25 * (limits[:, 1] - limits[:, 0])
    for restart, start in enumerate(starts):
        point = start.copy()
        value = float(objective(point))
        evaluations = 1
        step = initial_step.copy()
        for _ in range(iterations):
            improved = False
            for axis in range(len(point)):
                for direction in (-1.0, 1.0):
                    candidate = point.copy()
                    candidate[axis] = np.clip(
                        candidate[axis] + direction * step[axis], limits[axis, 0], limits[axis, 1]
                    )
                    candidate_value = float(objective(candidate))
                    evaluations += 1
                    if np.isfinite(candidate_value) and candidate_value < value:
                        point, value, improved = candidate, candidate_value, True
            if not improved:
                step *= 0.5
            if float(np.max(step)) <= 1e-10:
                break
        boundary = np.isclose(point, limits[:, 0]) | np.isclose(point, limits[:, 1])
        record = {
            "restart": restart, "initialization": start.tolist(),
            "coordinates": point.tolist(), "objective": value,
            "evaluations": evaluations, "converged": bool(np.max(step) <= 1e-6),
            "boundary_hit_indices": np.flatnonzero(boundary).tolist(),
        }
        records.append(record)
        if best is None or value < best[0]:
            best = value, point.copy()
    assert best is not None
    objectives = np.asarray([record["objective"] for record in records])
    payload = {
        "schema_version": 1,
        "optimizer": "deterministic_bounded_coordinate_search_v1",
        "seed": seed, "restart_count": restarts, "iteration_limit": iterations,
        "best_coordinates": best[1].tolist(), "best_objective": best[0],
        "restart_objective_min": float(np.min(objectives)),
        "restart_objective_max": float(np.max(objectives)),
        "restart_objective_std": float(np.std(objectives)),
        "restarts": records, "global_optimum_claimed": False,
    }
    payload["projection_id"] = content_id("dd-projection", payload)
    return payload


def publish_best_dd_projection(
    *, output: Path, optimization_result: Mapping[str, Any],
    foreign_generator: GeneratorMetadata, covariance_policy: str,
    discrepancy: str, parameter_transform_id: str,
) -> dict[str, Any]:
    """Publish a labelled projection floor without implying parameter truth."""

    if optimization_result.get("global_optimum_claimed") is not False:
        raise ValueError("projection artifact must explicitly disclaim a global optimum")
    if covariance_policy not in {
        "training_reference", "external_fixed", "holdout_truth_derived"
    }:
        raise ValueError("projection artifact requires a declared covariance policy")
    artifact = {
        "schema_version": 1,
        "projection_id": optimization_result["projection_id"],
        "interpretation": "best_discovered_internal_DD_compatibility_floor",
        "foreign_generator": foreign_generator.as_dict(),
        "common_parameter_truth": False,
        "DD_parameter_recovery_metric_permitted": False,
        "covariance_policy": covariance_policy,
        "discrepancy": discrepancy,
        "parameter_transform_id": parameter_transform_id,
        "optimization": dict(optimization_result),
    }
    atomic_json(output, artifact)
    return artifact


REQUIRED_PROVENANCE = {
    "git_commit", "git_dirty", "image_digest", "dependency_identity",
    "native_backend_identity", "master_corpus_id", "selection_id",
    "realization_id", "model_view_id", "resolved_configuration_hash",
    "checkpoint_hash", "model_family", "representation_class",
    "normalization_id", "target_transform_id", "uncertainty_mode",
    "covariance_policy", "evaluation_role",
}


def validate_production_provenance(
    provenance: Mapping[str, Any], *, profile: str
) -> list[str]:
    missing = sorted(
        key for key in REQUIRED_PROVENANCE
        if provenance.get(key) in {None, "", "unrecorded", "unknown"}
    )
    if profile in {"production", "publication"} and missing:
        raise RuntimeError("production provenance is incomplete: " + ", ".join(missing))
    return missing


def require_evaluation_access(
    *, role: str, artifact_id: str, unseal_ledger: Path | None = None
) -> None:
    if role not in EVALUATION_ROLES:
        raise ValueError(f"unknown evaluation role {role!r}")
    if role != "sealed_final":
        return
    if unseal_ledger is None or not unseal_ledger.is_file():
        raise PermissionError("sealed_final evaluation requires an explicit unseal record")
    records = json.loads(unseal_ledger.read_text(encoding="utf-8"))
    if not any(record.get("artifact_id") == artifact_id for record in records.get("records", [])):
        raise PermissionError(f"sealed_final artifact {artifact_id!r} has not been unsealed")


def unseal_evaluation(
    *, ledger: Path, artifact_id: str, reason: str, git_commit: str,
    configuration: Mapping[str, Any], artifact_identities: Mapping[str, Any],
) -> dict[str, Any]:
    if not artifact_id or not reason.strip() or not git_commit:
        raise ValueError("unseal requires artifact_id, commit, and a nonempty reason")
    existing = (
        json.loads(ledger.read_text(encoding="utf-8"))
        if ledger.is_file() else {"schema_version": 1, "records": []}
    )
    if any(record.get("artifact_id") == artifact_id for record in existing["records"]):
        raise FileExistsError(f"artifact version {artifact_id!r} was already unsealed")
    record = {
        "artifact_id": artifact_id,
        "unsealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "user": getpass.getuser(), "hostname": platform.node(),
        "configuration": dict(configuration),
        "configuration_sha256": hashlib.sha256(canonical_json(configuration).encode()).hexdigest(),
        "artifact_identities": dict(artifact_identities), "reason": reason,
        "sealed_after_operation": False,
    }
    existing["records"].append(record)
    atomic_json(ledger, existing)
    return record


def audit_native_failures(corpus: Path, *, bin_count: int = 5) -> dict[str, Any]:
    """Measure accepted/rejected proposal distortion from recorded corpus evidence."""

    if bin_count < 2:
        raise ValueError("bin_count must be at least two")
    manifest = json.loads((corpus / "corpus.json").read_text(encoding="utf-8"))
    accepted_blocks = []
    for record in manifest.get("core_shards", []):
        with np.load(corpus / record["path"], allow_pickle=False) as arrays:
            accepted_blocks.append(np.asarray(arrays["native_parameters"], dtype=np.float64))
    if not accepted_blocks:
        raise RuntimeError("native-failure audit requires generated core shards")
    accepted = np.concatenate(accepted_blocks)
    rejected_rows = []
    shard_rates = []
    failure_types: dict[str, int] = {}
    for path in sorted((corpus / "rejections").glob("shard-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        rejected = record.get("rejected", [])
        shard_rates.append({
            "shard_index": int(record["shard_index"]),
            "proposal_count": int(record["attempted"]),
            "accepted_count": int(record["accepted"]),
            "rejected_count": len(rejected),
            "acceptance_rate": float(record["accepted"] / record["attempted"]),
        })
        for item in rejected:
            rejected_rows.append(np.asarray(item["parameters"], dtype=np.float64))
            message = str(item.get("native_error", "unclassified")).strip().splitlines()[0]
            failure_type = message.split(":", 1)[0][:160] or "unclassified"
            failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
    rejected_array = (
        np.stack(rejected_rows) if rejected_rows else np.empty((0, accepted.shape[1]))
    )
    combined = np.concatenate((accepted, rejected_array))
    dimensions = []
    localized = False
    for index in range(combined.shape[1]):
        edges = np.unique(np.quantile(combined[:, index], np.linspace(0, 1, bin_count + 1)))
        if len(edges) < 2:
            continue
        accepted_hist, _ = np.histogram(accepted[:, index], bins=edges)
        rejected_hist, _ = np.histogram(rejected_array[:, index], bins=edges)
        proposals = accepted_hist + rejected_hist
        rates = np.divide(
            rejected_hist, proposals, out=np.zeros_like(rejected_hist, dtype=float), where=proposals > 0
        )
        spread = float(rates.max() - rates.min()) if len(rates) else 0.0
        localized = localized or spread > 0.05
        dimensions.append({
            "parameter_index": index, "bin_edges": edges.tolist(),
            "accepted_counts": accepted_hist.tolist(),
            "rejected_counts": rejected_hist.tolist(),
            "rejection_rates": rates.tolist(), "rejection_rate_spread": spread,
        })
    proposal_count = int(len(accepted) + len(rejected_array))
    rejection_rate = float(len(rejected_array) / proposal_count)
    status = (
        "negligible" if rejection_rate < 1e-3
        else "localized" if localized
        else "non_negligible_not_localized_by_marginal_bins"
    )
    return {
        "schema_version": 1, "corpus_manifest_sha256": manifest.get("manifest_sha256"),
        "proposal_count": proposal_count, "accepted_count": len(accepted),
        "rejected_count": len(rejected_array), "acceptance_rate": 1.0 - rejection_rate,
        "failure_types": failure_types, "acceptance_by_shard": shard_rates,
        "parameter_region_diagnostics": dimensions, "assessment": status,
        "accepted_distribution_is_original_prior": len(rejected_array) == 0,
        "inference_semantics_if_used": "conditional_on_simulator_validity",
    }
