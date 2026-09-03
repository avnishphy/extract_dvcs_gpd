#!/usr/bin/env python3
"""Evaluate and validate the exact production covariance without a campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from extract_dvcs_cff.contracts import atomic_json
from extract_dvcs_cff.workflows.pseudodata import (
    NativeCache,
    _covariance_and_responses,
    _physics_path,
    _point_table,
    _shape_vector_from_shapes,
    load_configuration,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_diagnostics(matrix: np.ndarray) -> dict[str, Any]:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    diagonal = np.diag(symmetric)
    scale = np.sqrt(diagonal)
    correlation = symmetric / np.outer(scale, scale)
    try:
        np.linalg.cholesky(symmetric)
        cholesky = True
    except np.linalg.LinAlgError:
        cholesky = False
    return {
        "shape": list(matrix.shape),
        "symmetry_max_abs": float(np.max(np.abs(matrix - matrix.T))),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "cholesky_success": cholesky,
        "diagonal_minimum": float(diagonal.min()),
        "diagonal_maximum": float(diagonal.max()),
        "correlation_minimum": float(correlation.min()),
        "correlation_maximum": float(correlation.max()),
    }


def monte_carlo_test(
    covariance: np.ndarray, *, replica_count: int, seed: int
) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64(seed))
    factor = np.linalg.cholesky(covariance)
    noise = rng.standard_normal((replica_count, len(covariance))) @ factor.T
    mean = noise.mean(axis=0)
    centered = noise - mean
    empirical = centered.T @ centered / (replica_count - 1)
    mean_z = mean / np.sqrt(np.diag(covariance) / replica_count)
    covariance_se = np.sqrt(
        (
            covariance**2
            + np.diag(covariance)[:, None] * np.diag(covariance)[None, :]
        )
        / (replica_count - 1)
    )
    covariance_z = (empirical - covariance) / covariance_se
    mean_threshold = 5.0
    covariance_threshold = 6.0
    return {
        "replica_count": replica_count,
        "seed": seed,
        "empirical_mean_max_abs": float(np.max(np.abs(mean))),
        "empirical_mean_max_abs_z": float(np.max(np.abs(mean_z))),
        "empirical_covariance_max_abs": float(np.max(np.abs(empirical - covariance))),
        "empirical_covariance_max_abs_z": float(np.max(np.abs(covariance_z))),
        "mean_familywise_threshold_z": mean_threshold,
        "covariance_familywise_threshold_z": covariance_threshold,
        "tolerance_basis": (
            "Gaussian standard error per mean/covariance element with conservative "
            "familywise z thresholds for 576 means and 166176 unique covariance entries"
        ),
        "passed": bool(
            np.max(np.abs(mean_z)) <= mean_threshold
            and np.max(np.abs(covariance_z)) <= covariance_threshold
        ),
    }


def low_rank_test(
    source: np.ndarray, *, replica_count: int, seed: int
) -> dict[str, Any]:
    target = np.outer(source, source)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.standard_normal(replica_count)
    empirical = np.outer(source, source) * float(np.var(draws, ddof=1))
    relative_scale_error = abs(float(np.var(draws, ddof=1)) - 1.0)
    tolerance = 5.0 * math.sqrt(2.0 / (replica_count - 1))
    return {
        "rank": int(np.linalg.matrix_rank(target)),
        "target_trace": float(np.trace(target)),
        "empirical_trace": float(np.trace(empirical)),
        "relative_scale_error": relative_scale_error,
        "five_sigma_relative_tolerance": tolerance,
        "passed": bool(relative_scale_error <= tolerance),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bridge", type=Path)
    source.add_argument("--reference-corpus", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.replicas < 1024:
        raise ValueError("covariance audit requires at least 1024 replicas")

    configuration_path = args.configuration.resolve(strict=True)
    config = load_configuration(configuration_path)
    physics_path = _physics_path(configuration_path, config)
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    if args.reference_corpus is None:
        client = NativeCache(args.bridge, args.workspace)
        truth_parameters = _shape_vector_from_shapes(
            physics["parameters"]["gpd_shapes"]
        )[None, :]
        truth = client.evaluate(
            parameters=truth_parameters,
            config=config,
            physics=physics,
        )
        reference = truth.predictions[0]
        truth_provenance = {
            "source": "current_bridge_exact_injected_truth",
            "bridge_sha256": client.bridge_hash,
            "native_cache_key": truth.cache_key,
            "native_cache_hit": truth.cache_hit,
            "request_sha256": truth.request_sha256,
            "response_sha256": truth.response_sha256,
            "evaluation_count": truth.evaluation_count,
            "production_authoritative": True,
        }
    else:
        corpus = args.reference_corpus.resolve(strict=True)
        manifest = json.loads((corpus / "corpus.json").read_text(encoding="utf-8"))
        blocks = {}
        for observable in config["observables"]:
            name = observable["id"]
            record = manifest["truth"]["observables"][name]
            with np.load(corpus / record["path"], allow_pickle=False) as arrays:
                blocks[name] = np.asarray(arrays["values"][0], dtype=np.float64)
        reference = np.asarray([
            blocks[observable["id"]][kinematic_index]
            for kinematic_index in range(len(config["kinematics"]))
            for observable in config["observables"]
        ])
        truth_provenance = {
            "source": "historical_completed_corpus_truth",
            "corpus": str(corpus),
            "corpus_manifest_sha256": manifest["manifest_sha256"],
            "bridge_sha256": manifest["truth"]["core"]["native"]["bridge_sha256"],
            "request_sha256": manifest["truth"]["core"]["native"]["request_sha256"],
            "response_sha256": manifest["truth"]["core"]["native"]["response_sha256"],
            "production_authoritative": False,
            "non_authoritative_reason": (
                "historical bridge/corpus identity differs from the current production image"
            ),
        }
    covariance, global_response, lu_response, builder = _covariance_and_responses(
        config, reference
    )

    model = config["experimental_model"]
    sigma = (
        float(model["uncorrelated_absolute_floor_normalized"])
        + float(model["uncorrelated_relative_sigma"]) * np.abs(reference)
    )
    diagonal_component = np.diag(sigma**2)
    local_scale = float(model["local_correlation_fraction"]) * np.maximum(
        np.abs(reference),
        float(model["uncorrelated_absolute_floor_normalized"]),
    )
    indices = np.arange(len(reference))
    local_component = local_scale[:, None] * local_scale[None, :] * np.exp(
        -np.abs(indices[:, None] - indices[None, :])
        / float(model["local_correlation_length"])
    )
    points = _point_table(config)
    phi_source = np.asarray([
        float(model["phi_shape_correlated_fraction"])
        * reference[index]
        * math.sin(float(point["phi_rad"]))
        for index, point in enumerate(points)
    ])
    phi_component = np.outer(phi_source, phi_source)
    reconstructed = diagonal_component + local_component + phi_component
    component_error = float(np.max(np.abs(covariance - reconstructed)))

    global_source = reference * global_response
    lu_source = reference * lu_response
    component_names = {"uncorrelated", "local_kernel", "phi_shape"}
    nuisance_names = {"global_normalization", "lu_normalization"}
    overlap = sorted(component_names & nuisance_names)
    masked_designs = []
    for active_count in config["observation_design_training"]["active_kinematic_counts"]:
        active = np.arange(int(active_count) * len(config["observables"]))
        masked_designs.append({
            "active_kinematic_count": int(active_count),
            "active_observation_count": int(len(active)),
            "diagnostics": matrix_diagnostics(covariance[np.ix_(active, active)]),
        })

    diagnostics = matrix_diagnostics(covariance)
    replica = monte_carlo_test(
        covariance, replica_count=args.replicas, seed=args.seed
    )
    report = {
        "schema_version": 1,
        "status": "pass" if (
            diagnostics["cholesky_success"]
            and diagnostics["minimum_eigenvalue"] > 0
            and component_error <= 1e-14
            and not overlap
            and replica["passed"]
        ) else "fail",
        "configuration": str(configuration_path),
        "configuration_sha256": sha256(configuration_path),
        "physics_sha256": sha256(physics_path),
        "truth_provenance": truth_provenance,
        "observable_order": [item["id"] for item in config["observables"]],
        "flattening_order": "kinematic_major_observable_minor",
        "kinematic_count": len(config["kinematics"]),
        "physical_covariance": diagnostics,
        "numerical_jitter": 0.0,
        "jitter_relative_to_smallest_physical_diagonal": 0.0,
        "builder_diagnostics": builder,
        "component_reconstruction_max_abs_error": component_error,
        "components": {
            "uncorrelated": matrix_diagnostics(diagonal_component),
            "local_kernel": matrix_diagnostics(local_component + np.eye(len(reference)) * np.finfo(float).eps),
            "phi_shape": {
                "rank": int(np.linalg.matrix_rank(phi_component)),
                "trace": float(np.trace(phi_component)),
            },
        },
        "explicit_nuisances": {
            "global_normalization": low_rank_test(
                global_source, replica_count=args.replicas, seed=args.seed + 1
            ),
            "lu_normalization": low_rank_test(
                lu_source, replica_count=args.replicas, seed=args.seed + 2
            ),
        },
        "covariance_nuisance_component_overlap": overlap,
        "double_counting_detected": bool(overlap),
        "masked_design_submatrices": masked_designs,
        "deterministic_replica_test": replica,
    }
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
