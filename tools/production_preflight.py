#!/usr/bin/env python3
"""Fail-closed preflight for the frozen JLab production campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any


NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, text=True, capture_output=True, check=False, timeout=timeout
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(
        self,
        check_id: str,
        passed: bool,
        detail: Any,
        *,
        blocking: bool = True,
    ) -> None:
        self.checks.append(
            {
                "id": check_id,
                "passed": bool(passed),
                "blocking": bool(blocking),
                "detail": detail,
            }
        )

    @property
    def blockers(self) -> list[dict[str, Any]]:
        return [item for item in self.checks if item["blocking"] and not item["passed"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("provenance/production-submission-manifest.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-native-runtime", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    manifest_path = (root / args.manifest).resolve() if not args.manifest.is_absolute() else args.manifest
    manifest = load_json(manifest_path)
    audit = Audit()

    audit.add("manifest.schema", manifest.get("schema_version") == 1, {
        "observed": manifest.get("schema_version"), "required": 1
    })
    project = manifest.get("project", {})
    corpus = manifest.get("corpus_plan", {})
    analysis = manifest.get("analysis_plan", {})
    runtime = manifest.get("runtime", {})
    paths = manifest.get("paths", {})

    for label, value in (
        ("project", project.get("name")),
        ("profile", project.get("profile")),
        ("selection", project.get("selection")),
        ("corpus", corpus.get("name")),
        ("corpus_workflow", corpus.get("workflow")),
        ("analysis_workflow", analysis.get("workflow")),
    ):
        audit.add(f"identity.{label}", isinstance(value, str) and bool(NAME.fullmatch(value)), value)

    resolved: dict[str, Path] = {}
    for key, hash_key in (
        ("experiment_path", "experiment_sha256"),
        ("workflow_path", "workflow_sha256"),
        ("physics_path", "physics_sha256"),
        ("selection_path", "selection_sha256"),
    ):
        path = root / str(project.get(key, ""))
        resolved[key] = path
        observed = sha256_file(path) if path.is_file() else None
        audit.add(f"hash.{key}", observed == project.get(hash_key), {
            "path": str(path), "declared": project.get(hash_key), "observed": observed
        })

    if resolved["workflow_path"].is_file():
        workflow = load_json(resolved["workflow_path"])
    else:
        workflow = {}
    if resolved["physics_path"].is_file():
        physics = load_json(resolved["physics_path"])
    else:
        physics = {}
    audit.add("schema.workflow", workflow.get("schema_version") == 8, workflow.get("schema_version"))
    audit.add("schema.physics", physics.get("schema_version") == 4, physics.get("schema_version"))
    audit.add("schema.retired_maf_field_absent", "num_bins" not in canonical(workflow).decode("utf-8"),
              "num_bins must not appear in the generated workflow")

    kinematics = workflow.get("kinematics", [])
    observed_kinematic_hash = hashlib.sha256(canonical(kinematics)).hexdigest()
    audit.add("kinematics.hash", observed_kinematic_hash == project.get("kinematic_table_canonical_sha256"), {
        "count": len(kinematics), "declared_count": project.get("kinematic_count"),
        "declared": project.get("kinematic_table_canonical_sha256"),
        "observed": observed_kinematic_hash,
    })
    audit.add("kinematics.count", len(kinematics) == project.get("kinematic_count"), len(kinematics))

    domain_failures: list[dict[str, Any]] = []
    domain = physics.get("observable_domain", {})
    mapping = {"x_b": "x_b", "t": "t_GeV2", "Q2": "Q2_GeV2", "beam_energy": "beam_energy_GeV", "phi": "phi_rad"}
    for domain_name, point_name in mapping.items():
        values = [float(point[point_name]) for point in kinematics if point_name in point]
        limits = domain.get(domain_name, {})
        if not values or "minimum" not in limits or "maximum" not in limits:
            domain_failures.append({"field": domain_name, "reason": "missing values or declared limits"})
            continue
        if min(values) < float(limits["minimum"]) or max(values) > float(limits["maximum"]):
            domain_failures.append({
                "field": domain_name,
                "actual": [min(values), max(values)],
                "declared": [limits["minimum"], limits["maximum"]],
            })
    audit.add("physics.kinematics_inside_declared_domain", not domain_failures, domain_failures)

    prior = manifest.get("prior", {})
    prior_payload = {"distribution": prior.get("distribution"), **prior.get("bounds", {})}
    prior_hash = hashlib.sha256(canonical(prior_payload)).hexdigest()
    audit.add("prior.hash", prior_hash == prior.get("canonical_sha256"), {
        "declared": prior.get("canonical_sha256"), "observed": prior_hash
    })

    registry_meta = manifest.get("physics_assumptions", {})
    registry_path = root / str(registry_meta.get("registry_path", ""))
    registry = load_json(registry_path) if registry_path.is_file() else {}
    identity_payload = {key: value for key, value in registry.items() if key != "registry_sha256"}
    registry_hash = hashlib.sha256(canonical(identity_payload)).hexdigest()
    audit.add("assumptions.hash", registry_hash == registry_meta.get("registry_sha256") == registry.get("registry_sha256"), {
        "manifest": registry_meta.get("registry_sha256"),
        "registry": registry.get("registry_sha256"), "observed": registry_hash,
    })
    approved = registry_meta.get("approval_status") == "approved" and registry.get("registry_status") == "approved"
    audit.add("assumptions.researcher_approved", approved, {
        "manifest": registry_meta.get("approval_status"), "registry": registry.get("registry_status")
    })

    gpd_truth_path = root / str(corpus.get("gpd_truth_request_path", ""))
    gpd_truth_hash = sha256_file(gpd_truth_path) if gpd_truth_path.is_file() else None
    audit.add("corpus.function_grid_request", gpd_truth_hash is not None and gpd_truth_hash == corpus.get("gpd_truth_request_sha256"), {
        "path": str(gpd_truth_path), "declared": corpus.get("gpd_truth_request_sha256"),
        "observed": gpd_truth_hash,
    })
    expected_shards = (
        int(corpus.get("accepted_native_parameter_groups", 0))
        + int(corpus.get("shard_size", 1)) - 1
    ) // int(corpus.get("shard_size", 1))
    expected_workers = (
        expected_shards + int(corpus.get("shards_per_worker", 1)) - 1
    ) // int(corpus.get("shards_per_worker", 1))
    audit.add("corpus.shard_count", expected_shards == corpus.get("atomic_shard_count"), expected_shards)
    audit.add("corpus.worker_count", expected_workers == corpus.get("worker_job_count"), expected_workers)
    audit.add("corpus.accepted_count", corpus.get("accepted_native_parameter_groups") == 16384,
              corpus.get("accepted_native_parameter_groups"))
    audit.add("corpus.reuse_policy", corpus.get("historical_reuse_compatible") is False, {
        "candidate": corpus.get("historical_reuse_candidate"),
        "compatible": corpus.get("historical_reuse_compatible"),
        "reason": corpus.get("historical_reuse_reason"),
    })

    profile = workflow.get("profiles", {}).get(project.get("profile"), {})
    expected_realizations = int(profile.get("native_parameter_count", 0)) * int(profile.get("noise_replicates_per_parameter", 0))
    audit.add("analysis.realization_count", expected_realizations == analysis.get("realization_count"), expected_realizations)
    audit.add("analysis.selection_count_sum", sum(project.get("selection_group_counts", {}).values()) == profile.get("native_parameter_count"),
              project.get("selection_group_counts"))
    audit.add("analysis.job_count", len(analysis.get("stages", [])) == analysis.get("job_count") == 12,
              {"declared": analysis.get("job_count"), "stage_count": len(analysis.get("stages", []))})

    sif = root / str(runtime.get("sif_path", ""))
    sif_hash = sha256_file(sif) if sif.is_file() else None
    audit.add("runtime.sif_digest", sif_hash == runtime.get("sif_sha256"), {
        "path": str(sif), "declared": runtime.get("sif_sha256"), "observed": sif_hash
    })
    lock_path = root / str(runtime.get("image_lock_path", ""))
    lock = load_json(lock_path) if lock_path.is_file() else {}
    locked = lock.get("images", {}).get("jlab_ifarm", {}).get("digest")
    registry = lock.get("registry")
    image_published = lock.get("published") is True and isinstance(locked, str) and locked.startswith("sha256:") and "OWNER" not in str(registry)
    audit.add("runtime.published_image_digest", image_published, {
        "published": lock.get("published"), "registry": registry, "digest": locked,
        "local_sif_sha256": sif_hash,
    })

    capabilities: dict[str, Any] | None = None
    if args.skip_native_runtime:
        audit.add("runtime.native_capabilities", True, "skipped by explicit test-only option", blocking=False)
    elif sif.is_file() and shutil.which("apptainer"):
        bridge = str(runtime.get("bridge_path_in_image"))
        binary = run(["apptainer", "exec", str(sif), "sha256sum", bridge])
        binary_hash = binary.stdout.split()[0] if binary.returncode == 0 and binary.stdout.split() else None
        audit.add("runtime.bridge_binary_digest", binary_hash == runtime.get("bridge_binary_sha256"), {
            "declared": runtime.get("bridge_binary_sha256"), "observed": binary_hash,
            "stderr": binary.stderr.strip(),
        })
        cap = run(["apptainer", "exec", str(sif), bridge, "--capabilities"])
        try:
            capabilities = json.loads(cap.stdout) if cap.returncode == 0 else None
        except json.JSONDecodeError:
            capabilities = None
        cap_text = canonical(capabilities).decode("utf-8") if capabilities is not None else ""
        cap_ok = (
            cap.returncode == 0
            and capabilities is not None
            and runtime.get("partons_git_revision") in cap_text
            and runtime.get("apfelxx_version") in cap_text
            and workflow.get("representation") in cap_text
        )
        audit.add("runtime.native_capabilities", cap_ok, {
            "returncode": cap.returncode, "partons_revision_required": runtime.get("partons_git_revision"),
            "apfelxx_required": runtime.get("apfelxx_version"), "route_required": workflow.get("representation"),
            "stderr": cap.stderr.strip(), "capabilities": capabilities,
        })
    else:
        audit.add("runtime.native_capabilities", False, "SIF or apptainer unavailable")

    resources_path = root / "jobs/jlab_ifarm/resources.env"
    resources_env = dotenv(resources_path) if resources_path.is_file() else {}
    required_env = ("JLAB_ACCOUNT", "DVCS_REPOSITORY", "DVCS_WORKSPACE", "DVCS_RESULTS", "DVCS_CACHE", "DVCS_DATABASE")
    missing_env = [key for key in required_env if not resources_env.get(key) or resources_env[key].startswith("REQUIRED_")]
    audit.add("swif.required_environment", not missing_env, {"missing": missing_env})
    mismatches = {
        key: {"configured": resources_env.get(key), "required": required}
        for key, required in (
            ("DVCS_PROJECT", project.get("name")), ("DVCS_CORPUS", corpus.get("name")),
            ("DVCS_SELECTION", project.get("selection")),
        ) if resources_env.get(key) != required
    }
    audit.add("swif.campaign_environment", not mismatches, mismatches)
    swif = shutil.which("swif2")
    swif_version = run([swif, "--version"]).stdout.strip() if swif else None
    audit.add("swif.command", swif is not None, {"path": swif, "version": swif_version})

    storage: dict[str, Any] = {}
    minimum_free = float(manifest.get("storage_estimate", {}).get("minimum_free_GiB", 0))
    for label in ("workspace_root", "cache_root", "result_root", "database_root", "log_root"):
        value = Path(str(paths.get(label, "")))
        path = value if value.is_absolute() else root / value
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            free_gib = shutil.disk_usage(probe).free / (1024 ** 3)
            writable = os.access(probe, os.W_OK)
        except OSError as error:
            free_gib, writable = 0.0, False
            storage[label] = {"path": str(path), "error": str(error)}
        else:
            storage[label] = {"path": str(path), "probe": str(probe), "writable": writable, "free_GiB": free_gib}
        audit.add(f"storage.{label}", writable and free_gib >= minimum_free, storage[label])
    quota = run(["quota", "-s"]) if shutil.which("quota") else None
    audit.add("storage.quota_query", quota is not None and quota.returncode == 0, {
        "returncode": quota.returncode if quota else None,
        "stdout": quota.stdout.strip() if quota else "",
        "stderr": quota.stderr.strip() if quota else "quota command unavailable",
    }, blocking=False)

    resource_plan = manifest.get("resources", {})
    required_resource_stages = {"corpus_worker", "corpus_merge", "selection", "materialize", "train", "evaluate", "exact_reevaluate", "compare", "holdout_worker", "holdout_merge", "plot"}
    missing_resources = sorted(required_resource_stages - set(resource_plan))
    invalid_resources = sorted(name for name, spec in resource_plan.items() if not all(key in spec for key in ("jobs", "cores_each", "ram_each", "disk_each", "walltime_each")))
    audit.add("resources.complete", not missing_resources and not invalid_resources, {
        "missing": missing_resources, "invalid": invalid_resources, "resources": resource_plan
    })

    covariance_path = root / str(manifest.get("statistical_model", {}).get("covariance_report", ""))
    covariance = load_json(covariance_path) if covariance_path.is_file() else {}
    covariance_hash = sha256_file(covariance_path) if covariance_path.is_file() else None
    cov_meta = manifest.get("statistical_model", {})
    audit.add("covariance.report_hash", covariance_hash == cov_meta.get("covariance_report_sha256"), {
        "declared": cov_meta.get("covariance_report_sha256"), "observed": covariance_hash
    })
    historical_truth = covariance.get("truth_provenance", {})
    current_truth_identity = (
        covariance.get("bridge_sha256") == runtime.get("bridge_binary_sha256")
        and covariance.get("truth_evaluation_count") == project.get("kinematic_count")
        and isinstance(covariance.get("truth_request_sha256"), str)
        and isinstance(covariance.get("truth_response_sha256"), str)
    )
    cov_ok = covariance.get("status") == "pass" and (
        historical_truth.get("production_authoritative") is True or current_truth_identity
    )
    audit.add("covariance.production_authoritative", cov_ok, {
        "status": covariance.get("status"),
        "truth_provenance": covariance.get("truth_provenance"),
        "current_truth_identity": current_truth_identity,
        "bridge_sha256": covariance.get("bridge_sha256"),
        "truth_evaluation_count": covariance.get("truth_evaluation_count"),
    })
    audit.add("covariance.no_double_counting", covariance.get("double_counting_detected") is False,
              covariance.get("covariance_nuisance_component_overlap"))

    rehearsal = manifest.get("rehearsal", {})
    rehearsal_path = root / str(rehearsal.get("report_path", ""))
    rehearsal_report = load_json(rehearsal_path) if rehearsal_path.is_file() else {}
    audit.add("rehearsal.production_vertical", rehearsal_report.get("status") == "pass", {
        "path": str(rehearsal_path), "manifest_status": rehearsal.get("status"),
        "report_status": rehearsal_report.get("status"),
    })

    git = run(["git", "status", "--porcelain", "--untracked-files=all"])
    dirty = [line for line in git.stdout.splitlines() if line.strip()]
    clean_ok = git.returncode == 0 and (not dirty or args.allow_dirty)
    audit.add("git.clean", clean_ok, {"allow_dirty": args.allow_dirty, "entries": dirty})
    head = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    branch = run(["git", "branch", "--show-current"]).stdout.strip()

    report = {
        "schema_version": 1,
        "status": "GO" if not audit.blockers else "NO-GO",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "git": {"branch": branch, "head": head},
        "summary": {
            "check_count": len(audit.checks),
            "passed_count": sum(item["passed"] for item in audit.checks),
            "blocking_failure_count": len(audit.blockers),
            "warning_count": sum(not item["passed"] and not item["blocking"] for item in audit.checks),
        },
        "checks": audit.checks,
        "blocking_failures": audit.blockers,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "GO" else 2


if __name__ == "__main__":
    sys.exit(main())
