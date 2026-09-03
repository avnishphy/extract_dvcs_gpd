"""Reusable exact-PARTONS corpora and group-aware neural selections.

The corpus owns only expensive, noise-free native results.  Experimental
noise, nuisance draws, latent transforms, and DeepSets contexts are derived
deterministically when a selection is materialized for inference.  No physics
formula is implemented here: every observable extension is obtained from the
versioned C++ PARTONS bridge.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from extract_dvcs_cff.workflows.pseudodata import (
    ADMITTED_OBSERVABLE_IDS,
    NativeCache,
    _DD_SHAPE_BOUNDS,
    _DD_SHAPE_FIELDS,
    _GPD_TYPES,
    _PARTON_CHANNELS,
    _covariance_and_responses,
    _effective_prediction,
    _parallel_partitioned_native_evaluation,
    _parallel_native_gpd_truth_evaluation,
    _physics_path,
    _point_table,
    _sample_shape_parameters,
    _shape_parameter_name,
    _shape_vector_from_shapes,
    canonical,
    load_configuration,
    pretty_json,
    scientific_configuration_sha256,
    sha256,
)


CORPUS_SCHEMA_VERSION = 2
SELECTION_SCHEMA_VERSION = 1
REALIZATION_SCHEMA_VERSION = 3
DEFAULT_SHARD_SIZE = 256
_MAX_CANDIDATE_MULTIPLIER = 10
ADMITTED_OBSERVABLE_METADATA = {
    "DVCSCrossSectionUUMinus": {
        "id": "DVCSCrossSectionUUMinus", "native_unit": "nb/GeV4",
        "normalization_scale": 1.0,
        "user_label": "Beam-spin half-sum / unpolarized cross section",
    },
    "DVCSCrossSectionDifferenceLUMinus": {
        "id": "DVCSCrossSectionDifferenceLUMinus", "native_unit": "nb/GeV4",
        "normalization_scale": 0.1, "user_label": "Beam-spin difference",
    },
    "DVCSAc": {"id": "DVCSAc", "native_unit": "1",
               "normalization_scale": 1.0,
               "user_label": "Beam-charge asymmetry"},
    "DVCSAluMinus": {"id": "DVCSAluMinus", "native_unit": "1",
                     "normalization_scale": 1.0,
                     "user_label": "Beam-spin asymmetry"},
    "DVCSAulMinus": {"id": "DVCSAulMinus", "native_unit": "1",
                     "normalization_scale": 1.0,
                     "user_label": "Longitudinal target-spin asymmetry"},
    "DVCSAllMinus": {"id": "DVCSAllMinus", "native_unit": "1",
                     "normalization_scale": 1.0,
                     "user_label": "Longitudinal double-spin asymmetry"},
}


def _nested_kinematic_order(
    kinematics: Sequence[Mapping[str, Any]], seed: int
) -> tuple[int, ...]:
    """Return a deterministic space-filling order; every prefix is nested."""

    features = np.asarray([
        (
            float(point["x_b"]) / 0.5,
            -float(point["t_GeV2"]) / 0.9,
            math.log(float(point["Q2_GeV2"])) / math.log(80.0),
            math.log(float(point["beam_energy_GeV"]) / 3.0)
            / math.log(200.0 / 3.0),
            math.sin(float(point["phi_rad"])),
            math.cos(float(point["phi_rad"])),
        )
        for point in kinematics
    ], dtype=np.float64)
    first = min(
        range(len(kinematics)),
        key=lambda index: hashlib.sha256(
            f"{seed}:{index}".encode("utf-8")
        ).hexdigest(),
    )
    chosen = [first]
    distances = np.sum((features - features[first]) ** 2, axis=1)
    distances[first] = -1.0
    while len(chosen) < len(kinematics):
        maximum = float(distances.max())
        tied = np.flatnonzero(
            np.isclose(distances, maximum, rtol=0.0, atol=1e-15)
        )
        next_index = int(min(tied))
        chosen.append(next_index)
        distances = np.minimum(
            distances,
            np.sum((features - features[next_index]) ** 2, axis=1),
        )
        distances[chosen] = -1.0
    return tuple(chosen)


def _masked_design_count(
    counts: Sequence[int], group_position: int, replica_index: int
) -> int:
    """Balance declared design counts deterministically across context rows."""

    if not counts or group_position < 0 or replica_index < 0:
        raise ValueError("masked design schedule inputs are invalid")
    return int(counts[(group_position + replica_index) % len(counts)])


def _atomic_bytes(path: Path, payload: bytes) -> None:
    """Commit one complete file with same-directory atomic replacement."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (pretty_json(value) + "\n").encode())


def _atomic_npz(
    path: Path, *, compression: str = "npz_deflate", **arrays: np.ndarray,
) -> dict[str, Any]:
    """Write, reopen, validate, hash, and atomically publish one shard."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as stream:
        writer = np.savez_compressed if compression == "npz_deflate" else np.savez
        if compression not in {"npz_deflate", "none"}:
            raise ValueError("unsupported NPZ compression policy")
        writer(stream, **{name: np.asarray(value) for name, value in arrays.items()})
        stream.flush()
        os.fsync(stream.fileno())
    with np.load(temporary, allow_pickle=False) as observed:
        if set(observed.files) != set(arrays):
            raise RuntimeError(f"atomic shard key mismatch: {temporary}")
        for name, expected in arrays.items():
            actual = observed[name]
            if actual.shape != np.asarray(expected).shape:
                raise RuntimeError(f"atomic shard shape mismatch: {name}")
            if actual.dtype != np.asarray(expected).dtype:
                raise RuntimeError(f"atomic shard dtype mismatch: {name}")
    os.replace(temporary, path)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "arrays": {
            name: {
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
            }
            for name, value in arrays.items()
        },
    }


def _archive_native_cache(cache_root: Path, destination: Path) -> dict[str, Any] | None:
    """Consolidate transient per-request evidence into one atomic archive.

    ``NativeCache`` deliberately stores one directory per bridge request.  That
    is useful while a request is running, but it is a poor durable corpus
    layout.  A shard therefore owns one compressed evidence archive and the
    transient request directories are removed only after that archive has been
    reopened successfully.
    """

    entries = sorted(path for path in cache_root.iterdir()) if cache_root.exists() else []
    if not entries:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    with tarfile.open(temporary, "w:gz") as archive:
        for entry in entries:
            archive.add(entry, arcname=entry.name, recursive=True)
    with tarfile.open(temporary, "r:gz") as archive:
        members = archive.getmembers()
        if not members:
            raise RuntimeError("native evidence archive is unexpectedly empty")
        if any(
            member.name.startswith("/") or ".." in Path(member.name).parts
            for member in members
        ):
            raise RuntimeError("unsafe member in native evidence archive")
    os.replace(temporary, destination)
    for entry in entries:
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    return {
        "path": str(destination),
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
        "request_entry_count": len(entries),
    }


def _manifest_hash(value: Mapping[str, Any]) -> str:
    content = dict(value)
    content.pop("manifest_sha256", None)
    return hashlib.sha256(canonical(content).encode()).hexdigest()


def _write_manifest(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["manifest_sha256"] = _manifest_hash(result)
    _atomic_json(path, result)
    return result


def _load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if value.get("manifest_sha256") != _manifest_hash(value):
        raise RuntimeError(f"manifest content hash mismatch: {path}")
    return value


def _parameter_names() -> list[str]:
    return [
        _shape_parameter_name(gpd, channel, field)
        for gpd in _GPD_TYPES
        for channel in _PARTON_CHANNELS
        for field in _DD_SHAPE_FIELDS
    ]


def _core_identity(config: Mapping[str, Any], physics: Mapping[str, Any],
                   profile: str, bridge_hash: str,
                   schema_version: int = CORPUS_SCHEMA_VERSION,
                   gpd_truth_request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    settings = config["profiles"][profile]
    physics_core = deepcopy(physics)
    physics_core["theory_configuration"].pop("observable_modules", None)
    result = {
        "parameter_count": int(settings["native_parameter_count"]),
        "parameter_seed": int(config["seeds"]["native_parameters"]),
        "parameter_names": _parameter_names(),
        "prior": {
            name: {"distribution": "uniform", "bounds": list(bounds)}
            for name, bounds in _DD_SHAPE_BOUNDS.items()
        },
        "kinematics": deepcopy(config["kinematics"]),
        "physics": physics_core,
        "bridge_sha256": bridge_hash,
    }
    if schema_version == 1:
        result["representation"] = config["representation"]
    else:
        result.update({
            "generator_family": "dd",
            "generator_representation": config["representation"],
            "generator_prior_id": "full_independent_dd_uniform_v1",
            "gpd_truth_request": deepcopy(gpd_truth_request),
        })
    return result


def _observable_metadata(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: deepcopy(item) for item in config["observables"]}


def create_corpus(*, corpus: Path, bridge: Path, configuration_path: Path,
                  profile: str, shard_size: int = DEFAULT_SHARD_SIZE,
                  gpd_truth_request_path: Path | None = None) -> dict[str, Any]:
    """Freeze a corpus definition without performing a native calculation."""

    if corpus.exists():
        raise FileExistsError(f"corpus already exists: {corpus}")
    if not 1 <= int(shard_size) <= 4096:
        raise ValueError("shard_size must be in [1,4096]")
    config = load_configuration(configuration_path)
    if profile not in config["profiles"]:
        raise ValueError(f"unknown profile {profile!r}")
    physics_path = _physics_path(configuration_path, config)
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    bridge_path = bridge.resolve(strict=True)
    bridge_hash = sha256(bridge_path)
    capabilities = subprocess.run(
        [str(bridge_path), "--capabilities"], text=True,
        capture_output=True, check=False,
    )
    if capabilities.returncode != 0:
        raise RuntimeError(
            f"bridge capabilities failed: {capabilities.stderr.strip()}"
        )
    backend = json.loads(capabilities.stdout)
    if gpd_truth_request_path is None:
        raise ValueError(
            "every new schema-2 master corpus requires canonical GPD truth; "
            "provide --gpd-truth-request or project/gpd_truth.json"
        )
    from extract_dvcs_cff.contracts import validate_gpd_truth_request

    gpd_truth_request = validate_gpd_truth_request(json.loads(
        gpd_truth_request_path.resolve(strict=True).read_text(encoding="utf-8")
    ))
    requested_q0 = float(gpd_truth_request["scale"]["Q0_squared_GeV2"])
    configured_q0 = float(physics["input_scale"]["Q0_squared"]["value"])
    if requested_q0 != configured_q0:
        raise ValueError(
            "GPD truth input scale differs from the native generator input scale"
        )
    if gpd_truth_request["generator_representation"] == "declared_by_master_corpus":
        gpd_truth_request["generator_representation"] = config["representation"]
    elif gpd_truth_request["generator_representation"] != config["representation"]:
        raise ValueError("GPD truth generator representation differs from corpus")
    native_truth = backend.get("capabilities", {}).get(
        "canonical_gpd_truth_v1", {}
    )
    if native_truth.get("available") is not True:
        raise RuntimeError(
            "native bridge lacks canonical_gpd_truth_v1: "
            + str(native_truth.get("reason", "capability not reported"))
        )
    core = _core_identity(
        config, physics, profile, bridge_hash,
        gpd_truth_request=gpd_truth_request,
    )
    core_hash = hashlib.sha256(canonical(core).encode()).hexdigest()
    corpus.mkdir(parents=True)
    (corpus / "configuration").mkdir()
    _atomic_json(corpus / "configuration" / "workflow.json", config)
    _atomic_json(corpus / "configuration" / "physics.json", physics)
    _atomic_json(corpus / "backend_provenance.json", backend)
    count = int(core["parameter_count"])
    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "status": "planned",
        "corpus_name": corpus.name,
        "corpus_root": str(corpus.resolve()),
        "core_identity_sha256": core_hash,
        "master_corpus_id": f"master-corpus-{core_hash}",
        "core_identity": core,
        "profile_source": profile,
        "shard_size": int(shard_size),
        "shard_count": math.ceil(count / int(shard_size)),
        "available_observables": {},
        "requested_observables": list(_observable_metadata(config)),
        "observable_metadata": deepcopy(ADMITTED_OBSERVABLE_METADATA),
        "core_shards": [],
        "truth": None,
        "gpd_truth": {
            "status": "planned", "request": gpd_truth_request,
            "coordinates": gpd_truth_request["coordinates"],
            "coordinate_table_sha256": gpd_truth_request[
                "coordinate_table_sha256"
            ],
            "value_shape": [count, len(gpd_truth_request["coordinates"])],
            "dtype": gpd_truth_request["dtype"],
            "compression": gpd_truth_request["compression"],
            "status_mask": True, "shards": [],
            "native_adapter_capability": "canonical_gpd_truth_v1",
        },
        "rejected_candidate_count": 0,
        "surrogate_used": False,
        "split_roles_stored_in_corpus": False,
        "noise_or_nuisance_replicas_stored_in_corpus": False,
        "native_backend_provenance": "backend_provenance.json",
        "configuration": "configuration/workflow.json",
        "physics_configuration": "configuration/physics.json",
    }
    return _write_manifest(corpus / "corpus.json", manifest)


def _configuration_for_observables(
    config: Mapping[str, Any], metadata: Mapping[str, Mapping[str, Any]],
    names: Sequence[str],
) -> dict[str, Any]:
    selected = deepcopy(config)
    selected["observables"] = [deepcopy(metadata[name]) for name in names]
    return selected


def _shard_relative(path: Path, corpus: Path) -> str:
    return str(path.relative_to(corpus))


def _generate_truth(*, corpus: Path, client: NativeCache,
                    config: Mapping[str, Any], physics: Mapping[str, Any],
                    force: bool) -> dict[str, Any]:
    truth_parameters = _shape_vector_from_shapes(
        physics["parameters"]["gpd_shapes"]
    )[None, :]
    result = client.evaluate(
        parameters=truth_parameters, config=config, physics=physics,
        force=force,
    )
    evidence = _archive_native_cache(
        client.cache_root, corpus / "evidence" / "truth-native.tar.gz"
    )
    core_path = corpus / "truth" / "core.npz"
    core_record = _atomic_npz(
        core_path,
        parameters=truth_parameters,
        cffs=result.cffs,
    )
    core_record["path"] = _shard_relative(core_path, corpus)
    core_record["native"] = {
        "request_sha256": result.request_sha256,
        "response_sha256": result.response_sha256,
        "bridge_sha256": result.bridge_sha256,
    }
    if evidence is not None:
        evidence["path"] = _shard_relative(Path(evidence["path"]), corpus)
        core_record["native_evidence"] = evidence
    names = [item["id"] for item in config["observables"]]
    observable_records: dict[str, Any] = {}
    for index, name in enumerate(names):
        path = corpus / "truth" / "observables" / f"{name}.npz"
        record = _atomic_npz(
            path, values=result.predictions[:, index::len(names)]
        )
        record["path"] = _shard_relative(path, corpus)
        observable_records[name] = record
    return {"core": core_record, "observables": observable_records}


def _generate_gpd_truth_shard(
    *, corpus: Path, client: NativeCache, shard_index: int,
    group_index: np.ndarray, parameters: np.ndarray,
    config: Mapping[str, Any], physics: Mapping[str, Any],
    truth_request: Mapping[str, Any], show_progress: bool | None,
    force: bool,
) -> dict[str, Any]:
    native = _parallel_native_gpd_truth_evaluation(
        client=client, parameters=parameters, config=config, physics=physics,
        truth_request=truth_request,
        native_workers=config["runtime"]["native_workers"],
        description=f"GPD truth shard {shard_index + 1}",
        show_progress=show_progress, force=force,
    )
    dtype = np.dtype(truth_request["dtype"])
    values = native.values.astype(dtype, copy=False)
    status_mask = native.status_mask.astype(bool, copy=False)
    path = corpus / "gpd_truth" / "shards" / f"shard-{shard_index:06d}.npz"
    record = _atomic_npz(
        path, compression=truth_request["compression"],
        group_index=group_index, values=values, status_mask=status_mask,
    )
    record["path"] = _shard_relative(path, corpus)
    record["shard_index"] = shard_index
    record["group_index_start"] = int(group_index[0])
    record["group_index_stop_exclusive"] = int(group_index[-1] + 1)
    record["valid_value_count"] = int(np.count_nonzero(status_mask))
    record["invalid_value_count"] = int(
        status_mask.size - np.count_nonzero(status_mask)
    )
    record["native_execution"] = {
        "worker_resolution": native.worker_resolution,
        "chunk_size": native.chunk_size,
        "task_count": native.task_count,
    }
    evidence = _archive_native_cache(
        client.cache_root,
        corpus / "evidence" / "gpd_truth" /
        f"shard-{shard_index:06d}-native.tar.gz",
    )
    if evidence is not None:
        evidence["path"] = _shard_relative(Path(evidence["path"]), corpus)
        record["native_evidence"] = evidence
    return record


def generate_corpus(*, corpus: Path, bridge: Path,
                    requested_observables: Sequence[str] | None = None,
                    show_progress: bool | None = None,
                    force_native: bool = False,
                    max_shards: int | None = None,
                    shard_start: int | None = None) -> dict[str, Any]:
    """Generate missing core shards or append missing observable extensions."""

    if max_shards is not None and (
        isinstance(max_shards, bool)
        or not isinstance(max_shards, int)
        or max_shards < 1
    ):
        raise ValueError("max_shards must be a positive integer")
    if shard_start is not None and (
        isinstance(shard_start, bool)
        or not isinstance(shard_start, int)
        or shard_start < 0
    ):
        raise ValueError("shard_start must be a nonnegative integer")

    manifest_path = corpus / "corpus.json"
    manifest = _load_manifest(manifest_path)
    if manifest["schema_version"] not in {1, CORPUS_SCHEMA_VERSION}:
        raise RuntimeError("unsupported master corpus schema")
    config_path = corpus / manifest["configuration"]
    physics_path = corpus / manifest["physics_configuration"]
    config = load_configuration(config_path)
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    bridge_path = bridge.resolve(strict=True)
    if sha256(bridge_path) != manifest["core_identity"]["bridge_sha256"]:
        raise RuntimeError("bridge hash differs from corpus definition")
    metadata = manifest["observable_metadata"]
    names = list(requested_observables or manifest["requested_observables"])
    expected = [name for name in ADMITTED_OBSERVABLE_IDS if name in names]
    if names != expected or any(name not in metadata for name in names):
        raise ValueError("requested observables are not the declared ordered subset")
    missing = [name for name in names
               if name not in manifest["available_observables"]]
    shard_count = int(manifest["shard_count"])
    existing_indices = [
        int(record["shard_index"]) for record in manifest["core_shards"]
    ]
    if (
        len(existing_indices) != len(set(existing_indices))
        or any(index < 0 or index >= shard_count for index in existing_indices)
    ):
        raise RuntimeError("corpus manifest contains invalid core shard indices")
    core_complete = len(existing_indices) == shard_count
    gpd_truth = manifest.get("gpd_truth")
    if int(manifest["schema_version"]) == CORPUS_SCHEMA_VERSION and not isinstance(
        gpd_truth, dict
    ):
        raise RuntimeError("schema-2 master corpus is missing required GPD truth")
    gpd_complete = int(manifest["schema_version"]) == 1 or (
        gpd_truth.get("status") == "complete"
        and len(gpd_truth.get("shards", ())) == shard_count
    )
    if core_complete and gpd_complete and not missing:
        return {
            "status": "complete_cache_hit",
            "corpus": manifest["corpus_name"],
            "missing_observables_generated": [],
            "native_called": False,
            "manifest_sha256": manifest["manifest_sha256"],
        }
    # Per-request cache directories are temporary working state.  Each native
    # unit is compacted into one evidence archive before its manifest commit.
    client = NativeCache(bridge_path, corpus / ".native_work")
    if manifest["truth"] is None:
        truth_config = _configuration_for_observables(
            config, metadata, names if names else list(metadata)[:1]
        )
        manifest["truth"] = _generate_truth(
            corpus=corpus, client=client, config=truth_config,
            physics=physics, force=force_native,
        )
        manifest = _write_manifest(manifest_path, manifest)

    if not core_complete:
        count = int(manifest["core_identity"]["parameter_count"])
        shard_size = int(manifest["shard_size"])
        combined_config = _configuration_for_observables(
            config, metadata, names
        )
        existing_index_set = set(existing_indices)
        if shard_start is None:
            selected_indices = [
                index for index in range(shard_count)
                if index not in existing_index_set
            ]
            if max_shards is not None:
                selected_indices = selected_indices[:max_shards]
        else:
            if shard_start >= shard_count:
                raise ValueError("shard_start must be smaller than shard_count")
            stop_shard = shard_count if max_shards is None else min(
                shard_count, shard_start + max_shards
            )
            selected_indices = [
                index for index in range(shard_start, stop_shard)
                if index not in existing_index_set
            ]
        for shard_index in selected_indices:
            target = min(shard_size, count - shard_index * shard_size)
            rng = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence([
                    int(manifest["core_identity"]["parameter_seed"]),
                    shard_index,
                    CORPUS_SCHEMA_VERSION,
                ])
            ))
            accepted_parameters: list[np.ndarray] = []
            accepted_predictions: list[np.ndarray] = []
            accepted_cffs: list[np.ndarray] = []
            rejected: list[dict[str, Any]] = []
            attempted = 0
            while sum(len(item) for item in accepted_parameters) < target:
                remaining = target - sum(len(item) for item in accepted_parameters)
                candidates = _sample_shape_parameters(rng, remaining)
                native = _parallel_partitioned_native_evaluation(
                    client=client, parameters=candidates,
                    config=combined_config, physics=physics,
                    native_workers=config["runtime"]["native_workers"],
                    description=f"Corpus shard {shard_index + 1}/{manifest['shard_count']}",
                    show_progress=show_progress, force=force_native,
                ).result
                valid = native.valid_mask
                if np.any(valid):
                    accepted_parameters.append(candidates[valid])
                    accepted_predictions.append(native.predictions[valid])
                    accepted_cffs.append(native.cffs[valid])
                for item in native.invalid_records:
                    record = dict(item)
                    record["shard_index"] = shard_index
                    record["candidate_index_in_shard_stream"] = (
                        attempted + int(item["candidate_index"])
                    )
                    rejected.append(record)
                attempted += len(candidates)
                if attempted > _MAX_CANDIDATE_MULTIPLIER * target:
                    raise RuntimeError(
                        f"shard {shard_index} exceeded invalid-draw limit"
                    )
            parameters = np.concatenate(accepted_parameters)[:target]
            predictions = np.concatenate(accepted_predictions)[:target]
            cffs = np.concatenate(accepted_cffs)[:target]
            group_index = np.arange(
                shard_index * shard_size,
                shard_index * shard_size + target,
                dtype=np.int64,
            )
            core_path = corpus / "core" / "shards" / f"shard-{shard_index:06d}.npz"
            core_record = _atomic_npz(
                core_path, group_index=group_index,
                native_parameters=parameters, native_cffs=cffs,
            )
            core_record["path"] = _shard_relative(core_path, corpus)
            core_record["shard_index"] = shard_index
            core_record["group_index_start"] = int(group_index[0])
            core_record["group_index_stop_exclusive"] = int(group_index[-1] + 1)
            evidence = _archive_native_cache(
                client.cache_root,
                corpus / "evidence" / "core" /
                f"shard-{shard_index:06d}-native.tar.gz",
            )
            if evidence is not None:
                evidence["path"] = _shard_relative(
                    Path(evidence["path"]), corpus
                )
                core_record["native_evidence"] = evidence
            manifest["core_shards"].append(core_record)
            observable_count = len(names)
            for observable_index, name in enumerate(names):
                observable_path = (
                    corpus / "observables" / name / "shards" /
                    f"shard-{shard_index:06d}.npz"
                )
                values = predictions[:, observable_index::observable_count]
                observable_record = _atomic_npz(
                    observable_path, group_index=group_index, values=values
                )
                observable_record["path"] = _shard_relative(
                    observable_path, corpus
                )
                observable_record["shard_index"] = shard_index
                extension = manifest["available_observables"].setdefault(
                    name, {"metadata": metadata[name], "shards": []}
                )
                extension["shards"].append(observable_record)
            if int(manifest["schema_version"]) == CORPUS_SCHEMA_VERSION:
                gpd_record = _generate_gpd_truth_shard(
                    corpus=corpus, client=client, shard_index=shard_index,
                    group_index=group_index, parameters=parameters,
                    config=config, physics=physics,
                    truth_request=manifest["gpd_truth"]["request"],
                    show_progress=show_progress, force=force_native,
                )
                manifest["gpd_truth"]["shards"].append(gpd_record)
                manifest["gpd_truth"]["status"] = "partial"
            rejection_path = (
                corpus / "rejections" / f"shard-{shard_index:06d}.json"
            )
            _atomic_json(rejection_path, {
                "schema_version": 1,
                "shard_index": shard_index,
                "attempted": attempted,
                "accepted": target,
                "rejected": rejected,
            })
            manifest["rejected_candidate_count"] += len(rejected)
            manifest["status"] = "partial"
            manifest = _write_manifest(manifest_path, manifest)
        manifest["core_shards"].sort(key=lambda record: int(record["shard_index"]))
        for extension in manifest["available_observables"].values():
            extension["shards"].sort(
                key=lambda record: int(record.get(
                    "shard_index", Path(record["path"]).stem.split("-")[-1]
                ))
            )
        if int(manifest["schema_version"]) == CORPUS_SCHEMA_VERSION:
            manifest["gpd_truth"]["shards"].sort(
                key=lambda record: int(record["shard_index"])
            )
        manifest = _write_manifest(manifest_path, manifest)
        core_complete = len(manifest["core_shards"]) == shard_count
        if not core_complete:
            return {
                "status": "partial",
                "corpus": manifest["corpus_name"],
                "completed_shards": len(manifest["core_shards"]),
                "generated_shard_indices": selected_indices,
                "shard_count": shard_count,
                "manifest_sha256": manifest["manifest_sha256"],
            }
        missing = []

    # Resume or repair a schema-2 campaign whose core shard was committed by
    # an earlier generator before its corresponding function-truth shard.
    if int(manifest["schema_version"]) == CORPUS_SCHEMA_VERSION:
        present = {
            int(record["shard_index"])
            for record in manifest["gpd_truth"]["shards"]
        }
        selected = [
            record for record in manifest["core_shards"]
            if int(record["shard_index"]) not in present
        ]
        for core_record in selected:
            shard_index = int(core_record["shard_index"])
            with np.load(corpus / core_record["path"], allow_pickle=False) as arrays:
                group_index = arrays["group_index"]
                parameters = arrays["native_parameters"]
            manifest["gpd_truth"]["shards"].append(_generate_gpd_truth_shard(
                corpus=corpus, client=client, shard_index=shard_index,
                group_index=group_index, parameters=parameters,
                config=config, physics=physics,
                truth_request=manifest["gpd_truth"]["request"],
                show_progress=show_progress, force=force_native,
            ))
            manifest["gpd_truth"]["status"] = "partial"
            manifest["gpd_truth"]["shards"].sort(
                key=lambda record: int(record["shard_index"])
            )
            manifest = _write_manifest(manifest_path, manifest)

    # Appending an observable replays only that native observable for the
    # immutable core parameters. CFF equality proves the same upstream route.
    for name in missing:
        singleton = _configuration_for_observables(config, metadata, [name])
        extension = {"metadata": metadata[name], "shards": []}
        with np.load(
            corpus / manifest["truth"]["core"]["path"], allow_pickle=False
        ) as truth_core:
            truth_parameters = truth_core["parameters"]
            truth_cffs = truth_core["cffs"]
        truth_native = client.evaluate(
            parameters=truth_parameters, config=singleton, physics=physics,
            force=force_native,
        )
        truth_evidence = _archive_native_cache(
            client.cache_root,
            corpus / "evidence" / "observables" / name /
            "truth-native.tar.gz",
        )
        if not np.allclose(truth_native.cffs, truth_cffs,
                           rtol=0.0, atol=1e-11):
            raise RuntimeError(
                f"truth observable extension {name} changed native CFFs"
            )
        truth_path = corpus / "truth" / "observables" / f"{name}.npz"
        truth_record = _atomic_npz(
            truth_path, values=truth_native.predictions
        )
        truth_record["path"] = _shard_relative(truth_path, corpus)
        if truth_evidence is not None:
            truth_evidence["path"] = _shard_relative(
                Path(truth_evidence["path"]), corpus
            )
            truth_record["native_evidence"] = truth_evidence
        manifest["truth"]["observables"][name] = truth_record
        for core_record in manifest["core_shards"]:
            core_path = corpus / core_record["path"]
            with np.load(core_path, allow_pickle=False) as core_arrays:
                group_index = core_arrays["group_index"]
                parameters = core_arrays["native_parameters"]
                expected_cffs = core_arrays["native_cffs"]
            native = _parallel_partitioned_native_evaluation(
                client=client, parameters=parameters, config=singleton,
                physics=physics,
                native_workers=config["runtime"]["native_workers"],
                description=f"Append {name}", show_progress=show_progress,
                force=force_native,
            ).result
            if not np.all(native.valid_mask):
                raise RuntimeError(
                    f"observable extension {name} has invalid native rows"
                )
            if not np.allclose(native.cffs, expected_cffs,
                               rtol=0.0, atol=1e-11):
                raise RuntimeError(
                    f"observable extension {name} changed native CFFs"
                )
            shard_index = int(core_record["shard_index"])
            path = (corpus / "observables" / name / "shards" /
                    f"shard-{shard_index:06d}.npz")
            record = _atomic_npz(
                path, group_index=group_index,
                values=native.predictions,
            )
            record["path"] = _shard_relative(path, corpus)
            record["shard_index"] = shard_index
            evidence = _archive_native_cache(
                client.cache_root,
                corpus / "evidence" / "observables" / name /
                f"shard-{shard_index:06d}-native.tar.gz",
            )
            if evidence is not None:
                evidence["path"] = _shard_relative(
                    Path(evidence["path"]), corpus
                )
                record["native_evidence"] = evidence
            extension["shards"].append(record)
        manifest["available_observables"][name] = extension
        manifest = _write_manifest(manifest_path, manifest)

    if int(manifest["schema_version"]) == CORPUS_SCHEMA_VERSION:
        if len(manifest["gpd_truth"]["shards"]) != shard_count:
            raise RuntimeError("canonical GPD truth shards are incomplete")
        manifest["gpd_truth"]["status"] = "complete"
    manifest["status"] = "complete"
    manifest = _write_manifest(manifest_path, manifest)
    for directory in (client.cache_root, client.cache_root.parent):
        try:
            directory.rmdir()
        except OSError:
            # Nonempty work state is retained after any unusual cache event;
            # verification still excludes it from the durable shard contract.
            break
    verification = verify_corpus(corpus=corpus, deep=True)
    return {
        "status": "complete",
        "corpus": manifest["corpus_name"],
        "native_called": True,
        "available_observables": sorted(manifest["available_observables"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "verification": verification,
    }


def plan_corpus(*, corpus: Path, requested_observables: Sequence[str]) -> dict[str, Any]:
    manifest = _load_manifest(corpus / "corpus.json")
    requested = list(requested_observables)
    expected = [name for name in ADMITTED_OBSERVABLE_IDS if name in requested]
    if requested != expected:
        raise ValueError("requested observables are not an admitted ordered subset")
    available = list(manifest["available_observables"])
    return {
        "status": manifest["status"],
        "corpus": manifest["corpus_name"],
        "parameter_count": manifest["core_identity"]["parameter_count"],
        "kinematic_count": len(manifest["core_identity"]["kinematics"]),
        "already_available": [name for name in requested if name in available],
        "missing_observables": [name for name in requested if name not in available],
        "existing_shards_will_be_modified": False,
        "review_required": True,
        "requested_observable_selection_sha256": hashlib.sha256(
            canonical(requested).encode()
        ).hexdigest(),
    }


def assert_corpus_compatible(*, corpus: Path, configuration_path: Path,
                             bridge: Path | None = None) -> dict[str, Any]:
    """Require unchanged GPD physics, parameters, kinematics, and backend."""

    manifest = _load_manifest(corpus / "corpus.json")
    config = load_configuration(configuration_path)
    profile = manifest["profile_source"]
    physics_path = _physics_path(configuration_path, config)
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    stored_bridge_hash = str(manifest["core_identity"]["bridge_sha256"])
    bridge_hash = (
        sha256(bridge.resolve(strict=True)) if bridge is not None
        else stored_bridge_hash
    )
    identity = _core_identity(
        config, physics, profile, bridge_hash,
        int(manifest["schema_version"]),
        (
            manifest.get("gpd_truth", {}).get("request")
            if isinstance(manifest.get("gpd_truth"), dict) else None
        ),
    )
    observed = hashlib.sha256(canonical(identity).encode()).hexdigest()
    if observed != manifest["core_identity_sha256"]:
        raise RuntimeError(
            "corpus core mismatch: GPD physics, priors/count, kinematics, "
            "parameter seed, or native bridge changed; create a new corpus"
        )
    return manifest


def verify_corpus(*, corpus: Path, deep: bool = False,
                  allow_partial: bool = False) -> dict[str, Any]:
    """Verify manifest structure and optionally every shard byte and array."""

    manifest = _load_manifest(corpus / "corpus.json")
    if int(manifest.get("schema_version", -1)) not in {1, CORPUS_SCHEMA_VERSION}:
        raise RuntimeError(
            "unsupported master corpus schema; supported historical/current "
            f"schemas are 1 and {CORPUS_SCHEMA_VERSION}"
        )
    expected_shards = int(manifest["shard_count"])
    completed_shards = len(manifest["core_shards"])
    core_indices = [
        int(record["shard_index"]) for record in manifest["core_shards"]
    ]
    if completed_shards > expected_shards:
        raise RuntimeError("corpus contains more core shards than declared")
    if (
        core_indices != sorted(core_indices)
        or len(core_indices) != len(set(core_indices))
        or any(index < 0 or index >= expected_shards for index in core_indices)
    ):
        raise RuntimeError("core shard indices are invalid, duplicated, or unordered")
    if completed_shards != expected_shards and not allow_partial:
        raise RuntimeError("corpus core is incomplete")
    seen: list[int] = []
    verified_files = 0
    records = list(manifest["core_shards"])
    for extension in manifest["available_observables"].values():
        required_shards = completed_shards if allow_partial else expected_shards
        if len(extension["shards"]) != required_shards:
            raise RuntimeError("observable extension is incomplete")
        extension_indices = [
            int(record.get(
                "shard_index", Path(record["path"]).stem.split("-")[-1]
            ))
            for record in extension["shards"]
        ]
        if extension_indices != core_indices:
            raise RuntimeError("observable extension shard indices differ from core")
        records.extend(extension["shards"])
    gpd_truth = manifest.get("gpd_truth")
    if int(manifest["schema_version"]) == CORPUS_SCHEMA_VERSION:
        if not isinstance(gpd_truth, dict):
            raise RuntimeError("schema-2 master corpus lacks canonical GPD truth")
        if not gpd_truth.get("coordinates") or not isinstance(
            gpd_truth.get("request"), dict
        ):
            raise RuntimeError("canonical GPD truth metadata is incomplete")
        gpd_records = gpd_truth.get("shards", [])
        required_shards = completed_shards if allow_partial else expected_shards
        if len(gpd_records) != required_shards:
            raise RuntimeError("canonical GPD truth shards are incomplete")
        gpd_indices = [int(record["shard_index"]) for record in gpd_records]
        if gpd_indices != core_indices:
            raise RuntimeError("GPD truth shard indices differ from core")
        expected_status = (
            "complete" if completed_shards == expected_shards else "partial"
        )
        if gpd_truth.get("status") != expected_status:
            raise RuntimeError("canonical GPD truth status is inconsistent")
        records.extend(gpd_records)
    if manifest["truth"] is not None:
        records.append(manifest["truth"]["core"])
        records.extend(manifest["truth"]["observables"].values())
    evidence_records: list[Mapping[str, Any]] = []
    for record in records:
        evidence = record.get("native_evidence")
        if evidence is not None:
            evidence_records.append(evidence)
        path = corpus / record["path"]
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or symlinked corpus file: {path}")
        if deep and sha256(path) != record["sha256"]:
            raise RuntimeError(f"corpus shard hash mismatch: {path}")
        if deep:
            with np.load(path, allow_pickle=False) as arrays:
                for name, contract in record["arrays"].items():
                    if name not in arrays.files:
                        raise RuntimeError(f"missing array {name} in {path}")
                    if list(arrays[name].shape) != contract["shape"]:
                        raise RuntimeError(f"array shape mismatch in {path}")
                    if str(arrays[name].dtype) != contract["dtype"]:
                        raise RuntimeError(f"array dtype mismatch in {path}")
                if "group_index" in arrays.files and "native_parameters" in arrays.files:
                    seen.extend(int(value) for value in arrays["group_index"])
                if {"group_index", "values", "status_mask"}.issubset(arrays.files):
                    values = arrays["values"]
                    status_mask = arrays["status_mask"]
                    if values.shape != status_mask.shape:
                        raise RuntimeError(f"GPD truth value/mask shape mismatch in {path}")
                    if values.shape[1] != len(gpd_truth["coordinates"]):
                        raise RuntimeError(f"GPD truth coordinate width mismatch in {path}")
                    if status_mask.dtype != np.dtype(bool):
                        raise RuntimeError(f"GPD truth status mask must be boolean in {path}")
                    if not np.all(np.isfinite(values)):
                        raise RuntimeError(f"GPD truth shard contains non-finite storage in {path}")
        verified_files += 1
    for evidence in evidence_records:
        path = corpus / evidence["path"]
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or symlinked evidence archive: {path}")
        if deep and sha256(path) != evidence["sha256"]:
            raise RuntimeError(f"native evidence hash mismatch: {path}")
        if deep:
            with tarfile.open(path, "r:gz") as archive:
                if not archive.getmembers():
                    raise RuntimeError(f"empty native evidence archive: {path}")
        verified_files += 1
    if deep:
        shard_size = int(manifest["shard_size"])
        parameter_count = int(manifest["core_identity"]["parameter_count"])
        expected = [
            group_index
            for shard_index in core_indices
            for group_index in range(
                shard_index * shard_size,
                min(parameter_count, (shard_index + 1) * shard_size),
            )
        ]
        if seen != expected:
            raise RuntimeError("core group indices do not match declared shards")
    partials = list(corpus.rglob("*.partial"))
    if partials:
        raise RuntimeError(f"unfinished atomic shard files remain: {partials}")
    return {
        "status": "partial_verified" if completed_shards != expected_shards else "verified",
        "deep": bool(deep),
        "corpus": manifest["corpus_name"],
        "manifest_sha256": manifest["manifest_sha256"],
        "verified_file_count": verified_files,
        "completed_shards": completed_shards,
        "shard_count": expected_shards,
        "parameter_count": manifest["core_identity"]["parameter_count"],
        "observable_count": len(manifest["available_observables"]),
        "gpd_truth_point_count": (
            len(gpd_truth["coordinates"]) if isinstance(gpd_truth, dict) else 0
        ),
    }


def export_corpus(*, corpus: Path, archive_path: Path,
                  allow_partial: bool = False) -> dict[str, Any]:
    """Write one portable, verified archive without altering the corpus."""

    verification = verify_corpus(corpus=corpus, deep=True, allow_partial=allow_partial)
    if archive_path.exists():
        raise FileExistsError(f"corpus archive already exists: {archive_path}")
    symlinks = [path for path in corpus.rglob("*") if path.is_symlink()]
    if symlinks:
        raise RuntimeError(f"corpus export refuses symlinks: {symlinks}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive_path.with_name(archive_path.name + ".partial")
    with tarfile.open(temporary, "w:gz") as archive:
        for path in sorted(item for item in corpus.rglob("*") if item.is_file()):
            archive.add(path, arcname=str(path.relative_to(corpus)), recursive=False)
    with tarfile.open(temporary, "r:gz") as archive:
        if not archive.getmembers():
            raise RuntimeError("corpus export archive is empty")
    os.replace(temporary, archive_path)
    return {
        "status": "checkpoint_exported" if allow_partial else "exported",
        "corpus": corpus.name,
        "archive": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "corpus_verification": verification,
    }


def import_corpus(*, archive_path: Path, corpus: Path,
                  allow_partial: bool = False) -> dict[str, Any]:
    """Safely import, rebind, and deeply verify one portable corpus archive."""

    if corpus.exists():
        raise FileExistsError(f"corpus already exists: {corpus}")
    source = archive_path.resolve(strict=True)
    corpus.mkdir(parents=True)
    try:
        with tarfile.open(source, "r:gz") as archive:
            members = archive.getmembers()
            if not members:
                raise RuntimeError("corpus import archive is empty")
            for member in members:
                parts = Path(member.name).parts
                if (
                    member.name.startswith("/") or ".." in parts
                    or member.issym() or member.islnk()
                    or not (member.isfile() or member.isdir())
                ):
                    raise RuntimeError(
                        f"unsafe corpus archive member: {member.name}"
                    )
            archive.extractall(corpus, filter="data")
        manifest_path = corpus / "corpus.json"
        manifest = _load_manifest(manifest_path)
        manifest["corpus_name"] = corpus.name
        manifest["corpus_root"] = str(corpus.resolve())
        manifest["import_provenance"] = {
            "archive_name": source.name,
            "archive_sha256": sha256(source),
        }
        manifest = _write_manifest(manifest_path, manifest)
        verification = verify_corpus(corpus=corpus, deep=True, allow_partial=allow_partial)
    except Exception:
        shutil.rmtree(corpus)
        raise
    return {
        "status": "checkpoint_imported" if allow_partial else "imported",
        "corpus": corpus.name,
        "path": str(corpus),
        "manifest_sha256": manifest["manifest_sha256"],
        "verification": verification,
    }


def _records_have_equal_arrays(
    left_corpus: Path, left: Mapping[str, Any],
    right_corpus: Path, right: Mapping[str, Any],
) -> bool:
    with np.load(left_corpus / left["path"], allow_pickle=False) as left_arrays, \
            np.load(right_corpus / right["path"], allow_pickle=False) as right_arrays:
        if set(left_arrays.files) != set(right_arrays.files):
            return False
        return all(
            np.array_equal(left_arrays[name], right_arrays[name])
            for name in left_arrays.files
        )


def merge_corpora(*, corpora: Sequence[Path], destination: Path,
                  consume_sources: bool = False) -> dict[str, Any]:
    """Merge disjoint, deeply verified shard batches into one complete corpus."""

    if len(corpora) < 2:
        raise ValueError("at least two corpus shard batches are required")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"corpus already exists: {destination}")
    sources = [path.resolve(strict=True) for path in corpora]
    if len(sources) != len(set(sources)):
        raise ValueError("corpus shard batch paths must be distinct")

    manifests: list[dict[str, Any]] = []
    for source in sources:
        if not source.is_dir() or source.is_symlink():
            raise RuntimeError(f"corpus shard batch must be a real directory: {source}")
        symlinks = [path for path in source.rglob("*") if path.is_symlink()]
        if symlinks:
            raise RuntimeError(f"corpus merge refuses symlinks: {symlinks}")
        verify_corpus(corpus=source, deep=True, allow_partial=True)
        manifests.append(_load_manifest(source / "corpus.json"))

    reference = manifests[0]
    identity_fields = (
        "schema_version", "core_identity_sha256", "shard_size", "shard_count",
        "profile_source", "requested_observables", "observable_metadata",
    )
    reference_identity = {field: reference[field] for field in identity_fields}
    reference_configuration = json.loads(
        (sources[0] / reference["configuration"]).read_text(encoding="utf-8")
    )
    reference_physics = json.loads(
        (sources[0] / reference["physics_configuration"]).read_text(encoding="utf-8")
    )
    reference_backend = json.loads(
        (sources[0] / reference["native_backend_provenance"]).read_text(
            encoding="utf-8"
        )
    )
    reference_observables = list(reference["available_observables"])
    reference_gpd = reference.get("gpd_truth")
    if int(reference["schema_version"]) == CORPUS_SCHEMA_VERSION and not isinstance(
        reference_gpd, dict
    ):
        raise RuntimeError("schema-2 corpus shard batch lacks canonical GPD truth")
    if reference["truth"] is None:
        raise RuntimeError("corpus shard batch is missing truth data")

    combined_core: dict[int, dict[str, Any]] = {}
    combined_observables: dict[str, dict[int, dict[str, Any]]] = {
        name: {} for name in reference_observables
    }
    combined_gpd: dict[int, dict[str, Any]] = {}
    shard_sources: dict[int, int] = {}
    for source_index, (source, manifest) in enumerate(zip(sources, manifests)):
        observed_identity = {field: manifest[field] for field in identity_fields}
        if observed_identity != reference_identity:
            raise RuntimeError("corpus shard batch identity mismatch")
        if list(manifest["available_observables"]) != reference_observables:
            raise RuntimeError("corpus shard batch observable mismatch")
        observed_gpd = manifest.get("gpd_truth")
        if isinstance(reference_gpd, dict):
            if not isinstance(observed_gpd, dict):
                raise RuntimeError("corpus shard batch GPD truth mismatch")
            gpd_metadata_keys = (
                "request", "coordinates", "coordinate_table_sha256",
                "value_shape", "dtype", "compression", "status_mask",
                "native_adapter_capability",
            )
            if any(
                observed_gpd.get(key) != reference_gpd.get(key)
                for key in gpd_metadata_keys
            ):
                raise RuntimeError("corpus shard batch GPD truth metadata mismatch")
        if json.loads((source / manifest["configuration"]).read_text()) != reference_configuration:
            raise RuntimeError("corpus shard batch workflow configuration mismatch")
        if json.loads((source / manifest["physics_configuration"]).read_text()) != reference_physics:
            raise RuntimeError("corpus shard batch physics configuration mismatch")
        if json.loads((source / manifest["native_backend_provenance"]).read_text()) != reference_backend:
            raise RuntimeError("corpus shard batch backend provenance mismatch")
        if manifest["truth"] is None or not _records_have_equal_arrays(
            sources[0], reference["truth"]["core"], source, manifest["truth"]["core"]
        ):
            raise RuntimeError("corpus shard batch truth core mismatch")
        for name in reference_observables:
            if not _records_have_equal_arrays(
                sources[0], reference["truth"]["observables"][name],
                source, manifest["truth"]["observables"][name],
            ):
                raise RuntimeError("corpus shard batch truth observable mismatch")
        for position, core_record in enumerate(manifest["core_shards"]):
            shard_index = int(core_record["shard_index"])
            if shard_index in combined_core:
                raise RuntimeError(f"duplicate corpus shard index: {shard_index}")
            combined_core[shard_index] = deepcopy(core_record)
            shard_sources[shard_index] = source_index
            for name in reference_observables:
                combined_observables[name][shard_index] = deepcopy(
                    manifest["available_observables"][name]["shards"][position]
                )
            if isinstance(reference_gpd, dict):
                combined_gpd[shard_index] = deepcopy(
                    observed_gpd["shards"][position]
                )

    shard_count = int(reference["shard_count"])
    if sorted(combined_core) != list(range(shard_count)):
        raise RuntimeError("corpus shard batches do not provide complete disjoint coverage")

    def copy_record(source: Path, record: Mapping[str, Any]) -> None:
        paths = [record["path"]]
        if record.get("native_evidence") is not None:
            paths.append(record["native_evidence"]["path"])
        for relative in paths:
            target = destination / relative
            if target.exists() or target.is_symlink():
                raise RuntimeError(f"merged corpus path collision: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if consume_sources:
                shutil.move(source / relative, target)
            else:
                shutil.copy2(source / relative, target)

    if consume_sources:
        shutil.move(sources[0], destination)
    else:
        shutil.copytree(sources[0], destination)
    try:
        first_indices = {
            int(record["shard_index"]) for record in manifests[0]["core_shards"]
        }
        for shard_index in range(shard_count):
            if shard_index in first_indices:
                continue
            source_index = shard_sources[shard_index]
            source = sources[source_index]
            copy_record(source, combined_core[shard_index])
            for name in reference_observables:
                copy_record(source, combined_observables[name][shard_index])
            if isinstance(reference_gpd, dict):
                copy_record(source, combined_gpd[shard_index])
            rejection = Path("rejections") / f"shard-{shard_index:06d}.json"
            rejection_target = destination / rejection
            if rejection_target.exists() or rejection_target.is_symlink():
                raise RuntimeError(f"merged corpus path collision: {rejection}")
            rejection_target.parent.mkdir(parents=True, exist_ok=True)
            if consume_sources:
                shutil.move(source / rejection, rejection_target)
            else:
                shutil.copy2(source / rejection, rejection_target)

        merged = deepcopy(reference)
        merged.pop("import_provenance", None)
        merged["corpus_name"] = destination.name
        merged["corpus_root"] = str(destination.resolve())
        merged["status"] = "complete"
        merged["core_shards"] = [combined_core[index] for index in range(shard_count)]
        merged["available_observables"] = {
            name: {
                "metadata": deepcopy(reference["available_observables"][name]["metadata"]),
                "shards": [
                    combined_observables[name][index] for index in range(shard_count)
                ],
            }
            for name in reference_observables
        }
        if isinstance(reference_gpd, dict):
            merged["gpd_truth"] = deepcopy(reference_gpd)
            merged["gpd_truth"]["status"] = "complete"
            merged["gpd_truth"]["shards"] = [
                combined_gpd[index] for index in range(shard_count)
            ]
        merged["rejected_candidate_count"] = sum(
            int(manifest["rejected_candidate_count"]) for manifest in manifests
        )
        merged["merge_provenance"] = {
            "source_count": len(sources),
            "sources": [
                {
                    "corpus": manifest["corpus_name"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "shard_indices": [
                        int(record["shard_index"])
                        for record in manifest["core_shards"]
                    ],
                }
                for manifest in manifests
            ],
        }
        merged = _write_manifest(destination / "corpus.json", merged)
        verification = verify_corpus(corpus=destination, deep=True)
        if consume_sources:
            for source in sources[1:]:
                shutil.rmtree(source)
    except Exception:
        shutil.rmtree(destination)
        raise
    return {
        "status": "merged",
        "corpus": destination.name,
        "source_count": len(sources),
        "sources_consumed": bool(consume_sources),
        "shard_count": shard_count,
        "manifest_sha256": merged["manifest_sha256"],
        "verification": verification,
    }


def create_selection(*, corpus: Path, selection_path: Path, profile: str,
                     configuration_path: Path, split_seed: int | None = None) -> dict[str, Any]:
    """Create one immutable group-aware development/outer-test selection."""

    if selection_path.exists():
        raise FileExistsError(f"selection already exists: {selection_path}")
    verify_corpus(corpus=corpus, deep=False)
    manifest = _load_manifest(corpus / "corpus.json")
    config = load_configuration(configuration_path)
    settings = config["profiles"][profile]
    count = int(manifest["core_identity"]["parameter_count"])
    if int(settings["native_parameter_count"]) != count:
        raise ValueError("profile parameter count does not match corpus")
    seed = int(split_seed if split_seed is not None else
               config["seeds"]["training_noise"] + 31001)
    rng = np.random.Generator(np.random.PCG64(seed))
    shuffled = rng.permutation(count)
    outer_count = max(1, int(round(count * float(settings["held_out_fraction"]))))
    development_count = count - outer_count
    validation_count = max(1, int(round(
        development_count * float(config["network"]["validation_fraction"])
    )))
    training_count = development_count - validation_count
    if training_count < 1:
        raise RuntimeError("selection training group is empty")
    selection = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_name": selection_path.stem,
        "corpus_name": manifest["corpus_name"],
        "corpus_core_identity_sha256": manifest["core_identity_sha256"],
        "profile_source": profile,
        "split_policy": "native_parameter_group_random_permutation_v1",
        "split_seed": seed,
        "training_group_indices": sorted(int(v) for v in shuffled[:training_count]),
        "validation_group_indices": sorted(
            int(v) for v in shuffled[training_count:development_count]
        ),
        "outer_test_group_indices": sorted(
            int(v) for v in shuffled[development_count:]
        ),
        "outer_test_locked": True,
        "group_overlap_counts": {
            "train_validation": 0,
            "train_outer_test": 0,
            "validation_outer_test": 0,
        },
        "k_fold_validation": False,
    }
    selected = _write_manifest(selection_path, selection)
    identity_payload = {
        key: value for key, value in selected.items()
        if key not in {"manifest_sha256", "selection_name"}
    }
    selected["selection_id"] = "selection-" + hashlib.sha256(
        canonical(identity_payload).encode()
    ).hexdigest()
    return _write_manifest(selection_path, selected)


def _load_exact_arrays(corpus: Path, manifest: Mapping[str, Any],
                       observable_names: Sequence[str],
                       show_progress: bool | None = None) -> tuple[
                           np.ndarray, np.ndarray, np.ndarray
                       ]:
    from extract_dvcs_cff.progress import progress_iter

    parameters: list[np.ndarray] = []
    cffs: list[np.ndarray] = []
    observable_blocks: dict[str, list[np.ndarray]] = {
        name: [] for name in observable_names
    }
    core_shards = manifest["core_shards"]
    shard_iterator = progress_iter(
        core_shards, total=len(core_shards),
        description="Loading corpus shards", enabled=show_progress,
        unit="shard", leave=True,
    )
    for shard_index, core_record in enumerate(shard_iterator):
        with np.load(corpus / core_record["path"], allow_pickle=False) as core:
            parameters.append(core["native_parameters"])
            cffs.append(core["native_cffs"])
        for name in observable_names:
            extension = manifest["available_observables"].get(name)
            if extension is None:
                raise RuntimeError(f"corpus lacks selected observable {name}")
            with np.load(
                corpus / extension["shards"][shard_index]["path"],
                allow_pickle=False,
            ) as values:
                observable_blocks[name].append(values["values"])
    native_parameters = np.concatenate(parameters)
    native_cffs = np.concatenate(cffs)
    by_observable = {
        name: np.concatenate(blocks)
        for name, blocks in observable_blocks.items()
    }
    native_predictions = np.stack(
        [
            by_observable[name][:, kinematic_index]
            for kinematic_index in range(by_observable[observable_names[0]].shape[1])
            for name in observable_names
        ],
        axis=1,
    )
    return native_parameters, native_predictions, native_cffs


def materialize_selection(*, corpus: Path, selection_path: Path,
                          configuration_path: Path, workspace: Path,
                          profile: str, bridge: Path | None = None,
                          workers: int | str = "all_available",
                          show_progress: bool | None = None) -> dict[str, Any]:
    """Serialize realization publication and reuse an identical result.

    Multi-GPU Optuna starts one CLI process per device.  Every process names
    the same deterministic realization, so only one may publish it while the
    others wait and verify/reuse the completed identity.
    """

    import fcntl

    workspace.mkdir(parents=True, exist_ok=True)
    if workspace.is_symlink():
        raise RuntimeError("realization workspace must not be a symlink")
    lock_path = workspace / ".materialization.lock"
    if lock_path.is_symlink():
        raise RuntimeError("realization lock must not be a symlink")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _materialize_selection_locked(
            corpus=corpus,
            selection_path=selection_path,
            configuration_path=configuration_path,
            workspace=workspace,
            profile=profile,
            bridge=bridge,
            workers=workers,
            show_progress=show_progress,
        )


def _materialize_selection_locked(*, corpus: Path, selection_path: Path,
                                  configuration_path: Path, workspace: Path,
                                  profile: str, bridge: Path | None = None,
                                  workers: int | str = "all_available",
                                  show_progress: bool | None = None) -> dict[str, Any]:
    """Build deterministic neural tensors from clean exact corpus groups."""

    from extract_dvcs_cff.inference.stage10 import (
        prepare_stage10_context, stage10_physical_to_latent,
    )
    from extract_dvcs_cff.native_parallel import available_affinity_cpus
    from extract_dvcs_cff.progress import progress_iter
    import torch

    verification = verify_corpus(corpus=corpus, deep=False)
    manifest = _load_manifest(corpus / "corpus.json")
    bridge_hash = str(manifest["core_identity"]["bridge_sha256"])
    if bridge is not None and sha256(bridge.resolve(strict=True)) != bridge_hash:
        raise RuntimeError("bridge hash differs from the immutable master corpus")
    selection = _load_manifest(selection_path)
    if selection["corpus_name"] != manifest["corpus_name"]:
        raise RuntimeError("selection references another corpus")
    if selection.get("profile_source") != profile:
        raise RuntimeError("selection was created for a different profile")
    if selection["corpus_core_identity_sha256"] != manifest["core_identity_sha256"]:
        raise RuntimeError("corpus physics/core identity changed after selection creation")
    role_groups = {
        "train": selection["training_group_indices"],
        "validation": selection["validation_group_indices"],
        "test": selection["outer_test_group_indices"],
    }
    role_sets = {name: set(int(value) for value in values)
                 for name, values in role_groups.items()}
    count = int(manifest["core_identity"]["parameter_count"])
    if any(not values for values in role_sets.values()):
        raise RuntimeError("selection contains an empty role")
    if (
        role_sets["train"] & role_sets["validation"]
        or role_sets["train"] & role_sets["test"]
        or role_sets["validation"] & role_sets["test"]
    ):
        raise RuntimeError("selection roles overlap")
    if set().union(*role_sets.values()) != set(range(count)):
        raise RuntimeError("selection does not partition every corpus group")
    config = load_configuration(configuration_path)
    if int(config["profiles"][profile]["native_parameter_count"]) != count:
        raise RuntimeError("profile native parameter count differs from corpus")
    observable_names = [item["id"] for item in config["observables"]]
    output = workspace / "generated"
    if output.is_symlink():
        raise RuntimeError("generated realization directory must not be a symlink")
    expected_contract_base = {
        "schema_version": 1,
        "configuration": str(configuration_path.resolve(strict=True)),
        "configuration_sha256": scientific_configuration_sha256(
            configuration_path
        ),
        "profile": profile,
        "bridge_sha256": bridge_hash,
        "real_data": False,
        "synthetic_split_policy": (
            "native_parameter_grouped_train_validation_test_v1"
        ),
    }
    array_manifest_path = output / "array_manifest.json"
    contract_path = workspace / "workspace_contract.json"
    if (
        array_manifest_path.is_file()
        and not array_manifest_path.is_symlink()
        and contract_path.is_file()
        and not contract_path.is_symlink()
    ):
        existing_arrays = json.loads(
            array_manifest_path.read_text(encoding="utf-8")
        )
        existing_contract = json.loads(
            contract_path.read_text(encoding="utf-8")
        )
        if not isinstance(existing_arrays, dict) or not isinstance(
            existing_contract, dict
        ):
            raise RuntimeError("materialized realization manifests are malformed")
        expected_identity = {
            "schema_version": 2,
            "corpus_manifest_sha256": manifest["manifest_sha256"],
            "selection_manifest_sha256": selection["manifest_sha256"],
            "configuration_sha256": scientific_configuration_sha256(
                configuration_path
            ),
            "bridge_sha256": bridge_hash,
            "realization_schema_version": REALIZATION_SCHEMA_VERSION,
        }
        observed_identity = {
            key: existing_arrays.get(key) for key in expected_identity
        }
        expected_contract = {
            **expected_contract_base,
            "corpus_manifest_sha256": manifest["manifest_sha256"],
            "selection_manifest_sha256": selection["manifest_sha256"],
            "realization_schema_version": REALIZATION_SCHEMA_VERSION,
            "realization_manifest_sha256": sha256(array_manifest_path),
        }
        records = existing_arrays.get("arrays")
        context_record = (
            records.get("contexts") if isinstance(records, dict) else None
        )
        complete_files = (
            isinstance(records, dict)
            and bool(records)
            and isinstance(context_record, dict)
            and isinstance(context_record.get("shape"), list)
            and bool(context_record["shape"])
        )
        if complete_files:
            for record in records.values():
                if not isinstance(record, dict):
                    complete_files = False
                    break
                relative = Path(str(record.get("path", "")))
                path = output / relative
                if (
                    relative.name != str(relative)
                    or path.is_symlink()
                    or not path.is_file()
                ):
                    complete_files = False
                    break
        if (
            observed_identity == expected_identity
            and existing_contract == expected_contract
            and complete_files
            and not any(output.glob("*.partial"))
        ):
            from extract_dvcs_cff.contracts import publish_layered_materialization

            layered = publish_layered_materialization(
                workspace=workspace, master_corpus_manifest=manifest,
                selection_manifest=selection, configuration=config,
            )
            _atomic_json(workspace / "materialization_progress.json", {
                "schema_version": 1,
                "phase": "complete",
                "completed": 1,
                "total": 1,
                "status": "reused",
            })
            return {
                "status": "reused",
                "native_called": False,
                "corpus_manifest_sha256": manifest["manifest_sha256"],
                "selection_manifest_sha256": selection["manifest_sha256"],
                "context_count": int(
                    existing_arrays["arrays"]["contexts"]["shape"][0]
                ),
                **layered,
            }
    parameters, native_predictions, native_cffs = _load_exact_arrays(
        corpus, manifest, observable_names, show_progress
    )
    with np.load(corpus / manifest["truth"]["core"]["path"], allow_pickle=False) as truth:
        truth_parameters = truth["parameters"][0]
        truth_cffs = truth["cffs"][0]
    truth_by_observable = {}
    for name in observable_names:
        record = manifest["truth"]["observables"].get(name)
        if record is None:
            raise RuntimeError(f"corpus lacks truth observable {name}")
        with np.load(corpus / record["path"], allow_pickle=False) as truth:
            truth_by_observable[name] = truth["values"][0]
    truth_predictions = np.asarray([
        truth_by_observable[name][kinematic_index]
        for kinematic_index in range(len(config["kinematics"]))
        for name in observable_names
    ])
    covariance, global_response, lu_response, covariance_diagnostics = (
        _covariance_and_responses(config, truth_predictions)
    )
    cholesky = np.linalg.cholesky(covariance)
    point_table = _point_table(config)
    roles = (
        ("train", selection["training_group_indices"]),
        ("validation", selection["validation_group_indices"]),
        ("test", selection["outer_test_group_indices"]),
    )
    replicates = int(config["profiles"][profile]["noise_replicates_per_parameter"])
    group_count = sum(len(group_indices) for _, group_indices in roles)
    available_workers = len(available_affinity_cpus())
    if workers == "all_available":
        materialization_workers = min(available_workers, group_count)
    elif isinstance(workers, int) and not isinstance(workers, bool):
        if not 1 <= workers <= 256:
            raise ValueError("materialization workers must be in [1,256]")
        materialization_workers = min(workers, available_workers, group_count)
    else:
        raise ValueError(
            "materialization workers must be an integer or 'all_available'"
        )
    context_count = group_count * replicates
    theta_physical = np.empty(
        (context_count, parameters.shape[1] + 2), dtype=np.float64
    )
    observations_normalized = np.empty(
        (context_count, len(point_table)), dtype=np.float64
    )
    full_context_builder = prepare_stage10_context(
        points=point_table,
        covariance=covariance,
        global_normalization_response=global_response,
        lu_normalization_response=lu_response,
    )
    contexts = np.empty(
        (context_count, full_context_builder.context_width), dtype=np.float32
    )
    active_kinematic_counts = np.empty(context_count, dtype=np.uint32)
    design_variants = np.empty(context_count, dtype=np.uint16)
    parameter_indices = np.empty(context_count, dtype=np.int64)
    role_codes = np.empty(context_count, dtype=np.uint8)
    role_code = {"train": 0, "validation": 1, "test": 2}
    design_contract = config["observation_design_training"]
    design_counts = (
        tuple(int(value) for value in design_contract["active_kinematic_counts"])
        if design_contract["enabled"]
        else (len(config["kinematics"]),)
    )
    observable_count = len(observable_names)
    design_indices: dict[tuple[int, int], np.ndarray] = {}
    design_builders: dict[tuple[int, int], Any] = {}
    design_cholesky: dict[tuple[int, int], np.ndarray] = {}
    design_kinematics: dict[tuple[int, int], tuple[int, ...]] = {}
    for active_count in design_counts:
        key = (0, active_count)
        if design_contract["enabled"]:
            ordering = _nested_kinematic_order(
                config["kinematics"],
                int(design_contract["selection_seed"]),
            )
            retained_kinematics = ordering[:active_count]
        else:
            retained_kinematics = tuple(range(len(config["kinematics"])))
        indices = np.asarray([
            kinematic_index * observable_count + observable_index
            for kinematic_index in retained_kinematics
            for observable_index in range(observable_count)
        ], dtype=np.int64)
        active_covariance = covariance[np.ix_(indices, indices)]
        design_indices[key] = indices
        design_kinematics[key] = retained_kinematics
        design_cholesky[key] = np.linalg.cholesky(active_covariance)
        design_builders[key] = prepare_stage10_context(
            points=[point_table[index] for index in indices],
            covariance=active_covariance,
            global_normalization_response=global_response[indices],
            lu_normalization_response=lu_response[indices],
            maximum_point_count=len(point_table),
        )
    progress_path = workspace / "materialization_progress.json"
    _atomic_json(progress_path, {
        "schema_version": 1, "phase": "contexts",
        "completed": 0, "total": group_count, "status": "running",
        "workers": materialization_workers,
    })
    noise_seed = int(config["seeds"]["training_noise"])
    ordered_groups = tuple(
        (role, int(group_index))
        for role, indices in roles
        for group_index in indices
    )
    group_items = tuple(
        (position, role, group_index)
        for position, (role, group_index) in enumerate(ordered_groups)
    )

    def materialize_group(item: tuple[int, str, int]) -> None:
        position, role, group_index = item
        row_start = position * replicates
        for replica in range(replicates):
            row_index = row_start + replica
            rng = np.random.Generator(np.random.PCG64(
                np.random.SeedSequence([
                    noise_seed, group_index, replica,
                    REALIZATION_SCHEMA_VERSION,
                ])
            ))
            eta_global, eta_lu = rng.standard_normal(2)
            theta_physical[row_index] = (
                *parameters[group_index], eta_global, eta_lu,
            )
            mean = _effective_prediction(
                native_predictions[group_index], eta_global, eta_lu,
                global_response, lu_response,
            )
            # Rotate count assignment by group. Validation's four replicas
            # expose every [12,30,60,96] design to each parameter; profiles
            # with fewer replicas remain balanced across parameter groups.
            design_key = (
                0,
                _masked_design_count(design_counts, position, replica),
            )
            indices = design_indices[design_key]
            observed = mean[indices] + design_cholesky[
                design_key
            ] @ rng.standard_normal(len(indices))
            observations_normalized[row_index].fill(0.0)
            observations_normalized[row_index, indices] = observed
            contexts[row_index] = design_builders[design_key].build(observed)
            design_variants[row_index] = design_key[0]
            active_kinematic_counts[row_index] = design_key[1]
            parameter_indices[row_index] = group_index
            role_codes[row_index] = role_code[role]

    def record_progress(results: Any) -> None:
        iterator = progress_iter(
            results, total=group_count,
            description="Materializing training contexts",
            enabled=show_progress, unit="parameter", leave=True,
        )
        for completed, _ in enumerate(iterator, 1):
            if completed % update_interval == 0 or completed == group_count:
                _atomic_json(progress_path, {
                    "schema_version": 1, "phase": "contexts",
                    "completed": completed, "total": group_count,
                    "status": "running", "workers": materialization_workers,
                })

    update_interval = max(1, group_count // 100)
    if materialization_workers == 1:
        record_progress(map(materialize_group, group_items))
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=materialization_workers) as executor:
            record_progress(executor.map(materialize_group, group_items))
    theta_latent = stage10_physical_to_latent(
        torch.from_numpy(theta_physical)
    ).numpy()
    train_indices = np.flatnonzero(role_codes == role_code["train"])
    validation_indices = np.flatnonzero(role_codes == role_code["validation"])
    test_indices = np.flatnonzero(role_codes == role_code["test"])
    pseudo_rng = np.random.Generator(np.random.PCG64(
        int(config["seeds"]["pseudodata"])
    ))
    truth_theta = np.concatenate((
        truth_parameters,
        [config["truth"]["eta_global_normalization"],
         config["truth"]["eta_lu_normalization"]],
    ))
    pseudo_mean = _effective_prediction(
        truth_predictions, truth_theta[-2], truth_theta[-1],
        global_response, lu_response,
    )
    pseudo_observed = pseudo_mean + cholesky @ pseudo_rng.standard_normal(
        len(point_table)
    )
    pseudo_context = full_context_builder.build(pseudo_observed)
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "observation_design_manifest.json", {
        "schema_version": 1,
        "enabled": bool(design_contract["enabled"]),
        "pooling_policy": design_contract["pooling_policy"],
        "selection_policy": design_contract["selection_policy"],
        "selection_seed": int(design_contract["selection_seed"]),
        "maximum_kinematic_count": len(config["kinematics"]),
        "maximum_token_count": len(point_table),
        "assignment_policy": (
            "active_counts[(parameter_group_position+replica_index)%count]"
        ),
        "noise_replicates_per_parameter": replicates,
        "designs": [
            {"variant": key[0], "active_kinematic_count": key[1],
             "kinematic_indices": list(design_kinematics[key])}
            for key in sorted(design_kinematics)
        ],
    })
    arrays = {
        "native_parameters": parameters,
        "native_predictions_normalized": native_predictions,
        "native_cffs": native_cffs,
        "theta_physical": theta_physical,
        "theta_latent": theta_latent,
        "observations_normalized": observations_normalized,
        "contexts": contexts,
        "covariance_normalized": covariance,
        "global_normalization_response": global_response,
        "lu_normalization_response": lu_response,
        "parameter_indices": parameter_indices,
        "context_active_kinematic_counts": active_kinematic_counts,
        "context_design_variants": design_variants,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "pseudodata_observed_normalized": pseudo_observed,
        "pseudodata_context": pseudo_context,
        "truth_theta_physical": truth_theta,
        "truth_cffs": truth_cffs,
        "truth_native_predictions_normalized": truth_predictions,
    }
    array_records: dict[str, Any] = {}
    _atomic_json(progress_path, {
        "schema_version": 1, "phase": "arrays",
        "completed": 0, "total": len(arrays), "status": "running",
    })
    array_iterator = progress_iter(
        arrays.items(), total=len(arrays),
        description="Writing training arrays", enabled=show_progress,
        unit="array", leave=True,
    )
    for array_index, (name, value) in enumerate(array_iterator, 1):
        path = output / f"{name}.npy"
        with tempfile.NamedTemporaryFile(dir=output, delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        array_records[name] = {
            "path": path.name, "sha256": sha256(path),
            "shape": list(value.shape), "dtype": str(value.dtype),
        }
        _atomic_json(progress_path, {
            "schema_version": 1, "phase": "arrays",
            "completed": array_index, "total": len(arrays),
            "status": "running",
        })
    _atomic_json(output / "array_manifest.json", {
        "schema_version": 2,
        "arrays": array_records,
        "corpus_manifest_sha256": manifest["manifest_sha256"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "configuration_sha256": scientific_configuration_sha256(
            configuration_path
        ),
        "bridge_sha256": bridge_hash,
        "realization_schema_version": REALIZATION_SCHEMA_VERSION,
    })
    _atomic_json(output / "generation_metrics.json", {
        "schema_version": 2,
        "profile": profile,
        "native_parameter_count": len(parameters),
        "training_context_count": len(theta_physical),
        "train_context_count": len(train_indices),
        "validation_context_count": len(validation_indices),
        "test_context_count": len(test_indices),
        "native_cache_hits": 0,
        "native_cache_misses": 0,
        "native_called": False,
        "corpus": verification,
        "selection_manifest_sha256": selection["manifest_sha256"],
        "materialization_workers": materialization_workers,
        "observation_design_training": {
            "enabled": bool(design_contract["enabled"]),
            "active_kinematic_counts": list(design_counts),
            "pooling_policy": design_contract["pooling_policy"],
            "manifest": "observation_design_manifest.json",
        },
        "covariance_diagnostics": covariance_diagnostics,
        "synthetic_split_policy": "external_native_parameter_group_selection_v1",
        "surrogate_used": False,
    })
    scales = {
        item["id"]: item["normalization_scale"]
        for item in config["observables"]
    }
    _atomic_json(output / "pseudodata.json", {
        "schema_version": 2,
        "dataset_id": "reusable-corpus-realization-v1",
        "synthetic": True,
        "real_data": False,
        "generator": "verified_exact_PARTONS_corpus_plus_declared_systematics",
        "point_order": point_table,
        "observable_normalization_scales": scales,
        "observed_normalized": pseudo_observed.tolist(),
        "observed_native": [
            float(pseudo_observed[index] * scales[point["observable_id"]])
            for index, point in enumerate(point_table)
        ],
        "truth_prediction_normalized": truth_predictions.tolist(),
        "covariance_normalized": covariance.tolist(),
        "covariance_diagnostics": covariance_diagnostics,
        "nuisance_groups": config["experimental_model"]["nuisance_groups"],
        "truth": config["truth"],
        "seed": config["seeds"]["pseudodata"],
        "corpus_manifest_sha256": manifest["manifest_sha256"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "backend_bridge_sha256": bridge_hash,
        "analysis_order_label": None,
    })
    _atomic_json(output / "invalid_simulation_map.json", {
        "schema_version": 2,
        "candidate_count": manifest["core_identity"]["parameter_count"]
                           + manifest["rejected_candidate_count"],
        "requested_valid_count": manifest["core_identity"]["parameter_count"],
        "valid_count": manifest["core_identity"]["parameter_count"],
        "invalid_count": manifest["rejected_candidate_count"],
        "invalid_records_location": "corpus/rejections/*.json",
        "policy": "exact_native_rejection_recorded_in_verified_corpus",
        "corpus_manifest_sha256": manifest["manifest_sha256"],
    })
    expected_contract = {
        **expected_contract_base,
        "corpus_manifest_sha256": manifest["manifest_sha256"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "realization_schema_version": REALIZATION_SCHEMA_VERSION,
        "realization_manifest_sha256": sha256(array_manifest_path),
    }
    _atomic_json(contract_path, expected_contract)
    _atomic_json(progress_path, {
        "schema_version": 1, "phase": "complete",
        "completed": 1, "total": 1, "status": "materialized",
        "workers": materialization_workers,
    })
    from extract_dvcs_cff.contracts import publish_layered_materialization

    layered = publish_layered_materialization(
        workspace=workspace, master_corpus_manifest=manifest,
        selection_manifest=selection, configuration=config,
    )
    return {
        "status": "materialized",
        "native_called": False,
        "corpus_manifest_sha256": manifest["manifest_sha256"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "context_count": len(theta_physical),
        "materialization_workers": materialization_workers,
        **layered,
    }
