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
    _physics_path,
    _point_table,
    _sample_shape_parameters,
    _shape_parameter_name,
    _shape_vector_from_shapes,
    canonical,
    load_configuration,
    pretty_json,
    sha256,
)


CORPUS_SCHEMA_VERSION = 1
SELECTION_SCHEMA_VERSION = 1
REALIZATION_SCHEMA_VERSION = 1
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


def _atomic_npz(path: Path, **arrays: np.ndarray) -> dict[str, Any]:
    """Write, reopen, validate, hash, and atomically publish one shard."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        temporary.unlink()
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **{name: np.asarray(value) for name, value in arrays.items()},
        )
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
                   profile: str, bridge_hash: str) -> dict[str, Any]:
    settings = config["profiles"][profile]
    physics_core = deepcopy(physics)
    physics_core["theory_configuration"].pop("observable_modules", None)
    return {
        "representation": config["representation"],
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


def _observable_metadata(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["id"]: deepcopy(item) for item in config["observables"]}


def create_corpus(*, corpus: Path, bridge: Path, configuration_path: Path,
                  profile: str, shard_size: int = DEFAULT_SHARD_SIZE) -> dict[str, Any]:
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
    core = _core_identity(config, physics, profile, bridge_hash)
    core_hash = hashlib.sha256(canonical(core).encode()).hexdigest()
    corpus.mkdir(parents=True)
    (corpus / "configuration").mkdir()
    _atomic_json(corpus / "configuration" / "workflow.json", config)
    _atomic_json(corpus / "configuration" / "physics.json", physics)
    capabilities = subprocess.run(
        [str(bridge_path), "--capabilities"], text=True,
        capture_output=True, check=False,
    )
    if capabilities.returncode != 0:
        raise RuntimeError(
            f"bridge capabilities failed: {capabilities.stderr.strip()}"
        )
    backend = json.loads(capabilities.stdout)
    _atomic_json(corpus / "backend_provenance.json", backend)
    count = int(core["parameter_count"])
    manifest = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "status": "planned",
        "corpus_name": corpus.name,
        "corpus_root": str(corpus.resolve()),
        "core_identity_sha256": core_hash,
        "core_identity": core,
        "profile_source": profile,
        "shard_size": int(shard_size),
        "shard_count": math.ceil(count / int(shard_size)),
        "available_observables": {},
        "requested_observables": list(_observable_metadata(config)),
        "observable_metadata": deepcopy(ADMITTED_OBSERVABLE_METADATA),
        "core_shards": [],
        "truth": None,
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


def generate_corpus(*, corpus: Path, bridge: Path,
                    requested_observables: Sequence[str] | None = None,
                    show_progress: bool | None = None,
                    force_native: bool = False) -> dict[str, Any]:
    """Generate missing core shards or append missing observable extensions."""

    manifest_path = corpus / "corpus.json"
    manifest = _load_manifest(manifest_path)
    if manifest["schema_version"] != CORPUS_SCHEMA_VERSION:
        raise RuntimeError("unsupported corpus schema")
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
    core_complete = len(manifest["core_shards"]) == manifest["shard_count"]
    if core_complete and not missing:
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
        if manifest["core_shards"]:
            start_shard = len(manifest["core_shards"])
        else:
            start_shard = 0
        count = int(manifest["core_identity"]["parameter_count"])
        shard_size = int(manifest["shard_size"])
        combined_config = _configuration_for_observables(
            config, metadata, names
        )
        for shard_index in range(start_shard, manifest["shard_count"]):
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
                extension = manifest["available_observables"].setdefault(
                    name, {"metadata": metadata[name], "shards": []}
                )
                extension["shards"].append(observable_record)
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
        core_complete = True
        missing = []

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
                             bridge: Path) -> dict[str, Any]:
    """Require unchanged GPD physics, parameters, kinematics, and backend."""

    manifest = _load_manifest(corpus / "corpus.json")
    config = load_configuration(configuration_path)
    profile = manifest["profile_source"]
    physics_path = _physics_path(configuration_path, config)
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    identity = _core_identity(
        config, physics, profile, sha256(bridge.resolve(strict=True))
    )
    observed = hashlib.sha256(canonical(identity).encode()).hexdigest()
    if observed != manifest["core_identity_sha256"]:
        raise RuntimeError(
            "corpus core mismatch: GPD physics, priors/count, kinematics, "
            "parameter seed, or native bridge changed; create a new corpus"
        )
    return manifest


def verify_corpus(*, corpus: Path, deep: bool = False) -> dict[str, Any]:
    """Verify manifest structure and optionally every shard byte and array."""

    manifest = _load_manifest(corpus / "corpus.json")
    expected_shards = int(manifest["shard_count"])
    if len(manifest["core_shards"]) != expected_shards:
        raise RuntimeError("corpus core is incomplete")
    seen: list[int] = []
    verified_files = 0
    records = list(manifest["core_shards"])
    for extension in manifest["available_observables"].values():
        if len(extension["shards"]) != expected_shards:
            raise RuntimeError("observable extension is incomplete")
        records.extend(extension["shards"])
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
        expected = list(range(int(manifest["core_identity"]["parameter_count"])))
        if seen != expected:
            raise RuntimeError("core group indices are not unique and contiguous")
    partials = list(corpus.rglob("*.partial"))
    if partials:
        raise RuntimeError(f"unfinished atomic shard files remain: {partials}")
    return {
        "status": "verified",
        "deep": bool(deep),
        "corpus": manifest["corpus_name"],
        "manifest_sha256": manifest["manifest_sha256"],
        "verified_file_count": verified_files,
        "parameter_count": manifest["core_identity"]["parameter_count"],
        "observable_count": len(manifest["available_observables"]),
    }


def export_corpus(*, corpus: Path, archive_path: Path) -> dict[str, Any]:
    """Write one portable, verified archive without altering the corpus."""

    verification = verify_corpus(corpus=corpus, deep=True)
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
        "status": "exported",
        "corpus": corpus.name,
        "archive": str(archive_path),
        "archive_sha256": sha256(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "corpus_verification": verification,
    }


def import_corpus(*, archive_path: Path, corpus: Path) -> dict[str, Any]:
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
        verification = verify_corpus(corpus=corpus, deep=True)
    except Exception:
        shutil.rmtree(corpus)
        raise
    return {
        "status": "imported",
        "corpus": corpus.name,
        "path": str(corpus),
        "manifest_sha256": manifest["manifest_sha256"],
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
    return _write_manifest(selection_path, selection)


def _load_exact_arrays(corpus: Path, manifest: Mapping[str, Any],
                       observable_names: Sequence[str]) -> tuple[
                           np.ndarray, np.ndarray, np.ndarray
                       ]:
    parameters: list[np.ndarray] = []
    cffs: list[np.ndarray] = []
    observable_blocks: dict[str, list[np.ndarray]] = {
        name: [] for name in observable_names
    }
    for shard_index, core_record in enumerate(manifest["core_shards"]):
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
                          profile: str, bridge: Path) -> dict[str, Any]:
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
        )


def _materialize_selection_locked(*, corpus: Path, selection_path: Path,
                                  configuration_path: Path, workspace: Path,
                                  profile: str, bridge: Path) -> dict[str, Any]:
    """Build deterministic neural tensors from clean exact corpus groups."""

    from extract_dvcs_cff.inference.stage10 import (
        build_stage10_context, stage10_physical_to_latent,
    )
    import torch

    verification = verify_corpus(corpus=corpus, deep=False)
    manifest = _load_manifest(corpus / "corpus.json")
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
        "configuration_sha256": sha256(configuration_path.resolve(strict=True)),
        "profile": profile,
        "bridge_sha256": sha256(bridge.resolve(strict=True)),
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
            "configuration_sha256": sha256(configuration_path),
            "bridge_sha256": sha256(bridge.resolve(strict=True)),
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
            return {
                "status": "reused",
                "native_called": False,
                "corpus_manifest_sha256": manifest["manifest_sha256"],
                "selection_manifest_sha256": selection["manifest_sha256"],
                "context_count": int(
                    existing_arrays["arrays"]["contexts"]["shape"][0]
                ),
            }
    parameters, native_predictions, native_cffs = _load_exact_arrays(
        corpus, manifest, observable_names
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
    theta_rows: list[np.ndarray] = []
    observations: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    parameter_indices: list[int] = []
    row_roles: list[str] = []
    noise_seed = int(config["seeds"]["training_noise"])
    for role, group_indices in roles:
        for group_index in group_indices:
            for replica in range(replicates):
                rng = np.random.Generator(np.random.PCG64(
                    np.random.SeedSequence([
                        noise_seed, int(group_index), replica,
                        REALIZATION_SCHEMA_VERSION,
                    ])
                ))
                eta_global, eta_lu = rng.standard_normal(2)
                theta = np.concatenate((
                    parameters[group_index], [eta_global, eta_lu]
                ))
                mean = _effective_prediction(
                    native_predictions[group_index], eta_global, eta_lu,
                    global_response, lu_response,
                )
                observed = mean + cholesky @ rng.standard_normal(len(point_table))
                context = build_stage10_context(
                    points=point_table, observed=observed,
                    covariance=covariance,
                    global_normalization_response=global_response,
                    lu_normalization_response=lu_response,
                )
                theta_rows.append(theta)
                observations.append(observed)
                contexts.append(context)
                parameter_indices.append(int(group_index))
                row_roles.append(role)
    theta_physical = np.asarray(theta_rows, dtype=np.float64)
    theta_latent = stage10_physical_to_latent(
        torch.from_numpy(theta_physical)
    ).numpy()
    role_array = np.asarray(row_roles)
    train_indices = np.flatnonzero(role_array == "train")
    validation_indices = np.flatnonzero(role_array == "validation")
    test_indices = np.flatnonzero(role_array == "test")
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
    pseudo_context = build_stage10_context(
        points=point_table, observed=pseudo_observed,
        covariance=covariance,
        global_normalization_response=global_response,
        lu_normalization_response=lu_response,
    )
    output.mkdir(parents=True, exist_ok=True)
    arrays = {
        "native_parameters": parameters,
        "native_predictions_normalized": native_predictions,
        "native_cffs": native_cffs,
        "theta_physical": theta_physical,
        "theta_latent": theta_latent,
        "observations_normalized": np.asarray(observations),
        "contexts": np.asarray(contexts, dtype=np.float32),
        "covariance_normalized": covariance,
        "global_normalization_response": global_response,
        "lu_normalization_response": lu_response,
        "parameter_indices": np.asarray(parameter_indices, dtype=np.int64),
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
    for name, value in arrays.items():
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
    _atomic_json(output / "array_manifest.json", {
        "schema_version": 2,
        "arrays": array_records,
        "corpus_manifest_sha256": manifest["manifest_sha256"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "configuration_sha256": sha256(configuration_path),
        "bridge_sha256": sha256(bridge.resolve(strict=True)),
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
        "backend_bridge_sha256": sha256(bridge.resolve(strict=True)),
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
    return {
        "status": "materialized",
        "native_called": False,
        "corpus_manifest_sha256": manifest["manifest_sha256"],
        "selection_manifest_sha256": selection["manifest_sha256"],
        "context_count": len(theta_physical),
    }
