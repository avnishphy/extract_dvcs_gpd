"""Auditable Stage 10 pseudodata, NPE, and comparison workflow.

The orchestration in this module never evaluates a physics formula. Native
predictions come only from the versioned C++ bridge. Python adds declared
experimental covariance, nuisance shifts, random noise, neural posterior
training, and validation. Exact native responses are content-addressed and
may be reused only after every request byte and the bridge executable hash
match.
"""

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from extract_dvcs_cff.native_parallel import resolve_native_workers


# A monolithic 4,096-parameter bridge request can run for nearly an hour
# without returning control to Python.  Two-parameter requests are the
# measured Stage 11 parallel unit: 22 isolated workers achieved an 8.12x
# speedup while preserving every retained array bit-for-bit.  They also return
# control frequently enough for the interactive counter to advance.  This is
# an orchestration constant, not a physics parameter.
_NATIVE_GENERATION_CHUNK_SIZE = 2

# Submit at least this many parameters per wave. Larger CPU allocations scale
# the wave to one two-parameter task per usable worker, while the incremental
# acceptance loop still avoids enqueueing an entire validation corpus.
_NATIVE_GENERATION_MIN_WAVE_SIZE = 64


def _native_generation_wave_size(
    native_workers: int | str,
    *,
    remaining: int,
    available_candidates: int,
) -> int:
    """Size a generation wave to keep every usable native worker busy."""

    worker_capacity = resolve_native_workers(native_workers).resolved_workers
    target = max(
        _NATIVE_GENERATION_MIN_WAVE_SIZE,
        worker_capacity * _NATIVE_GENERATION_CHUNK_SIZE,
    )
    return min(remaining, target, available_candidates)

# Physics-informed PARTONS models are external validation datasets only. They
# must never be introduced into DD corpus generation or model selection.
_NATIVE_VALIDATION_MODELS = (
    "GPDGK11",
    "GPDGK16",
    "GPDGK19",
    "GPDVGG99",
)
_NATIVE_VALIDATION_SEED_OFFSETS = {
    # Preserve the already published GK16/VGG99 fixtures. New models receive
    # new fixed offsets; list insertion or display reordering changes nothing.
    "GPDGK11": {"noise": 11002, "posterior": 12200},
    "GPDGK16": {"noise": 11000, "posterior": 12000},
    "GPDGK19": {"noise": 11003, "posterior": 12300},
    "GPDVGG99": {"noise": 11001, "posterior": 12100},
}

_GPD_TYPES = ("H", "E", "Htilde", "Etilde")
_PARTON_CHANNELS = ("u", "d", "s", "gluon")
_DD_SHAPE_FIELDS = (
    "normalization",
    "a",
    "c",
    "profile_b",
    "t_slope_GeV_minus2",
)
_DD_SHAPE_BOUNDS = {
    "normalization": (-4.0, 4.0),
    "a": (0.1, 1.0),
    "c": (2.0, 6.0),
    "profile_b": (1.0, 4.0),
    "t_slope_GeV_minus2": (0.0, 2.0),
}
_PHYSICS_PARAMETER_COUNT = (
    len(_GPD_TYPES) * len(_PARTON_CHANNELS) * len(_DD_SHAPE_FIELDS)
)
_POSTERIOR_PARAMETER_COUNT = _PHYSICS_PARAMETER_COUNT + 2
ADMITTED_OBSERVABLE_IDS = (
    "DVCSCrossSectionUUMinus",
    "DVCSCrossSectionDifferenceLUMinus",
    "DVCSAc",
    "DVCSAluMinus",
    "DVCSAulMinus",
    "DVCSAllMinus",
)


def _shape_parameter_name(gpd_type: str, channel: str, field: str) -> str:
    return f"{gpd_type}_{channel}_{field}"


def _shape_vector_from_shapes(shapes: Mapping[str, Any]) -> np.ndarray:
    """Serialize complete independent DD shapes in stable wire order."""

    return np.asarray(
        [
            float(
                shapes[gpd_type][channel][
                    "t_slope"
                    if field == "t_slope_GeV_minus2"
                    else field
                ]["value"]
                if field == "t_slope_GeV_minus2"
                else shapes[gpd_type][channel][field]
            )
            for gpd_type in _GPD_TYPES
            for channel in _PARTON_CHANNELS
            for field in _DD_SHAPE_FIELDS
        ],
        dtype=np.float64,
    )


def _sample_shape_parameters(
    rng: np.random.Generator, count: int
) -> np.ndarray:
    """Draw all 80 DD controls independently from their declared priors."""

    columns = []
    epsilon = 1.0e-6
    for _gpd_type in _GPD_TYPES:
        for _channel in _PARTON_CHANNELS:
            for field in _DD_SHAPE_FIELDS:
                lower, upper = _DD_SHAPE_BOUNDS[field]
                columns.append(
                    rng.uniform(lower + epsilon, upper - epsilon, count)
                )
    return np.column_stack(columns)


def canonical(value: Any) -> str:
    """Serialize deterministic JSON and reject NaN/Infinity."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def pretty_json(value: Any) -> str:
    """Serialize deterministic, human-readable JSON for persisted records."""

    return json.dumps(
        value,
        sort_keys=True,
        indent=4,
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256(path: Path) -> str:
    """Return the SHA-256 of one real file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Write a stable UTF-8 JSON record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty_json(value) + "\n", encoding="utf-8")


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"{context} key mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def load_configuration(path: Path) -> dict[str, Any]:
    """Load and fail-close the Stage 10 workflow's top-level contract."""

    config = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    _exact_keys(
        config,
        {
            "schema_version",
            "workflow",
            "physics_configuration",
            "bridge_schema_version",
            "representation",
            "real_data",
            "data_provenance",
            "runtime",
            "truth",
            "kinematics",
            "observables",
            "diagnostics",
            "experimental_model",
            "profiles",
            "context_contract",
            "network",
            "hyperparameter_optimization",
            "seeds",
            "frozen_validation_gates",
        },
        "configuration",
    )
    if (
        config["schema_version"] != 8
        or config["bridge_schema_version"] != 1
        or config["workflow"]
        != "stage11_full_independent_dd_multiq2_pseudodata_v1"
        or config["representation"]
        != "stage11_lo_multiq2_full_independent_dd_v1"
        or config["real_data"] is not False
    ):
        raise ValueError("configuration is not the frozen synthetic workflow")
    if set(config["profiles"]) != {"quick", "validation"}:
        raise ValueError("profiles must be exactly quick and validation")
    provenance = config["data_provenance"]
    if not isinstance(provenance, dict) or provenance.get("mode") not in {
        "manual",
        "gpddatabase_native_scale_kinematics_v1",
        "gpddatabase_native_scale_kinematics_v2",
    }:
        raise ValueError("unsupported data_provenance mode")
    if (
        provenance["mode"] in {
            "gpddatabase_native_scale_kinematics_v1",
            "gpddatabase_native_scale_kinematics_v2",
        }
        and provenance.get("measurements_quarantined") is not True
    ):
        raise ValueError("real measurements must remain quarantined")
    _exact_keys(
        config["frozen_validation_gates"],
        {
            "minimum_conventional_effective_sample_size",
            "maximum_ensemble_sliced_wasserstein",
            "maximum_marginal_wasserstein_over_conventional_sd",
            "maximum_test_nll_minus_validation_nll",
            "maximum_coverage_standard_error",
            "minimum_posterior_predictive_point_90_coverage",
            "minimum_neural_to_conventional_stress_width_ratio",
            "maximum_exact_reevaluation_invalid_fraction",
        },
        "frozen_validation_gates",
    )
    _exact_keys(
        config["runtime"],
        {
            "accelerator",
            "cpu_threads",
            "deterministic_algorithms",
            "native_workers",
        },
        "runtime",
    )
    if config["runtime"]["accelerator"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("runtime.accelerator must be auto, cpu, or cuda")
    if not isinstance(config["runtime"]["cpu_threads"], int) or not (
        1 <= config["runtime"]["cpu_threads"] <= 256
    ):
        raise ValueError("runtime.cpu_threads must be an integer in [1,256]")
    if config["runtime"]["deterministic_algorithms"] is not True:
        raise ValueError("this release requires deterministic algorithms")
    resolve_native_workers(config["runtime"]["native_workers"])
    optimization = config["hyperparameter_optimization"]
    _exact_keys(
        optimization,
        {
            "package",
            "sampler",
            "objective",
            "trials",
            "pruner",
            "pruner_startup_trials",
            "pruner_warmup_epochs",
            "search_space",
        },
        "hyperparameter_optimization",
    )
    if (
        optimization["package"] != "Optuna"
        or optimization["sampler"] != "TPESampler"
        or optimization["pruner"] != "MedianPruner"
        or optimization["objective"]
        != "best_validation_negative_log_density"
    ):
        raise ValueError("unsupported hyperparameter optimization contract")
    _exact_keys(
        optimization["trials"],
        {"quick", "validation"},
        "hyperparameter_optimization.trials",
    )
    for profile, count in optimization["trials"].items():
        if not isinstance(count, int) or not 1 <= count <= 1000:
            raise ValueError(
                f"hyperparameter_optimization.trials.{profile} must be an "
                "integer in [1,1000]"
            )
    for name in ("pruner_startup_trials", "pruner_warmup_epochs"):
        value = optimization[name]
        if not isinstance(value, int) or not 0 <= value <= 1000:
            raise ValueError(f"hyperparameter_optimization.{name} invalid")
    search = optimization["search_space"]
    categorical = {
        "point_hidden",
        "dataset_hidden",
        "embedding_features",
        "flow_hidden_features",
        "num_transforms",
        "num_bins",
        "training_batch_size",
    }
    _exact_keys(
        search,
        categorical | {"learning_rate"},
        "hyperparameter_optimization.search_space",
    )
    for name in categorical:
        choices = search[name]
        if (
            not isinstance(choices, list)
            or not choices
            or any(not isinstance(item, int) or item <= 0 for item in choices)
            or len(choices) != len(set(choices))
        ):
            raise ValueError(
                f"search_space.{name} must contain unique positive integers"
            )
    learning_rate = search["learning_rate"]
    if (
        not isinstance(learning_rate, list)
        or len(learning_rate) != 2
        or any(
            not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in learning_rate
        )
        or not 0.0 < float(learning_rate[0]) < float(learning_rate[1])
    ):
        raise ValueError(
            "search_space.learning_rate must be [positive_min, larger_max]"
        )
    for profile, settings in config["profiles"].items():
        seeds = settings["ensemble_seeds"]
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError(
                f"profiles.{profile}.ensemble_seeds must be nonempty and unique"
            )
        active_count = settings.get("active_ensemble_member_count")
        if (
            not isinstance(active_count, int)
            or isinstance(active_count, bool)
            or not 1 <= active_count <= len(seeds)
        ):
            raise ValueError(
                f"profiles.{profile}.active_ensemble_member_count must be "
                "an integer in [1,len(ensemble_seeds)]"
            )
    if len(config["kinematics"]) < 4 or not config["observables"]:
        raise ValueError("workflow requires >=4 kinematics and observables")
    observable_ids = [item["id"] for item in config["observables"]]
    admitted_order = [
        name for name in ADMITTED_OBSERVABLE_IDS if name in observable_ids
    ]
    if observable_ids != admitted_order:
        raise ValueError(
            "observables must be an ordered unique nonempty subset of the "
            "six admitted native modules"
        )
    for gpd_type in ("H", "E", "Htilde", "Etilde"):
        key = f"{gpd_type}_shadow_coefficient"
        if not -1.0 <= float(config["truth"][key]) <= 1.0:
            raise ValueError(f"truth.{key} must lie in [-1,1]")
    return config


def _physics_path(configuration_path: Path, config: Mapping[str, Any]) -> Path:
    """Resolve either a project-local or repository-relative physics file.

    The accepted developer configuration names a path relative to the
    repository root.  Isolated user projects instead keep a private
    `physics.json` beside their generated engine configuration.  Supporting
    both locations lets the user layer reuse this exact workflow without
    copying physics code or writing into the repository's canonical configs.
    """

    configuration_path = configuration_path.resolve(strict=True)
    declared = Path(str(config["physics_configuration"]))
    if declared.is_absolute():
        return declared.resolve(strict=True)
    beside_configuration = configuration_path.parent / declared
    if beside_configuration.is_file():
        return beside_configuration.resolve(strict=True)
    root = configuration_path.parents[2]
    return (root / declared).resolve(strict=True)


def _point_table(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand each kinematic into the declared six-observable token order."""

    points: list[dict[str, Any]] = []
    for kinematics in config["kinematics"]:
        for observable in config["observables"]:
            points.append(
                {
                    **kinematics,
                    "observable_id": observable["id"],
                }
            )
    return points


def _native_request(
    *,
    request_id: str,
    parameters: Sequence[float],
    kinematics: Mapping[str, float],
    physics: Mapping[str, Any],
    observable_names: Sequence[str] = ADMITTED_OBSERVABLE_IDS,
) -> dict[str, Any]:
    """Build one strict bridge request without duplicating native physics."""

    expected = _PHYSICS_PARAMETER_COUNT
    if len(parameters) != expected:
        raise ValueError(
            f"native DD parameter vector must have length {expected}"
        )
    offset = 0
    gpd_shapes: dict[str, dict[str, dict[str, Any]]] = {}
    for gpd_type in _GPD_TYPES:
        gpd_shapes[gpd_type] = {}
        for channel in _PARTON_CHANNELS:
            values = {
                field: float(parameters[offset + field_index])
                for field_index, field in enumerate(_DD_SHAPE_FIELDS)
            }
            offset += len(_DD_SHAPE_FIELDS)
            gpd_shapes[gpd_type][channel] = {
                "normalization": values["normalization"],
                "a": values["a"],
                "c": values["c"],
                "profile_b": values["profile_b"],
                "t_slope": {
                    "value": values["t_slope_GeV_minus2"],
                    "unit": "GeV-2",
                },
            }
    theory = dict(physics["theory_configuration"])
    theory["observable_modules"] = list(observable_names)
    return {
        "request_id": request_id,
        "representation": physics["representation"],
        "parameters": {
            "gpd_shapes": gpd_shapes,
            "shadow_coefficients": physics["parameters"][
                "shadow_coefficients"
            ],
            "shadow_channel_amplitudes": physics["parameters"][
                "shadow_channel_amplitudes"
            ],
        },
        "quadrature_order": physics["parameters"]["quadrature_order"],
        "q0_squared": physics["input_scale"]["Q0_squared"],
        "observable_kinematics": {
            "x_b": kinematics["x_b"],
            "t": {"value": kinematics["t_GeV2"], "unit": "GeV2"},
            "Q2": {"value": kinematics["Q2_GeV2"], "unit": "GeV2"},
            "beam_energy": {
                "value": kinematics["beam_energy_GeV"],
                "unit": "GeV",
            },
            "phi": {"value": kinematics["phi_rad"], "unit": "rad"},
        },
        "evolution_configuration": physics["evolution_configuration"],
        "theory_configuration": theory,
    }


@dataclass(frozen=True)
class NativeResult:
    """Exact normalized predictions and immutable cache provenance."""

    predictions: np.ndarray
    cffs: np.ndarray
    gpds: np.ndarray
    cache_key: str
    cache_hit: bool
    bridge_sha256: str
    request_sha256: str
    response_sha256: str
    elapsed_seconds: float
    evaluation_count: int


@dataclass(frozen=True)
class PartitionedNativeResult:
    """Native predictions with explicit records for simulator failures.

    PARTONS rejects a whole batch when one evaluation is non-finite.  The
    recursive partitioner below preserves fast batches in valid regions while
    isolating failures down to one parameter point.  A failed point is never
    replaced by a Python formula or an interpolated value.
    """

    predictions: np.ndarray
    cffs: np.ndarray
    gpds: np.ndarray
    valid_mask: np.ndarray
    invalid_records: tuple[dict[str, Any], ...]
    successful_batches: tuple[NativeResult, ...]


@dataclass(frozen=True)
class ParallelNativeResult:
    """Ordered partitioned result plus process-isolated execution metadata."""

    result: PartitionedNativeResult
    worker_resolution: dict[str, Any]
    chunk_size: int
    task_count: int


class NativeCache:
    """Content-addressed client for atomic C++ bridge batches."""

    def __init__(self, bridge: Path, workspace: Path) -> None:
        self.bridge = bridge.resolve(strict=True)
        if not self.bridge.is_file():
            raise ValueError("bridge must be a regular executable file")
        self.cache_root = workspace / "cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.bridge_hash = sha256(self.bridge)

    def evaluate(
        self,
        *,
        parameters: np.ndarray,
        config: Mapping[str, Any],
        physics: Mapping[str, Any],
        gpd_x_grid: Sequence[float] = (),
        force: bool = False,
        operation: str = "batch_evaluate_pseudodata_dvcs",
    ) -> NativeResult:
        """Evaluate all parameter/kinematic pairs or load an exact cache hit."""

        allowed_operations = {
            "batch_evaluate_pseudodata_dvcs",
            "batch_evaluate_post_training_comparison_dvcs",
        }
        if operation not in allowed_operations:
            raise ValueError("unsupported native-cache operation")

        requests = []
        for parameter_index, row in enumerate(parameters):
            for point_index, kinematics in enumerate(config["kinematics"]):
                request = _native_request(
                        request_id=(
                            f"theta-{parameter_index:06d}-"
                            f"kin-{point_index:03d}"
                        ),
                        parameters=row,
                        kinematics=kinematics,
                        physics=physics,
                        observable_names=[
                            item["id"] for item in config["observables"]
                        ],
                    )
                if len(gpd_x_grid) != 0:
                    request["gpd_diagnostic_x"] = [
                        float(value) for value in gpd_x_grid
                    ]
                requests.append(request)
        payload = {
            "schema_version": 1,
            "operation": operation,
            "requests": requests,
        }
        request_bytes = (pretty_json(payload) + "\n").encode()
        request_hash = hashlib.sha256(request_bytes).hexdigest()
        key = hashlib.sha256(
            canonical(
                {
                    "cache_schema_version": 1,
                    "bridge_sha256": self.bridge_hash,
                    "request_sha256": request_hash,
                }
            ).encode()
        ).hexdigest()
        directory = self.cache_root / key
        request_path = directory / "request.json"
        response_path = directory / "response.json"
        metadata_path = directory / "metadata.json"
        stderr_path = directory / "stderr.txt"
        exit_path = directory / "exit_code.txt"

        cache_hit = False
        started = time.perf_counter()
        if not force and response_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            cache_hit = bool(
                metadata.get("bridge_sha256") == self.bridge_hash
                and metadata.get("request_sha256") == request_hash
                and metadata.get("response_sha256") == sha256(response_path)
                and metadata.get("exit_code") == 0
            )
            if not cache_hit:
                raise RuntimeError(
                    f"stale or corrupt native cache entry: {directory}"
                )
        if not cache_hit:
            directory.mkdir(parents=True, exist_ok=True)
            request_path.write_bytes(request_bytes)
            process = subprocess.run(
                [str(self.bridge), "--input", str(request_path)],
                text=True,
                capture_output=True,
                check=False,
            )
            # Persist raw process evidence before parsing.  A crash, truncated
            # JSON response, or native exception must remain auditable and may
            # never be converted into a cache hit or a simulator fallback.
            raw_stdout_path = directory / "stdout.raw.txt"
            raw_stdout_path.write_text(process.stdout, encoding="utf-8")
            stderr_path.write_text(process.stderr, encoding="utf-8")
            exit_path.write_text(f"{process.returncode}\n", encoding="utf-8")
            if process.returncode != 0:
                detail = ""
                try:
                    error_response = json.loads(process.stdout)
                except json.JSONDecodeError:
                    error_response = None
                if isinstance(error_response, dict):
                    error = error_response.get("error")
                    if isinstance(error, dict):
                        code = error.get("code", "unknown_native_error")
                        message = error.get("message", "no native message")
                        detail = f": {code}: {message}"
                raise RuntimeError(
                    "native bridge batch failed with exit "
                    f"{process.returncode}{detail}; raw stdout "
                    f"{raw_stdout_path}; stderr {stderr_path}"
                )
            try:
                raw_response = json.loads(process.stdout)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "native bridge returned malformed JSON; see "
                    f"{raw_stdout_path} and {stderr_path}"
                ) from error
            response_path.write_text(
                pretty_json(raw_response) + "\n", encoding="utf-8"
            )
            response_hash = sha256(response_path)
            write_json(
                metadata_path,
                {
                    "cache_schema_version": 1,
                    "bridge": str(self.bridge),
                    "bridge_sha256": self.bridge_hash,
                    "request_sha256": request_hash,
                    "response_sha256": response_hash,
                    "exit_code": process.returncode,
                    "operation": operation,
                    "evaluation_count": len(requests),
                    "surrogate_used": False,
                },
            )
        elapsed = time.perf_counter() - started
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if (
            response.get("status") != "ok"
            or response.get("operation") != operation
            or len(response.get("evaluations", ())) != len(requests)
        ):
            raise RuntimeError("native response violates the batch contract")
        if operation == "batch_evaluate_post_training_comparison_dvcs":
            comparison_contract = response.get("comparison_contract", {})
            if comparison_contract != {
                "role": "post_training_diagnostic_only",
                "real_data_fit": False,
                "posterior_updated": False,
                "source_Q2_retained": True,
                "data_evolved": False,
                "gpd_evolved_to_each_datum_Q2": True,
            }:
                raise RuntimeError(
                    "native response violates the real-comparison quarantine"
                )

        observable_ids = [item["id"] for item in config["observables"]]
        scales = {
            item["id"]: float(item["normalization_scale"])
            for item in config["observables"]
        }
        predictions = np.empty(
            (len(parameters), len(config["kinematics"]) * len(observable_ids)),
            dtype=np.float64,
        )
        cffs = np.empty(
            (len(parameters), len(config["kinematics"]), 4, 2),
            dtype=np.float64,
        )
        gpds = np.empty(
            (
                len(parameters),
                len(config["kinematics"]),
                len(gpd_x_grid),
                4,
                11,
            ),
            dtype=np.float64,
        )
        cff_names = ("H", "E", "Htilde", "Etilde")
        for flat_index, evaluation in enumerate(response["evaluations"]):
            parameter_index, kinematic_index = divmod(
                flat_index, len(config["kinematics"])
            )
            for cff_index, cff_name in enumerate(cff_names):
                cff = evaluation["result"]["cffs"][cff_name]
                cffs[parameter_index, kinematic_index, cff_index] = (
                    float(cff["real"]),
                    float(cff["imaginary"]),
                )
            diagnostics = evaluation["result"]["gpd_diagnostics"]
            if len(diagnostics) != len(gpd_x_grid):
                raise RuntimeError("native GPD diagnostic grid has wrong size")
            for x_index, point in enumerate(diagnostics):
                if not math.isclose(
                    float(point["x"]),
                    float(gpd_x_grid[x_index]),
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise RuntimeError("native GPD diagnostic x order changed")
                for gpd_index, gpd_name in enumerate(cff_names):
                    gpd = point["gpd"][gpd_name]
                    gpds[
                        parameter_index, kinematic_index, x_index, gpd_index
                    ] = (
                        float(gpd["u"]["value"]),
                        float(gpd["u"]["plus"]),
                        float(gpd["u"]["minus"]),
                        float(gpd["d"]["value"]),
                        float(gpd["d"]["plus"]),
                        float(gpd["d"]["minus"]),
                        float(gpd["s"]["value"]),
                        float(gpd["s"]["plus"]),
                        float(gpd["s"]["minus"]),
                        float(gpd["gluon"]),
                        float(gpd[
                            "charge_squared_weighted_c_even_quark_sum"
                        ]),
                    )
            for observable_index, observable_id in enumerate(observable_ids):
                item = evaluation["result"]["observables"][observable_id]
                expected_unit = (
                    "NB"
                    if next(
                        entry
                        for entry in config["observables"]
                        if entry["id"] == observable_id
                    )["native_unit"]
                    == "nb/GeV4"
                    else "NONE"
                )
                if item["native_unit_symbol"] != expected_unit:
                    raise RuntimeError("unexpected native observable unit")
                column = (
                    kinematic_index * len(observable_ids) + observable_index
                )
                predictions[parameter_index, column] = (
                    float(item["value"]) / scales[observable_id]
                )
        if not np.all(np.isfinite(predictions)) or not np.all(
            np.isfinite(cffs)
        ) or not np.all(np.isfinite(gpds)):
            raise RuntimeError("native response contains a non-finite value")
        return NativeResult(
            predictions=predictions,
            cffs=cffs,
            gpds=gpds,
            cache_key=key,
            cache_hit=cache_hit,
            bridge_sha256=self.bridge_hash,
            request_sha256=request_hash,
            response_sha256=sha256(response_path),
            elapsed_seconds=elapsed,
            evaluation_count=len(requests),
        )


def _partitioned_native_evaluation(
    *,
    client: NativeCache,
    parameters: np.ndarray,
    config: Mapping[str, Any],
    physics: Mapping[str, Any],
    force: bool = False,
    operation: str = "batch_evaluate_pseudodata_dvcs",
    gpd_x_grid: Sequence[float] = (),
) -> PartitionedNativeResult:
    """Evaluate and recursively isolate exact-native invalid simulations."""

    parameters = np.asarray(parameters, dtype=np.float64)
    point_count = len(config["kinematics"]) * len(config["observables"])
    predictions = np.full(
        (len(parameters), point_count), np.nan, dtype=np.float64
    )
    cffs = np.full(
        (len(parameters), len(config["kinematics"]), 4, 2),
        np.nan,
        dtype=np.float64,
    )
    gpds = np.full(
        (
            len(parameters),
            len(config["kinematics"]),
            len(gpd_x_grid),
            4,
            11,
        ),
        np.nan,
        dtype=np.float64,
    )
    valid_mask = np.zeros(len(parameters), dtype=bool)
    invalid_records: list[dict[str, Any]] = []
    successful_batches: list[NativeResult] = []

    def evaluate_slice(indices: np.ndarray) -> None:
        try:
            result = client.evaluate(
                parameters=parameters[indices],
                config=config,
                physics=physics,
                gpd_x_grid=gpd_x_grid,
                force=force,
                operation=operation,
            )
        except RuntimeError as error:
            if len(indices) > 1:
                middle = len(indices) // 2
                evaluate_slice(indices[:middle])
                evaluate_slice(indices[middle:])
                return
            index = int(indices[0])
            invalid_records.append(
                {
                    "candidate_index": index,
                    "parameters": parameters[index].tolist(),
                    "native_error": str(error),
                    "replacement": None,
                }
            )
            return
        predictions[indices] = result.predictions
        cffs[indices] = result.cffs
        gpds[indices] = result.gpds
        valid_mask[indices] = True
        successful_batches.append(result)

    if len(parameters):
        evaluate_slice(np.arange(len(parameters), dtype=np.int64))
    return PartitionedNativeResult(
        predictions=predictions,
        cffs=cffs,
        gpds=gpds,
        valid_mask=valid_mask,
        invalid_records=tuple(invalid_records),
        successful_batches=tuple(successful_batches),
    )


def _parallel_partitioned_native_evaluation(
    *,
    client: NativeCache,
    parameters: np.ndarray,
    config: Mapping[str, Any],
    physics: Mapping[str, Any],
    native_workers: int | str,
    description: str,
    show_progress: bool | None,
    chunk_size: int = _NATIVE_GENERATION_CHUNK_SIZE,
    force: bool = False,
    operation: str = "batch_evaluate_pseudodata_dvcs",
    gpd_x_grid: Sequence[float] = (),
) -> ParallelNativeResult:
    """Evaluate ordered parameter chunks in isolated bridge subprocesses.

    Thread-pool workers supervise independent executable processes; no
    PARTONS object is shared. Results are restored to input order regardless
    of completion order, and every completed atomic chunk advances stderr
    progress even when native rows are invalid.
    """

    from extract_dvcs_cff.progress import progress_bar

    parameters = np.asarray(parameters, dtype=np.float64)
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise ValueError("native evaluation chunk_size must be an integer")
    if chunk_size < 1:
        raise ValueError("native evaluation chunk_size must be positive")
    chunk_specs = [
        (
            chunk_index,
            start,
            parameters[start : min(start + chunk_size, len(parameters))],
        )
        for chunk_index, start in enumerate(
            range(0, len(parameters), chunk_size), start=1
        )
    ]
    worker_resolution = resolve_native_workers(
        native_workers, task_count=len(chunk_specs)
    )

    def evaluate_chunk(
        specification: tuple[int, int, np.ndarray],
    ) -> tuple[int, int, np.ndarray, PartitionedNativeResult]:
        chunk_index, start, chunk = specification
        partition = _partitioned_native_evaluation(
            client=client,
            parameters=chunk,
            config=config,
            physics=physics,
            force=force,
            operation=operation,
            gpd_x_grid=gpd_x_grid,
        )
        return chunk_index, start, chunk, partition

    progress = progress_bar(
        total=len(parameters),
        description=description,
        enabled=show_progress,
        leave=False,
        unit="sample",
    )
    completed_chunks: dict[
        int, tuple[int, int, np.ndarray, PartitionedNativeResult]
    ] = {}
    completed_count = 0
    valid_count = 0
    invalid_count = 0

    def record_completion(
        completed_index: int,
        value: tuple[int, int, np.ndarray, PartitionedNativeResult],
    ) -> None:
        nonlocal completed_count, valid_count, invalid_count
        completed_chunks[value[0]] = value
        chunk_count = len(value[2])
        chunk_valid = int(np.count_nonzero(value[3].valid_mask))
        completed_count += chunk_count
        valid_count += chunk_valid
        invalid_count += chunk_count - chunk_valid
        progress.update(chunk_count)
        progress.set_postfix_str(
            f"batch {completed_index}/{len(chunk_specs)}; "
            f"completed={completed_count}; valid={valid_count}; "
            f"invalid={invalid_count}; "
            f"workers={worker_resolution.resolved_workers}",
            refresh=False,
        )
        progress.refresh()

    try:
        if worker_resolution.resolved_workers == 1:
            for completed_index, specification in enumerate(
                chunk_specs, start=1
            ):
                record_completion(
                    completed_index, evaluate_chunk(specification)
                )
        else:
            with ThreadPoolExecutor(
                max_workers=worker_resolution.resolved_workers,
                thread_name_prefix="partons-evaluation",
            ) as executor:
                futures = {
                    executor.submit(evaluate_chunk, specification): (
                        specification[0]
                    )
                    for specification in chunk_specs
                }
                for completed_index, future in enumerate(
                    as_completed(futures), start=1
                ):
                    record_completion(completed_index, future.result())
    finally:
        progress.close()

    point_count = len(config["kinematics"]) * len(config["observables"])
    predictions = np.full(
        (len(parameters), point_count), np.nan, dtype=np.float64
    )
    cffs = np.full(
        (len(parameters), len(config["kinematics"]), 4, 2),
        np.nan,
        dtype=np.float64,
    )
    gpds = np.full(
        (
            len(parameters),
            len(config["kinematics"]),
            len(gpd_x_grid),
            4,
            11,
        ),
        np.nan,
        dtype=np.float64,
    )
    valid_mask = np.zeros(len(parameters), dtype=bool)
    invalid_records: list[dict[str, Any]] = []
    successful_batches: list[NativeResult] = []
    for chunk_index in sorted(completed_chunks):
        _, start, chunk, partition = completed_chunks[chunk_index]
        stop = start + len(chunk)
        predictions[start:stop] = partition.predictions
        cffs[start:stop] = partition.cffs
        gpds[start:stop] = partition.gpds
        valid_mask[start:stop] = partition.valid_mask
        for record in partition.invalid_records:
            adjusted = dict(record)
            adjusted["candidate_index"] = (
                start + int(record["candidate_index"])
            )
            invalid_records.append(adjusted)
        successful_batches.extend(partition.successful_batches)
    return ParallelNativeResult(
        result=PartitionedNativeResult(
            predictions=predictions,
            cffs=cffs,
            gpds=gpds,
            valid_mask=valid_mask,
            invalid_records=tuple(invalid_records),
            successful_batches=tuple(successful_batches),
        ),
        worker_resolution=worker_resolution.as_dict(),
        chunk_size=chunk_size,
        task_count=len(chunk_specs),
    )


def _covariance_and_responses(
    config: Mapping[str, Any], reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Construct fixed full covariance and explicit nuisance responses."""

    model = config["experimental_model"]
    count = len(reference)
    point_table = _point_table(config)
    relative = float(model["uncorrelated_relative_sigma"])
    floor = float(model["uncorrelated_absolute_floor_normalized"])
    sigma = floor + relative * np.abs(reference)
    covariance = np.diag(sigma**2)

    local_fraction = float(model["local_correlation_fraction"])
    local_scale = local_fraction * np.maximum(np.abs(reference), floor)
    indices = np.arange(count)
    kernel = np.exp(
        -np.abs(indices[:, None] - indices[None, :])
        / float(model["local_correlation_length"])
    )
    covariance += local_scale[:, None] * local_scale[None, :] * kernel

    phi_fraction = float(model["phi_shape_correlated_fraction"])
    phi_source = np.asarray(
        [
            phi_fraction
            * reference[index]
            * math.sin(float(point["phi_rad"]))
            for index, point in enumerate(point_table)
        ],
        dtype=np.float64,
    )
    covariance += np.outer(phi_source, phi_source)
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= 0.0:
        raise RuntimeError("constructed covariance is not positive definite")

    nuisance_groups = {
        item["name"]: item for item in model["nuisance_groups"]
    }
    required_nuisances = {
        "eta_global_normalization",
        "eta_lu_normalization",
    }
    if set(nuisance_groups) != required_nuisances:
        raise ValueError(
            "experimental_model.nuisance_groups must contain exactly "
            f"{sorted(required_nuisances)}"
        )
    global_nuisance = nuisance_groups["eta_global_normalization"]
    lu_nuisance = nuisance_groups["eta_lu_normalization"]
    if (
        global_nuisance["mode"] != "multiplicative"
        or global_nuisance["prior"] != "standard_normal"
        or global_nuisance["applies_to"] != "all"
        or lu_nuisance["mode"] != "multiplicative"
        or lu_nuisance["prior"] != "standard_normal"
        or lu_nuisance["applies_to"]
        != "DVCSCrossSectionDifferenceLUMinus"
    ):
        raise ValueError("nuisance group semantics violate the frozen contract")
    global_fraction = float(global_nuisance["fractional_response"])
    lu_fraction = float(lu_nuisance["fractional_response"])
    if (
        not math.isfinite(global_fraction)
        or global_fraction < 0.0
        or not math.isfinite(lu_fraction)
        or lu_fraction < 0.0
    ):
        raise ValueError("nuisance fractional responses must be nonnegative")

    global_response = np.full(count, global_fraction, dtype=np.float64)
    lu_response = np.asarray(
        [
            lu_fraction
            if point["observable_id"]
            == "DVCSCrossSectionDifferenceLUMinus"
            else 0.0
            for point in point_table
        ],
        dtype=np.float64,
    )
    diagnostics = {
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
        "log_determinant": float(np.linalg.slogdet(covariance)[1]),
        "off_diagonal_nonzero": bool(
            np.any(covariance - np.diag(np.diag(covariance)))
        ),
        "regularization": None,
        "covariance_policy": model["covariance_policy"],
        "correlated_sources": {
            "local_kernel": local_scale.tolist(),
            "phi_shape": phi_source.tolist(),
        },
    }
    return covariance, global_response, lu_response, diagnostics


def _effective_prediction(
    base: np.ndarray,
    eta_global: float,
    eta_lu: float,
    global_response: np.ndarray,
    lu_response: np.ndarray,
) -> np.ndarray:
    """Apply the two declared linear fractional nuisance responses."""

    return base * (
        1.0 + eta_global * global_response + eta_lu * lu_response
    )


def _save_array(directory: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = directory / f"{name}.npy"
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    return {
        "path": path.name,
        "sha256": sha256(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _workspace_contract(
    workspace: Path,
    config_path: Path,
    profile: str,
    bridge_hash: str,
) -> None:
    """Reject cross-configuration reuse of one user workspace."""

    path = workspace / "workspace_contract.json"
    expected = {
        "schema_version": 1,
        "configuration": str(config_path.resolve(strict=True)),
        "configuration_sha256": sha256(config_path.resolve(strict=True)),
        "profile": profile,
        "bridge_sha256": bridge_hash,
        "real_data": False,
        "synthetic_split_policy": (
            "native_parameter_grouped_train_validation_test_v1"
        ),
    }
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != expected:
            raise RuntimeError(
                "workspace contract mismatch; choose a new workspace after "
                "changing config, profile, or bridge"
            )
    else:
        write_json(path, expected)


def generate_pseudodata(
    *,
    bridge: Path,
    configuration_path: Path,
    workspace: Path,
    profile: str,
    force_native: bool = False,
    show_progress: bool | None = None,
) -> dict[str, Any]:
    """Generate exact-native simulations and full-systematics pseudodata."""

    from extract_dvcs_cff.inference.stage10 import (
        build_stage10_context,
        stage10_physical_to_latent,
    )
    from extract_dvcs_cff.progress import progress_bar, progress_iter
    import torch

    config = load_configuration(configuration_path)
    if profile not in config["profiles"]:
        raise ValueError(f"unknown profile {profile!r}")
    settings = config["profiles"][profile]
    physics_path = _physics_path(configuration_path, config)
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    workspace.mkdir(parents=True, exist_ok=True)
    client = NativeCache(bridge, workspace)
    _workspace_contract(
        workspace, configuration_path, profile, client.bridge_hash
    )
    output = workspace / "generated"
    output.mkdir(parents=True, exist_ok=True)

    count = int(settings["native_parameter_count"])
    rng = np.random.Generator(
        np.random.PCG64(int(config["seeds"]["native_parameters"]))
    )
    truth = config["truth"]
    truth_parameters = _shape_vector_from_shapes(
        physics["parameters"]["gpd_shapes"]
    )[None, :]
    try:
        truth_native = client.evaluate(
            parameters=truth_parameters,
            config=config,
            physics=physics,
            force=force_native,
        )
    except RuntimeError as error:
        write_json(
            output / "invalid_simulation_map.json",
            {
                "schema_version": 1,
                "status": "failed_truth_preflight",
                "phase": "injected_truth_native_preflight",
                "candidate_count": 0,
                "requested_valid_count": count,
                "valid_count": 0,
                "invalid_count": 1,
                "invalid_records": [
                    {
                        "candidate_index": None,
                        "parameters": truth_parameters[0].tolist(),
                        "native_error": str(error),
                        "replacement": None,
                    }
                ],
                "policy": "abort_before_prior_corpus_generation",
            },
        )
        raise RuntimeError(
            "injected-truth native preflight failed before prior-corpus "
            f"generation: {error}"
        ) from error
    accepted_parameters: list[np.ndarray] = []
    accepted_predictions: list[np.ndarray] = []
    accepted_cffs: list[np.ndarray] = []
    invalid_records: list[dict[str, Any]] = []
    native_batches: list[NativeResult] = []
    candidate_count = 0
    accepted_count = 0
    native_worker_resolutions: list[dict[str, Any]] = []
    # Draw the same first candidate bank as the pre-wave implementation, then
    # expose it to PARTONS incrementally.  This preserves seed-level corpus
    # reproducibility while bounding submitted native work.
    candidate_pool = _sample_shape_parameters(rng, count)
    candidate_pool_offset = 0
    native_progress = progress_bar(
        total=count,
        description="Accepted PARTONS simulations",
        enabled=show_progress,
        leave=False,
        unit="parameter",
    )
    while sum(len(item) for item in accepted_parameters) < count:
        remaining = count - sum(len(item) for item in accepted_parameters)
        if candidate_pool_offset == len(candidate_pool):
            candidate_pool = _sample_shape_parameters(rng, remaining)
            candidate_pool_offset = 0
        wave_size = _native_generation_wave_size(
            config["runtime"]["native_workers"],
            remaining=remaining,
            available_candidates=len(candidate_pool) - candidate_pool_offset,
        )
        candidates = candidate_pool[
            candidate_pool_offset : candidate_pool_offset + wave_size
        ]
        candidate_pool_offset += wave_size
        chunk_specs = [
            (
                chunk_index,
                start,
                candidates[
                    start : min(
                        start + _NATIVE_GENERATION_CHUNK_SIZE,
                        len(candidates),
                    )
                ],
            )
            for chunk_index, start in enumerate(
                range(0, len(candidates), _NATIVE_GENERATION_CHUNK_SIZE),
                start=1,
            )
        ]
        worker_resolution = resolve_native_workers(
            config["runtime"]["native_workers"],
            task_count=len(chunk_specs),
        )
        native_worker_resolutions.append(worker_resolution.as_dict())
        native_progress.set_postfix_str(
            f"workers={worker_resolution.resolved_workers}; "
            f"batch={_NATIVE_GENERATION_CHUNK_SIZE} parameters",
            refresh=True,
        )

        def evaluate_chunk(
            specification: tuple[int, int, np.ndarray],
        ) -> tuple[int, int, np.ndarray, PartitionedNativeResult]:
            chunk_index, start, chunk = specification
            partition = _partitioned_native_evaluation(
                client=client,
                parameters=chunk,
                config=config,
                physics=physics,
                force=force_native,
            )
            return chunk_index, start, chunk, partition

        completed_chunks: dict[
            int, tuple[int, int, np.ndarray, PartitionedNativeResult]
        ] = {}
        wave_attempted = 0
        if worker_resolution.resolved_workers == 1:
            for specification in chunk_specs:
                result = evaluate_chunk(specification)
                completed_chunks[result[0]] = result
                valid_count = int(np.count_nonzero(result[3].valid_mask))
                native_progress.update(valid_count)
                wave_attempted += len(result[2])
                accepted_count += valid_count
                attempted_total = candidate_count + wave_attempted
                rejected_total = attempted_total - accepted_count
                native_progress.set_postfix_str(
                    f"wave batch {result[0]}/{len(chunk_specs)}; "
                    f"attempted={attempted_total}; "
                    f"accepted={accepted_count}; "
                    f"rejected={rejected_total}; "
                    "workers=1",
                    refresh=False,
                )
                # tqdm rate-limits normal redraws.  An explicit refresh after
                # update guarantees that the terminal shows this completed
                # atomic PARTONS batch instead of lagging by one batch.
                native_progress.refresh()
        else:
            with ThreadPoolExecutor(
                max_workers=worker_resolution.resolved_workers,
                thread_name_prefix="partons-bridge",
            ) as executor:
                futures = {
                    executor.submit(evaluate_chunk, specification): specification[0]
                    for specification in chunk_specs
                }
                for completed_index, future in enumerate(
                    as_completed(futures), start=1
                ):
                    result = future.result()
                    completed_chunks[result[0]] = result
                    valid_count = int(
                        np.count_nonzero(result[3].valid_mask)
                    )
                    native_progress.update(valid_count)
                    wave_attempted += len(result[2])
                    accepted_count += valid_count
                    attempted_total = candidate_count + wave_attempted
                    rejected_total = attempted_total - accepted_count
                    native_progress.set_postfix_str(
                        f"wave batch {completed_index}/{len(chunk_specs)}; "
                        f"attempted={attempted_total}; "
                        f"accepted={accepted_count}; "
                        f"rejected={rejected_total}; "
                        f"workers={worker_resolution.resolved_workers}",
                        refresh=False,
                    )
                    native_progress.refresh()

        # Completion order is intentionally ignored.  Candidate order defines
        # corpus order and therefore remains byte-stable between serial and
        # parallel execution.
        for chunk_index in sorted(completed_chunks):
            _, start, chunk, partition = completed_chunks[chunk_index]
            if np.any(partition.valid_mask):
                accepted_parameters.append(chunk[partition.valid_mask])
                accepted_predictions.append(
                    partition.predictions[partition.valid_mask]
                )
                accepted_cffs.append(partition.cffs[partition.valid_mask])
            for record in partition.invalid_records:
                adjusted = dict(record)
                adjusted["candidate_index"] = (
                    candidate_count
                    + start
                    + int(record["candidate_index"])
                )
                invalid_records.append(adjusted)
            native_batches.extend(partition.successful_batches)
        candidate_count += len(candidates)
        if candidate_count >= _NATIVE_GENERATION_MIN_WAVE_SIZE and not any(
            len(item) for item in accepted_parameters
        ):
            native_progress.close()
            failure = {
                "schema_version": 1,
                "status": "failed_zero_acceptance_preflight",
                "phase": "prior_corpus_first_wave",
                "candidate_count": candidate_count,
                "requested_valid_count": count,
                "valid_count": 0,
                "invalid_count": len(invalid_records),
                "invalid_fraction_of_candidates": 1.0,
                "invalid_records": invalid_records,
                "policy": (
                    "abort_after_bounded_zero_acceptance_wave;"
                    "no_surrogate_no_imputation"
                ),
            }
            write_json(output / "invalid_simulation_map.json", failure)
            raise RuntimeError(
                "the first bounded native generation wave accepted 0 of "
                f"{candidate_count} prior draws; inspect "
                f"{output / 'invalid_simulation_map.json'}"
            )
        if candidate_count > 10 * count:
            native_progress.close()
            write_json(
                output / "invalid_simulation_map.json",
                {
                    "schema_version": 1,
                    "status": "failed_excessive_invalid_fraction",
                    "phase": "prior_corpus_generation",
                    "candidate_count": candidate_count,
                    "requested_valid_count": count,
                    "valid_count": sum(
                        len(item) for item in accepted_parameters
                    ),
                    "invalid_count": len(invalid_records),
                    "invalid_fraction_of_candidates": (
                        len(invalid_records) / candidate_count
                    ),
                    "invalid_records": invalid_records,
                    "policy": (
                        "abort_after_ten_times_requested_candidates;"
                        "no_surrogate_no_imputation"
                    ),
                },
            )
            raise RuntimeError(
                "native invalid-simulation fraction prevented generation; "
                "inspect invalid_simulation_map.json"
            )
    native_progress.close()
    parameters = np.concatenate(accepted_parameters, axis=0)[:count]
    native_predictions = np.concatenate(
        accepted_predictions, axis=0
    )[:count]
    native_cffs = np.concatenate(accepted_cffs, axis=0)[:count]
    reference = truth_native.predictions[0]
    covariance, global_response, lu_response, covariance_diagnostics = (
        _covariance_and_responses(config, reference)
    )
    cholesky = np.linalg.cholesky(covariance)
    point_table = _point_table(config)

    replicates = int(settings["noise_replicates_per_parameter"])
    context_width = (
        len(point_table) * len(config["context_contract"]["point_features"])
        + len(config["context_contract"]["global_features"])
    )
    total = count * replicates
    theta = np.empty(
        (total, _POSTERIOR_PARAMETER_COUNT), dtype=np.float64
    )
    observations = np.empty((total, len(point_table)), dtype=np.float64)
    contexts = np.empty((total, context_width), dtype=np.float32)
    parameter_indices = np.repeat(np.arange(count), replicates)
    noise_rng = np.random.Generator(
        np.random.PCG64(int(config["seeds"]["training_noise"]))
    )
    context_rows = progress_iter(
        enumerate(parameter_indices),
        total=total,
        description="Noise and DeepSets contexts",
        enabled=show_progress,
        unit="context",
    )
    for row, parameter_index in context_rows:
        eta_global, eta_lu = noise_rng.standard_normal(2)
        theta[row] = (
            *parameters[parameter_index],
            eta_global,
            eta_lu,
        )
        mean = _effective_prediction(
            native_predictions[parameter_index],
            eta_global,
            eta_lu,
            global_response,
            lu_response,
        )
        observations[row] = mean + cholesky @ noise_rng.standard_normal(
            len(point_table)
        )
        contexts[row] = build_stage10_context(
            points=point_table,
            observed=observations[row],
            covariance=covariance,
            global_normalization_response=global_response,
            lu_normalization_response=lu_response,
        )
    latent = stage10_physical_to_latent(
        torch.from_numpy(theta)
    ).numpy()

    held_out_count = max(
        1, int(round(count * float(settings["held_out_fraction"])))
    )
    pretest_count = count - held_out_count
    validation_parameter_count = max(
        1,
        int(
            round(
                pretest_count
                * float(config["network"]["validation_fraction"])
            )
        ),
    )
    training_parameter_count = pretest_count - validation_parameter_count
    if training_parameter_count < 1:
        raise RuntimeError("grouped training split is empty")
    training_parameters = np.arange(0, training_parameter_count)
    validation_parameters = np.arange(
        training_parameter_count, pretest_count
    )
    held_out_parameters = np.arange(pretest_count, count)
    train_indices = np.flatnonzero(
        np.isin(parameter_indices, training_parameters)
    )
    validation_indices = np.flatnonzero(
        np.isin(parameter_indices, validation_parameters)
    )
    test_indices = np.flatnonzero(
        np.isin(parameter_indices, held_out_parameters)
    )
    if not len(train_indices) or not len(validation_indices) or not len(test_indices):
        raise RuntimeError("grouped train/validation/test split is empty")
    row_membership = np.concatenate(
        (train_indices, validation_indices, test_indices)
    )
    if (
        len(np.unique(row_membership)) != total
        or not np.array_equal(np.sort(row_membership), np.arange(total))
    ):
        raise RuntimeError("grouped split does not partition every context")

    pseudo_rng = np.random.Generator(
        np.random.PCG64(int(config["seeds"]["pseudodata"]))
    )
    truth_theta = np.concatenate(
        (
            _shape_vector_from_shapes(physics["parameters"]["gpd_shapes"]),
            np.asarray(
                [
                    truth["eta_global_normalization"],
                    truth["eta_lu_normalization"],
                ],
                dtype=np.float64,
            ),
        )
    )
    pseudo_mean = _effective_prediction(
        reference,
        truth_theta[_PHYSICS_PARAMETER_COUNT],
        truth_theta[_PHYSICS_PARAMETER_COUNT + 1],
        global_response,
        lu_response,
    )
    pseudo_observed = pseudo_mean + cholesky @ pseudo_rng.standard_normal(
        len(point_table)
    )
    pseudo_context = build_stage10_context(
        points=point_table,
        observed=pseudo_observed,
        covariance=covariance,
        global_normalization_response=global_response,
        lu_normalization_response=lu_response,
    )

    arrays: dict[str, Any] = {}
    for name, value in (
        ("native_parameters", parameters),
        ("native_predictions_normalized", native_predictions),
        ("native_cffs", native_cffs),
        ("theta_physical", theta),
        ("theta_latent", latent),
        ("observations_normalized", observations),
        ("contexts", contexts),
        ("covariance_normalized", covariance),
        ("global_normalization_response", global_response),
        ("lu_normalization_response", lu_response),
        ("parameter_indices", parameter_indices),
        ("train_indices", train_indices),
        ("validation_indices", validation_indices),
        ("test_indices", test_indices),
        ("pseudodata_observed_normalized", pseudo_observed),
        ("pseudodata_context", pseudo_context),
        ("truth_theta_physical", truth_theta),
        ("truth_cffs", truth_native.cffs[0]),
        ("truth_native_predictions_normalized", reference),
    ):
        arrays[name] = _save_array(output, name, np.asarray(value))

    scales = {
        item["id"]: item["normalization_scale"]
        for item in config["observables"]
    }
    dataset = {
        "schema_version": 1,
        "dataset_id": "stage10-user-pseudodata-v1",
        "synthetic": True,
        "real_data": False,
        "kinematics_source": config["data_provenance"],
        "generator": "exact_native_PARTONS_plus_declared_systematics",
        "point_order": point_table,
        "observable_normalization_scales": scales,
        "native_unit": "nb/GeV4",
        "observed_normalized": pseudo_observed.tolist(),
        "observed_native": [
            float(pseudo_observed[index] * scales[point["observable_id"]])
            for index, point in enumerate(point_table)
        ],
        "truth_prediction_normalized": reference.tolist(),
        "covariance_normalized": covariance.tolist(),
        "covariance_diagnostics": covariance_diagnostics,
        "nuisance_groups": config["experimental_model"]["nuisance_groups"],
        "truth": config["truth"],
        "seed": config["seeds"]["pseudodata"],
        "physics_configuration": str(physics_path),
        "physics_configuration_sha256": sha256(physics_path),
        "native_cache_key": truth_native.cache_key,
        "backend_bridge_sha256": truth_native.bridge_sha256,
        "analysis_order_label": None,
    }
    write_json(output / "pseudodata.json", dataset)
    invalid = {
        "schema_version": 1,
        "candidate_count": candidate_count,
        "requested_valid_count": count,
        "valid_count": count,
        "invalid_count": len(invalid_records),
        "invalid_fraction_of_candidates": (
            len(invalid_records) / candidate_count
        ),
        "invalid_records": invalid_records,
        "policy": (
            "exact_native_rejection_with_recorded_parameter_and_error;"
            "no_surrogate_no_imputation"
        ),
    }
    write_json(output / "invalid_simulation_map.json", invalid)
    runtime = {
        "schema_version": 1,
        "profile": profile,
        "native_parameter_count": count + 1,
        "native_point_evaluation_count": (
            sum(item.evaluation_count for item in native_batches)
            + truth_native.evaluation_count
        ),
        "training_context_count": total,
        "train_context_count": len(train_indices),
        "validation_context_count": len(validation_indices),
        "test_context_count": len(test_indices),
        "train_native_parameter_count": len(training_parameters),
        "validation_native_parameter_count": len(validation_parameters),
        "test_native_parameter_count": len(held_out_parameters),
        "synthetic_split_policy": (
            "native_parameter_grouped_train_validation_test_v1"
        ),
        "native_parameter_overlap_counts": {
            "train_validation": 0,
            "train_test": 0,
            "validation_test": 0,
        },
        "native_cache_hits": sum(
            int(item.cache_hit) for item in native_batches
        )
        + int(truth_native.cache_hit),
        "native_cache_misses": sum(
            int(not item.cache_hit) for item in native_batches
        )
        + int(not truth_native.cache_hit),
        "native_batch_count": len(native_batches) + 1,
        "native_generation_chunk_size": _NATIVE_GENERATION_CHUNK_SIZE,
        "native_generation_wave_size": min(
            count,
            max(
                _NATIVE_GENERATION_MIN_WAVE_SIZE,
                max(
                    item["resolved_workers"]
                    for item in native_worker_resolutions
                ) * _NATIVE_GENERATION_CHUNK_SIZE,
            ),
        ),
        "native_worker_request": config["runtime"]["native_workers"],
        "native_worker_resolutions": native_worker_resolutions,
        "native_max_concurrent_workers": max(
            item["resolved_workers"] for item in native_worker_resolutions
        ),
        "native_isolation": (
            "independent_partons_bridge_subprocesses;"
            "no_shared_native_module_state"
        ),
        "native_client_elapsed_seconds": (
            sum(item.elapsed_seconds for item in native_batches)
            + truth_native.elapsed_seconds
        ),
        "native_candidate_count": candidate_count,
        "native_invalid_count": len(invalid_records),
        "native_invalid_fraction": len(invalid_records) / candidate_count,
        "surrogate_used": False,
    }
    write_json(output / "generation_metrics.json", runtime)
    write_json(
        output / "array_manifest.json",
        {
            "schema_version": 1,
            "arrays": arrays,
            "configuration_sha256": sha256(configuration_path),
            "physics_configuration_sha256": sha256(physics_path),
            "bridge_sha256": truth_native.bridge_sha256,
        },
    )
    return runtime


class _MetricTracker:
    """Minimal in-memory tracker satisfying the sbi training interface."""

    def __init__(self) -> None:
        self.metrics: list[dict[str, Any]] = []
        self.params: dict[str, Any] = {}

    @property
    def log_dir(self) -> None:
        return None

    def log_metric(
        self, name: str, value: float, step: int | None = None
    ) -> None:
        self.metrics.append(
            {"name": name, "value": float(value), "step": step}
        )

    def log_metrics(
        self, metrics: Mapping[str, float], step: int | None = None
    ) -> None:
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def log_params(self, params: Mapping[str, Any]) -> None:
        self.params.update(params)

    def add_figure(
        self, name: str, figure: Any, step: int | None = None
    ) -> None:
        del name, figure, step

    def flush(self) -> None:
        return None


def _load_generated(workspace: Path) -> dict[str, np.ndarray]:
    directory = workspace / "generated"
    manifest_path = directory / "array_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "corpus selection must be materialized first; missing "
            f"{manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("arrays")
    if not isinstance(records, dict):
        raise RuntimeError("generated array manifest is malformed")
    required = (
        "theta_physical",
        "theta_latent",
        "contexts",
        "parameter_indices",
        "train_indices",
        "validation_indices",
        "test_indices",
        "native_parameters",
        "native_predictions_normalized",
        "native_cffs",
        "covariance_normalized",
        "global_normalization_response",
        "lu_normalization_response",
        "pseudodata_observed_normalized",
        "pseudodata_context",
        "truth_theta_physical",
        "truth_cffs",
        "truth_native_predictions_normalized",
    )
    values = {}
    for name in required:
        path = directory / f"{name}.npy"
        if not path.exists():
            raise RuntimeError(
                f"corpus selection must be materialized first; missing {path}"
            )
        record = records.get(name)
        if not isinstance(record, dict) or record.get("sha256") != sha256(path):
            raise RuntimeError(f"generated array hash mismatch: {path}")
        value = np.load(path, allow_pickle=False)
        if record.get("shape") != list(value.shape) or record.get(
            "dtype"
        ) != str(value.dtype):
            raise RuntimeError(f"generated array contract mismatch: {path}")
        values[name] = value
    return values


def train_model(
    *,
    configuration_path: Path,
    workspace: Path,
    profile: str,
    evaluate_test: bool = True,
    show_progress: bool | None = None,
) -> dict[str, Any]:
    """Train each declared deterministic NPE ensemble member.

    ``evaluate_test=False`` is reserved for hyperparameter trials.  It keeps
    held-out test scores entirely out of model selection; only the explicit
    native-parameter-grouped validation split contributes an Optuna objective.
    """

    import sbi
    import torch
    import warnings

    from extract_dvcs_cff.inference.stage10 import (
        GroupedValidationNPE,
        load_stage10_posterior,
        stage10_density_builder,
        stage10_latent_to_physical,
        stage10_prior,
    )
    from extract_dvcs_cff.inference.device import (
        cuda_runtime_metrics,
        finish_distributed_training,
        initialize_distributed_training,
        resolve_torch_device,
    )
    from extract_dvcs_cff.progress import progress_enabled, progress_iter

    config = load_configuration(configuration_path)
    settings = config["profiles"][profile]
    runtime = config["runtime"]
    device_resolution = resolve_torch_device(runtime["accelerator"])
    distributed_rank, distributed_world_size = (
        initialize_distributed_training(device_resolution)
    )
    device = torch.device(device_resolution.resolved)
    arrays = _load_generated(workspace)
    theta = torch.from_numpy(
        arrays["theta_latent"].astype(np.float32, copy=False)
    ).to(device)
    context = torch.from_numpy(
        arrays["contexts"].astype(np.float32, copy=False)
    ).to(device)
    train_indices = torch.from_numpy(
        arrays["train_indices"].astype(np.int64, copy=False)
    ).to(device)
    validation_indices = torch.from_numpy(
        arrays["validation_indices"].astype(np.int64, copy=False)
    ).to(device)
    test_indices = torch.from_numpy(
        arrays["test_indices"].astype(np.int64, copy=False)
    ).to(device)
    if not torch.isfinite(theta).all() or not torch.isfinite(context).all():
        raise RuntimeError("training tensors contain non-finite values")
    fit_indices = torch.cat((train_indices, validation_indices))
    training_input_manifest_sha256 = sha256(
        workspace / "generated" / "array_manifest.json"
    )

    output = workspace / "training"
    output.mkdir(parents=True, exist_ok=True)
    seed_metrics = {}
    all_seeds = settings["ensemble_seeds"]
    if distributed_world_size > len(all_seeds):
        raise RuntimeError(
            "allocated GPUs exceed ensemble members; reduce GPUs or add seeds"
        )
    seeds = all_seeds[distributed_rank::distributed_world_size]
    seed_iterator = progress_iter(
        seeds,
        total=len(seeds),
        description=(
            "NPE ensemble members "
            f"[{device_resolution.resolved}: "
            f"{device_resolution.cuda_device_name or 'host CPU'}]"
        ),
        enabled=show_progress,
        unit="model",
    )
    for seed in seed_iterator:
        seed_directory = output / f"seed_{seed}"
        model_directory = seed_directory / "model"
        model_directory.mkdir(parents=True, exist_ok=True)
        state_path = model_directory / "density_estimator_state.pt"
        metrics_path = seed_directory / "training_metrics.json"
        if state_path.exists() or metrics_path.exists():
            if not state_path.exists() or not metrics_path.exists():
                raise RuntimeError(
                    f"incomplete training restart state in {seed_directory}"
                )
            retained = json.loads(metrics_path.read_text(encoding="utf-8"))
            if retained.get("state_sha256") != sha256(state_path):
                raise RuntimeError(f"model state hash mismatch for seed {seed}")
            if retained.get("workflow") != config["workflow"]:
                raise RuntimeError(
                    "checkpoint predates the current validation-only ensemble "
                    "selection contract; use a new user project"
                )
            if retained.get("device", {}).get("resolved", "").split(":")[0] != device.type:
                raise RuntimeError(
                    f"seed {seed} was trained on a different accelerator; "
                    "use a new user project for this execution policy"
                )
            if retained.get("synthetic_split_policy") != (
                "native_parameter_grouped_train_validation_test_v1"
            ):
                raise RuntimeError(
                    "checkpoint predates grouped validation; use a new user "
                    "project"
                )
            if retained.get("training_input_manifest_sha256") != (
                training_input_manifest_sha256
            ):
                raise RuntimeError(
                    "checkpoint was trained from different generated arrays; "
                    "use a new user project"
                )
            seed_metrics[str(seed)] = retained
            continue

        if device.type == "cpu":
            torch.set_num_threads(int(runtime["cpu_threads"]))
        torch.use_deterministic_algorithms(
            bool(runtime["deterministic_algorithms"])
        )
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
            torch.cuda.reset_peak_memory_stats()
        np.random.seed(int(seed))
        tracker = _MetricTracker()
        grouped_train_indices = torch.arange(
            len(train_indices), dtype=torch.long
        )
        grouped_validation_indices = torch.arange(
            len(train_indices), len(fit_indices), dtype=torch.long
        )
        inference = GroupedValidationNPE(
            prior=stage10_prior(device),
            density_estimator=stage10_density_builder(config),
            device=str(device),
            tracker=tracker,
            show_progress_bars=progress_enabled(show_progress),
            grouped_train_indices=grouped_train_indices,
            grouped_validation_indices=grouped_validation_indices,
        )
        network = config["network"]
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            estimator = inference.append_simulations(
                theta[fit_indices], context[fit_indices]
            ).train(
                training_batch_size=int(network["training_batch_size"]),
                learning_rate=float(network["learning_rate"]),
                validation_fraction=float(network["validation_fraction"]),
                stop_after_epochs=int(settings["stop_after_epochs"]),
                max_num_epochs=int(settings["max_num_epochs"]),
                show_train_summary=False,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        estimator.eval()
        with torch.no_grad():
            train_log_prob = estimator.log_prob(
                theta[train_indices], context[train_indices]
            )
        if not torch.isfinite(train_log_prob).all():
            raise RuntimeError("trained estimator returned non-finite scores")
        torch.save(estimator.state_dict(), state_path)

        # A posterior smoke test catches support/conditioning mistakes before
        # the more expensive coverage and exact-reevaluation steps.
        posterior = inference.build_posterior(estimator)
        smoke_context = context[validation_indices[0]]
        torch.manual_seed(int(seed) + 1)
        smoke_latent = posterior.sample(
            (128,), x=smoke_context, show_progress_bars=False
        )
        smoke_physical = stage10_latent_to_physical(smoke_latent)
        smoke_finite = bool(torch.isfinite(smoke_physical).all())
        summary = inference.summary
        metrics = {
            "schema_version": 1,
            "workflow": config["workflow"],
            "profile": profile,
            "seed": int(seed),
            "training_loss": "-log q_phi(theta, eta | D, m)",
            "torch_version": torch.__version__,
            "sbi_version": sbi.__version__,
            "device": device_resolution.as_dict(),
            "train_context_count": int(len(train_indices)),
            "validation_context_count": int(len(validation_indices)),
            "test_context_count": int(len(test_indices)),
            "synthetic_split_policy": (
                "native_parameter_grouped_train_validation_test_v1"
            ),
            "training_input_manifest_sha256": (
                training_input_manifest_sha256
            ),
            "internal_validation_grouped_by_native_parameter": True,
            "outer_test_used_for_training_or_early_stopping": False,
            "trainable_parameter_count": int(
                sum(
                    parameter.numel()
                    for parameter in estimator.parameters()
                    if parameter.requires_grad
                )
            ),
            "epochs_trained": int(summary["epochs_trained"][-1]),
            "training_loss_by_epoch": [
                float(value) for value in summary["training_loss"]
            ],
            "validation_loss_by_epoch": [
                float(value) for value in summary["validation_loss"]
            ],
            # sbi 0.26 reports the positive validation NLL under its
            # historical ``best_validation_loss`` summary key.  Publish both
            # conventions explicitly so user-facing score comparisons cannot
            # silently mix signs.
            "best_validation_log_density": float(
                -summary["best_validation_loss"][-1]
            ),
            "best_validation_negative_log_density": float(
                summary["best_validation_loss"][-1]
            ),
            "mean_train_negative_log_density": float(
                -train_log_prob.mean().item()
            ),
            "mean_test_negative_log_density": None,
            "test_minus_validation_negative_log_density": None,
            "test_set_evaluated": False,
            "validation_only_ensemble_selection_status": "candidate_pending",
            "state_sha256": sha256(state_path),
            "training_elapsed_seconds": elapsed,
            "captured_warnings": [
                str(item.message) for item in captured
            ],
            "smoke_sample_count": 128,
            "smoke_finite_and_in_support": smoke_finite,
            "deterministic_algorithms": bool(
                runtime["deterministic_algorithms"]
            ),
            "torch_num_threads": torch.get_num_threads(),
            **cuda_runtime_metrics(device_resolution),
            "surrogate_used": False,
            "real_data_used": False,
        }
        if not smoke_finite:
            raise RuntimeError("posterior smoke sample is non-finite")
        write_json(metrics_path, metrics)
        seed_metrics[str(seed)] = metrics

    if distributed_world_size > 1:
        torch.distributed.barrier()
        if distributed_rank != 0:
            finish_distributed_training()
            return {
                "schema_version": 1,
                "status": "distributed_worker_complete",
                "rank": distributed_rank,
                "world_size": distributed_world_size,
                "trained_seeds": seeds,
            }
        seed_metrics = {
            str(seed): json.loads(
                (output / f"seed_{seed}" / "training_metrics.json")
                .read_text(encoding="utf-8")
            )
            for seed in all_seeds
        }
        seeds = all_seeds

    # Candidate ranking is frozen exclusively by grouped synthetic validation
    # NLL. Outer-test tensors have not been evaluated above. This prevents one
    # unstable random initialization from entering the released mixture while
    # preserving a genuinely held-out test for the selected estimator.
    active_count = int(settings["active_ensemble_member_count"])
    ranked_seeds = sorted(
        (int(seed) for seed in seeds),
        key=lambda seed: (
            seed_metrics[str(seed)]["best_validation_negative_log_density"],
            seed,
        ),
    )
    active_seeds = ranked_seeds[:active_count]
    for seed in seeds:
        member = seed_metrics[str(seed)]
        selected = int(seed) in active_seeds
        member["selected_by_grouped_validation_nll"] = selected
        member["validation_only_ensemble_selection_status"] = (
            "active" if selected else "rejected_before_outer_test"
        )
        if evaluate_test and selected:
            state_path = (
                output
                / f"seed_{seed}"
                / "model"
                / "density_estimator_state.pt"
            )
            posterior = load_stage10_posterior(
                config,
                state_path,
                theta[fit_indices[:2]],
                context[fit_indices[:2]],
                device=str(device),
            )
            with torch.no_grad():
                test_log_prob = posterior.posterior_estimator.log_prob(
                    theta[test_indices], context[test_indices]
                )
            if not torch.isfinite(test_log_prob).all():
                raise RuntimeError(
                    "selected estimator returned non-finite outer-test scores"
                )
            member["mean_test_negative_log_density"] = float(
                -test_log_prob.mean().item()
            )
            member["test_minus_validation_negative_log_density"] = float(
                -test_log_prob.mean().item()
                - member["best_validation_negative_log_density"]
            )
            member["test_set_evaluated"] = True
        else:
            member["mean_test_negative_log_density"] = None
            member["test_minus_validation_negative_log_density"] = None
            member["test_set_evaluated"] = False
        write_json(
            output / f"seed_{seed}" / "training_metrics.json", member
        )

    active_metrics = [seed_metrics[str(seed)] for seed in active_seeds]
    aggregate = {
        "schema_version": 1,
        "profile": profile,
        "candidate_ensemble_seeds": settings["ensemble_seeds"],
        "active_ensemble_member_count": active_count,
        "active_ensemble_seeds": active_seeds,
        "ensemble_selection_metric": (
            "lowest_grouped_validation_negative_log_density"
        ),
        "outer_test_used_for_ensemble_selection": False,
        "neural_device": device_resolution.as_dict(),
        "maximum_cuda_peak_allocated_bytes": max(
            item["cuda_peak_allocated_bytes"] or 0
            for item in active_metrics
        ),
        "maximum_cuda_peak_reserved_bytes": max(
            item["cuda_peak_reserved_bytes"] or 0
            for item in active_metrics
        ),
        "mean_train_negative_log_density": float(
            np.mean(
                [
                    item["mean_train_negative_log_density"]
                    for item in active_metrics
                ]
            )
        ),
        "mean_test_negative_log_density": (
            float(
                np.mean(
                    [
                        item["mean_test_negative_log_density"]
                        for item in active_metrics
                    ]
                )
            )
            if evaluate_test
            else None
        ),
        "worst_test_minus_validation_negative_log_density": (
            float(
                max(
                    item["test_minus_validation_negative_log_density"]
                    for item in active_metrics
                )
            )
            if evaluate_test
            else None
        ),
        "test_set_evaluated": evaluate_test,
        "synthetic_split_policy": (
            "native_parameter_grouped_train_validation_test_v1"
        ),
        "outer_test_used_for_training_or_early_stopping": False,
        "members": seed_metrics,
        "distributed_training": {
            "strategy": "deterministic_ensemble_member_sharding",
            "world_size": distributed_world_size,
            "one_rank_per_visible_gpu": distributed_world_size > 1,
        },
    }
    write_json(output / "training_summary.json", aggregate)
    finish_distributed_training()
    return aggregate


def _ensemble_samples(
    *,
    config: Mapping[str, Any],
    workspace: Path,
    profile: str,
    context: np.ndarray,
    sample_count: int,
    seed_offset: int,
) -> np.ndarray:
    """Draw an equal-size deterministic mixture of ensemble members."""

    import torch

    from extract_dvcs_cff.inference.stage10 import (
        load_stage10_posterior,
        stage10_latent_to_physical,
    )
    from extract_dvcs_cff.inference.device import resolve_torch_device

    arrays = _load_generated(workspace)
    resolution = resolve_torch_device(config["runtime"]["accelerator"])
    device = torch.device(resolution.resolved)
    example_theta = torch.from_numpy(
        arrays["theta_latent"][:2].astype(np.float32, copy=False)
    ).to(device)
    example_context = torch.from_numpy(
        arrays["contexts"][:2].astype(np.float32, copy=False)
    ).to(device)
    condition = torch.from_numpy(
        context.astype(np.float32, copy=False)
    ).to(device)
    training_summary_path = workspace / "training" / "training_summary.json"
    if not training_summary_path.is_file():
        raise RuntimeError("train must run first; missing training_summary.json")
    training_summary = json.loads(
        training_summary_path.read_text(encoding="utf-8")
    )
    seeds = training_summary.get("active_ensemble_seeds")
    declared_candidates = config["profiles"][profile]["ensemble_seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds)
        != config["profiles"][profile]["active_ensemble_member_count"]
        or not set(seeds).issubset(set(declared_candidates))
        or training_summary.get("outer_test_used_for_ensemble_selection")
        is not False
    ):
        raise RuntimeError("active ensemble selection provenance mismatch")
    base_count, remainder = divmod(sample_count, len(seeds))
    samples = []
    for member_index, seed in enumerate(seeds):
        member_count = base_count + int(member_index < remainder)
        if member_count == 0:
            continue
        state = (
            workspace
            / "training"
            / f"seed_{seed}"
            / "model"
            / "density_estimator_state.pt"
        )
        if not state.exists():
            raise RuntimeError(f"train must run first; missing {state}")
        posterior = load_stage10_posterior(
            config,
            state,
            example_theta,
            example_context,
            device=str(device),
        )
        torch.manual_seed(int(seed) + seed_offset)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed) + seed_offset)
        latent = posterior.sample(
            (member_count,), x=condition, show_progress_bars=False
        )
        physical = (
            stage10_latent_to_physical(latent).detach().cpu().numpy()
        )
        if not np.all(np.isfinite(physical)):
            raise RuntimeError(f"posterior seed {seed} returned non-finite")
        samples.append(physical)
    mixture = np.concatenate(samples, axis=0)
    if len(mixture) != sample_count:
        raise RuntimeError("ensemble mixture allocation is inconsistent")
    return mixture


def _coverage_record(
    samples_by_trial: Sequence[np.ndarray],
    truths: np.ndarray,
    levels: Sequence[float],
) -> dict[str, Any]:
    """Compute equal-tailed empirical coverage and standardized deviations."""

    from extract_dvcs_cff.inference.stage10 import STAGE10_PARAMETER_NAMES

    result: dict[str, Any] = {}
    trial_count = len(samples_by_trial)
    for level in levels:
        covered = np.zeros((trial_count, truths.shape[1]), dtype=bool)
        tail = (1.0 - float(level)) / 2.0
        for trial, samples in enumerate(samples_by_trial):
            lower = np.quantile(samples, tail, axis=0)
            upper = np.quantile(samples, 1.0 - tail, axis=0)
            covered[trial] = (
                (truths[trial] >= lower) & (truths[trial] <= upper)
            )
        empirical = covered.mean(axis=0)
        standard_error = math.sqrt(
            float(level) * (1.0 - float(level)) / trial_count
        )
        result[str(level)] = {
            name: {
                "empirical_coverage": float(empirical[index]),
                "nominal_coverage": float(level),
                "standard_error": standard_error,
                "absolute_deviation_in_standard_errors": float(
                    abs(empirical[index] - float(level)) / standard_error
                ),
            }
            for index, name in enumerate(STAGE10_PARAMETER_NAMES)
        }
    return result


def shadow_width_diagnostic(
    posterior_samples: np.ndarray,
    *,
    minimum_width_ratio: float | None = None,
) -> dict[str, Any]:
    """Describe all inferred DD widths relative to their uniform priors.

    The historical function name is retained for artifact compatibility.
    Shadows are fixed—not inferred—in the milestone contract.
    """

    samples = np.asarray(posterior_samples, dtype=np.float64)
    if (
        samples.ndim != 2
        or samples.shape[1] != _POSTERIOR_PARAMETER_COUNT
        or len(samples) < 20
        or not np.all(np.isfinite(samples))
    ):
        raise ValueError("posterior_samples has incompatible milestone shape")
    directions = {}
    offset = 0
    for gpd_type in _GPD_TYPES:
        for channel in _PARTON_CHANNELS:
            for field in _DD_SHAPE_FIELDS:
                name = _shape_parameter_name(gpd_type, channel, field)
                lower, upper = _DD_SHAPE_BOUNDS[field]
                prior_width90 = 0.9 * (upper - lower)
                width = float(
                    np.quantile(samples[:, offset], 0.95)
                    - np.quantile(samples[:, offset], 0.05)
                )
                directions[name] = {
                    "prior_width90": prior_width90,
                    "posterior_width90": width,
                    "posterior_prior_width90_ratio": width / prior_width90,
                    "passes_requested_descriptive_floor": (
                        None
                        if minimum_width_ratio is None
                        else bool(
                            width / prior_width90 >= minimum_width_ratio
                        )
                    ),
                }
                offset += 1
    ratio = min(
        value["posterior_prior_width90_ratio"]
        for value in directions.values()
    )
    return {
        "posterior_prior_width90_ratio": ratio,
        "directions": directions,
        "requested_descriptive_floor": (
            None
            if minimum_width_ratio is None
            else float(minimum_width_ratio)
        ),
        "passes_requested_descriptive_floor": (
            None
            if minimum_width_ratio is None
            else bool(ratio >= minimum_width_ratio)
        ),
        "interpretation": (
            "descriptive prior-relative width over all 80 inferred DD "
            "controls; shadow settings are fixed simulator inputs"
        ),
    }


def stress_width_reference_diagnostic(
    neural_samples: np.ndarray,
    conventional_samples: np.ndarray,
    *,
    minimum_width_ratio: float,
) -> dict[str, Any]:
    """Reject neural false certainty using the exact posterior as reference.

    The conventional samples are drawn from the same prior and exact-native
    likelihood as the neural target.  A neural central-90% interval narrower
    than the declared fraction of the conventional interval fails.  This
    comparison permits genuine likelihood information to narrow either
    posterior relative to the prior.
    """

    neural = np.asarray(neural_samples, dtype=np.float64)
    conventional = np.asarray(conventional_samples, dtype=np.float64)
    for label, samples in (
        ("neural_samples", neural),
        ("conventional_samples", conventional),
    ):
        if (
            samples.ndim != 2
            or samples.shape[1] != _POSTERIOR_PARAMETER_COUNT
            or len(samples) < 20
            or not np.all(np.isfinite(samples))
        ):
            raise ValueError(f"{label} has incompatible milestone shape")
    if not 0.0 < float(minimum_width_ratio) <= 1.0:
        raise ValueError("minimum_width_ratio must be in (0,1]")

    directions = {}
    for offset in range(_PHYSICS_PARAMETER_COUNT):
        gpd_block, field_index = divmod(offset, len(_DD_SHAPE_FIELDS))
        gpd_index, channel_index = divmod(
            gpd_block, len(_PARTON_CHANNELS)
        )
        name = _shape_parameter_name(
            _GPD_TYPES[gpd_index],
            _PARTON_CHANNELS[channel_index],
            _DD_SHAPE_FIELDS[field_index],
        )
        conventional_width = float(
            np.quantile(conventional[:, offset], 0.95)
            - np.quantile(conventional[:, offset], 0.05)
        )
        neural_width = float(
            np.quantile(neural[:, offset], 0.95)
            - np.quantile(neural[:, offset], 0.05)
        )
        if conventional_width <= 0.0:
            raise ValueError(
                f"conventional {name} stress direction has zero width"
            )
        ratio = neural_width / conventional_width
        directions[name] = {
            "conventional_width90": conventional_width,
            "neural_width90": neural_width,
            "neural_conventional_width90_ratio": ratio,
            "passed": bool(ratio >= minimum_width_ratio),
        }
    ratio = min(
        value["neural_conventional_width90_ratio"]
        for value in directions.values()
    )
    return {
        "neural_conventional_width90_ratio": ratio,
        "directions": directions,
        "minimum_width_ratio": float(minimum_width_ratio),
        "passed": bool(ratio >= minimum_width_ratio),
        "interpretation": (
            "false-certainty check against the conventional posterior from "
            "the same exact-native likelihood; not a CFF-null claim"
        ),
    }


def evaluate_model(
    *,
    bridge: Path,
    configuration_path: Path,
    workspace: Path,
    profile: str,
    show_progress: bool | None = None,
) -> dict[str, Any]:
    """Run held-out coverage and exact-native posterior prediction."""

    config = load_configuration(configuration_path)
    from extract_dvcs_cff.progress import progress_iter

    settings = config["profiles"][profile]
    arrays = _load_generated(workspace)
    test_indices = arrays["test_indices"].astype(np.int64)
    trial_count = min(
        int(settings["coverage_trial_count"]), len(test_indices)
    )
    selected = test_indices[
        np.linspace(0, len(test_indices) - 1, trial_count, dtype=np.int64)
    ]
    samples_by_trial = []
    trial_sample_count = min(
        512, int(settings["posterior_sample_count"])
    )
    coverage_trials = progress_iter(
        enumerate(selected),
        total=trial_count,
        description="Posterior coverage trials",
        enabled=show_progress,
        unit="trial",
    )
    for trial, row in coverage_trials:
        samples_by_trial.append(
            _ensemble_samples(
                config=config,
                workspace=workspace,
                profile=profile,
                context=arrays["contexts"][row],
                sample_count=trial_sample_count,
                seed_offset=1000 + trial,
            )
        )
    truths = arrays["theta_physical"][selected]
    coverage = _coverage_record(
        samples_by_trial, truths, (0.5, 0.8, 0.9)
    )
    max_coverage_deviation = max(
        item["absolute_deviation_in_standard_errors"]
        for level in coverage.values()
        for item in level.values()
    )

    posterior_samples = _ensemble_samples(
        config=config,
        workspace=workspace,
        profile=profile,
        context=arrays["pseudodata_context"],
        sample_count=int(settings["posterior_sample_count"]),
        seed_offset=2000,
    )
    evaluation_directory = workspace / "evaluation"
    evaluation_directory.mkdir(parents=True, exist_ok=True)
    with (evaluation_directory / "posterior_samples.npy").open("wb") as stream:
        np.save(stream, posterior_samples, allow_pickle=False)

    exact_count = int(settings["exact_reevaluation_sample_count"])
    exact_indices = np.linspace(
        0, len(posterior_samples) - 1, exact_count, dtype=np.int64
    )
    exact_parameters = posterior_samples[
        exact_indices, :_PHYSICS_PARAMETER_COUNT
    ]
    physics_path = _physics_path(configuration_path, config)
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    client = NativeCache(bridge, workspace)
    exact_parallel = _parallel_partitioned_native_evaluation(
        client=client,
        parameters=exact_parameters,
        config=config,
        physics=physics,
        native_workers=config["runtime"]["native_workers"],
        description="Exact posterior reevaluation",
        show_progress=show_progress,
    )
    exact_native = exact_parallel.result
    valid_exact_indices = np.flatnonzero(exact_native.valid_mask)
    if len(valid_exact_indices) == 0:
        raise RuntimeError("every exact posterior reevaluation was invalid")
    valid_cffs = exact_native.cffs[valid_exact_indices]
    with (evaluation_directory / "exact_native_cffs.npy").open("wb") as stream:
        np.save(stream, valid_cffs, allow_pickle=False)
    with (
        evaluation_directory / "exact_native_parameter_samples.npy"
    ).open("wb") as stream:
        np.save(
            stream,
            exact_parameters[valid_exact_indices],
            allow_pickle=False,
        )
    grid = config["diagnostics"]["gpd_x_grid"]
    gpd_x = np.linspace(
        float(grid["minimum"]),
        float(grid["maximum"]),
        int(grid["point_count"]),
        dtype=np.float64,
    )
    diagnostic_config = {
        **config,
        "kinematics": [
            config["diagnostics"]["gpd_reference_kinematics"]
        ],
    }
    diagnostic_parameters = np.vstack(
        (
            arrays["truth_theta_physical"][
                None, :_PHYSICS_PARAMETER_COUNT
            ],
            exact_parameters[valid_exact_indices],
        )
    )
    diagnostic_parallel = _parallel_partitioned_native_evaluation(
        client=client,
        parameters=diagnostic_parameters,
        config=diagnostic_config,
        physics=physics,
        native_workers=config["runtime"]["native_workers"],
        description="Exact GPD diagnostics",
        show_progress=show_progress,
        gpd_x_grid=gpd_x,
    )
    diagnostic_native = diagnostic_parallel.result
    if diagnostic_native.invalid_records:
        raise RuntimeError(
            "exact GPD diagnostics contain invalid native samples: "
            f"{len(diagnostic_native.invalid_records)} of "
            f"{len(diagnostic_parameters)}"
        )
    truth_gpds = diagnostic_native.gpds[0, 0]
    posterior_gpds = diagnostic_native.gpds[1:, 0]
    with (evaluation_directory / "gpd_x_grid.npy").open("wb") as stream:
        np.save(stream, gpd_x, allow_pickle=False)
    with (evaluation_directory / "truth_gpds.npy").open("wb") as stream:
        np.save(stream, truth_gpds, allow_pickle=False)
    with (
        evaluation_directory / "posterior_exact_native_gpds.npy"
    ).open("wb") as stream:
        np.save(stream, posterior_gpds, allow_pickle=False)
    global_response = arrays["global_normalization_response"]
    lu_response = arrays["lu_normalization_response"]
    exact_predictions = np.asarray(
        [
            _effective_prediction(
                exact_native.predictions[index],
                posterior_samples[
                    exact_indices[index], _PHYSICS_PARAMETER_COUNT
                ],
                posterior_samples[
                    exact_indices[index], _PHYSICS_PARAMETER_COUNT + 1
                ],
                global_response,
                lu_response,
            )
            for index in valid_exact_indices
        ]
    )
    covariance = arrays["covariance_normalized"]
    rng = np.random.Generator(
        np.random.PCG64(int(config["seeds"]["comparison"]))
    )
    predictive = exact_predictions + (
        rng.standard_normal(exact_predictions.shape)
        @ np.linalg.cholesky(covariance).T
    )
    observed = arrays["pseudodata_observed_normalized"]
    lower = np.quantile(predictive, 0.05, axis=0)
    upper = np.quantile(predictive, 0.95, axis=0)
    predictive_coverage = float(np.mean((observed >= lower) & (observed <= upper)))

    gates = config["frozen_validation_gates"]
    stress_prior_widths = shadow_width_diagnostic(posterior_samples)
    training = json.loads(
        (workspace / "training" / "training_summary.json").read_text(
            encoding="utf-8"
        )
    )
    checks = {
        "test_density_generalizes": bool(
            training[
                "worst_test_minus_validation_negative_log_density"
            ]
            <= gates["maximum_test_nll_minus_validation_nll"]
        ),
        "coverage_calibrated": bool(
            max_coverage_deviation
            <= gates["maximum_coverage_standard_error"]
        ),
        "posterior_predictive_coverage": bool(
            predictive_coverage
            >= gates["minimum_posterior_predictive_point_90_coverage"]
        ),
        "exact_reevaluation_valid": bool(
            len(exact_native.invalid_records) / exact_count
            <= gates["maximum_exact_reevaluation_invalid_fraction"]
        ),
    }
    record = {
        "schema_version": 1,
        "profile": profile,
        "coverage_trial_count": trial_count,
        "posterior_samples_per_coverage_trial": trial_sample_count,
        "coverage": coverage,
        "maximum_coverage_deviation_in_standard_errors": (
            max_coverage_deviation
        ),
        "posterior_predictive": {
            "exact_native_sample_count": len(valid_exact_indices),
            "pointwise_90_coverage": predictive_coverage,
            "observed": observed.tolist(),
            "lower90": lower.tolist(),
            "median": np.quantile(predictive, 0.5, axis=0).tolist(),
            "upper90": upper.tolist(),
            "injected_truth_native": arrays[
                "truth_native_predictions_normalized"
            ].tolist(),
        },
        "cff_posterior": {
            "types": ["H", "E", "Htilde", "Etilde"],
            "components": ["real", "imaginary"],
            "kinematics": config["kinematics"],
            "truth": arrays["truth_cffs"].tolist(),
            "lower90": np.quantile(valid_cffs, 0.05, axis=0).tolist(),
            "median": np.quantile(valid_cffs, 0.5, axis=0).tolist(),
            "upper90": np.quantile(valid_cffs, 0.95, axis=0).tolist(),
        },
        "gpd_posterior": {
            "types": ["H", "E", "Htilde", "Etilde"],
            "components": [
                "u_value",
                "u_plus",
                "u_minus",
                "d_value",
                "d_plus",
                "d_minus",
                "s_value",
                "s_plus",
                "s_minus",
                "gluon",
                "charge_squared_weighted_c_even_quark_sum",
            ],
            "reference_kinematics": config["diagnostics"][
                "gpd_reference_kinematics"
            ],
            "x_grid": gpd_x.tolist(),
            "truth": truth_gpds.tolist(),
            "lower90": np.quantile(
                posterior_gpds, 0.05, axis=0
            ).tolist(),
            "median": np.quantile(
                posterior_gpds, 0.5, axis=0
            ).tolist(),
            "upper90": np.quantile(
                posterior_gpds, 0.95, axis=0
            ).tolist(),
            "native_cache_keys": [
                item.cache_key
                for item in diagnostic_native.successful_batches
            ],
        },
        "stress_direction_prior_widths": stress_prior_widths,
        "test_scores": {
            "mean_train_negative_log_density": training[
                "mean_train_negative_log_density"
            ],
            "mean_test_negative_log_density": training[
                "mean_test_negative_log_density"
            ],
            "worst_test_minus_validation_negative_log_density": training[
                "worst_test_minus_validation_negative_log_density"
            ],
        },
        "exact_reevaluation": {
            "native_cache_keys": [
                item.cache_key for item in exact_native.successful_batches
            ],
            "native_evaluation_count": sum(
                item.evaluation_count
                for item in exact_native.successful_batches
            ),
            "requested_count": exact_count,
            "valid_count": len(valid_exact_indices),
            "invalid_count": len(exact_native.invalid_records),
            "invalid_fraction": len(exact_native.invalid_records)
            / exact_count,
            "invalid_records": list(exact_native.invalid_records),
            "surrogate_used": False,
            "parallel_execution": {
                "chunk_size": exact_parallel.chunk_size,
                "task_count": exact_parallel.task_count,
                "worker_resolution": exact_parallel.worker_resolution,
            },
        },
        "gpd_diagnostic_reevaluation": {
            "requested_count": len(diagnostic_parameters),
            "valid_count": int(
                np.count_nonzero(diagnostic_native.valid_mask)
            ),
            "invalid_count": len(diagnostic_native.invalid_records),
            "native_cache_keys": [
                item.cache_key
                for item in diagnostic_native.successful_batches
            ],
            "parallel_execution": {
                "chunk_size": diagnostic_parallel.chunk_size,
                "task_count": diagnostic_parallel.task_count,
                "worker_resolution": diagnostic_parallel.worker_resolution,
            },
            "surrogate_used": False,
        },
        "gate_checks": checks,
        "passed": bool(all(checks.values())),
        "real_data_used": False,
    }
    write_json(evaluation_directory / "evaluation_metrics.json", record)
    return record


def _sliced_wasserstein(
    first: np.ndarray,
    second: np.ndarray,
    rng: np.random.Generator,
    directions: int = 128,
) -> float:
    from scipy.stats import wasserstein_distance

    projections = rng.normal(size=(directions, first.shape[1]))
    projections /= np.linalg.norm(projections, axis=1)[:, None]
    return float(
        np.mean(
            [
                wasserstein_distance(
                    first @ direction, second @ direction
                )
                for direction in projections
            ]
        )
    )


def compare_posteriors(
    *,
    configuration_path: Path,
    workspace: Path,
    profile: str,
    show_progress: bool | None = None,
) -> dict[str, Any]:
    """Compare neural inference with exact-native prior importance sampling."""

    from scipy.stats import wasserstein_distance

    from extract_dvcs_cff.inference.stage10 import STAGE10_PARAMETER_NAMES
    from extract_dvcs_cff.progress import progress_iter

    config = load_configuration(configuration_path)
    settings = config["profiles"][profile]
    arrays = _load_generated(workspace)
    native_parameters = arrays["native_parameters"]
    native_predictions = arrays["native_predictions_normalized"]
    nuisance_draws = int(
        settings["conventional_nuisance_draws_per_parameter"]
    )
    rng = np.random.Generator(
        np.random.PCG64(int(config["seeds"]["conventional_nuisance"]))
    )
    eta = rng.standard_normal(
        (len(native_parameters), nuisance_draws, 2)
    )
    native_dimension = int(native_parameters.shape[1])
    nuisance_dimension = int(eta.shape[2])
    posterior_dimension = len(STAGE10_PARAMETER_NAMES)
    if native_dimension + nuisance_dimension != posterior_dimension:
        raise RuntimeError(
            "conventional posterior dimension mismatch: "
            f"native={native_dimension}, nuisance={nuisance_dimension}, "
            f"declared={posterior_dimension}"
        )
    proposal_theta = np.empty(
        (len(native_parameters) * nuisance_draws, posterior_dimension),
        dtype=np.float64,
    )
    proposal_prediction = np.empty(
        (
            len(proposal_theta),
            native_predictions.shape[1],
        ),
        dtype=np.float64,
    )
    global_response = arrays["global_normalization_response"]
    lu_response = arrays["lu_normalization_response"]
    row = 0
    parameter_rows = progress_iter(
        enumerate(native_parameters),
        total=len(native_parameters),
        description="Conventional posterior proposals",
        enabled=show_progress,
        unit="parameter",
    )
    for parameter_index, parameters in parameter_rows:
        for nuisance_index in range(nuisance_draws):
            proposal_theta[row] = (
                *parameters,
                *eta[parameter_index, nuisance_index],
            )
            proposal_prediction[row] = _effective_prediction(
                native_predictions[parameter_index],
                eta[parameter_index, nuisance_index, 0],
                eta[parameter_index, nuisance_index, 1],
                global_response,
                lu_response,
            )
            row += 1

    observed = arrays["pseudodata_observed_normalized"]
    cholesky = np.linalg.cholesky(arrays["covariance_normalized"])
    residual = observed[None, :] - proposal_prediction
    whitened = np.linalg.solve(cholesky, residual.T).T
    log_likelihood = -0.5 * np.sum(whitened**2, axis=1)
    log_likelihood -= np.max(log_likelihood)
    weights = np.exp(log_likelihood)
    weights /= np.sum(weights)
    ess = float(1.0 / np.sum(weights**2))
    sample_count = int(settings["posterior_sample_count"])
    resample_rng = np.random.Generator(
        np.random.PCG64(int(config["seeds"]["comparison"]))
    )
    resampled_indices = resample_rng.choice(
        len(proposal_theta), size=sample_count, replace=True, p=weights
    )
    conventional = proposal_theta[resampled_indices]
    neural = _ensemble_samples(
        config=config,
        workspace=workspace,
        profile=profile,
        context=arrays["pseudodata_context"],
        sample_count=sample_count,
        seed_offset=3000,
    )
    mean = conventional.mean(axis=0)
    sd = conventional.std(axis=0, ddof=1)
    if np.any(sd <= 0.0):
        raise RuntimeError("conventional posterior has zero marginal width")
    standardized_conventional = (conventional - mean) / sd
    standardized_neural = (neural - mean) / sd
    comparison_rng = np.random.Generator(
        np.random.PCG64(int(config["seeds"]["comparison"]) + 1)
    )
    sliced = _sliced_wasserstein(
        standardized_conventional,
        standardized_neural,
        comparison_rng,
    )
    marginal = {
        name: float(
            wasserstein_distance(
                conventional[:, index], neural[:, index]
            )
            / sd[index]
        )
        for index, name in enumerate(STAGE10_PARAMETER_NAMES)
    }
    intervals = {}
    truth = arrays["truth_theta_physical"]
    for index, name in enumerate(STAGE10_PARAMETER_NAMES):
        intervals[name] = {
            "truth": float(truth[index]),
            "conventional_q05": float(
                np.quantile(conventional[:, index], 0.05)
            ),
            "conventional_q50": float(
                np.quantile(conventional[:, index], 0.5)
            ),
            "conventional_q95": float(
                np.quantile(conventional[:, index], 0.95)
            ),
            "neural_q05": float(np.quantile(neural[:, index], 0.05)),
            "neural_q50": float(np.quantile(neural[:, index], 0.5)),
            "neural_q95": float(np.quantile(neural[:, index], 0.95)),
        }
    gates = config["frozen_validation_gates"]
    stress_width_reference = stress_width_reference_diagnostic(
        neural,
        conventional,
        minimum_width_ratio=gates[
            "minimum_neural_to_conventional_stress_width_ratio"
        ],
    )
    checks = {
        "conventional_effective_sample_size": bool(
            ess >= gates["minimum_conventional_effective_sample_size"]
        ),
        "ensemble_sliced_wasserstein": bool(
            sliced <= gates["maximum_ensemble_sliced_wasserstein"]
        ),
        "marginal_wasserstein": bool(
            max(marginal.values())
            <= gates[
                "maximum_marginal_wasserstein_over_conventional_sd"
            ]
        ),
        "stress_directions_not_artificially_narrower_than_conventional": (
            bool(stress_width_reference["passed"])
        ),
    }
    output = workspace / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "conventional_posterior_samples.npy").open("wb") as stream:
        np.save(stream, conventional, allow_pickle=False)
    with (output / "neural_posterior_samples.npy").open("wb") as stream:
        np.save(stream, neural, allow_pickle=False)
    record = {
        "schema_version": 1,
        "profile": profile,
        "conventional_method": (
            "self_normalized_prior_importance_sampling_with_exact_native_"
            "predictions_and_explicit_prior_nuisance_draws"
        ),
        "proposal_count": len(proposal_theta),
        "posterior_dimension": posterior_dimension,
        "native_parameter_dimension": native_dimension,
        "nuisance_dimension": nuisance_dimension,
        "effective_sample_size": ess,
        "sample_count": sample_count,
        "ensemble_sliced_wasserstein": sliced,
        "marginal_wasserstein_over_conventional_sd": marginal,
        "intervals": intervals,
        "stress_direction_width_reference": stress_width_reference,
        "gate_checks": checks,
        "passed": bool(all(checks.values())),
        "surrogate_used": False,
        "real_data_used": False,
    }
    write_json(output / "comparison_metrics.json", record)
    return record


def _holdout_manifest(workspace: Path) -> dict[str, str]:
    """Hash every artifact that is forbidden to change during holdout use."""

    protected: list[Path] = []
    for relative in ("generated", "training", "optimization"):
        directory = workspace / relative
        if directory.is_dir():
            protected.extend(path for path in directory.rglob("*") if path.is_file())
    for relative in (
        "workspace_contract.json",
        "evaluation/evaluation_metrics.json",
        "comparison/comparison_metrics.json",
    ):
        path = workspace / relative
        if path.is_file():
            protected.append(path)
    return {
        str(path.relative_to(workspace)): sha256(path)
        for path in sorted(set(protected))
    }


def _native_holdout_request(
    *,
    model: str,
    request_id: str,
    kinematics: Mapping[str, float],
    physics: Mapping[str, Any],
    gpd_x: Sequence[float] = (),
) -> dict[str, Any]:
    """Build the strict post-training named-model bridge request."""

    request: dict[str, Any] = {
        "request_id": request_id,
        "representation": "stage11_native_model_holdout_v1",
        "model": model,
        "observable_kinematics": {
            "x_b": kinematics["x_b"],
            "t": {"value": kinematics["t_GeV2"], "unit": "GeV2"},
            "Q2": {"value": kinematics["Q2_GeV2"], "unit": "GeV2"},
            "beam_energy": {
                "value": kinematics["beam_energy_GeV"],
                "unit": "GeV",
            },
            "phi": {"value": kinematics["phi_rad"], "unit": "rad"},
        },
        "theory_configuration": physics["theory_configuration"],
    }
    if len(gpd_x) != 0:
        request["gpd_diagnostic_x"] = [float(value) for value in gpd_x]
    return request


def _gpd_diagnostic_array(
    diagnostics: Sequence[Mapping[str, Any]], gpd_x: np.ndarray
) -> np.ndarray:
    """Convert native structured GPD diagnostics to the frozen component order."""

    result = np.empty((len(gpd_x), 4, 11), dtype=np.float64)
    for x_index, point in enumerate(diagnostics):
        if not math.isclose(
            float(point["x"]), float(gpd_x[x_index]), rel_tol=0.0, abs_tol=1e-15
        ):
            raise RuntimeError("named-model GPD diagnostic x order changed")
        for gpd_index, gpd_name in enumerate(("H", "E", "Htilde", "Etilde")):
            gpd = point["gpd"][gpd_name]
            result[x_index, gpd_index] = (
                float(gpd["u"]["value"]),
                float(gpd["u"]["plus"]),
                float(gpd["u"]["minus"]),
                float(gpd["d"]["value"]),
                float(gpd["d"]["plus"]),
                float(gpd["d"]["minus"]),
                float(gpd["s"]["value"]),
                float(gpd["s"]["plus"]),
                float(gpd["s"]["minus"]),
                float(gpd["gluon"]),
                float(gpd["charge_squared_weighted_c_even_quark_sum"]),
            )
    if not np.all(np.isfinite(result)):
        raise RuntimeError("named-model GPD diagnostics are non-finite")
    return result


def _evaluate_native_named_model(
    *,
    bridge: Path,
    workspace: Path,
    config: Mapping[str, Any],
    physics: Mapping[str, Any],
    model: str,
    gpd_x: np.ndarray,
) -> dict[str, Any]:
    """Evaluate one named model exactly and persist its raw native transaction."""

    requests = [
        _native_holdout_request(
            model=model,
            request_id=f"{model}-kin-{index:03d}",
            kinematics=kinematics,
            physics=physics,
        )
        for index, kinematics in enumerate(config["kinematics"])
    ]
    requests.append(
        _native_holdout_request(
            model=model,
            request_id=f"{model}-gpd-diagnostic",
            kinematics=config["diagnostics"]["gpd_reference_kinematics"],
            physics=physics,
            gpd_x=gpd_x,
        )
    )
    payload = {
        "schema_version": 1,
        "operation": "batch_evaluate_native_model_holdout_dvcs",
        "requests": requests,
    }
    directory = workspace / "holdout" / model / "native_truth"
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "request.json"
    response_path = directory / "response.json"
    stdout_path = directory / "stdout.txt"
    stderr_path = directory / "stderr.txt"
    exit_path = directory / "exit_code.txt"
    request_path.write_text(pretty_json(payload) + "\n", encoding="utf-8")
    process = subprocess.run(
        [str(bridge.resolve(strict=True)), "--input", str(request_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    stdout_path.write_text(process.stdout, encoding="utf-8")
    stderr_path.write_text(process.stderr, encoding="utf-8")
    exit_path.write_text(f"{process.returncode}\n", encoding="utf-8")
    if process.returncode != 0 or process.stderr:
        raise RuntimeError(
            f"native {model} holdout failed; see {stdout_path}, "
            f"{stderr_path}, and {exit_path}"
        )
    response = json.loads(process.stdout)
    response_path.write_text(pretty_json(response) + "\n", encoding="utf-8")
    evaluations = response.get("evaluations", ())
    if (
        response.get("status") != "ok"
        or response.get("operation")
        != "batch_evaluate_native_model_holdout_dvcs"
        or len(evaluations) != len(requests)
    ):
        raise RuntimeError("named-model native response violates its contract")
    required_holdout = {
        "training_use": False,
        "validation_use": False,
        "optuna_use": False,
        "early_stopping_use": False,
        "post_training_only": True,
        "real_data_used": False,
        "external_native_validation_use": True,
        "replaces_dd_outer_test": False,
    }
    if any(item.get("holdout_contract") != required_holdout for item in evaluations):
        raise RuntimeError("native named-model leakage contract changed")

    observable_ids = [item["id"] for item in config["observables"]]
    scales = {
        item["id"]: float(item["normalization_scale"])
        for item in config["observables"]
    }
    predictions = np.empty(len(config["kinematics"]) * len(observable_ids))
    cffs = np.empty((len(config["kinematics"]), 4, 2))
    for kinematic_index, evaluation in enumerate(evaluations[:-1]):
        native = evaluation["result"]
        for observable_index, observable_id in enumerate(observable_ids):
            predictions[
                kinematic_index * len(observable_ids) + observable_index
            ] = float(native["observables"][observable_id]["value"]) / scales[
                observable_id
            ]
        for gpd_index, gpd_name in enumerate(("H", "E", "Htilde", "Etilde")):
            cff = native["cffs"][gpd_name]
            cffs[kinematic_index, gpd_index] = (
                float(cff["real"]),
                float(cff["imaginary"]),
            )
    gpds = _gpd_diagnostic_array(
        evaluations[-1]["result"]["gpd_diagnostics"], gpd_x
    )
    if not np.all(np.isfinite(predictions)) or not np.all(np.isfinite(cffs)):
        raise RuntimeError("named-model truth contains non-finite values")
    return {
        "predictions": predictions,
        "cffs": cffs,
        "gpds": gpds,
        "request_sha256": sha256(request_path),
        "response_sha256": sha256(response_path),
        "bridge_sha256": sha256(bridge.resolve(strict=True)),
        "source_provenance": evaluations[0]["source_provenance"],
        "forward_pdf": evaluations[0].get("forward_pdf"),
    }


def _write_native_holdout_plots(
    *,
    output: Path,
    config: Mapping[str, Any],
    model: str,
    record: Mapping[str, Any],
) -> dict[str, str]:
    """Write observable, CFF, and GPD truth-versus-extraction diagnostics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    phi = np.asarray([item["phi_rad"] for item in config["kinematics"]])
    observable = record["observable_comparison"]
    truth = np.asarray(observable["native_model_truth"])
    lower = np.asarray(observable["extracted_lower90"])
    median = np.asarray(observable["extracted_median"])
    upper = np.asarray(observable["extracted_upper90"])
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for observable_index, (axis, item) in enumerate(
        zip(axes.flat, config["observables"])
    ):
        indices = np.arange(observable_index, len(truth), len(config["observables"]))
        order = np.argsort(phi)
        axis.fill_between(
            phi[order], lower[indices][order], upper[indices][order], alpha=0.25,
            label="frozen NPE + exact DD PARTONS 90%",
        )
        axis.plot(phi[order], median[indices][order], "o-", label="extracted median")
        axis.plot(phi[order], truth[indices][order], "s--", label=f"exact {model}")
        axis.set(title=item["user_label"], xlabel=r"$\phi$ [rad]", ylabel="normalized observable")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    observable_path = output / "observable_truth_vs_extraction.png"
    figure.suptitle(f"{model}: unseen native-model observables")
    figure.savefig(observable_path, dpi=150)
    plt.close(figure)

    cff = record["cff_comparison"]
    cff_truth = np.asarray(cff["native_model_truth"])
    cff_lower = np.asarray(cff["extracted_lower90"])
    cff_median = np.asarray(cff["extracted_median"])
    cff_upper = np.asarray(cff["extracted_upper90"])
    figure, axes = plt.subplots(4, 2, figsize=(13, 14), constrained_layout=True)
    for gpd_index, name in enumerate(cff["types"]):
        for component, label in enumerate(("real", "imaginary")):
            axis = axes[gpd_index, component]
            order = np.argsort(phi)
            axis.fill_between(phi[order], cff_lower[:, gpd_index, component][order], cff_upper[:, gpd_index, component][order], alpha=0.25)
            axis.plot(phi[order], cff_median[:, gpd_index, component][order], "o-", label="extracted median")
            axis.plot(phi[order], cff_truth[:, gpd_index, component][order], "s--", label=f"exact {model}")
            axis.set(title=f"{label} {name} CFF", xlabel=r"$\phi$ [rad]")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    cff_path = output / "cff_truth_vs_extraction.png"
    figure.suptitle(f"{model}: exact-native CFF comparison")
    figure.savefig(cff_path, dpi=150)
    plt.close(figure)

    gpd = record["gpd_comparison"]
    x = np.asarray(gpd["x_grid"])
    gpd_truth = np.asarray(gpd["native_model_truth"])
    gpd_lower = np.asarray(gpd["extracted_lower90"])
    gpd_median = np.asarray(gpd["extracted_median"])
    gpd_upper = np.asarray(gpd["extracted_upper90"])
    figure, axes = plt.subplots(4, 4, figsize=(17, 14), constrained_layout=True)
    components = ((0, "u"), (3, "d"), (6, "s"), (9, "gluon"))
    for gpd_index, name in enumerate(gpd["types"]):
        for column, (component, flavor) in enumerate(components):
            axis = axes[gpd_index, column]
            axis.fill_between(x, gpd_lower[:, gpd_index, component], gpd_upper[:, gpd_index, component], alpha=0.25)
            axis.plot(x, gpd_median[:, gpd_index, component], label="extracted median")
            axis.plot(x, gpd_truth[:, gpd_index, component], "--", label=f"exact {model}")
            axis.set(title=f"{name}, {flavor}", xlabel=r"$x$")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7)
    gpd_path = output / "gpd_truth_vs_extraction.png"
    figure.suptitle(f"{model}: unseen native-model GPD comparison")
    figure.savefig(gpd_path, dpi=150)
    plt.close(figure)
    return {
        "observables": str(observable_path),
        "cffs": str(cff_path),
        "gpds": str(gpd_path),
    }


def evaluate_native_model_holdouts(
    *,
    bridge: Path,
    configuration_path: Path,
    workspace: Path,
    profile: str,
    show_progress: bool | None = None,
) -> dict[str, Any]:
    """Externally test the frozen DD NPE on unseen native model families.

    This operation runs only after grouped DD closure.  It neither replaces
    the DD outer test nor exposes native-model values to learning or model
    selection.
    """

    from extract_dvcs_cff.inference.stage10 import build_stage10_context
    from extract_dvcs_cff.progress import progress_iter

    config = load_configuration(configuration_path)
    repository = configuration_path.resolve(strict=True).parents[2]
    blind_path = (
        repository
        / "configs/validation/stage11_blind_fresh_kinematics_v1.json"
    ).resolve(strict=True)
    blind = json.loads(blind_path.read_text(encoding="utf-8"))
    expected_hash = blind.pop("manifest_content_sha256", None)
    observed_hash = hashlib.sha256(
        canonical(blind).encode("utf-8")
    ).hexdigest()
    if expected_hash != observed_hash:
        raise RuntimeError("blind kinematics manifest hash changed")
    if (
        blind.get("protocol") != "output_blind_fresh_kinematics_v1"
        or blind.get("selection_frozen_before_native_outputs") is not True
        or blind.get("native_outputs_generated") is not False
        or blind.get("training_use") is not False
        or blind.get("architecture_selection_use") is not False
        or blind.get("optuna_objective_use") is not False
    ):
        raise RuntimeError("blind kinematics leakage contract changed")
    blind_points = blind.get("npe_dataset_points", ())
    if len(blind_points) != len(config["kinematics"]):
        raise RuntimeError("blind NPE dataset must preserve token count")
    holdout_config = {
        **config,
        "kinematics": [
            {
                "x_b": float(item["x_b"]),
                "t_GeV2": float(item["t_GeV2"]),
                "Q2_GeV2": float(item["Q2_GeV2"]),
                "beam_energy_GeV": float(item["beam_energy_GeV"]),
                "phi_rad": float(item["phi_rad"]),
            }
            for item in blind_points
        ],
        "diagnostics": {
            **config["diagnostics"],
            "gpd_reference_kinematics": {
                "x_b": float(blind_points[0]["x_b"]),
                "t_GeV2": float(blind_points[0]["t_GeV2"]),
                "Q2_GeV2": float(blind_points[0]["Q2_GeV2"]),
                "beam_energy_GeV": float(blind_points[0]["beam_energy_GeV"]),
                "phi_rad": float(blind_points[0]["phi_rad"]),
            },
        },
    }
    evaluation_path = workspace / "evaluation/evaluation_metrics.json"
    comparison_path = workspace / "comparison/comparison_metrics.json"
    if not evaluation_path.is_file() or not comparison_path.is_file():
        raise RuntimeError("passed synthetic closure must exist before holdout")
    closure = {
        "evaluation": json.loads(evaluation_path.read_text(encoding="utf-8")),
        "comparison": json.loads(comparison_path.read_text(encoding="utf-8")),
    }
    if not closure["evaluation"].get("passed") or not closure["comparison"].get("passed"):
        raise RuntimeError("synthetic closure gates failed; native holdout is blocked")
    before = _holdout_manifest(workspace)
    forbidden_names = _NATIVE_VALIDATION_MODELS
    training_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for relative in ("generated", "training", "optimization")
        for path in (workspace / relative).rglob("*.json")
        if (workspace / relative).is_dir()
    )
    if any(name in training_text for name in forbidden_names):
        raise RuntimeError("named holdout model leaked into pre-holdout artifacts")

    physics_path = _physics_path(configuration_path, config)
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    _load_generated(workspace)  # integrity check; holdout never edits corpus
    point_table = _point_table(holdout_config)
    grid = config["diagnostics"]["gpd_x_grid"]
    gpd_x = np.linspace(grid["minimum"], grid["maximum"], grid["point_count"])
    exact_count = int(config["profiles"][profile]["exact_reevaluation_sample_count"])
    client = NativeCache(bridge, workspace)
    records: dict[str, Any] = {}
    for model in progress_iter(
            _NATIVE_VALIDATION_MODELS,
            total=len(_NATIVE_VALIDATION_MODELS),
            description="External native-model validation",
            enabled=show_progress,
            unit="model",
        ):
        truth = _evaluate_native_named_model(
            bridge=bridge,
            workspace=workspace,
            config=holdout_config,
            physics=physics,
            model=model,
            gpd_x=gpd_x,
        )
        seed_offsets = _NATIVE_VALIDATION_SEED_OFFSETS[model]
        covariance, global_response, lu_response, covariance_diagnostics = (
            _covariance_and_responses(holdout_config, truth["predictions"])
        )
        noise_seed = int(config["seeds"]["comparison"]) + seed_offsets["noise"]
        noise_rng = np.random.Generator(np.random.PCG64(noise_seed))
        observed = truth["predictions"] + np.linalg.cholesky(covariance) @ noise_rng.standard_normal(len(point_table))
        context = build_stage10_context(
            points=point_table,
            observed=observed,
            covariance=covariance,
            global_normalization_response=global_response,
            lu_normalization_response=lu_response,
        )
        posterior = _ensemble_samples(
            config=config,
            workspace=workspace,
            profile=profile,
            context=context,
            sample_count=int(config["profiles"][profile]["posterior_sample_count"]),
            seed_offset=seed_offsets["posterior"],
        )
        exact_indices = np.linspace(0, len(posterior) - 1, exact_count, dtype=np.int64)
        exact = _partitioned_native_evaluation(
            client=client,
            parameters=posterior[
                exact_indices, :_PHYSICS_PARAMETER_COUNT
            ],
            config=holdout_config,
            physics=physics,
            operation="batch_evaluate_post_training_comparison_dvcs",
        )
        valid = np.flatnonzero(exact.valid_mask)
        if len(valid) == 0:
            raise RuntimeError(f"all extracted {model} DD samples are native-invalid")
        predictions = np.asarray([
            _effective_prediction(
                exact.predictions[index],
                posterior[exact_indices[index], _PHYSICS_PARAMETER_COUNT],
                posterior[
                    exact_indices[index], _PHYSICS_PARAMETER_COUNT + 1
                ],
                global_response,
                lu_response,
            )
            for index in valid
        ])
        diagnostic_config = {
            **holdout_config,
            "kinematics": [
                holdout_config["diagnostics"]["gpd_reference_kinematics"]
            ],
        }
        diagnostic = client.evaluate(
            parameters=posterior[
                exact_indices[valid], :_PHYSICS_PARAMETER_COUNT
            ],
            config=diagnostic_config,
            physics=physics,
            gpd_x_grid=gpd_x,
            operation="batch_evaluate_post_training_comparison_dvcs",
        )
        extracted_gpds = diagnostic.gpds[:, 0]
        observable_lower = np.quantile(predictions, 0.05, axis=0)
        observable_median = np.quantile(predictions, 0.5, axis=0)
        observable_upper = np.quantile(predictions, 0.95, axis=0)
        cff_lower = np.quantile(exact.cffs[valid], 0.05, axis=0)
        cff_median = np.quantile(exact.cffs[valid], 0.5, axis=0)
        cff_upper = np.quantile(exact.cffs[valid], 0.95, axis=0)
        gpd_lower = np.quantile(extracted_gpds, 0.05, axis=0)
        gpd_median = np.quantile(extracted_gpds, 0.5, axis=0)
        gpd_upper = np.quantile(extracted_gpds, 0.95, axis=0)
        observable_coverage = float(np.mean((truth["predictions"] >= observable_lower) & (truth["predictions"] <= observable_upper)))
        cff_coverage = float(np.mean((truth["cffs"] >= cff_lower) & (truth["cffs"] <= cff_upper)))
        gpd_coverage = float(np.mean((truth["gpds"] >= gpd_lower) & (truth["gpds"] <= gpd_upper)))
        record: dict[str, Any] = {
            "model": model,
            "role": "post_training_external_native_validation_test",
            "conditioning": {
                "observed_normalized": observed.tolist(),
                "native_model_mean_normalized": truth["predictions"].tolist(),
                "noise_seed": noise_seed,
                "covariance_source": "frozen_synthetic_training_contract",
                "covariance_diagnostics": covariance_diagnostics,
                "nuisance_truth": [0.0, 0.0],
            },
            "observable_comparison": {
                "native_model_truth": truth["predictions"].tolist(),
                "extracted_lower90": observable_lower.tolist(),
                "extracted_median": observable_median.tolist(),
                "extracted_upper90": observable_upper.tolist(),
                "pointwise_90_truth_coverage": observable_coverage,
            },
            "cff_comparison": {
                "types": ["H", "E", "Htilde", "Etilde"],
                "components": ["real", "imaginary"],
                "native_model_truth": truth["cffs"].tolist(),
                "extracted_lower90": cff_lower.tolist(),
                "extracted_median": cff_median.tolist(),
                "extracted_upper90": cff_upper.tolist(),
                "pointwise_90_truth_coverage": cff_coverage,
            },
            "gpd_comparison": {
                "types": ["H", "E", "Htilde", "Etilde"],
                "components": ["u_value", "u_plus", "u_minus", "d_value", "d_plus", "d_minus", "s_value", "s_plus", "s_minus", "gluon", "charge_squared_weighted_c_even_quark_sum"],
                "x_grid": gpd_x.tolist(),
                "native_model_truth": truth["gpds"].tolist(),
                "extracted_lower90": gpd_lower.tolist(),
                "extracted_median": gpd_median.tolist(),
                "extracted_upper90": gpd_upper.tolist(),
                "pointwise_90_truth_coverage": gpd_coverage,
            },
            "exact_reevaluation": {
                "requested_count": exact_count,
                "valid_count": len(valid),
                "invalid_count": len(exact.invalid_records),
                "invalid_records": list(exact.invalid_records),
                "surrogate_used": False,
            },
            "native_truth_provenance": {
                key: value for key, value in truth.items()
                if key not in {"predictions", "cffs", "gpds"}
            },
            "posterior_parameter_truth": None,
            "parameter_recovery_claimed": False,
            "real_data_used": False,
        }
        model_output = workspace / "holdout" / model
        with (model_output / "posterior_samples.npy").open("wb") as stream:
            np.save(stream, posterior, allow_pickle=False)
        write_json(model_output / "metrics.json", record)
        record["plots"] = _write_native_holdout_plots(
            output=model_output / "plots", config=holdout_config, model=model, record=record
        )
        write_json(model_output / "metrics.json", record)
        records[model] = record

    after = _holdout_manifest(workspace)
    immutable = before == after
    if not immutable:
        raise RuntimeError("holdout mutated protected training or closure artifacts")
    minimum_observable_coverage = min(
        item["observable_comparison"]["pointwise_90_truth_coverage"]
        for item in records.values()
    )
    robustness_enabled = bool(
        minimum_observable_coverage
        >= config["frozen_validation_gates"]["minimum_posterior_predictive_point_90_coverage"]
        and all(item["exact_reevaluation"]["invalid_count"] == 0 for item in records.values())
    )
    execution_passed = bool(
        immutable
        and all(
            item["exact_reevaluation"]["invalid_count"] == 0
            for item in records.values()
        )
    )
    kinematic_provenance = {
        "protocol": blind["protocol"],
        "manifest": str(blind_path),
        "manifest_content_sha256": expected_hash,
        "catalog_sha256": blind["catalog_sha256"],
        "database_revision": blind["database_revision"],
    }
    summary = {
        "schema_version": 2,
        "profile": profile,
        # `passed` is retained as an execution-integrity alias for simple
        # clients. Predictive precision has its own non-conflated gate below.
        "passed": execution_passed,
        "execution_passed": execution_passed,
        "predictive_precision_passed": robustness_enabled,
        "status": (
            "pass"
            if robustness_enabled
            else "complete_with_scientific_mismatch"
        ),
        "method_improvement_required": not robustness_enabled,
        "models": records,
        "dataset_roles": {
            "dd_learning_distribution": "project_DD_pseudodata_only",
            "dd_grouped_outer_test_fraction": float(
                config["profiles"][profile]["held_out_fraction"]
            ),
            "dd_outer_test_preserved": True,
            "external_native_validation_models": sorted(records),
            "external_native_validation_post_training_only": True,
            "external_native_validation_replaces_dd_outer_test": False,
        },
        "external_validation_kinematics": {
            "uses_same_kinematic_points_as_dd_pseudodata": False,
            "fresh_kinematics_output_blind": True,
            "kinematics_sha256": hashlib.sha256(
                canonical(holdout_config["kinematics"]).encode("utf-8")
            ).hexdigest(),
            "point_count": len(holdout_config["kinematics"]),
            "real_catalog_like": True,
            "source": kinematic_provenance,
            "measurement_values_used": False,
            "data_evolved": False,
            "named_model_evaluated_at_each_source_Q2": True,
        },
        "leakage_audit": {
            "protected_manifest_before": before,
            "protected_manifest_after": after,
            "protected_artifacts_unchanged": immutable,
            "named_models_absent_from_pre_holdout_json": True,
            "training_use": False,
            "validation_use": False,
            "external_native_validation_use": True,
            "replaces_dd_outer_test": False,
            "optuna_use": False,
            "ensemble_selection_use": False,
        },
        "minimum_observable_pointwise_90_truth_coverage": minimum_observable_coverage,
        "robustness_claim_enabled": robustness_enabled,
        "threshold_source": "frozen minimum posterior-predictive coverage gate",
        "real_data_used": False,
        "real_data_fit_enabled": False,
    }
    write_json(workspace / "holdout" / "summary.json", summary)
    return summary


def compare_real_observables(
    *,
    bridge: Path,
    configuration_path: Path,
    workspace: Path,
    profile: str,
    database_root: Path,
    show_progress: bool | None = None,
    sample_count: int | None = None,
) -> dict[str, Any]:
    """Compare quarantined real ALU values with frozen-NPE predictions.

    The NPE supplies posterior parameter samples conditioned only on the
    synthetic dataset.  Observable predictions are then computed by exact
    PARTONS reevaluation; the neural network is not an observable surrogate.
    Real values never enter training, a likelihood, hyperparameter tuning, or
    a posterior update in this diagnostic.
    """

    from extract_dvcs_cff.data.gpddatabase import (
        load_real_comparison_observations,
    )
    from extract_dvcs_cff.progress import progress_iter

    config = load_configuration(configuration_path)
    posterior_path = workspace / "comparison" / "neural_posterior_samples.npy"
    metrics_path = workspace / "comparison" / "comparison_metrics.json"
    posterior = np.load(posterior_path.resolve(strict=True), allow_pickle=False)
    comparison_metrics = json.loads(metrics_path.resolve(strict=True).read_text(
        encoding="utf-8"
    ))
    if (
        posterior.ndim != 2
        or posterior.shape[1] != _POSTERIOR_PARAMETER_COUNT
    ):
        raise RuntimeError(
            "frozen NPE posterior has incompatible milestone dimension"
        )
    if not np.all(np.isfinite(posterior)):
        raise RuntimeError("frozen NPE posterior contains non-finite values")
    if comparison_metrics.get("real_data_used") is not False:
        raise RuntimeError("posterior comparison lacks synthetic-only proof")
    synthetic_gate_passed = comparison_metrics.get("passed") is True

    source = load_real_comparison_observations(database_root)
    observations = source["observations"]
    default_exact_count = min(
        int(config["profiles"][profile]["exact_reevaluation_sample_count"]),
        len(posterior),
    )
    exact_count = default_exact_count if sample_count is None else sample_count
    if isinstance(exact_count, bool) or not isinstance(exact_count, int):
        raise ValueError("real-comparison sample count must be an integer")
    if not 1 <= exact_count <= len(posterior):
        raise ValueError(
            "real-comparison sample count must be between 1 and the frozen "
            "posterior sample count"
        )
    indices = np.linspace(0, len(posterior) - 1, exact_count, dtype=np.int64)
    parameters = posterior[indices, :_PHYSICS_PARAMETER_COUNT]
    evaluation_config = {
        **config,
        "kinematics": [
            {
                "x_b": item["x_b"],
                "t_GeV2": item["t_GeV2"],
                "Q2_GeV2": item["evaluation_Q2_GeV2"],
                "beam_energy_GeV": item["beam_energy_GeV"],
                "phi_rad": item["phi_rad"],
            }
            for item in observations
        ],
    }
    physics_path = _physics_path(configuration_path, config)
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    client = NativeCache(bridge, workspace)

    # Separate subprocesses are required because local PARTONS is not proven
    # thread-safe.  Each subprocess owns one independent PARTONS session.
    chunk_size = 8
    chunks = [
        (chunk_index, start, parameters[start : start + chunk_size])
        for chunk_index, start in enumerate(
            range(0, len(parameters), chunk_size), start=1
        )
    ]
    worker_resolution = resolve_native_workers(
        config["runtime"]["native_workers"], task_count=len(chunks)
    )

    def evaluate_chunk(
        specification: tuple[int, int, np.ndarray],
    ) -> tuple[int, int, PartitionedNativeResult]:
        chunk_index, start, values = specification
        result = _partitioned_native_evaluation(
            client=client,
            parameters=values,
            config=evaluation_config,
            physics=physics,
            operation="batch_evaluate_post_training_comparison_dvcs",
        )
        return chunk_index, start, result

    completed: dict[int, tuple[int, int, PartitionedNativeResult]] = {}
    if worker_resolution.resolved_workers == 1:
        for specification in progress_iter(
            chunks,
            total=len(chunks),
            description="Exact real-data diagnostic reevaluations",
            enabled=show_progress,
            unit="batch",
        ):
            result = evaluate_chunk(specification)
            completed[result[0]] = result
    else:
        # Submit in deterministic chunks; only collection order varies.
        with ThreadPoolExecutor(
            max_workers=worker_resolution.resolved_workers,
            thread_name_prefix="partons-real-comparison",
        ) as executor:
            futures = {
                executor.submit(evaluate_chunk, specification): specification[0]
                for specification in chunks
            }
            for future in progress_iter(
                as_completed(futures),
                total=len(futures),
                description="Exact real-data diagnostic reevaluations",
                enabled=show_progress,
                unit="batch",
            ):
                result = future.result()
                completed[result[0]] = result

    prediction_blocks = []
    invalid_records: list[dict[str, Any]] = []
    valid_parameter_indices: list[int] = []
    for chunk_index in sorted(completed):
        _, start, result = completed[chunk_index]
        local_valid = np.flatnonzero(result.valid_mask)
        if len(local_valid):
            prediction_blocks.append(result.predictions[local_valid])
            valid_parameter_indices.extend((start + local_valid).tolist())
        invalid_records.extend(
            {**item, "posterior_sample_index": start + item["candidate_index"]}
            for item in result.invalid_records
        )
    if not prediction_blocks:
        raise RuntimeError("all exact real-comparison reevaluations failed")
    predictions = np.vstack(prediction_blocks)
    observable_ids = [item["id"] for item in config["observables"]]
    alu_index = observable_ids.index("DVCSAluMinus")
    observable_count = len(observable_ids)
    alu_columns = np.arange(len(observations)) * observable_count + alu_index
    alu_predictions = predictions[:, alu_columns]
    if not np.all(np.isfinite(alu_predictions)):
        raise RuntimeError("real-comparison ALU prediction is non-finite")

    rows = []
    for point_index, item in enumerate(observations):
        samples = alu_predictions[:, point_index]
        rows.append(
            {
                **item,
                "prediction_q05": float(np.quantile(samples, 0.05)),
                "prediction_q50": float(np.quantile(samples, 0.50)),
                "prediction_q95": float(np.quantile(samples, 0.95)),
                "observed_minus_prediction_median_over_statistical_sigma": (
                    float(
                        (item["observed_value"] - np.median(samples))
                        / item["statistical_uncertainty"]
                    )
                ),
            }
        )

    output = workspace / "real_data_comparison"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "exact_native_alu_predictions.npy").open("wb") as stream:
        np.save(stream, alu_predictions, allow_pickle=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort([item["phi_rad"] for item in rows])
    x = np.arange(len(rows))
    observed = np.asarray([rows[index]["observed_value"] for index in order])
    uncertainty = np.asarray(
        [rows[index]["statistical_uncertainty"] for index in order]
    )
    lower = np.asarray([rows[index]["prediction_q05"] for index in order])
    median = np.asarray([rows[index]["prediction_q50"] for index in order])
    upper = np.asarray([rows[index]["prediction_q95"] for index in order])
    figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    axis.errorbar(
        x - 0.12,
        median,
        yerr=np.vstack((median - lower, upper - median)),
        fmt="s",
        capsize=2,
        label="frozen NPE posterior + exact PARTONS (median and 90%)",
    )
    axis.errorbar(x + 0.12, observed, yerr=uncertainty, fmt="o", capsize=2,
                  label="CLAS ALU ± database stat_unc field")
    axis.set(
        xlabel="accepted database point (ordered by phi; xB and t vary)",
        ylabel=r"$A_{LU}$",
        title=(
            r"CLAS $A_{LU}$ vs frozen synthetic-NPE prediction "
            "(candidate mapping; no real-data fit)\n"
            r"Source $Q^2$ retained; input GPD evolved from "
            r"$Q_0^2=1$ GeV$^2$; data not evolved; synthetic posterior gate: "
            f"{'PASS' if synthetic_gate_passed else 'FAIL'}"
        ),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    overlay_path = output / "real_ALU_vs_neural_posterior_prediction.png"
    figure.savefig(overlay_path, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 7), constrained_layout=True)
    axis.errorbar(
        median,
        observed,
        xerr=np.vstack((median - lower, upper - median)),
        yerr=uncertainty,
        fmt="o",
        capsize=2,
        label="x: predictive 90%; y: database stat_unc field",
    )
    limits = [
        float(min(np.min(lower), np.min(observed - uncertainty))),
        float(max(np.max(upper), np.max(observed + uncertainty))),
    ]
    axis.plot(limits, limits, "--", color="black", label="observed = predicted")
    axis.set(
        xlim=limits,
        ylim=limits,
        xlabel="exact-PARTONS median from frozen NPE posterior",
        ylabel=r"observed CLAS $A_{LU}$",
        title="Observed versus predicted (diagnostic only)",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    scatter_path = output / "real_ALU_observed_vs_predicted.png"
    figure.savefig(scatter_path, dpi=180)
    plt.close(figure)

    native_prediction_set_complete = len(invalid_records) == 0
    if not native_prediction_set_complete:
        diagnostic_status = "diagnostic_incomplete_invalid_native_samples"
    elif not synthetic_gate_passed:
        diagnostic_status = "diagnostic_complete_unvalidated_posterior"
    else:
        diagnostic_status = "diagnostic_complete_with_mapping_limitations"
    record = {
        "schema_version": 1,
        "status": diagnostic_status,
        "interpretation": (
            "comparison of real observations with exact PARTONS predictions "
            "from a posterior learned exclusively from pseudodata; not a fit"
        ),
        "real_data_used_for_training": False,
        "real_data_used_for_hyperparameter_optimization": False,
        "real_data_used_in_likelihood": False,
        "posterior_updated": False,
        "synthetic_posterior_gate_passed": synthetic_gate_passed,
        "synthetic_posterior_gate_checks": comparison_metrics.get(
            "gate_checks"
        ),
        "neural_observable_surrogate_used": False,
        "prediction_pipeline": "frozen_NPE_samples_then_exact_PARTONS",
        "observable_mapping": {
            "database_observable": "ALU",
            "native_observable": "DVCSAluMinus",
            "native_definition": (
                "(sigma_beam_helicity_plus_minus_sigma_beam_helicity_minus)"
                "/(sigma_beam_helicity_plus_plus_sigma_beam_helicity_minus),"
                " electron_beam_charge_minus_one"
            ),
            "status": (
                "candidate_same_named_direct_observable; experimental sign "
                "and phi conventions not benchmarked pointwise"
            ),
        },
        "source": {key: value for key, value in source.items() if key != "observations"},
        "observable_coverage": {
            "compared": ["ALU"],
            "skipped": {
                "CrossSectionUU_and_CrossSectionDifferenceLU": (
                    "dataset-specific differential unit convention unresolved"
                ),
                "AUL_and_ALL": "target-polarization sign convention not yet source-audited",
                "Ac": "W-based kinematics and beam-charge convention not supported by this slice",
                "Fourier_moments": "no audited direct native moment mapping",
            },
        },
        "posterior_sample_file": str(posterior_path.resolve(strict=True)),
        "posterior_sample_sha256": sha256(posterior_path.resolve(strict=True)),
        "posterior_samples_requested": exact_count,
        "profile_default_posterior_samples": default_exact_count,
        "posterior_samples_valid": len(valid_parameter_indices),
        "native_prediction_set_complete": native_prediction_set_complete,
        "invalid_native_evaluations": invalid_records,
        "native_worker_resolution": worker_resolution.as_dict(),
        "bridge": str(bridge.resolve(strict=True)),
        "bridge_sha256": sha256(bridge.resolve(strict=True)),
        "physics_configuration": str(physics_path),
        "physics_configuration_sha256": sha256(physics_path),
        "rows": rows,
        "plots": {
            "ordered_overlay": str(overlay_path),
            "observed_vs_predicted": str(scatter_path),
        },
        "gate_decision": None,
        "agreement_claim_enabled": False,
    }
    write_json(output / "real_observable_comparison.json", record)
    return record


def doctor(
    *,
    bridge: Path,
    configuration_path: Path,
) -> dict[str, Any]:
    """Verify the exact local backend and neural runtime before a user run."""

    config = load_configuration(configuration_path)
    physics_path = _physics_path(configuration_path, config)
    bridge_path = bridge.resolve(strict=True)
    process = subprocess.run(
        [str(bridge_path), "--capabilities"],
        text=True,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"bridge capabilities failed: {process.stderr.strip()}"
        )
    capabilities = json.loads(process.stdout)
    pseudodata = capabilities["capabilities"].get("pseudodata_dvcs", {})
    if (
        pseudodata.get("available") is not True
        or pseudodata.get("representation") != config["representation"]
        or pseudodata.get("analysis_order_label") is not None
    ):
        raise RuntimeError("bridge lacks the frozen milestone capability")
    import scipy
    import sbi
    import torch
    from extract_dvcs_cff.inference.device import resolve_torch_device

    device = resolve_torch_device(config["runtime"]["accelerator"])
    native_workers = resolve_native_workers(
        config["runtime"]["native_workers"]
    )

    record = {
        "schema_version": 1,
        "status": "ok",
        "configuration": str(configuration_path.resolve(strict=True)),
        "configuration_sha256": sha256(configuration_path.resolve(strict=True)),
        "physics_configuration": str(physics_path),
        "physics_configuration_sha256": sha256(physics_path),
        "bridge": str(bridge_path),
        "bridge_sha256": sha256(bridge_path),
        "bridge_version": capabilities["backend"]["bridge_version"],
        "partons_version": capabilities["backend"]["partons"]["version"],
        "partons_git_revision": capabilities["backend"]["partons"][
            "git_revision"
        ],
        "apfelxx_version": capabilities["backend"]["apfelxx"]["version"],
        "python_dependencies": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "sbi": sbi.__version__,
        },
        "neural_device": device.as_dict(),
        "native_accelerator": "cpu",
        "native_worker_resolution": native_workers.as_dict(),
        "native_generation_parameters_per_batch": (
            _NATIVE_GENERATION_CHUNK_SIZE
        ),
        "native_generation_parameters_per_wave": (
            max(
                _NATIVE_GENERATION_MIN_WAVE_SIZE,
                native_workers.resolved_workers
                * _NATIVE_GENERATION_CHUNK_SIZE,
            )
        ),
        "native_progress_update_unit": (
            "one_completed_atomic_partons_batch_with_attempted_accepted_rejected"
        ),
        "native_evaluation_parameters_per_batch": (
            _NATIVE_GENERATION_CHUNK_SIZE
        ),
        "native_evaluation_parallel_phases": [
            "exact_posterior_reevaluation",
            "exact_gpd_diagnostics",
        ],
        "native_evaluation_progress_update_unit": (
            "one_completed_atomic_partons_batch_with_completed_valid_invalid"
        ),
        "native_libraries": capabilities["backend"]["native_libraries"],
        "real_data": False,
        "analysis_order_label": None,
    }
    return record


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    comparison = summary["comparison"]
    evaluation = summary["evaluation"]
    training = summary["training"]
    marginal_distances = comparison[
        "marginal_wasserstein_over_conventional_sd"
    ]
    largest_marginal = max(marginal_distances, key=marginal_distances.get)
    lines = [
        "# Pseudodata workflow summary",
        "",
        f"- Overall status: `{'pass' if summary['passed'] else 'fail'}`",
        f"- Profile: `{summary['profile']}`",
        "- Data: synthetic only; no real data were read",
        (
            "- Neural device: "
            f"`{training['neural_device']['resolved']}` / "
            f"`{training['neural_device']['cuda_device_name'] or 'host CPU'}`; "
            "peak Torch allocation "
            f"`{training['maximum_cuda_peak_allocated_bytes']}` bytes"
        ),
        (
            "- Mean train/test negative log density: "
            f"`{training['mean_train_negative_log_density']:.6g}` / "
            f"`{training['mean_test_negative_log_density']:.6g}`"
        ),
        (
            "- Neural/conventional sliced Wasserstein: "
            f"`{comparison['ensemble_sliced_wasserstein']:.6g}`"
        ),
        (
            "- Largest marginal Wasserstein/conventional SD: "
            f"`{largest_marginal}` = "
            f"`{marginal_distances[largest_marginal]:.6g}`"
        ),
        (
            "- Conventional effective sample size: "
            f"`{comparison['effective_sample_size']:.1f}`"
        ),
        (
            "- Posterior-predictive pointwise 90% coverage: "
            f"`{evaluation['posterior_predictive']['pointwise_90_coverage']:.3f}`"
        ),
        (
            "- Minimum neural/conventional stress-direction 90% width ratio: "
            f"`{comparison['stress_direction_width_reference']['neural_conventional_width90_ratio']:.3f}`"
        ),
        "",
        "Machine-readable details are in `summary.json`, "
        "`training/training_summary.json`, "
        "`evaluation/evaluation_metrics.json`, and "
        "`comparison/comparison_metrics.json`.",
        "",
        "A pass means that predeclared tolerances were met; it does not mean "
        "that the two posterior densities are identical. Inspect per-seed "
        "scores, every marginal, and joint sample correlations.",
        "",
        "The stress-width comparison is a neural false-certainty diagnostic "
        "against the exact conventional posterior, not evidence that an "
        "injected stress coefficient was recovered or that it is CFF-null.",
        "",
    ]
    return "\n".join(lines)


def _write_diagnostic_plots(
    *,
    workspace: Path,
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> dict[str, str]:
    """Write stable, user-readable diagnostics from recorded metrics.

    Plotting consumes saved arrays and JSON only; it never evaluates physics,
    changes a posterior, or decides a stage gate.  The JSON records remain the
    authoritative machine-readable results.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from extract_dvcs_cff.inference.stage10 import STAGE10_PARAMETER_NAMES

    output = workspace / "plots"
    output.mkdir(parents=True, exist_ok=True)

    # Each sbi trace is a positive negative-log-density objective.  Showing
    # train and validation on one axis makes overfitting visible to a user.
    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for seed, member in training["members"].items():
        train = np.asarray(member["training_loss_by_epoch"])
        validation = np.asarray(member["validation_loss_by_epoch"])
        axis.plot(np.arange(1, len(train) + 1), train, label=f"train {seed}")
        axis.plot(
            np.arange(1, len(validation) + 1),
            validation,
            linestyle="--",
            label=f"validation {seed}",
        )
    axis.set(
        xlabel="epoch",
        ylabel="negative log density",
        title="NPE training and validation scores",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    training_path = output / "training_validation_nll.png"
    figure.savefig(training_path, dpi=160)
    plt.close(figure)

    conventional = np.load(
        workspace / "comparison" / "conventional_posterior_samples.npy",
        allow_pickle=False,
    )
    neural = np.load(
        workspace / "comparison" / "neural_posterior_samples.npy",
        allow_pickle=False,
    )
    intervals = comparison["intervals"]
    # Keep the overview compact: the two nuisance coordinates are plotted
    # here, while all 80 physics marginals are split into four readable GPD
    # sheets below. A single 82-panel page is technically complete but not a
    # usable scientific diagnostic.
    nuisance_names = STAGE10_PARAMETER_NAMES[-2:]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for offset, (name, axis) in enumerate(zip(nuisance_names, axes)):
        index = len(STAGE10_PARAMETER_NAMES) - 2 + offset
        axis.hist(conventional[:, index], bins=32, density=True, alpha=0.5,
                  label="conventional")
        axis.hist(neural[:, index], bins=32, density=True, alpha=0.5,
                  label="NPE")
        axis.axvline(intervals[name]["truth"], color="black", linewidth=1.5,
                     label="injected truth")
        axis.set_title(name)
        axis.grid(alpha=0.2)
        axis.legend(fontsize="small")
    figure.suptitle("Normalization-nuisance posterior comparison")
    comparison_path = output / "posterior_comparison.png"
    figure.savefig(comparison_path, dpi=160)
    plt.close(figure)

    physics_marginal_paths = []
    for gpd_index, gpd_name in enumerate(_GPD_TYPES):
        figure, axes = plt.subplots(
            len(_DD_SHAPE_FIELDS), len(_PARTON_CHANNELS),
            figsize=(18, 15), constrained_layout=True,
        )
        for channel_index, channel in enumerate(_PARTON_CHANNELS):
            for field_index, field in enumerate(_DD_SHAPE_FIELDS):
                index = (
                    gpd_index * len(_PARTON_CHANNELS) * len(_DD_SHAPE_FIELDS)
                    + channel_index * len(_DD_SHAPE_FIELDS) + field_index
                )
                name = STAGE10_PARAMETER_NAMES[index]
                axis = axes[field_index, channel_index]
                axis.hist(conventional[:, index], bins=28, density=True,
                          alpha=0.5, label="conventional")
                axis.hist(neural[:, index], bins=28, density=True,
                          alpha=0.5, label="NPE")
                axis.axvline(intervals[name]["truth"], color="black",
                             linewidth=1.2, label="truth")
                axis.set_title(f"{channel}: {field}")
                axis.grid(alpha=0.2)
        axes[0, 0].legend(fontsize="x-small")
        figure.suptitle(
            f"{gpd_name}: all independent DD-shape posterior marginals"
        )
        path = output / f"posterior_marginals_{gpd_name}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        physics_marginal_paths.append(path)

    predictive = evaluation["posterior_predictive"]
    observed = np.asarray(predictive["observed"])
    lower = np.asarray(predictive["lower90"])
    upper = np.asarray(predictive["upper90"])
    median = np.asarray(predictive["median"])
    truth_prediction = np.asarray(predictive["injected_truth_native"])
    dataset = json.loads(
        (workspace / "generated" / "pseudodata.json").read_text(
            encoding="utf-8"
        )
    )
    point_order = dataset["point_order"]

    # The split is by independent native parameter draw, not by noisy replica
    # and not by kinematic point. PCA is display-only and makes leakage or a
    # grossly unbalanced split visible without pretending two components
    # summarize the 80-dimensional physics space.
    native_parameters = np.load(
        workspace / "generated/native_parameters.npy", allow_pickle=False
    )
    parameter_indices = np.load(
        workspace / "generated/parameter_indices.npy", allow_pickle=False
    )
    centered = native_parameters - np.mean(native_parameters, axis=0)
    scale = np.std(centered, axis=0)
    standardized = centered / np.where(scale > 0, scale, 1.0)
    _, _, rotation = np.linalg.svd(standardized, full_matrices=False)
    projected = standardized @ rotation[:2].T
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for role, color in (("train", "C0"), ("validation", "C1"), ("test", "C2")):
        replica_indices = np.load(
            workspace / f"generated/{role}_indices.npy", allow_pickle=False
        )
        groups = np.unique(parameter_indices[replica_indices])
        axis.scatter(projected[groups, 0], projected[groups, 1], s=18,
                     alpha=0.7, label=f"{role}: {len(groups)} native draws",
                     color=color)
    axis.set(
        xlabel="display PCA component 1",
        ylabel="display PCA component 2",
        title="Grouped DD train / internal-validation / outer-test split",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small")
    split_path = output / "dd_train_validation_test_split.png"
    figure.savefig(split_path, dpi=160)
    plt.close(figure)

    real_coverage_path = None
    mapping_path = workspace.parents[1] / "real_data_mapping_readiness.json"
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        coverage = mapping.get("measurement_free_kinematic_coverage", {})
        histograms = coverage.get("histograms", {})
        if set(histograms) == {"x_b", "minus_t_GeV2", "Q2_GeV2", "phi_rad"}:
            pseudo_sites = dataset.get("kinematics")
            if pseudo_sites is None:
                observable_count_for_sites = len({
                    point["observable_id"] for point in dataset["point_order"]
                })
                pseudo_sites = [
                    {
                        key: point[key]
                        for key in (
                            "x_b", "t_GeV2", "Q2_GeV2",
                            "beam_energy_GeV", "phi_rad",
                        )
                    }
                    for point in dataset["point_order"][
                        ::observable_count_for_sites
                    ]
                ]
            pseudo_values = {
                "x_b": [item["x_b"] for item in pseudo_sites],
                "minus_t_GeV2": [-item["t_GeV2"] for item in pseudo_sites],
                "Q2_GeV2": [item["Q2_GeV2"] for item in pseudo_sites],
                "phi_rad": [item["phi_rad"] for item in pseudo_sites],
            }
            labels_kin = {
                "x_b": r"$x_B$",
                "minus_t_GeV2": r"$-t$ [GeV$^2$]",
                "Q2_GeV2": r"$Q^2$ [GeV$^2$]",
                "phi_rad": r"$\phi$ [rad]",
            }
            figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
            for axis, key in zip(np.asarray(axes).reshape(-1), labels_kin):
                edges = np.asarray(histograms[key]["bin_edges"])
                counts = np.asarray(histograms[key]["counts"])
                axis.stairs(counts / max(1, counts.sum()), edges, fill=True,
                            alpha=0.35, label="real catalog kinematics (normalized)")
                for value in pseudo_values[key]:
                    axis.axvline(value, color="C1", alpha=0.65, linewidth=1)
                axis.set(xlabel=labels_kin[key], ylabel="catalog fraction")
                axis.grid(alpha=0.25)
            axes[0, 0].legend(fontsize="small")
            figure.suptitle(
                "Measurement-free real-catalog coverage and pseudodata sites\n"
                "orange lines are the common kinematic sites used by every DD split"
            )
            real_coverage_path = output / "real_catalog_vs_pseudodata_kinematics.png"
            figure.savefig(real_coverage_path, dpi=160)
            plt.close(figure)
    observable_ids = list(dict.fromkeys(
        point["observable_id"] for point in point_order
    ))
    labels = {
        "DVCSCrossSectionUUMinus": (
            "Beam-spin half-sum / unpolarized cross section"
        ),
        "DVCSCrossSectionDifferenceLUMinus": "Beam-spin difference",
        "DVCSAc": "Beam-charge asymmetry",
        "DVCSAluMinus": "Beam-spin asymmetry",
        "DVCSAulMinus": "Longitudinal target-spin asymmetry",
        "DVCSAllMinus": "Longitudinal double-spin asymmetry",
    }
    covariance = np.load(
        workspace / "generated" / "covariance_normalized.npy",
        allow_pickle=False,
    )
    figure, axes = plt.subplots(
        3, 2, figsize=(14, 12), constrained_layout=True
    )
    for axis, observable_id in zip(np.asarray(axes).reshape(-1), observable_ids):
        indices = np.asarray(
            [
                index
                for index, point_record in enumerate(point_order)
                if point_record["observable_id"] == observable_id
            ],
            dtype=np.int64,
        )
        phi = np.asarray(
            [float(point_order[index]["phi_rad"]) for index in indices]
        )
        order = np.argsort(phi)
        indices = indices[order]
        phi = phi[order]
        axis.fill_between(
            phi,
            lower[indices],
            upper[indices],
            alpha=0.28,
            label="posterior predictive 90%",
        )
        axis.plot(phi, median[indices], "-", label="posterior median")
        axis.plot(
            phi,
            truth_prediction[indices],
            "--",
            label="injected native truth",
        )
        axis.errorbar(
            phi,
            observed[indices],
            yerr=np.sqrt(np.diag(covariance))[indices],
            fmt="o",
            capsize=2,
            label="pseudodata ± marginal σ",
        )
        axis.set(
            xlabel=r"$\phi$ [rad]",
            ylabel="normalized value",
            title=labels[observable_id],
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize="x-small")
    figure.suptitle("Exact-native observable posterior predictions")
    predictive_path = output / "posterior_predictive.png"
    figure.savefig(predictive_path, dpi=160)
    plt.close(figure)

    cff = evaluation["cff_posterior"]
    cff_truth = np.asarray(cff["truth"])
    cff_lower = np.asarray(cff["lower90"])
    cff_median = np.asarray(cff["median"])
    cff_upper = np.asarray(cff["upper90"])
    kinematic_index = np.arange(cff_truth.shape[0])
    figure, axes = plt.subplots(
        4, 2, figsize=(14, 14), constrained_layout=True
    )
    for gpd_index, cff_name in enumerate(cff["types"]):
        for component_index, component in enumerate(cff["components"]):
            axis = axes[gpd_index, component_index]
            axis.fill_between(
                kinematic_index,
                cff_lower[:, gpd_index, component_index],
                cff_upper[:, gpd_index, component_index],
                alpha=0.28,
                label="posterior 90%",
            )
            axis.plot(
                kinematic_index,
                cff_median[:, gpd_index, component_index],
                label="posterior median",
            )
            axis.plot(
                kinematic_index,
                cff_truth[:, gpd_index, component_index],
                "o--",
                label="injected truth",
            )
            axis.set(
                xlabel="declared kinematic index",
                ylabel=f"{component} part",
                title=f"{component.capitalize()} part of CFF {cff_name}",
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize="x-small")
    figure.suptitle("Exact-native CFF posterior diagnostics")
    cff_path = output / "cff_real_imaginary.png"
    figure.savefig(cff_path, dpi=160)
    plt.close(figure)

    gpd = evaluation["gpd_posterior"]
    x_grid = np.asarray(gpd["x_grid"])
    gpd_truth = np.asarray(gpd["truth"])
    gpd_lower = np.asarray(gpd["lower90"])
    gpd_median = np.asarray(gpd["median"])
    gpd_upper = np.asarray(gpd["upper90"])
    figure, axes = plt.subplots(
        4, 4, figsize=(22, 14), constrained_layout=True
    )
    components = (
        (1, r"$F^{u,+}$"),
        (4, r"$F^{d,+}$"),
        (7, r"$F^{s,+}$"),
        (9, r"$F^g$ (PARTONS convention)"),
    )
    for gpd_index, gpd_name in enumerate(gpd["types"]):
        for column, (component_index, component_label) in enumerate(components):
            axis = axes[gpd_index, column]
            axis.fill_between(
                x_grid,
                gpd_lower[:, gpd_index, component_index],
                gpd_upper[:, gpd_index, component_index],
                alpha=0.28,
                label="posterior 90%",
            )
            axis.plot(
                x_grid,
                gpd_median[:, gpd_index, component_index],
                label="posterior median",
            )
            axis.plot(
                x_grid,
                gpd_truth[:, gpd_index, component_index],
                "--",
                label="injected truth",
            )
            axis.axvline(0.0, color="black", linewidth=0.6, alpha=0.5)
            axis.set(
                xlabel=r"$x$",
                ylabel=component_label,
                title=f"{gpd_name}: {component_label}",
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize="x-small")
    reference = gpd["reference_kinematics"]
    figure.suptitle(
        "Exact-native GPD posterior diagnostics at "
        rf"$x_B={reference['x_b']}$, "
        rf"$t={reference['t_GeV2']}\,\mathrm{{GeV}}^2$, "
        rf"$Q^2={reference['Q2_GeV2']}\,\mathrm{{GeV}}^2$"
    )
    gpd_path = output / "gpd_posterior_predictions.png"
    figure.savefig(gpd_path, dpi=160)
    plt.close(figure)

    # The combined figure above is the compact overview.  Separate flavor
    # files expose every native quark component (value, C-even plus, and C-odd
    # minus) so a user never has to infer d/s behavior from an aggregate.
    flavor_components = {
        "u": (
            (0, r"$F^u(x,\xi,t)$"),
            (1, r"$F^{u,+}$"),
            (2, r"$F^{u,-}$"),
        ),
        "d": (
            (3, r"$F^d(x,\xi,t)$"),
            (4, r"$F^{d,+}$"),
            (5, r"$F^{d,-}$"),
        ),
        "s": (
            (6, r"$F^s(x,\xi,t)$"),
            (7, r"$F^{s,+}$"),
            (8, r"$F^{s,-}$"),
        ),
        "gluon": ((9, r"$F^g$ (PARTONS convention)"),),
    }
    flavor_paths = []
    for flavor, flavor_columns in flavor_components.items():
        figure, axes = plt.subplots(
            4,
            len(flavor_columns),
            figsize=(6 * len(flavor_columns), 13),
            squeeze=False,
            constrained_layout=True,
        )
        for gpd_index, gpd_name in enumerate(gpd["types"]):
            for column, (component_index, component_label) in enumerate(
                flavor_columns
            ):
                axis = axes[gpd_index, column]
                axis.fill_between(
                    x_grid,
                    gpd_lower[:, gpd_index, component_index],
                    gpd_upper[:, gpd_index, component_index],
                    alpha=0.28,
                    label="posterior 90%",
                )
                axis.plot(
                    x_grid,
                    gpd_median[:, gpd_index, component_index],
                    label="posterior median",
                )
                axis.plot(
                    x_grid,
                    gpd_truth[:, gpd_index, component_index],
                    "--",
                    label="injected truth",
                )
                axis.axvline(0.0, color="black", linewidth=0.6, alpha=0.5)
                axis.set(
                    xlabel=r"$x$",
                    ylabel=component_label,
                    title=f"{gpd_name}: {component_label}",
                )
                axis.grid(alpha=0.25)
                axis.legend(fontsize="x-small")
        figure.suptitle(
            f"Exact-native {flavor} GPD posterior diagnostics"
        )
        flavor_path = output / f"gpd_{flavor}_components.png"
        figure.savefig(flavor_path, dpi=160)
        plt.close(figure)
        flavor_paths.append(flavor_path)

    # PARTONS also returns the charge-squared-weighted C-even light-quark sum
    # that feeds the verified LO CFF path.  Plot it separately from individual
    # flavors because it is a derived combination, not another parton flavor.
    figure, axes = plt.subplots(
        4, 1, figsize=(9, 13), squeeze=False, constrained_layout=True
    )
    for gpd_index, gpd_name in enumerate(gpd["types"]):
        axis = axes[gpd_index, 0]
        axis.fill_between(
            x_grid,
            gpd_lower[:, gpd_index, 10],
            gpd_upper[:, gpd_index, 10],
            alpha=0.28,
            label="posterior 90%",
        )
        axis.plot(
            x_grid,
            gpd_median[:, gpd_index, 10],
            label="posterior median",
        )
        axis.plot(
            x_grid,
            gpd_truth[:, gpd_index, 10],
            "--",
            label="injected truth",
        )
        axis.axvline(0.0, color="black", linewidth=0.6, alpha=0.5)
        axis.set(
            xlabel=r"$x$",
            ylabel=r"$\sum_q e_q^2 F^{q,+}$",
            title=f"{gpd_name}: charge-squared-weighted C-even quark sum",
        )
        axis.grid(alpha=0.25)
        axis.legend(fontsize="x-small")
    figure.suptitle("Exact-native light-quark combination entering LO CFFs")
    weighted_sum_path = output / "gpd_charge_weighted_c_even_sum.png"
    figure.savefig(weighted_sum_path, dpi=160)
    plt.close(figure)

    # Correlations are essential for diagnosing apparently accurate marginal
    # peaks that actually sit on broad or degenerate joint directions.
    correlations = (
        ("Conventional", np.corrcoef(conventional, rowvar=False)),
        ("Neural", np.corrcoef(neural, rowvar=False)),
    )
    figure, axes = plt.subplots(
        1, 2, figsize=(19, 8), constrained_layout=True
    )
    correlation_image = None
    for axis, (method, correlation) in zip(axes, correlations):
        correlation_image = axis.imshow(
            np.nan_to_num(correlation),
            vmin=-1.0,
            vmax=1.0,
            cmap="coolwarm",
        )
        axis.set_xticks(
            np.arange(len(STAGE10_PARAMETER_NAMES)),
            STAGE10_PARAMETER_NAMES,
            rotation=90,
            fontsize=6,
        )
        axis.set_yticks(
            np.arange(len(STAGE10_PARAMETER_NAMES)),
            STAGE10_PARAMETER_NAMES,
            fontsize=6,
        )
        axis.set_title(f"{method} posterior correlation")
    figure.colorbar(
        correlation_image, ax=axes, shrink=0.8, label="Pearson correlation"
    )
    correlation_path = output / "posterior_correlations.png"
    figure.savefig(correlation_path, dpi=160)
    plt.close(figure)

    # Display empirical minus nominal coverage for every posterior coordinate.
    levels = sorted(float(value) for value in evaluation["coverage"])
    calibration = np.asarray(
        [
            [
                evaluation["coverage"][str(level)][name][
                    "empirical_coverage"
                ]
                - level
                for name in STAGE10_PARAMETER_NAMES
            ]
            for level in levels
        ]
    )
    figure, axis = plt.subplots(figsize=(17, 4), constrained_layout=True)
    calibration_image = axis.imshow(
        calibration,
        aspect="auto",
        vmin=-0.5,
        vmax=0.5,
        cmap="coolwarm",
    )
    axis.set_xticks(
        np.arange(len(STAGE10_PARAMETER_NAMES)),
        STAGE10_PARAMETER_NAMES,
        rotation=90,
        fontsize=7,
    )
    axis.set_yticks(np.arange(len(levels)), [f"{value:.0%}" for value in levels])
    axis.set(
        xlabel="posterior coordinate",
        ylabel="nominal interval",
        title="Empirical coverage minus nominal coverage",
    )
    figure.colorbar(calibration_image, ax=axis, label="coverage difference")
    calibration_path = output / "coverage_calibration.png"
    figure.savefig(calibration_path, dpi=160)
    plt.close(figure)

    # Pulls use the declared experimental marginal standard deviation.  They
    # are a residual diagnostic and intentionally do not replace the full-
    # covariance posterior-predictive gate stored in evaluation_metrics.json.
    marginal_sigma = np.sqrt(np.diag(covariance))
    pulls = (observed - median) / marginal_sigma
    figure, axes = plt.subplots(
        3, 2, figsize=(14, 12), constrained_layout=True
    )
    for axis, observable_id in zip(np.asarray(axes).reshape(-1), observable_ids):
        indices = np.asarray(
            [
                index
                for index, point_record in enumerate(point_order)
                if point_record["observable_id"] == observable_id
            ],
            dtype=np.int64,
        )
        phi = np.asarray(
            [float(point_order[index]["phi_rad"]) for index in indices]
        )
        order = np.argsort(phi)
        axis.plot(phi[order], pulls[indices][order], "o-")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.axhline(1.0, color="grey", linestyle="--", linewidth=0.7)
        axis.axhline(-1.0, color="grey", linestyle="--", linewidth=0.7)
        axis.axhline(2.0, color="grey", linestyle=":", linewidth=0.7)
        axis.axhline(-2.0, color="grey", linestyle=":", linewidth=0.7)
        axis.set(
            xlabel=r"$\phi$ [rad]",
            ylabel=r"$(y-\mathrm{median})/\sigma_{\mathrm{marginal}}$",
            title=labels[observable_id],
        )
        axis.grid(alpha=0.25)
    figure.suptitle("Pseudodata residual pulls by native observable")
    pull_path = output / "posterior_predictive_pulls.png"
    figure.savefig(pull_path, dpi=160)
    plt.close(figure)

    plot_paths = [
            training_path,
            split_path,
            comparison_path,
            *physics_marginal_paths,
            predictive_path,
            cff_path,
            gpd_path,
            *flavor_paths,
            weighted_sum_path,
            correlation_path,
            calibration_path,
            pull_path,
        ]
    if real_coverage_path is not None:
        plot_paths.append(real_coverage_path)
    return {
        path.name: sha256(path)
        for path in plot_paths
    }


def plot_saved_results(
    *,
    configuration_path: Path,
    workspace: Path,
    profile: str,
) -> dict[str, Any]:
    """Render diagnostics from completed, compatible saved results only.

    This public workflow deliberately has no bridge, device, or progress
    argument: it performs no PARTONS evaluation, neural training/sampling, or
    posterior calculation.  It reads the authoritative records produced by
    ``generate``, ``train``, ``evaluate``, and ``compare`` and writes only PNG
    views plus a content-hashed manifest.  Scientific gate failures are
    reported in the manifest but never suppress diagnostic plots.
    """

    load_configuration(configuration_path)
    required = (
        workspace / "generated" / "pseudodata.json",
        workspace / "generated" / "native_parameters.npy",
        workspace / "generated" / "parameter_indices.npy",
        workspace / "generated" / "train_indices.npy",
        workspace / "generated" / "validation_indices.npy",
        workspace / "generated" / "test_indices.npy",
        workspace / "generated" / "covariance_normalized.npy",
        workspace / "training" / "training_summary.json",
        workspace / "evaluation" / "evaluation_metrics.json",
        workspace / "comparison" / "comparison_metrics.json",
        workspace / "comparison" / "conventional_posterior_samples.npy",
        workspace / "comparison" / "neural_posterior_samples.npy",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "plot requires completed generate, train, evaluate, and compare "
            "results; missing: " + ", ".join(missing)
        )

    training_path = workspace / "training" / "training_summary.json"
    evaluation_path = workspace / "evaluation" / "evaluation_metrics.json"
    comparison_path = workspace / "comparison" / "comparison_metrics.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    for label, record in (
        ("training", training),
        ("evaluation", evaluation),
        ("comparison", comparison),
    ):
        recorded_profile = record.get("profile")
        if recorded_profile is not None and recorded_profile != profile:
            raise RuntimeError(
                f"{label} result profile mismatch: requested {profile!r}, "
                f"recorded {recorded_profile!r}"
            )

    plot_hashes = _write_diagnostic_plots(
        workspace=workspace,
        training=training,
        evaluation=evaluation,
        comparison=comparison,
    )
    inputs = {
        str(path.relative_to(workspace)): sha256(path)
        for path in required
    }
    optional_mapping = workspace.parents[1] / "real_data_mapping_readiness.json"
    if optional_mapping.is_file():
        inputs[str(optional_mapping.relative_to(workspace.parents[1]))] = (
            sha256(optional_mapping)
        )
    record = {
        "schema_version": 1,
        "status": "complete",
        "profile": profile,
        "presentation_only": True,
        "native_backend_called": False,
        "neural_inference_called": False,
        "configuration_sha256": sha256(configuration_path),
        "input_artifacts": inputs,
        "plots": plot_hashes,
        "plot_count": len(plot_hashes),
        "scientific_gate_status": {
            "evaluation_passed": evaluation.get("passed"),
            "evaluation_gate_checks": evaluation.get("gate_checks"),
            "comparison_passed": comparison.get("passed"),
            "comparison_gate_checks": comparison.get("gate_checks"),
        },
    }
    write_json(workspace / "plots" / "plot_manifest.json", record)
    return record


def run_workflow(
    *,
    bridge: Path,
    configuration_path: Path,
    workspace: Path,
    profile: str,
    show_progress: bool | None = None,
) -> dict[str, Any]:
    """Execute doctor → generate → train → evaluate → compare → summary."""

    from extract_dvcs_cff.progress import progress_bar

    workspace.mkdir(parents=True, exist_ok=True)
    overall = progress_bar(
        total=8,
        description="Complete DVCS workflow",
        enabled=show_progress,
    )
    health = doctor(
        bridge=bridge, configuration_path=configuration_path
    )
    write_json(workspace / "doctor.json", health)
    overall.update()
    generation = generate_pseudodata(
        bridge=bridge,
        configuration_path=configuration_path,
        workspace=workspace,
        profile=profile,
        show_progress=show_progress,
    )
    overall.update()
    deterministic_before = {
        name: sha256(workspace / "generated" / name)
        for name in (
            "array_manifest.json",
            "pseudodata.json",
            "theta_physical.npy",
            "contexts.npy",
        )
    }
    restart = generate_pseudodata(
        bridge=bridge,
        configuration_path=configuration_path,
        workspace=workspace,
        profile=profile,
        show_progress=show_progress,
    )
    deterministic_after = {
        name: sha256(workspace / "generated" / name)
        for name in deterministic_before
    }
    restart_record = {
        "schema_version": 1,
        "second_generation_cache_hits": restart["native_cache_hits"],
        "second_generation_cache_misses": restart["native_cache_misses"],
        "expected_cache_hits": generation["native_batch_count"],
        "native_batch_count": restart["native_batch_count"],
        "deterministic_hashes_before": deterministic_before,
        "deterministic_hashes_after": deterministic_after,
        "byte_reproducible": deterministic_before == deterministic_after,
        "passed": bool(
            restart["native_cache_misses"] == 0
            and restart["native_cache_hits"]
            == generation["native_batch_count"]
            and restart["native_batch_count"]
            == generation["native_batch_count"]
            and deterministic_before == deterministic_after
        ),
    }
    write_json(workspace / "restart_cache_metrics.json", restart_record)
    overall.update()
    training = train_model(
        configuration_path=configuration_path,
        workspace=workspace,
        profile=profile,
        show_progress=show_progress,
    )
    overall.update()
    evaluation = evaluate_model(
        bridge=bridge,
        configuration_path=configuration_path,
        workspace=workspace,
        profile=profile,
        show_progress=show_progress,
    )
    overall.update()
    comparison = compare_posteriors(
        configuration_path=configuration_path,
        workspace=workspace,
        profile=profile,
        show_progress=show_progress,
    )
    overall.update()
    plot_record = plot_saved_results(
        configuration_path=configuration_path,
        workspace=workspace,
        profile=profile,
    )
    plot_hashes = plot_record["plots"]
    overall.update()
    passed = bool(
        restart_record["passed"]
        and evaluation["passed"]
        and comparison["passed"]
    )
    summary = {
        "schema_version": 1,
        "workflow": "full_independent_dd_multiq2_pseudodata_v1",
        "profile": profile,
        "passed": passed,
        "doctor": health,
        "generation": generation,
        "restart_and_cache": restart_record,
        "training": training,
        "evaluation": evaluation,
        "comparison": comparison,
        "diagnostic_plots": plot_hashes,
        "real_data_used": False,
        "analysis_order_label": None,
        "next_stage_started": False,
    }
    write_json(workspace / "summary.json", summary)
    (workspace / "summary.md").write_text(
        _summary_markdown(summary), encoding="utf-8"
    )
    overall.update()
    overall.close()
    return summary
