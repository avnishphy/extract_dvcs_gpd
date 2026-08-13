"""Safe user-project interface for the synthetic DVCS posterior workflow.

The scientific implementation remains in ``workflows.pseudodata``.  This
module adds only a containment and translation layer:

* editable files live below the Git-ignored ``user/workspaces`` directory;
* users never edit canonical development configurations or accepted evidence;
* a small public JSON document is translated into the frozen engine schema;
* each profile writes to a separate content-hashed result directory.

There is deliberately no command that deletes, resets, or overwrites an
existing project.  Create a new project name for a changed experiment.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

from extract_dvcs_cff.workflows.pseudodata import (
    compare_real_observables,
    compare_posteriors,
    doctor,
    evaluate_model,
    evaluate_native_model_holdouts,
    generate_pseudodata,
    load_configuration,
    plot_saved_results,
    run_workflow,
    sha256,
    train_model,
    write_json,
)
from extract_dvcs_cff.optimization import optimize_hyperparameters
from extract_dvcs_cff.data.gpddatabase import (
    build_kinematic_catalog,
    build_real_data_mapping_readiness,
    require_physical_fixed_target_kinematics,
    select_native_scale_points,
)


PROJECT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
EXPERIMENT_KEYS_V1 = {
    "schema_version",
    "experiment_name",
    "truth",
    "kinematics",
    "uncertainties",
    "profiles",
    "network",
    "seeds",
    "validation_gates",
}
EXPERIMENT_KEYS_V2 = {
    "schema_version",
    "experiment",
    "injected_truth",
    "synthetic_dataset",
    "inference",
    "validation_gates",
}
EXPERIMENT_KEYS_V3 = EXPERIMENT_KEYS_V2 | {"output_diagnostics"}
EXPERIMENT_METADATA_KEYS = {"name", "description"}
INJECTED_TRUTH_KEYS = {"gpd_parameters", "nuisance_parameters"}
GPD_TRUTH_KEYS = {
    "H",
    "E",
    "Htilde",
    "Etilde",
    "shadow_coefficients",
    "shadow_channel_amplitudes",
}
GPD_SHAPE_KEYS = {
    "normalization",
    "a",
    "c",
    "profile_b",
    "t_slope_GeV_minus2",
}
NUISANCE_TRUTH_KEYS = {
    "eta_global_normalization",
    "eta_lu_normalization",
}
SYNTHETIC_DATASET_KEYS_V5 = {"kinematics", "uncertainty_model"}
SYNTHETIC_DATASET_KEYS = SYNTHETIC_DATASET_KEYS_V5 | {"kinematics_source"}
KINEMATICS_SOURCE_KEYS = {
    "mode",
    "database_root",
    "database_revision",
    "catalog_sha256",
    "selection_metric",
    "selected_records",
    "measurements_quarantined",
    "data_evolved",
    "gpd_evolved_to_each_datum_Q2",
}
SELECTED_RECORD_KEYS = {
    "record_id",
    "dataset_id",
    "source_uuid",
    "source_path",
    "collaboration",
    "reference",
    "source_Q2_GeV2",
    "source_beam_energy_GeV",
    "generated_Q2_GeV2",
    "Q2_projected",
    "generated_beam_energy_GeV",
    "measurement_values_used",
}
INFERENCE_KEYS = {
    "profiles",
    "neural_posterior",
    "random_seeds",
    "runtime",
    "hyperparameter_optimization",
}
RUNTIME_KEYS_V5 = {"accelerator", "cpu_threads", "deterministic_algorithms"}
RUNTIME_KEYS = RUNTIME_KEYS_V5 | {"native_workers"}
TRUTH_KEYS = {
    "H_shadow_coefficient",
    "E_shadow_coefficient",
    "Htilde_shadow_coefficient",
    "Etilde_shadow_coefficient",
    "eta_global_normalization",
    "eta_lu_normalization",
}
KINEMATIC_KEYS = {
    "x_b",
    "t_GeV2",
    "Q2_GeV2",
    "beam_energy_GeV",
    "phi_rad",
}
UNCERTAINTY_KEYS = {
    "uncorrelated_relative_sigma",
    "uncorrelated_absolute_floor_normalized",
    "local_correlation_fraction",
    "local_correlation_length",
    "phi_shape_correlated_fraction",
    "global_normalization_fraction",
    "lu_normalization_fraction",
}
DIAGNOSTIC_KEYS = {
    "gpd_reference_kinematics",
    "gpd_x_grid",
    "credible_interval",
}
GPD_GRID_KEYS = {"minimum", "maximum", "point_count"}


def _exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    """Fail on typos instead of silently ignoring a requested control."""

    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"{context} key mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )


def _repository_root(argument: Path | None) -> Path:
    """Resolve and verify the repository that owns the user launcher."""

    candidate = argument
    if candidate is None:
        environment = os.environ.get("DVCS_INFER_REPOSITORY_ROOT")
        candidate = Path(environment) if environment else Path.cwd()
    root = candidate.resolve(strict=True)
    required = (
        root / "configs/inference/stage10_pseudodata_workflow.json",
        root / "configs/physics/stage10_pseudodata_systematics_v1.json",
        root / "user/README.md",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError(
            f"{root} is not an extract_dvcs_cff user release"
        )
    return root


def _user_root(repository: Path) -> Path:
    """Return the sole location in which the user interface may write."""

    configured = os.environ.get("DVCS_WORKSPACE_ROOT")
    path = Path(configured) if configured else repository / "user" / "workspaces"
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeError("user/workspaces must not be a symlink")
    return path.resolve(strict=True)


def _database_root(repository: Path) -> Path:
    """Resolve the read-only database mount without assuming a sibling repo."""

    configured = os.environ.get("DVCS_GPDDATABASE_ROOT")
    return Path(configured) if configured else repository.parent / "gpddatabase"


def _validate_name(name: str) -> None:
    if not PROJECT_NAME.fullmatch(name):
        raise ValueError(
            "project name must start with an alphanumeric character and "
            "contain only letters, digits, '_' or '-'"
        )


def _project_path(repository: Path, name: str, *, existing: bool) -> Path:
    """Resolve one direct child and reject traversal and symlink escapes."""

    _validate_name(name)
    root = _user_root(repository)
    candidate = root / name
    if existing:
        project = candidate.resolve(strict=True)
        if candidate.is_symlink() or not project.is_dir():
            raise RuntimeError("user project must be a real directory")
        if project.parent != root:
            raise RuntimeError("user project escapes user/workspaces")
        return project
    if candidate.exists() or candidate.is_symlink():
        raise FileExistsError(
            f"project {name!r} already exists; choose a new name"
        )
    return candidate


def _canonical_paths(repository: Path) -> tuple[Path, Path]:
    return (
        repository
        / "configs"
        / "inference"
        / "stage10_pseudodata_workflow.json",
        repository
        / "configs"
        / "physics"
        / "stage10_pseudodata_systematics_v1.json",
    )


def _public_experiment(
    name: str,
    canonical_configuration: Mapping[str, Any],
    kinematics_source: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a readable public schema from documented engine controls.

    The hierarchy follows how a user thinks about an experiment: identity,
    injected physics/nuisance truth, synthetic measurements, inference, then
    validation.  Internal workflow and representation identifiers are absent.
    """

    model = canonical_configuration["experimental_model"]
    nuisance = {
        item["name"]: item for item in model["nuisance_groups"]
    }
    truth = canonical_configuration["truth"]
    return {
        "schema_version": 7,
        "experiment": {
            "name": name,
            "description": (
                "Editable synthetic DVCS posterior experiment; no real data"
            ),
        },
        "injected_truth": {
            "gpd_parameters": {
                name: {
                    channel: {
                        "normalization": canonical_configuration["_physics"]
                        ["parameters"]["gpd_shapes"][name][channel]
                        ["normalization"],
                        "a": canonical_configuration["_physics"]
                        ["parameters"]["gpd_shapes"][name][channel]["a"],
                        "c": canonical_configuration["_physics"]
                        ["parameters"]["gpd_shapes"][name][channel]["c"],
                        "profile_b": canonical_configuration["_physics"]
                        ["parameters"]["gpd_shapes"][name][channel]
                        ["profile_b"],
                        "t_slope_GeV_minus2": canonical_configuration
                        ["_physics"]["parameters"]["gpd_shapes"][name]
                        [channel]["t_slope"]["value"],
                    }
                    for channel in ("u", "d", "s", "gluon")
                }
                for name in ("H", "E", "Htilde", "Etilde")
            } | {
                "shadow_coefficients": deepcopy(
                    canonical_configuration["_physics"]["parameters"]
                    ["shadow_coefficients"]
                ),
                "shadow_channel_amplitudes": deepcopy(
                    canonical_configuration["_physics"]["parameters"]
                    ["shadow_channel_amplitudes"]
                ),
            },
            "nuisance_parameters": {
                "eta_global_normalization": truth[
                    "eta_global_normalization"
                ],
                "eta_lu_normalization": truth["eta_lu_normalization"],
            },
        },
        "synthetic_dataset": {
            "kinematics": deepcopy(canonical_configuration["kinematics"]),
            "kinematics_source": deepcopy(kinematics_source),
            "uncertainty_model": {
                "uncorrelated_relative_sigma": model[
                    "uncorrelated_relative_sigma"
                ],
                "uncorrelated_absolute_floor_normalized": model[
                    "uncorrelated_absolute_floor_normalized"
                ],
                "local_correlation_fraction": model[
                    "local_correlation_fraction"
                ],
                "local_correlation_length": model[
                    "local_correlation_length"
                ],
                "phi_shape_correlated_fraction": model[
                    "phi_shape_correlated_fraction"
                ],
                "global_normalization_fraction": nuisance[
                    "eta_global_normalization"
                ]["fractional_response"],
                "lu_normalization_fraction": nuisance[
                    "eta_lu_normalization"
                ]["fractional_response"],
            },
        },
        "inference": {
            "profiles": deepcopy(canonical_configuration["profiles"]),
            "neural_posterior": deepcopy(canonical_configuration["network"]),
            "random_seeds": deepcopy(canonical_configuration["seeds"]),
            "runtime": deepcopy(canonical_configuration["runtime"]),
            "hyperparameter_optimization": deepcopy(
                canonical_configuration["hyperparameter_optimization"]
            ),
        },
        "output_diagnostics": deepcopy(
            canonical_configuration["diagnostics"]
        ),
        "validation_gates": deepcopy(
            canonical_configuration["frozen_validation_gates"]
        ),
    }


def _write_public_json(path: Path, value: Any) -> None:
    """Write user-editable JSON with stable ordering and four-space indent."""

    path.write_text(
        json.dumps(
            value,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _generated_readme(name: str) -> str:
    """Give every project self-contained commands and result locations."""

    return f"""# DVCS pseudodata project: {name}

This directory is your isolated sandbox. It is ignored by Git, so you may
edit `experiment.json`, create notes, and run experiments without changing
the framework or its accepted development evidence.

Run commands from the repository root:

```bash
./dvcs doctor {name}
./dvcs run {name} --profile quick
```

For step-by-step execution:

```bash
./dvcs generate {name} --profile quick
./dvcs train {name} --profile quick
./dvcs evaluate {name} --profile quick
./dvcs compare {name} --profile quick
./dvcs plot {name} --profile quick
```

`plot` consumes compatible saved results without repeating PARTONS or neural
inference. For custom analysis, copy the compact notebook from
`../../scripts/` into `../../scripts/workspaces/` and change its `PROJECT` and
`PROFILE` values.

Edit `injected_truth.gpd_parameters.shadow_coefficients` in `experiment.json`
to change the four type-level shadow injections for `H`, `E`, `Htilde`, and
`Etilde`; each accepted range is `[-1, 1]`. The neighboring
`shadow_channel_amplitudes` block controls how each type-level coefficient is
distributed over `u`, `d`, `s`, and `gluon`. These are synthetic stress-test
directions, not four native PARTONS shadow modules. The NPE infers all five
DD controls for every H/E/Htilde/Etilde × u/d/s/gluon block (80 physics
coordinates) plus two normalization nuisances. Shadows are fixed simulator
settings in this milestone; changing one defines a new simulator family.

The measurement-free `real_data_mapping_readiness.json` inventories candidate
observable mappings and missing audits. It does not enable a real-data fit.
All controls are explained in `../../README.md`. The exact simulator,
uncertainty equations, priors,
flavor conventions, and pseudodata-to-NPE sequence are in
`../../WORKFLOW_AND_PHYSICS.md`.
The exhaustive schema-7 field reference is
`../../../docs/EXPERIMENT_JSON_REFERENCE.md`.
Interpretation of every result is in
`../../PHYSICS_AND_RESULTS_GUIDE.md`.

Interactive workflow commands show live progress on stderr while preserving
the final JSON result on stdout. Exact PARTONS generation advances after each
completed deterministic two-parameter native block and forces an immediate
redraw after incrementing the accepted counter. The postfix separately shows
attempted, accepted, and rejected totals. Injected truth is preflighted before
the seed-preserving prior bank is submitted in 64-parameter waves.
`runtime.native_workers` defaults to
`"all_available"`, meaning independent PARTONS bridge subprocesses up to the
CPU-affinity limit; no PARTONS object is shared between workers. `doctor`
reports the requested and resolved worker counts and the batch/update unit.
Use `--no-progress` for a quiet scripted run.

After neural coverage trials, `evaluate` shows `Exact posterior reevaluation`
and `Exact GPD diagnostics`. Both use the same process-isolated worker pool and
two-sample native batches. Their postfix reports completed, valid, invalid,
and worker counts; PARTONS remains CPU-only.

New projects borrow `xB`, `t`, `phi`, `Q2`, and beam energy from the pinned
read-only `gpddatabase` catalog. `kinematics_source` records every source row.
Selector v2 requires fixed-target `0 < y = Q2/(2*M_p*E*xB) < 1`; the public
loader and native bridge recheck it before simulation.
The GPD starts at `Q0^2=1 GeV2` and the native backend evolves it to each
retained datum scale; data are never evolved. No database measurement or
uncertainty is used.

Outputs for the quick profile appear in `results/quick/`. Start with
`results/quick/summary.md` and the PNG files under `results/quick/plots/`:
training history, parameter posteriors and correlations, observable
predictions and pulls, coverage, real/imaginary CFFs, and combined plus
flavor-separated GPD functions. The private `.engine/`
directory is generated from your experiment and should not be edited; it
contains no independent physics implementation.

Changing `experiment.json` after generating results intentionally makes the
existing result contract fail. Use a new project name so results from
different configurations cannot be mixed accidentally.
"""


def initialize_project(repository: Path, name: str) -> dict[str, Any]:
    """Create one new editable project without touching framework inputs."""

    project = _project_path(repository, name, existing=False)
    canonical_path, physics_path = _canonical_paths(repository)
    canonical_configuration = load_configuration(canonical_path)
    canonical_configuration["_physics"] = json.loads(
        physics_path.read_text(encoding="utf-8")
    )
    database_root = _database_root(repository)
    if database_root.is_dir():
        catalog = build_kinematic_catalog(database_root)
        mapping_readiness = build_real_data_mapping_readiness(catalog)
        selected, kinematics_source = select_native_scale_points(
            catalog,
            canonical_configuration["kinematics"],
        )
        canonical_configuration["kinematics"] = selected
        kinematics_source["database_root"] = str(
            database_root.resolve(strict=True)
        )
    else:
        # This explicit mode supports source distributions and isolated tests
        # where the protected sibling is intentionally not installed.
        kinematics_source = {"mode": "manual"}
        mapping_readiness = {
            "schema_version": 1,
            "status": "gpddatabase_not_installed_real_likelihood_disabled",
            "real_data_fit_enabled": False,
        }
    project.mkdir(parents=False)
    (project / "results").mkdir()
    _write_public_json(
        project / "experiment.json",
        _public_experiment(
            name, canonical_configuration, kinematics_source
        ),
    )
    _write_public_json(
        project / "real_data_mapping_readiness.json", mapping_readiness
    )
    (project / "README.md").write_text(
        _generated_readme(name), encoding="utf-8"
    )
    return {
        "status": "created",
        "project": name,
        "experiment": str(project / "experiment.json"),
        "readme": str(project / "README.md"),
        "real_data_mapping_readiness": str(
            project / "real_data_mapping_readiness.json"
        ),
        "next_command": f"./dvcs doctor {name}",
    }


def _load_experiment(project: Path) -> dict[str, Any]:
    """Load the full-independent public schema; reject reduced schemas.

    Schema v1 remains readable only for historical developer tests. New user
    New projects receive schema v7. Earlier schemas encoded a different,
    reduced posterior and are never migrated silently.
    """

    path = project / "experiment.json"
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("experiment.json must contain one JSON object")
    if value.get("schema_version") == 1:
        _exact_keys(value, EXPERIMENT_KEYS_V1, "experiment")
        if value["experiment_name"] != project.name:
            raise ValueError(
                "experiment_name must equal the project directory"
            )
        validation_gates = deepcopy(value["validation_gates"])
        retired_width_key = "minimum_shadow_posterior_prior_width_ratio"
        current_width_key = (
            "minimum_neural_to_conventional_stress_width_ratio"
        )
        if retired_width_key in validation_gates:
            if current_width_key in validation_gates:
                raise ValueError(
                    "validation_gates cannot contain both retired and current "
                    "stress-width keys"
                )
            validation_gates[current_width_key] = validation_gates.pop(
                retired_width_key
            )
        normalized = {
            "truth": value["truth"],
            "kinematics": value["kinematics"],
            "uncertainties": value["uncertainties"],
            "profiles": value["profiles"],
            "network": value["network"],
            "seeds": value["seeds"],
            "validation_gates": validation_gates,
            "kinematics_source": {"mode": "manual"},
        }
    elif value.get("schema_version") == 7:
        public_schema = int(value["schema_version"])
        _exact_keys(value, EXPERIMENT_KEYS_V3, "experiment")
        _exact_keys(
            value["experiment"],
            EXPERIMENT_METADATA_KEYS,
            "experiment metadata",
        )
        _exact_keys(
            value["injected_truth"],
            INJECTED_TRUTH_KEYS,
            "injected_truth",
        )
        _exact_keys(
            value["injected_truth"]["gpd_parameters"],
            GPD_TRUTH_KEYS,
            "injected_truth.gpd_parameters",
        )
        for gpd_type in ("H", "E", "Htilde", "Etilde"):
            _exact_keys(value["injected_truth"]["gpd_parameters"][gpd_type],
                {"u", "d", "s", "gluon"},
                f"injected_truth.gpd_parameters.{gpd_type}")
            for channel in ("u", "d", "s", "gluon"):
                _exact_keys(
                    value["injected_truth"]["gpd_parameters"][gpd_type][channel],
                    GPD_SHAPE_KEYS,
                    f"injected_truth.gpd_parameters.{gpd_type}.{channel}",
                )
        for block in ("shadow_coefficients", "shadow_channel_amplitudes"):
            _exact_keys(
                value["injected_truth"]["gpd_parameters"][block],
                {"H", "E", "Htilde", "Etilde"},
                f"injected_truth.gpd_parameters.{block}",
            )
        for gpd_type in ("H", "E", "Htilde", "Etilde"):
            _exact_keys(
                value["injected_truth"]["gpd_parameters"]
                ["shadow_channel_amplitudes"][gpd_type],
                {"u", "d", "s", "gluon"},
                f"shadow_channel_amplitudes.{gpd_type}",
            )
        _exact_keys(
            value["injected_truth"]["nuisance_parameters"],
            NUISANCE_TRUTH_KEYS,
            "injected_truth.nuisance_parameters",
        )
        _exact_keys(
            value["synthetic_dataset"],
            (
                SYNTHETIC_DATASET_KEYS
                if public_schema == 7
                else SYNTHETIC_DATASET_KEYS_V5
            ),
            "synthetic_dataset",
        )
        if public_schema == 7:
            source = value["synthetic_dataset"]["kinematics_source"]
            if not isinstance(source, dict):
                raise ValueError("kinematics_source must be an object")
            if source.get("mode") == "manual":
                _exact_keys(source, {"mode"}, "kinematics_source")
            elif source.get("mode") in {
                "gpddatabase_native_scale_kinematics_v1",
                "gpddatabase_native_scale_kinematics_v2",
            }:
                _exact_keys(
                    source, KINEMATICS_SOURCE_KEYS, "kinematics_source"
                )
                if source["measurements_quarantined"] is not True:
                    raise ValueError(
                        "gpddatabase measurements must remain quarantined"
                    )
                if source["data_evolved"] is not False or source[
                    "gpd_evolved_to_each_datum_Q2"
                ] is not True:
                    raise ValueError("multi-Q2 provenance policy changed")
                if not isinstance(source["selected_records"], list):
                    raise ValueError("selected_records must be a list")
                for index, record in enumerate(source["selected_records"]):
                    _exact_keys(
                        record,
                        SELECTED_RECORD_KEYS,
                        f"selected_records[{index}]",
                    )
                    if record["measurement_values_used"] is not False:
                        raise ValueError(
                            "selected records cannot use measured values"
                        )
            else:
                raise ValueError("unsupported kinematics_source.mode")
        _exact_keys(value["inference"], INFERENCE_KEYS, "inference")
        _exact_keys(
            value["inference"]["runtime"],
            RUNTIME_KEYS if public_schema == 7 else RUNTIME_KEYS_V5,
            "inference.runtime",
        )
        _exact_keys(
            value["output_diagnostics"],
            DIAGNOSTIC_KEYS,
            "output_diagnostics",
        )
        _exact_keys(
            value["output_diagnostics"]["gpd_reference_kinematics"],
            KINEMATIC_KEYS,
            "output_diagnostics.gpd_reference_kinematics",
        )
        _exact_keys(
            value["output_diagnostics"]["gpd_x_grid"],
            GPD_GRID_KEYS,
            "output_diagnostics.gpd_x_grid",
        )
        if value["experiment"]["name"] != project.name:
            raise ValueError(
                "experiment.name must equal the project directory"
            )
        if not isinstance(value["experiment"]["description"], str):
            raise ValueError("experiment.description must be a string")
        validation_gates = deepcopy(value["validation_gates"])
        retired_width_key = "minimum_shadow_posterior_prior_width_ratio"
        current_width_key = (
            "minimum_neural_to_conventional_stress_width_ratio"
        )
        if retired_width_key in validation_gates:
            if current_width_key in validation_gates:
                raise ValueError(
                    "validation_gates cannot contain both retired and current "
                    "stress-width keys"
                )
            validation_gates[current_width_key] = validation_gates.pop(
                retired_width_key
            )
        normalized = {
            "truth": {
                **{
                    f"{gpd_type}_shadow_coefficient": value["injected_truth"]
                    ["gpd_parameters"]["shadow_coefficients"][gpd_type]
                    for gpd_type in ("H", "E", "Htilde", "Etilde")
                },
                **value["injected_truth"]["nuisance_parameters"],
            },
            "gpd_shapes": {
                name: value["injected_truth"]["gpd_parameters"][name]
                for name in ("H", "E", "Htilde", "Etilde")
            },
            "shadow_amplitudes": value["injected_truth"]["gpd_parameters"]
            ["shadow_channel_amplitudes"],
            "kinematics": value["synthetic_dataset"]["kinematics"],
            "kinematics_source": (
                value["synthetic_dataset"]["kinematics_source"]
                if public_schema == 7
                else {"mode": "manual"}
            ),
            "uncertainties": value["synthetic_dataset"][
                "uncertainty_model"
            ],
            "profiles": value["inference"]["profiles"],
            "network": value["inference"]["neural_posterior"],
            "runtime": {
                **value["inference"]["runtime"],
                "native_workers": value["inference"]["runtime"].get(
                    "native_workers", "all_available"
                ),
            },
            "optimization": value["inference"][
                "hyperparameter_optimization"
            ],
            "seeds": value["inference"]["random_seeds"],
            "validation_gates": validation_gates,
            "diagnostics": value["output_diagnostics"],
        }
    elif value.get("schema_version") in {2, 3, 4, 5, 6}:
        raise ValueError(
            "this experiment schema is retired; create a new project to use "
            "the full-independent multi-Q2 schema_version 7"
        )
    else:
        raise ValueError("unsupported experiment schema_version")
    accelerator_override = os.environ.get("DVCS_ACCELERATOR")
    threads_override = os.environ.get("DVCS_CPU_THREADS")
    workers_override = os.environ.get("DVCS_NATIVE_WORKERS")
    if accelerator_override:
        normalized["runtime"]["accelerator"] = accelerator_override
    if threads_override:
        normalized["runtime"]["cpu_threads"] = int(threads_override)
    if workers_override:
        normalized["runtime"]["native_workers"] = (
            workers_override
            if workers_override == "all_available"
            else int(workers_override)
        )
    _exact_keys(normalized["truth"], TRUTH_KEYS, "truth")
    _exact_keys(
        normalized["uncertainties"], UNCERTAINTY_KEYS, "uncertainties"
    )
    if len(normalized["kinematics"]) < 4:
        raise ValueError("at least four kinematics are required")
    for index, point in enumerate(normalized["kinematics"]):
        _exact_keys(point, KINEMATIC_KEYS, f"kinematics[{index}]")
        for key, item in point.items():
            if not isinstance(item, (int, float)) or not math.isfinite(item):
                raise ValueError(f"kinematics[{index}].{key} must be finite")
        if not 0.12 <= float(point["x_b"]) <= 0.28:
            raise ValueError(f"kinematics[{index}].x_b is outside [0.12,0.28]")
        if not -0.2 <= float(point["t_GeV2"]) <= -0.08:
            raise ValueError(
                f"kinematics[{index}].t_GeV2 is outside [-0.2,-0.08]"
            )
        if not 1.0 <= float(point["Q2_GeV2"]) <= 80.0:
            raise ValueError("Q2_GeV2 must lie in [1,80]")
        if not 3.0 <= float(point["beam_energy_GeV"]) <= 200.0:
            raise ValueError(
                "beam_energy_GeV must lie in [3,200]"
            )
        require_physical_fixed_target_kinematics(
            x_b=float(point["x_b"]),
            Q2_GeV2=float(point["Q2_GeV2"]),
            beam_energy_GeV=float(point["beam_energy_GeV"]),
            context=f"kinematics[{index}]",
        )
        if not 0.0 <= float(point["phi_rad"]) <= 2.0 * math.pi:
            raise ValueError(f"kinematics[{index}].phi_rad is outside [0,2pi]")
    truth = normalized["truth"]
    for gpd_type in ("H", "E", "Htilde", "Etilde"):
        for channel, shape in normalized["gpd_shapes"][gpd_type].items():
            if not -4.0 < float(shape["normalization"]) < 4.0:
                raise ValueError(f"{gpd_type}.{channel}.normalization outside (-4,4)")
            if not 0.1 < float(shape["a"]) < 1.0 or not 2.0 < float(shape["c"]) < 6.0:
                raise ValueError(f"{gpd_type}.{channel} requires 0.1<a<1, 2<c<6")
            if not 1.0 < float(shape["profile_b"]) < 4.0:
                raise ValueError(f"{gpd_type}.{channel}.profile_b outside (1,4)")
            if not 0.0 < float(shape["t_slope_GeV_minus2"]) < 2.0:
                raise ValueError(f"{gpd_type}.{channel}.t_slope outside (0,2)")
        shadow = float(truth[f"{gpd_type}_shadow_coefficient"])
        if not -1.0 <= shadow <= 1.0:
            raise ValueError(f"truth.{gpd_type}_shadow_coefficient outside [-1,1]")
        for channel, amplitude in normalized["shadow_amplitudes"][gpd_type].items():
            if not -4.0 <= float(amplitude) <= 4.0:
                raise ValueError(f"{gpd_type}.{channel} shadow amplitude outside [-4,4]")
    for key, item in truth.items():
        if not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"truth.{key} must be finite")
    for key, item in normalized["uncertainties"].items():
        if not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(f"uncertainties.{key} must be finite")
        if float(item) < 0.0:
            raise ValueError(f"uncertainties.{key} must be nonnegative")
    if float(normalized["uncertainties"]["local_correlation_length"]) == 0.0:
        raise ValueError("local_correlation_length must be positive")
    grid = normalized["diagnostics"]["gpd_x_grid"]
    if not -1.0 <= float(grid["minimum"]) < float(grid["maximum"]) <= 1.0:
        raise ValueError("GPD diagnostic x range must lie inside [-1,1]")
    if (
        not isinstance(grid["point_count"], int)
        or not 3 <= grid["point_count"] <= 129
    ):
        raise ValueError("GPD diagnostic point_count must be an integer in [3,129]")
    credible = float(normalized["diagnostics"]["credible_interval"])
    if not 0.0 < credible < 1.0:
        raise ValueError("credible_interval must lie in (0,1)")
    return normalized


def _validate_kinematics_provenance(experiment: Mapping[str, Any]) -> None:
    """Revalidate a database-backed selection before creating engine input."""

    source = experiment["kinematics_source"]
    if source["mode"] == "manual":
        return
    catalog = build_kinematic_catalog(Path(source["database_root"]))
    if catalog["database_revision"] != source["database_revision"]:
        raise RuntimeError("gpddatabase revision changed since project creation")
    if catalog["catalog_sha256"] != source["catalog_sha256"]:
        raise RuntimeError("gpddatabase kinematic catalog changed")
    records = {
        item["record_id"]: item for item in catalog["kinematic_records"]
    }
    selected = source["selected_records"]
    points = experiment["kinematics"]
    if len(selected) != len(points):
        raise ValueError("selected_records and kinematics length mismatch")
    for index, (provenance, point) in enumerate(zip(selected, points, strict=True)):
        record = records.get(provenance["record_id"])
        if record is None:
            raise RuntimeError(f"source record {provenance['record_id']} vanished")
        values = record["kinematics"]
        units = record["kinematic_units"]
        phi = float(values["phi"])
        if units["phi"] == "deg":
            phi = math.radians(phi)
        phi %= 2.0 * math.pi
        expected = {
            "x_b": float(values["xB"]),
            "t_GeV2": float(values["t"]),
            "phi_rad": phi,
            "Q2_GeV2": float(provenance["generated_Q2_GeV2"]),
            "beam_energy_GeV": float(
                provenance["generated_beam_energy_GeV"]
            ),
        }
        for key, value in expected.items():
            if not math.isclose(
                float(point[key]), value, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"kinematics[{index}].{key} no longer matches its "
                    "gpddatabase provenance; use mode='manual' after editing"
                )


def _require_matching_shape(
    public: Mapping[str, Any],
    canonical: Mapping[str, Any],
    context: str,
) -> None:
    """Keep adjustable blocks complete so omitted controls are never hidden."""

    _exact_keys(public, set(canonical), context)


def prepare_engine(
    repository: Path, project: Path
) -> tuple[Path, dict[str, Any]]:
    """Translate a public experiment into the private frozen engine schema."""

    experiment = _load_experiment(project)
    _validate_kinematics_provenance(experiment)
    canonical_path, physics_path = _canonical_paths(repository)
    configuration = load_configuration(canonical_path)
    _require_matching_shape(
        experiment["profiles"], configuration["profiles"], "profiles"
    )
    for profile in ("quick", "validation"):
        _require_matching_shape(
            experiment["profiles"][profile],
            configuration["profiles"][profile],
            f"profiles.{profile}",
        )
    _require_matching_shape(
        experiment["network"], configuration["network"], "network"
    )
    _require_matching_shape(
        experiment["runtime"], configuration["runtime"], "runtime"
    )
    _require_matching_shape(
        experiment["optimization"],
        configuration["hyperparameter_optimization"],
        "hyperparameter_optimization",
    )
    _require_matching_shape(
        experiment["seeds"], configuration["seeds"], "seeds"
    )
    _require_matching_shape(
        experiment["validation_gates"],
        configuration["frozen_validation_gates"],
        "validation_gates",
    )

    configuration["physics_configuration"] = "physics.json"
    configuration["truth"] = deepcopy(experiment["truth"])
    configuration["kinematics"] = deepcopy(experiment["kinematics"])
    configuration["data_provenance"] = deepcopy(
        experiment["kinematics_source"]
    )
    configuration["diagnostics"] = deepcopy(experiment["diagnostics"])
    configuration["profiles"] = deepcopy(experiment["profiles"])
    configuration["network"] = deepcopy(experiment["network"])
    configuration["runtime"] = deepcopy(experiment["runtime"])
    configuration["hyperparameter_optimization"] = deepcopy(
        experiment["optimization"]
    )
    configuration["seeds"] = deepcopy(experiment["seeds"])
    configuration["frozen_validation_gates"] = deepcopy(
        experiment["validation_gates"]
    )
    model = configuration["experimental_model"]
    controls = experiment["uncertainties"]
    for key in (
        "uncorrelated_relative_sigma",
        "uncorrelated_absolute_floor_normalized",
        "local_correlation_fraction",
        "local_correlation_length",
        "phi_shape_correlated_fraction",
    ):
        model[key] = controls[key]
    responses = {
        "eta_global_normalization": controls[
            "global_normalization_fraction"
        ],
        "eta_lu_normalization": controls["lu_normalization_fraction"],
    }
    for nuisance in model["nuisance_groups"]:
        nuisance["fractional_response"] = responses[nuisance["name"]]

    engine = project / ".engine"
    engine.mkdir(exist_ok=True)
    engine_physics = engine / "physics.json"
    physics = json.loads(physics_path.read_text(encoding="utf-8"))
    for name, public_channels in experiment["gpd_shapes"].items():
        for channel, public_shape in public_channels.items():
            shape = physics["parameters"]["gpd_shapes"][name][channel]
            shape["normalization"] = float(public_shape["normalization"])
            shape["profile_b"] = float(public_shape["profile_b"])
            shape["t_slope"]["value"] = float(public_shape["t_slope_GeV_minus2"])
            shape["a"] = float(public_shape["a"])
            shape["c"] = float(public_shape["c"])
    physics["parameters"]["shadow_coefficients"] = {
        name: float(experiment["truth"][f"{name}_shadow_coefficient"])
        for name in ("H", "E", "Htilde", "Etilde")
    }
    physics["parameters"]["shadow_channel_amplitudes"] = deepcopy(
        experiment["shadow_amplitudes"]
    )
    write_json(engine_physics, physics)
    engine_configuration = engine / "workflow.json"
    write_json(engine_configuration, configuration)
    # Reuse the workflow's strict parser as the final translation gate.
    load_configuration(engine_configuration)
    return engine_configuration, experiment


def _bridge_path(repository: Path, argument: Path | None) -> Path:
    candidate = (
        argument
        if argument is not None
        else repository
        / "build"
        / "user_native"
        / "cpp"
        / "partons_bridge"
        / "partons_bridge"
    )
    return candidate.resolve(strict=True)


def assert_result_contract(
    *,
    configuration: Path,
    workspace: Path,
    profile: str,
    bridge: Path,
) -> None:
    """Refuse stepwise reuse after any experiment/profile/backend change.

    Generation creates the authoritative contract.  The full ``run`` command
    reaches the same check through generation, but direct ``train``,
    ``evaluate``, and ``compare`` calls need this preflight before consuming
    arrays from an earlier step.
    """

    path = workspace / "workspace_contract.json"
    if not path.is_file():
        raise RuntimeError(
            f"generate must run first; missing {path}"
        )
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "configuration": str(configuration.resolve(strict=True)),
        "configuration_sha256": sha256(configuration.resolve(strict=True)),
        "profile": profile,
        "bridge_sha256": sha256(bridge.resolve(strict=True)),
        "real_data": False,
        "synthetic_split_policy": (
            "native_parameter_grouped_train_validation_test_v1"
        ),
    }
    if observed != expected:
        raise RuntimeError(
            "result contract mismatch; create a new project after changing "
            "experiment.json, profile, or native bridge"
        )


def _public_result(
    command: str,
    project: Path,
    profile: str | None,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Emit concise user metrics while retaining full JSON in results."""

    public: dict[str, Any] = {
        "command": command,
        "project": project.name,
        "status": (
            "pass"
            if result.get("passed") is True
            else "fail"
            if result.get("passed") is False
            else result.get("status", "complete")
        ),
    }
    if profile is not None:
        output = project / "results" / profile
        public["profile"] = profile
        public["results"] = str(output)
    if command == "generate":
        public["native_parameter_count"] = result["native_parameter_count"]
        public["native_invalid_count"] = result["native_invalid_count"]
    elif command == "doctor":
        public["neural_device"] = result["neural_device"]
        public["native_accelerator"] = result["native_accelerator"]
    elif command == "train":
        if result.get("status") == "distributed_worker_complete":
            public["distributed_rank"] = result["rank"]
            public["distributed_world_size"] = result["world_size"]
            public["trained_seeds"] = result["trained_seeds"]
            return public
        public["mean_train_negative_log_density"] = result[
            "mean_train_negative_log_density"
        ]
        public["mean_test_negative_log_density"] = result[
            "mean_test_negative_log_density"
        ]
        public["neural_device"] = result["neural_device"]
        public["maximum_cuda_peak_allocated_bytes"] = result[
            "maximum_cuda_peak_allocated_bytes"
        ]
    elif command == "optimize":
        public["best_validation_negative_log_density"] = result[
            "best_objective"
        ]
        public["best_parameters"] = result["best_parameters"]
        public["summary"] = str(
            project
            / "results"
            / str(profile)
            / "optimization"
            / "optimization_summary.json"
        )
    elif command == "evaluate":
        public["gate_checks"] = result["gate_checks"]
    elif command == "compare":
        public["effective_sample_size"] = result[
            "effective_sample_size"
        ]
        public["ensemble_sliced_wasserstein"] = result[
            "ensemble_sliced_wasserstein"
        ]
    elif command == "plot":
        public["plots"] = result["plot_count"]
        public["plot_directory"] = str(
            project / "results" / str(profile) / "plots"
        )
        public["evaluation_passed"] = result["scientific_gate_status"][
            "evaluation_passed"
        ]
        public["comparison_passed"] = result["scientific_gate_status"][
            "comparison_passed"
        ]
    elif command == "compare-real":
        public["diagnostic_only"] = True
        public["synthetic_posterior_gate_passed"] = result[
            "synthetic_posterior_gate_passed"
        ]
        public["agreement_claim_enabled"] = False
        public["native_prediction_set_complete"] = result[
            "native_prediction_set_complete"
        ]
        public["native_invalid_count"] = len(
            result["invalid_native_evaluations"]
        )
        public["observable_types_compared"] = result[
            "observable_coverage"
        ]["compared"]
        public["observation_count"] = len(result["rows"])
        public["plots"] = result["plots"]
        public["record"] = str(
            project
            / "results"
            / str(profile)
            / "real_data_comparison"
            / "real_observable_comparison.json"
        )
    elif command == "holdout":
        public["status"] = result["status"]
        public["execution_passed"] = result["execution_passed"]
        public["predictive_precision_passed"] = result[
            "predictive_precision_passed"
        ]
        public["method_improvement_required"] = result[
            "method_improvement_required"
        ]
        public["robustness_claim_enabled"] = result[
            "robustness_claim_enabled"
        ]
        public["minimum_observable_pointwise_90_truth_coverage"] = result[
            "minimum_observable_pointwise_90_truth_coverage"
        ]
        public["training_artifacts_unchanged"] = result["leakage_audit"][
            "protected_artifacts_unchanged"
        ]
        public["dd_outer_test_preserved"] = result["dataset_roles"][
            "dd_outer_test_preserved"
        ]
        public["real_data_fit_enabled"] = False
        public["summary"] = str(
            project / "results" / str(profile) / "holdout" / "summary.json"
        )
    elif command == "run":
        public["summary"] = str(project / "results" / profile / "summary.md")
        public["mean_train_negative_log_density"] = result["training"][
            "mean_train_negative_log_density"
        ]
        public["mean_test_negative_log_density"] = result["training"][
            "mean_test_negative_log_density"
        ]
        public["effective_sample_size"] = result["comparison"][
            "effective_sample_size"
        ]
        public["ensemble_sliced_wasserstein"] = result["comparison"][
            "ensemble_sliced_wasserstein"
        ]
        public["neural_device"] = result["training"]["neural_device"]
        public["maximum_cuda_peak_allocated_bytes"] = result["training"][
            "maximum_cuda_peak_allocated_bytes"
        ]
    return public


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dvcs-infer",
        description=(
            "Create isolated DVCS pseudodata projects, train the posterior "
            "network, inspect scores, and compare posterior methods."
        ),
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--bridge",
        type=Path,
        default=None,
        help="advanced override for the exact native bridge executable",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser(
        "init", help="create a new isolated editable project"
    )
    initialize.add_argument("project")
    commands.add_parser("list", help="list isolated user projects")
    show = commands.add_parser("show", help="show project controls and paths")
    show.add_argument("project")
    for command, help_text in (
        ("doctor", "verify the exact-native and neural runtime"),
        ("generate", "generate native simulations and pseudodata"),
        ("train", "train or resume the neural posterior ensemble"),
        ("optimize", "tune neural hyperparameters on validation NLL"),
        ("evaluate", "run coverage and exact-predictive checks"),
        ("compare", "compare neural and conventional posteriors"),
        ("plot", "plot compatible saved results without rerunning inference"),
        (
            "holdout",
            "run fresh output-blind GK11/GK16/GK19/VGG99 validation",
        ),
        (
            "compare-real",
            "compare quarantined real observables with frozen-NPE predictions",
        ),
        ("run", "run generation, training, evaluation, comparison, and plots"),
    ):
        subparser = commands.add_parser(command, help=help_text)
        subparser.add_argument("project")
        if command != "doctor":
            subparser.add_argument(
                "--profile",
                choices=("quick", "validation"),
                default="quick",
            )
            if command != "plot":
                subparser.add_argument(
                    "--no-progress",
                    action="store_true",
                    help=(
                        "disable live stderr progress bars (bars are already "
                        "disabled when stderr is not an interactive terminal)"
                    ),
                )
        if command == "generate":
            subparser.add_argument(
                "--force-native",
                action="store_true",
                help="recompute exact simulations instead of exact cache hits",
            )
        if command == "optimize":
            subparser.add_argument(
                "--trials",
                type=int,
                default=None,
                help="new trials to add (default from experiment.json)",
            )
        if command == "compare-real":
            subparser.add_argument(
                "--samples",
                type=int,
                default=None,
                help=(
                    "exact posterior samples to reevaluate (default: the "
                    "profile exact-reevaluation count)"
                ),
            )
    return parser


def main() -> int:
    """Dispatch one contained user action and return fail-closed exit codes."""

    try:
        args = _parser().parse_args()
        repository = _repository_root(args.repository_root)
        if args.command == "init":
            result = initialize_project(repository, args.project)
        elif args.command == "list":
            root = _user_root(repository)
            result = {
                "projects": sorted(
                    path.name
                    for path in root.iterdir()
                    if path.is_dir() and not path.is_symlink()
                )
            }
        else:
            project = _project_path(
                repository, args.project, existing=True
            )
            configuration, experiment = prepare_engine(repository, project)
            if args.command == "show":
                result = {
                    "project": project.name,
                    "experiment": str(project / "experiment.json"),
                    "results": str(project / "results"),
                    "truth": experiment["truth"],
                    "profiles": sorted(experiment["profiles"]),
                    "synthetic_only": True,
                }
            else:
                bridge = _bridge_path(repository, args.bridge)
                if args.command == "doctor":
                    internal = doctor(
                        bridge=bridge,
                        configuration_path=configuration,
                    )
                    write_json(project / "results" / "doctor.json", internal)
                    result = _public_result(
                        "doctor", project, None, internal
                    )
                else:
                    profile = args.profile
                    workspace = project / "results" / profile
                    common = {
                        "configuration_path": configuration,
                        "workspace": workspace,
                        "profile": profile,
                        "show_progress": (
                            False
                            if getattr(args, "no_progress", False)
                            else None
                        ),
                    }
                    if args.command in {
                        "train",
                        "optimize",
                        "evaluate",
                        "compare",
                        "plot",
                        "compare-real",
                        "holdout",
                    }:
                        assert_result_contract(
                            configuration=configuration,
                            workspace=workspace,
                            profile=profile,
                            bridge=bridge,
                        )
                    if args.command == "generate":
                        internal = generate_pseudodata(
                            bridge=bridge,
                            force_native=args.force_native,
                            **common,
                        )
                    elif args.command == "train":
                        internal = train_model(**common)
                    elif args.command == "optimize":
                        internal = optimize_hyperparameters(
                            n_trials=args.trials, **common
                        )
                    elif args.command == "evaluate":
                        internal = evaluate_model(
                            bridge=bridge, **common
                        )
                    elif args.command == "compare":
                        internal = compare_posteriors(**common)
                    elif args.command == "plot":
                        internal = plot_saved_results(
                            configuration_path=configuration,
                            workspace=workspace,
                            profile=profile,
                        )
                    elif args.command == "compare-real":
                        internal = compare_real_observables(
                            bridge=bridge,
                            database_root=_database_root(repository),
                            sample_count=args.samples,
                            **common,
                        )
                    elif args.command == "holdout":
                        internal = evaluate_native_model_holdouts(
                            bridge=bridge,
                            **common,
                        )
                    else:
                        internal = run_workflow(bridge=bridge, **common)
                    result = _public_result(
                        args.command, project, profile, internal
                    )
        print(json.dumps(result, indent=4, sort_keys=True))
        return 1 if result.get("status") == "fail" else 0
    except (
        FileNotFoundError,
        FileExistsError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"dvcs-infer: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
