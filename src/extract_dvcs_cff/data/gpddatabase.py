"""Read-only, measurement-quarantined access to the local GPD database.

The connector parses the protected DVCS YAML files directly.  It does not
import the database singleton, create caches, or retain measured values and
uncertainties in the kinematic catalog used for synthetic generation.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import yaml


VERIFIED_GPDDATABASE_REVISION = (
    "1e9e97fd417ce1d6d44fc73550bdbf32cca4eeb1"
)

# PARTONS defines the proton mass with this value in
# FundamentalPhysicalConstants.h.  The fixed-target DVCS process then uses
# y = Q2 / (2 M_p E x_B).  Keeping the audited value here lets catalog
# selection reject impossible beam-energy/Q2 combinations before launching
# the expensive native simulator.
PARTONS_PROTON_MASS_GEV = 0.938272013

# This deliberately narrow allowlist is the only measurement-bearing path in
# the connector.  Expanding it requires a source-backed convention audit for
# each observable and dataset; catalog construction above remains strictly
# measurement-free.
REAL_COMPARISON_SOURCES = {
    "QfefWWW2": {
        "source_path": "CLAS_2023_2.yaml",
        "observable": "ALU",
        "observable_unit": "none",
        "beam_type": "e-",
        "hadron_type": "p",
        "beam_energy_GeV": 10.6,
    }
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _revision(root: Path) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    process = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "cannot identify gpddatabase revision: " + process.stderr.strip()
        )
    return process.stdout.strip()


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def fixed_target_inelasticity(
    *, x_b: float, Q2_GeV2: float, beam_energy_GeV: float
) -> float:
    """Return PARTONS' fixed-target inelasticity y for proton DVCS."""

    x_b = _finite_number(x_b, "x_b")
    q2 = _finite_number(Q2_GeV2, "Q2_GeV2")
    beam_energy = _finite_number(beam_energy_GeV, "beam_energy_GeV")
    if x_b <= 0.0 or q2 <= 0.0 or beam_energy <= 0.0:
        raise ValueError("x_b, Q2_GeV2, and beam_energy_GeV must be positive")
    return q2 / (2.0 * PARTONS_PROTON_MASS_GEV * beam_energy * x_b)


def exact_dvcs_t_limits_GeV2(
    *, x_b: float, Q2_GeV2: float
) -> tuple[float, float]:
    """Return exact finite-Q2 ``(t_backward, t_forward)`` DVCS limits."""

    x_b = _finite_number(x_b, "x_b")
    q2 = _finite_number(Q2_GeV2, "Q2_GeV2")
    if not 0.0 < x_b < 1.0:
        raise ValueError("x_b must lie in (0,1)")
    if q2 <= 0.0:
        raise ValueError("Q2_GeV2 must be positive")
    epsilon2 = 4.0 * PARTONS_PROTON_MASS_GEV**2 * x_b**2 / q2
    root = math.sqrt(1.0 + epsilon2)
    denominator = 4.0 * x_b * (1.0 - x_b) + epsilon2
    t_forward = -q2 * (
        2.0 * (1.0 - x_b) * (1.0 - root) + epsilon2
    ) / denominator
    t_backward = -q2 * (
        2.0 * (1.0 - x_b) * (1.0 + root) + epsilon2
    ) / denominator
    return t_backward, t_forward


def require_physical_fixed_target_kinematics(
    *,
    x_b: float,
    Q2_GeV2: float,
    beam_energy_GeV: float,
    context: str,
    t_GeV2: float | None = None,
    phi_rad: float | None = None,
) -> float:
    """Validate exact physical fixed-target DVCS kinematics."""

    x_b = _finite_number(x_b, f"{context}.x_b")
    q2 = _finite_number(Q2_GeV2, f"{context}.Q2_GeV2")
    beam_energy = _finite_number(
        beam_energy_GeV, f"{context}.beam_energy_GeV"
    )
    if not 0.0 < x_b < 1.0:
        raise ValueError(f"{context}.x_b must lie in (0,1)")
    y = fixed_target_inelasticity(
        x_b=x_b,
        Q2_GeV2=q2,
        beam_energy_GeV=beam_energy,
    )
    if not 0.0 < y < 1.0:
        raise ValueError(
            f"{context} is not physical fixed-target DVCS: "
            f"y=Q2/(2*M_p*E*x_b)={y:.12g} must lie in (0,1)"
        )
    if t_GeV2 is not None:
        t_value = _finite_number(t_GeV2, f"{context}.t_GeV2")
        t_backward, t_forward = exact_dvcs_t_limits_GeV2(
            x_b=x_b, Q2_GeV2=q2
        )
        tolerance = 1e-12 * max(1.0, abs(t_backward), abs(t_forward))
        if t_value < t_backward - tolerance or t_value > t_forward + tolerance:
            raise ValueError(
                f"{context}.t_GeV2={t_value:.12g} is outside exact finite-Q2 "
                f"DVCS limits [{t_backward:.12g},{t_forward:.12g}]"
            )
    if phi_rad is not None:
        phi = _finite_number(phi_rad, f"{context}.phi_rad")
        if not 0.0 <= phi <= 2.0 * math.pi:
            raise ValueError(f"{context}.phi_rad must lie in [0,2pi]")
    return y


@lru_cache(maxsize=4)
def build_kinematic_catalog(
    database_root: Path,
    *,
    expected_revision: str = VERIFIED_GPDDATABASE_REVISION,
) -> dict[str, Any]:
    """Validate DVCS YAML and return metadata/kinematics without measurements."""

    root = database_root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("gpddatabase root must be a real directory")
    revision = _revision(root)
    if revision != expected_revision:
        raise RuntimeError(
            f"gpddatabase revision {revision} does not match verified "
            f"revision {expected_revision}"
        )
    data_root = (root / "gpddatabase" / "data" / "DVCS").resolve(strict=True)
    if not data_root.is_relative_to(root):
        raise RuntimeError("DVCS data directory escapes gpddatabase root")
    paths = sorted(data_root.rglob("*.yaml"))
    if not paths:
        raise RuntimeError("gpddatabase contains no DVCS YAML files")

    datasets: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    uuids: set[str] = set()
    source_hashes: list[dict[str, str]] = []
    for path in paths:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(data_root) or not resolved.is_file():
            raise RuntimeError(f"database path escapes DVCS root: {path}")
        relative = str(resolved.relative_to(data_root))
        document = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"{relative}: YAML root must be a mapping")
        uuid = document.get("uuid")
        if not isinstance(uuid, str) or not uuid or uuid in uuids:
            raise ValueError(f"{relative}: missing or duplicate UUID")
        uuids.add(uuid)
        general = document.get("general_info")
        if not isinstance(general, dict) or general.get("data_type") != "DVCS":
            raise ValueError(f"{relative}: expected DVCS general_info")
        conditions = general.get("conditions", {})
        if not isinstance(conditions, dict):
            raise ValueError(f"{relative}: conditions must be a mapping")
        beam_energy = conditions.get("lepton_beam_energy")
        if beam_energy is not None:
            beam_energy = _finite_number(
                beam_energy, f"{relative}: lepton_beam_energy"
            )
        source_hashes.append({"path": relative, "sha256": _sha256(resolved)})
        wrappers = document.get("data")
        if not isinstance(wrappers, list):
            raise ValueError(f"{relative}: data must be a list")
        for dataset_index, wrapper in enumerate(wrappers):
            if not isinstance(wrapper, dict) or set(wrapper) != {"data_set"}:
                raise ValueError(f"{relative}: invalid data_set wrapper")
            dataset = wrapper["data_set"]
            kinematics = dataset.get("kinematics", {})
            observable = dataset.get("observable", {})
            names = kinematics.get("name")
            units = kinematics.get("unit")
            rows = kinematics.get("value")
            observable_names = observable.get("name")
            observable_units = observable.get("unit")
            observable_values = observable.get("value")
            if not all(
                isinstance(item, list)
                for item in (
                    names,
                    units,
                    rows,
                    observable_names,
                    observable_units,
                    observable_values,
                )
            ):
                raise ValueError(f"{relative}: malformed dataset arrays")
            if len(names) != len(units) or len(observable_names) != len(
                observable_units
            ):
                raise ValueError(f"{relative}: name/unit length mismatch")
            if len(rows) != len(observable_values):
                raise ValueError(
                    f"{relative}: kinematic/observable row count mismatch"
                )
            dataset_id = f"{uuid}:{dataset_index}"
            datasets.append(
                {
                    "dataset_id": dataset_id,
                    "source_uuid": uuid,
                    "source_path": relative,
                    "dataset_index": dataset_index,
                    "label": str(dataset.get("label", "")),
                    "collaboration": str(general.get("collaboration", "")),
                    "reference": str(general.get("reference", "")),
                    "beam_energy_GeV": beam_energy,
                    "lepton_beam_type": conditions.get("lepton_beam_type"),
                    "hadron_beam_type": conditions.get("hadron_beam_type"),
                    "kinematic_names": list(names),
                    "kinematic_units": list(units),
                    "observable_names": list(observable_names),
                    "observable_units": list(observable_units),
                    "uncertainty_components_present": {
                        "statistical": "stat_unc" in observable,
                        "systematic": "sys_unc" in observable,
                        "normalization": "norm_unc" in observable,
                    },
                    "covariance_field_present": "covariance" in observable,
                    "row_count": len(rows),
                }
            )
            for row_index, row in enumerate(rows):
                if not isinstance(row, list) or len(row) != len(names):
                    raise ValueError(f"{relative}: malformed kinematic row")
                # Validate the corresponding measurement shape, then discard
                # it.  No measured value or uncertainty enters this catalog.
                measured = observable_values[row_index]
                if not isinstance(measured, list) or len(measured) != len(
                    observable_names
                ):
                    raise ValueError(f"{relative}: malformed observable row")
                values = {
                    str(name): _finite_number(
                        value, f"{relative}: row {row_index} {name}"
                    )
                    for name, value in zip(names, row, strict=True)
                }
                records.append(
                    {
                        "record_id": f"{dataset_id}:{row_index}",
                        "dataset_id": dataset_id,
                        "row_index": row_index,
                        "kinematics": values,
                        "kinematic_units": dict(
                            zip(names, units, strict=True)
                        ),
                    }
                )
    catalog = {
        "schema_version": 1,
        "source": "gpddatabase_read_only_DVCS_YAML",
        "database_root": str(root),
        "database_revision": revision,
        "source_file_count": len(paths),
        "dataset_count": len(datasets),
        "kinematic_record_count": len(records),
        "measurement_values_included": False,
        "uncertainties_included": False,
        "source_files": source_hashes,
        "datasets": datasets,
        "kinematic_records": records,
    }
    encoded = json.dumps(
        catalog, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    catalog["catalog_sha256"] = hashlib.sha256(encoded).hexdigest()
    return catalog


def select_projected_fixed_scale_points(
    catalog: Mapping[str, Any],
    anchors: Sequence[Mapping[str, float]],
    *,
    fixed_Q2_GeV2: float,
    fixed_beam_energy_GeV: float,
    x_b_bounds: tuple[float, float] = (0.12, 0.28),
    t_GeV2_bounds: tuple[float, float] = (-0.2, -0.08),
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Select real xB/t/phi points and explicitly project Q2/beam energy.

    This is a synthetic fixed-scale compatibility mode, not use of the real
    measurements and not an exact reproduction of their full kinematics.
    """

    dataset_metadata = {
        item["dataset_id"]: item for item in catalog["datasets"]
    }
    candidates: list[tuple[Mapping[str, Any], float, float, float, float]] = []
    for record in catalog["kinematic_records"]:
        values = record["kinematics"]
        units = record["kinematic_units"]
        if not {"xB", "t", "Q2", "phi"}.issubset(values):
            continue
        if units["xB"] != "none" or units["t"] != "GeV2" or units[
            "Q2"
        ] != "GeV2":
            continue
        phi = float(values["phi"])
        if units["phi"] == "deg":
            phi = math.radians(phi)
        elif units["phi"] != "rad":
            continue
        x_b = float(values["xB"])
        t_value = float(values["t"])
        if not x_b_bounds[0] <= x_b <= x_b_bounds[1]:
            continue
        if not t_GeV2_bounds[0] <= t_value <= t_GeV2_bounds[1]:
            continue
        candidates.append(
            (
                record,
                x_b,
                t_value,
                float(values["Q2"]),
                phi % (2.0 * math.pi),
            )
        )
    if len(candidates) < len(anchors):
        raise RuntimeError("too few database points with xB/t/Q2/phi")

    selected: list[dict[str, float]] = []
    provenance: list[dict[str, Any]] = []
    used: set[str] = set()
    for anchor in anchors:
        def distance(item: tuple[Mapping[str, Any], float, float, float, float]) -> tuple[float, str]:
            record, x_b, t_value, q2, phi = item
            phi_delta = abs(phi - float(anchor["phi_rad"]))
            phi_delta = min(phi_delta, 2.0 * math.pi - phi_delta)
            score = (
                ((x_b - float(anchor["x_b"])) / 0.05) ** 2
                + ((t_value - float(anchor["t_GeV2"])) / 0.05) ** 2
                + (phi_delta / 0.5) ** 2
                + ((q2 - fixed_Q2_GeV2) / 2.0) ** 2
            )
            return score, str(record["record_id"])

        available = [
            item for item in candidates if item[0]["record_id"] not in used
        ]
        record, x_b, t_value, source_q2, phi = min(available, key=distance)
        source_dataset = dataset_metadata[record["dataset_id"]]
        used.add(str(record["record_id"]))
        selected.append(
            {
                "x_b": x_b,
                "t_GeV2": t_value,
                "Q2_GeV2": float(fixed_Q2_GeV2),
                "beam_energy_GeV": float(fixed_beam_energy_GeV),
                "phi_rad": phi,
            }
        )
        provenance.append(
            {
                "record_id": record["record_id"],
                "dataset_id": record["dataset_id"],
                "source_uuid": source_dataset["source_uuid"],
                "source_path": source_dataset["source_path"],
                "collaboration": source_dataset["collaboration"],
                "reference": source_dataset["reference"],
                "source_Q2_GeV2": source_q2,
                "source_beam_energy_GeV": source_dataset[
                    "beam_energy_GeV"
                ],
                "generated_Q2_GeV2": float(fixed_Q2_GeV2),
                "Q2_projected": not math.isclose(source_q2, fixed_Q2_GeV2),
                "generated_beam_energy_GeV": float(fixed_beam_energy_GeV),
                "measurement_values_used": False,
            }
        )
    return selected, {
        "mode": "gpddatabase_xB_t_phi_projected_fixed_scale",
        "catalog_sha256": catalog["catalog_sha256"],
        "database_revision": catalog["database_revision"],
        "selection_metric": (
            "nearest_without_replacement_scaled_xB_t_circular_phi_Q2"
        ),
        "selected_records": provenance,
        "measurements_quarantined": True,
    }


def select_native_scale_points(
    catalog: Mapping[str, Any],
    anchors: Sequence[Mapping[str, float]],
    *,
    q2_bounds_GeV2: tuple[float, float] = (1.0, 80.0),
    beam_energy_bounds_GeV: tuple[float, float] = (3.0, 200.0),
    x_b_bounds: tuple[float, float] = (0.12, 0.28),
    t_GeV2_bounds: tuple[float, float] = (-0.2, -0.08),
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    """Select measurement-free catalog coordinates and retain Q2/beam energy.

    Selection uses only metadata and kinematics.  No measured observable or
    uncertainty is exposed.  Unlike the retired compatibility selector, Q2
    is never projected: the input GPD will be evolved to each retained datum
    scale by the exact native simulator.
    """

    metadata = {item["dataset_id"]: item for item in catalog["datasets"]}
    candidates: list[tuple[Mapping[str, Any], float, float, float, float, float]] = []
    for record in catalog["kinematic_records"]:
        values = record["kinematics"]
        units = record["kinematic_units"]
        if not {"xB", "t", "Q2", "phi"}.issubset(values):
            continue
        dataset = metadata[record["dataset_id"]]
        beam = dataset["beam_energy_GeV"]
        if beam is None:
            continue
        if (
            units.get("xB") != "none"
            or units.get("t") != "GeV2"
            or units.get("Q2") != "GeV2"
        ):
            continue
        phi = float(values["phi"])
        if units.get("phi") == "deg":
            phi = math.radians(phi)
        elif units.get("phi") != "rad":
            continue
        x_b = float(values["xB"])
        t_value = float(values["t"])
        q2 = float(values["Q2"])
        beam = float(beam)
        if not (
            x_b_bounds[0] <= x_b <= x_b_bounds[1]
            and t_GeV2_bounds[0] <= t_value <= t_GeV2_bounds[1]
            and q2_bounds_GeV2[0] <= q2 <= q2_bounds_GeV2[1]
            and beam_energy_bounds_GeV[0] <= beam <= beam_energy_bounds_GeV[1]
        ):
            continue
        try:
            require_physical_fixed_target_kinematics(
                x_b=x_b,
                Q2_GeV2=q2,
                beam_energy_GeV=beam,
                context=str(record["record_id"]),
            )
        except ValueError:
            continue
        candidates.append((record, x_b, t_value, q2, phi % (2 * math.pi), beam))
    if len(candidates) < len(anchors):
        raise RuntimeError("too few catalog points in the native-scale domain")

    selected: list[dict[str, float]] = []
    provenance: list[dict[str, Any]] = []
    used: set[str] = set()
    for anchor in anchors:
        def distance(item: tuple[Mapping[str, Any], float, float, float, float, float]) -> tuple[float, str]:
            record, x_b, t_value, q2, phi, beam = item
            phi_delta = abs(phi - float(anchor["phi_rad"]))
            phi_delta = min(phi_delta, 2 * math.pi - phi_delta)
            score = (
                ((x_b - float(anchor["x_b"])) / 0.05) ** 2
                + ((t_value - float(anchor["t_GeV2"])) / 0.05) ** 2
                + (phi_delta / 0.5) ** 2
                + (math.log(q2 / float(anchor["Q2_GeV2"])) / 0.5) ** 2
                + (math.log(beam / float(anchor["beam_energy_GeV"])) / 0.5) ** 2
            )
            return score, str(record["record_id"])

        available = [item for item in candidates if item[0]["record_id"] not in used]
        record, x_b, t_value, q2, phi, beam = min(available, key=distance)
        dataset = metadata[record["dataset_id"]]
        used.add(str(record["record_id"]))
        selected.append({
            "x_b": x_b,
            "t_GeV2": t_value,
            "Q2_GeV2": q2,
            "beam_energy_GeV": beam,
            "phi_rad": phi,
        })
        provenance.append({
            "record_id": record["record_id"],
            "dataset_id": record["dataset_id"],
            "source_uuid": dataset["source_uuid"],
            "source_path": dataset["source_path"],
            "collaboration": dataset["collaboration"],
            "reference": dataset["reference"],
            "source_Q2_GeV2": q2,
            "source_beam_energy_GeV": beam,
            "generated_Q2_GeV2": q2,
            "Q2_projected": False,
            "generated_beam_energy_GeV": beam,
            "measurement_values_used": False,
        })
    return selected, {
        "mode": "gpddatabase_native_scale_kinematics_v2",
        "catalog_sha256": catalog["catalog_sha256"],
        "database_revision": catalog["database_revision"],
        "selection_metric": (
            "physical_y_nearest_without_replacement_log_Q2_beam_xB_t_phi_v2"
        ),
        "selected_records": provenance,
        "measurements_quarantined": True,
        "data_evolved": False,
        "gpd_evolved_to_each_datum_Q2": True,
    }


def build_real_data_mapping_readiness(
    catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Inventory source observables without opening a real-data likelihood.

    A direct-name mapping is a *candidate* until units, beam/target signs,
    azimuthal conventions, uncertainty decomposition, and covariance are
    audited dataset by dataset.  Fourier coefficients and virtual-photon
    quantities are deliberately not manufactured from pointwise PARTONS
    observables.
    """

    direct_candidates = {
        "CrossSectionUU": "DVCSCrossSectionUUMinus",
        "CrossSectionDifferenceLU": "DVCSCrossSectionDifferenceLUMinus",
        "Ac": "DVCSAc",
        "ALU": "DVCSAluMinus",
        "AUL": "DVCSAulMinus",
        "ALL": "DVCSAllMinus",
    }
    rows: list[dict[str, Any]] = []
    candidate_rows = 0
    for dataset in catalog["datasets"]:
        mappings = []
        for source_name, source_unit in zip(
            dataset["observable_names"],
            dataset["observable_units"],
            strict=True,
        ):
            native = direct_candidates.get(source_name)
            if native is not None:
                status = "candidate_requires_dataset_convention_audit"
                candidate_rows += int(dataset["row_count"])
            elif source_name == "TSlope":
                status = "unmapped_derived_t_slope_not_pointwise_observable"
            elif source_name == "CrossSectionUUVirtualPhotoProduction":
                status = "unmapped_virtual_photon_cross_section_convention"
            else:
                status = "unmapped_fourier_or_transverse_spin_component"
            mappings.append({
                "source_observable": source_name,
                "source_unit": source_unit,
                "native_candidate": native,
                "status": status,
            })
        rows.append({
            "dataset_id": dataset["dataset_id"],
            "source_path": dataset["source_path"],
            "source_uuid": dataset["source_uuid"],
            "row_count": dataset["row_count"],
            "beam_energy_GeV": dataset["beam_energy_GeV"],
            "kinematic_names": dataset["kinematic_names"],
            "kinematic_units": dataset["kinematic_units"],
            "uncertainty_components_present": dataset[
                "uncertainty_components_present"
            ],
            "covariance_field_present": dataset[
                "covariance_field_present"
            ],
            "observable_mappings": mappings,
        })
    coverage_values = {"x_b": [], "minus_t_GeV2": [], "Q2_GeV2": [], "phi_rad": []}
    for record in catalog["kinematic_records"]:
        values = record["kinematics"]
        units = record["kinematic_units"]
        if not {"xB", "t", "Q2", "phi"}.issubset(values):
            continue
        if units.get("xB") != "none" or units.get("t") != "GeV2" or units.get("Q2") != "GeV2":
            continue
        phi = float(values["phi"])
        if units.get("phi") == "deg":
            phi = math.radians(phi)
        elif units.get("phi") != "rad":
            continue
        x_b, t_value, q2 = float(values["xB"]), float(values["t"]), float(values["Q2"])
        if not (0 < x_b < 1 and t_value < 0 and 1 <= q2 <= 80):
            continue
        coverage_values["x_b"].append(x_b)
        coverage_values["minus_t_GeV2"].append(-t_value)
        coverage_values["Q2_GeV2"].append(q2)
        coverage_values["phi_rad"].append(phi % (2 * math.pi))

    def histogram(values: Sequence[float], low: float, high: float) -> dict[str, Any]:
        bins = 24
        width = (high - low) / bins
        counts = [0] * bins
        for value in values:
            index = min(bins - 1, max(0, int((value - low) / width)))
            counts[index] += 1
        return {
            "bin_edges": [low + width * index for index in range(bins + 1)],
            "counts": counts,
        }

    coverage = {
        "complete_record_count": len(coverage_values["x_b"]),
        "histograms": {
            "x_b": histogram(coverage_values["x_b"], 0.0, 1.0),
            "minus_t_GeV2": histogram(coverage_values["minus_t_GeV2"], 0.0, 2.0),
            "Q2_GeV2": histogram(coverage_values["Q2_GeV2"], 1.0, 80.0),
            "phi_rad": histogram(coverage_values["phi_rad"], 0.0, 2 * math.pi),
        },
    }
    return {
        "schema_version": 1,
        "status": "mapping_inventory_ready_real_likelihood_disabled",
        "catalog_sha256": catalog["catalog_sha256"],
        "database_revision": catalog["database_revision"],
        "dataset_count": len(rows),
        "candidate_mapping_row_assignments": candidate_rows,
        "data_evolved": False,
        "gpd_input_scale_squared_GeV2": 1.0,
        "gpd_evolved_to_each_source_Q2": True,
        "real_data_fit_enabled": False,
        "measurement_free_kinematic_coverage": coverage,
        "mandatory_before_fit": [
            "dataset_by_dataset_observable_definition_and_sign_audit",
            "differential_unit_conversion_benchmark",
            "systematic_and_normalization_nuisance_assignment",
            "published_covariance_or_explicit_missing-correlation_policy",
            "validated_full-independent-DD_synthetic_and_blind_native_campaign",
        ],
        "datasets": rows,
    }


def load_real_comparison_observations(
    database_root: Path,
    *,
    source_uuid: str = "QfefWWW2",
    q2_GeV2_bounds: tuple[float, float] = (1.0, 80.0),
    t_GeV2_bounds: tuple[float, float] = (-1.0, 0.0),
    expected_revision: str = VERIFIED_GPDDATABASE_REVISION,
) -> dict[str, Any]:
    """Load one structure-audited ALU slice without mutating the database.

    The returned measurements are for a *post-training diagnostic only*.
    Each source Q2 is retained exactly.  The input GPD is evaluated at Q0 and
    evolved by the native backend to that datum's Q2; experimental data are
    never evolved or projected.
    """

    if source_uuid not in REAL_COMPARISON_SOURCES:
        raise ValueError("source UUID is not audited for real comparison")
    if not 1.0 <= q2_GeV2_bounds[0] < q2_GeV2_bounds[1] <= 80.0:
        raise ValueError("Q2 comparison bounds must lie inside [1,80] GeV2")
    if not t_GeV2_bounds[0] < t_GeV2_bounds[1] <= 0.0:
        raise ValueError("invalid negative-t comparison bounds")

    root = database_root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("gpddatabase root must be a real directory")
    revision = _revision(root)
    if revision != expected_revision:
        raise RuntimeError(
            f"gpddatabase revision {revision} does not match verified "
            f"revision {expected_revision}"
        )
    contract = REAL_COMPARISON_SOURCES[source_uuid]
    path = (
        root / "gpddatabase" / "data" / "DVCS" / contract["source_path"]
    ).resolve(strict=True)
    data_root = (root / "gpddatabase" / "data" / "DVCS").resolve(strict=True)
    if not path.is_relative_to(data_root) or not path.is_file():
        raise RuntimeError("comparison source escapes the protected DVCS tree")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("uuid") != source_uuid:
        raise ValueError("comparison source UUID changed")
    general = document.get("general_info")
    if not isinstance(general, dict) or general.get("data_type") != "DVCS":
        raise ValueError("comparison source is not DVCS")
    conditions = general.get("conditions")
    expected_conditions = {
        "lepton_beam_type": contract["beam_type"],
        "lepton_beam_energy": contract["beam_energy_GeV"],
        "hadron_beam_type": contract["hadron_type"],
    }
    if not isinstance(conditions, dict) or any(
        conditions.get(key) != value
        for key, value in expected_conditions.items()
    ):
        raise ValueError("comparison source beam/target conditions changed")
    wrappers = document.get("data")
    if not isinstance(wrappers, list) or len(wrappers) != 1:
        raise ValueError("comparison source dataset structure changed")
    dataset = wrappers[0].get("data_set", {})
    kinematics = dataset.get("kinematics", {})
    observable = dataset.get("observable", {})
    if kinematics.get("name") != ["xB", "t", "Q2", "phi"] or kinematics.get(
        "unit"
    ) != ["none", "GeV2", "GeV2", "rad"]:
        raise ValueError("comparison kinematic convention changed")
    if observable.get("name") != [contract["observable"]] or observable.get(
        "unit"
    ) != [contract["observable_unit"]]:
        raise ValueError("comparison observable convention changed")
    rows = kinematics.get("value")
    values = observable.get("value")
    statistical = observable.get("stat_unc")
    systematic = observable.get("sys_unc")
    if not all(isinstance(item, list) for item in (rows, values, statistical)):
        raise ValueError("comparison measurement arrays are malformed")
    if systematic is not None:
        raise ValueError("systematic uncertainty appeared without an audit")
    if not (len(rows) == len(values) == len(statistical)):
        raise ValueError("comparison measurement row counts differ")

    accepted: list[dict[str, Any]] = []
    for row_index, (row, measured, stat_unc) in enumerate(
        zip(rows, values, statistical, strict=True)
    ):
        if not (
            isinstance(row, list)
            and len(row) == 4
            and isinstance(measured, list)
            and len(measured) == 1
            and isinstance(stat_unc, list)
            and len(stat_unc) == 1
        ):
            raise ValueError(f"malformed comparison row {row_index}")
        x_b, t_value, source_q2, phi = (
            _finite_number(value, f"comparison row {row_index}")
            for value in row
        )
        observed = _finite_number(measured[0], f"observable row {row_index}")
        sigma = _finite_number(stat_unc[0], f"stat_unc row {row_index}")
        if sigma <= 0.0:
            raise ValueError(f"non-positive statistical uncertainty at {row_index}")
        if not q2_GeV2_bounds[0] <= source_q2 <= q2_GeV2_bounds[1]:
            continue
        if not (0.0 < x_b < 1.0 and t_GeV2_bounds[0] <= t_value < t_GeV2_bounds[1]):
            continue
        if not 0.0 <= phi <= 2.0 * math.pi:
            raise ValueError(f"phi outside [0,2pi] at comparison row {row_index}")
        accepted.append(
            {
                "record_id": f"{source_uuid}:0:{row_index}",
                "row_index": row_index,
                "observable": contract["observable"],
                "observable_unit": contract["observable_unit"],
                "x_b": x_b,
                "t_GeV2": t_value,
                "source_Q2_GeV2": source_q2,
                "evaluation_Q2_GeV2": source_q2,
                "Q2_projection_GeV2": 0.0,
                "beam_energy_GeV": float(contract["beam_energy_GeV"]),
                "phi_rad": phi,
                "observed_value": observed,
                "statistical_uncertainty": sigma,
                "systematic_uncertainty": None,
            }
        )
    if not accepted:
        raise RuntimeError("no comparison observations satisfy the audited slice")
    # Keep this optional diagnostic bounded while retaining the source Q2
    # range. Selection uses kinematics only and is deterministic.
    accepted.sort(key=lambda item: (
        item["source_Q2_GeV2"], item["x_b"], item["t_GeV2"],
        item["phi_rad"], item["record_id"],
    ))
    if len(accepted) > 24:
        selected_indices = [
            round(index * (len(accepted) - 1) / 23) for index in range(24)
        ]
        accepted = [accepted[index] for index in selected_indices]
    return {
        "schema_version": 1,
        "role": "post_training_diagnostic_only",
        "source_uuid": source_uuid,
        "source_path": str(path.relative_to(data_root)),
        "source_sha256": _sha256(path),
        "database_revision": revision,
        "collaboration": general.get("collaboration"),
        "reference": general.get("reference"),
        "observable": contract["observable"],
        "observable_unit": contract["observable_unit"],
        "uncertainty_field": "gpddatabase.observable.stat_unc",
        "uncertainty_decomposition_audited": False,
        "source_Q2_retained": True,
        "data_evolved": False,
        "gpd_evolved_to_each_datum_Q2": True,
        "real_data_used_for_training": False,
        "posterior_updated": False,
        "selection": {
            "Q2_GeV2_bounds": list(q2_GeV2_bounds),
            "t_GeV2_bounds": list(t_GeV2_bounds),
            "maximum_point_count": 24,
            "downselection": "kinematic_Q2_ordered_quantiles_no_outputs",
        },
        "observations": accepted,
    }
