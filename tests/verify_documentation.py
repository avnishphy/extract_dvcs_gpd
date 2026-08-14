#!/usr/bin/env python3
"""Verify that public documentation is complete, linked, and code-aligned."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extract_dvcs_cff.cli.user import _generated_readme, _public_experiment
from extract_dvcs_cff.workflows.pseudodata import load_configuration


REQUIRED = {
    "INDEX.md",
    "INSTALLATION.md",
    "USER_GUIDE.md",
    "CORPUS_AND_DATA_SELECTION.md",
    "CLI_REFERENCE.md",
    "EXPERIMENT_JSON_REFERENCE.md",
    "WORKFLOW_AND_PHYSICS.md",
    "RESULTS_AND_INTERPRETATION.md",
    "ARCHITECTURE.md",
    "NATIVE_BRIDGE.md",
    "DATA_CONTRACTS.md",
    "RESOURCE_MANAGEMENT.md",
    "CONTAINERS.md",
    "JLAB_IFARM.md",
    "DEPENDENCIES.md",
    "REPRODUCIBILITY.md",
    "SECURITY_AND_DATA.md",
    "TROUBLESHOOTING.md",
    "KNOWN_LIMITATIONS.md",
    "ACCEPTANCE.md",
    "UPDATING_FROM_EXTRACT_DVCS_CFF.md",
    "REFERENCES.md",
}


def leaf_paths(value: object, path: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else key
            yield from leaf_paths(item, child)
    elif isinstance(value, list):
        if value:
            yield from leaf_paths(value[0], f"{path}[]")
        else:
            yield f"{path}[]"
    else:
        yield path


def normalize_repeated(path: str) -> str:
    path = re.sub(
        r"^injected_truth\.gpd_parameters\.(H|E|Htilde|Etilde)\."
        r"(u|d|s|gluon)\.",
        "injected_truth.gpd_parameters.{GPD}.{channel}.",
        path,
    )
    path = re.sub(
        r"^injected_truth\.gpd_parameters\.shadow_coefficients\."
        r"(H|E|Htilde|Etilde)$",
        "injected_truth.gpd_parameters.shadow_coefficients.{GPD}",
        path,
    )
    path = re.sub(
        r"^injected_truth\.gpd_parameters\.shadow_channel_amplitudes\."
        r"(H|E|Htilde|Etilde)\.(u|d|s|gluon)$",
        "injected_truth.gpd_parameters.shadow_channel_amplitudes."
        "{GPD}.{channel}",
        path,
    )
    path = re.sub(
        r"^inference\.profiles\.(quick|validation)\.",
        "inference.profiles.{profile}.",
        path,
    )
    return re.sub(
        r"^inference\.hyperparameter_optimization\.trials\."
        r"(quick|validation)$",
        "inference.hyperparameter_optimization.trials.{profile}",
        path,
    )


docs = ROOT / "docs"
observed = {path.name for path in docs.glob("*.md")}
assert not REQUIRED - observed, sorted(REQUIRED - observed)
for name in REQUIRED:
    text = (docs / name).read_text(encoding="utf-8")
    assert text.startswith("# "), name
    assert len(text.split()) >= 100, f"documentation page is too sparse: {name}"

markdown_files = [ROOT / "README.md", *docs.glob("*.md"), *(ROOT / "user").glob("*.md")]
link_pattern = re.compile(r"\[[^]]*\]\(([^)]+)\)")
for source in markdown_files:
    for raw in link_pattern.findall(source.read_text(encoding="utf-8")):
        target = raw.strip().split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        resolved = (source.parent / target).resolve()
        assert resolved.exists(), f"broken local link: {source.relative_to(ROOT)} -> {raw}"

config = load_configuration(ROOT / "configs/inference/stage10_pseudodata_workflow.json")
config["_physics"] = json.loads(
    (ROOT / "configs/physics/stage10_pseudodata_systematics_v1.json").read_text()
)
source = {
    "mode": "gpddatabase_native_scale_kinematics_v2",
    "database_root": "/protected/gpddatabase",
    "database_revision": "revision",
    "catalog_sha256": "0" * 64,
    "selection_metric": "deterministic_metric",
    "selected_records": [{
        "record_id": "record:0", "dataset_id": "dataset:0",
        "source_uuid": "uuid", "source_path": "source.yaml",
        "collaboration": "example", "reference": "reference",
        "source_Q2_GeV2": 1.2, "source_beam_energy_GeV": 5.75,
        "generated_Q2_GeV2": 1.2, "Q2_projected": False,
        "generated_beam_energy_GeV": 5.75,
        "measurement_values_used": False,
    }],
    "measurements_quarantined": True,
    "data_evolved": False,
    "gpd_evolved_to_each_datum_Q2": True,
}
template = _public_experiment("documentation-test", config, source)
reference = (docs / "EXPERIMENT_JSON_REFERENCE.md").read_text(encoding="utf-8")
paths = {normalize_repeated(path) for path in leaf_paths(template)}
missing = sorted(path for path in paths if path not in reference)
assert not missing, f"undocumented experiment leaves: {missing}"
assert len(paths) >= 110

root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
for key in (
    "docs/INDEX.md",
    "docs/USER_GUIDE.md",
    "docs/CORPUS_AND_DATA_SELECTION.md",
    "docs/EXPERIMENT_JSON_REFERENCE.md",
):
    assert key in root_readme

project_readme = _generated_readme("documentation-test")
for key in (
    "docs/USER_GUIDE.md",
    "docs/CORPUS_AND_DATA_SELECTION.md",
    "docs/EXPERIMENT_JSON_REFERENCE.md",
    "docs/WORKFLOW_AND_PHYSICS.md",
    "docs/RESULTS_AND_INTERPRETATION.md",
):
    assert key in project_readme
assert "../" not in project_readme

print(f"documentation verification: PASS ({len(REQUIRED)} required pages, {len(paths)} schema leaves)")
