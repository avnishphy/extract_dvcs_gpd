#!/usr/bin/env python3
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

SKIP_PARTS = {".git", ".dvcs", "build", "cache", "results", "workspace"}
for path in ROOT.rglob("*.json"):
    if SKIP_PARTS.intersection(path.relative_to(ROOT).parts):
        continue
    json.loads(path.read_text(encoding="utf-8"))
for path in ROOT.rglob("*.py"):
    if SKIP_PARTS.intersection(path.relative_to(ROOT).parts):
        continue
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

for forbidden in ("docs/plans", "docs/reports", "artifacts", "stage_status.yaml", "user/workspaces"):
    assert not (ROOT / forbidden).exists(), forbidden

matches = subprocess.run(
    ["rg", "-n", "/home/debian/" + "physics_analysis", str(ROOT),
     "--glob", "!provenance/upstream-import.json"],
    text=True, capture_output=True, check=False,
)
assert matches.returncode == 1, matches.stdout

state = json.loads((ROOT / "provenance/upstream-import.json").read_text())
assert state["upstream_commit"] == "7d690f69af60082d0bca8eb22a3a44144a98cd30"
assert state["upstream_dirty"] is False
assert len(state["files"]) >= 80
images = json.loads((ROOT / "provenance/images.lock.json").read_text())
assert images["published"] is False
assert all(item["digest"] is None for item in images["images"].values())

sys.path.insert(0, str(ROOT / "src"))
from extract_dvcs_cff.native_parallel import resolve_native_workers
with patch("extract_dvcs_cff.native_parallel.os.sched_getaffinity", return_value={2, 4, 6}):
    resolution = resolve_native_workers("all_available", task_count=8)
    assert resolution.resolved_workers == 3
    assert resolution.affinity_cpus == (2, 4, 6)
with patch("extract_dvcs_cff.native_parallel.os.sched_getaffinity", return_value={1, 3}):
    assert resolve_native_workers(8).resolved_workers == 2
with patch("extract_dvcs_cff.native_parallel.os.sched_getaffinity", return_value={0, 1, 2, 3}), patch.dict("os.environ", {"SLURM_CPUS_PER_TASK": "2"}):
    assert resolve_native_workers("all_available").resolved_workers == 2

lock = json.loads((ROOT / "provenance/dependencies.lock.json").read_text())
assert lock["native_sources"]["lhapdf"]["sha256"] == "6b8b7e38dc26a977a24f5a321215b7054c14a4469d04134d70cb93a860eeeea7"
project = tomllib.loads((ROOT / "pyproject.toml").read_text())
direct = project["project"]["dependencies"] + project["project"]["optional-dependencies"]["neural"]
direct_versions = dict(item.split("==", 1) for item in direct)
lock_names = {"PyYAML": "PyYAML", **{name: name for name in lock["python"] if isinstance(lock["python"][name], str)}}
for project_name, lock_name in lock_names.items():
    if project_name in direct_versions:
        assert direct_versions[project_name] == lock["python"][lock_name]
assert set(direct_versions) <= set(lock["python"])
assert direct_versions["torch"] == "2.12.1"

constraints = []
for line in (ROOT / "requirements/runtime-constraints.txt").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#"):
        assert line.count("==") == 1, line
        constraints.append(line.split("==", 1)[0].lower())
assert len(constraints) == len(set(constraints))
assert {"filelock", "fsspec", "nflows", "scikit-learn", "tensorboard"} <= set(constraints)
for recipe in (ROOT / "containers/Dockerfile", ROOT / "scripts/build-native-source.sh"):
    text = recipe.read_text()
    assert "torch==2.12.1" in text
    assert "runtime-constraints.txt" in text

audit = json.loads((ROOT / "provenance/dependency-audit-2026-08-13.json").read_text())
assert audit["python"]["torch"]["locked"] == "2.12.1"
assert audit["security"]["direct_pin_findings"][0]["advisory"] == "GHSA-rrmf-rvhw-rf47"
print("distribution static verification: PASS")
