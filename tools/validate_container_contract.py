#!/usr/bin/env python3
"""Validate Docker/Apptainer/native-build pins against dependency authority."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def require(text: str, token: str, source: str) -> None:
    if token not in text:
        raise RuntimeError(f"{source} does not contain locked token {token!r}")


def main() -> int:
    lock = json.loads((ROOT / "provenance/dependencies.lock.json").read_text())
    docker = (ROOT / "containers/Dockerfile").read_text()
    apptainer = (ROOT / "containers/apptainer.def").read_text()
    native = (ROOT / "scripts/build-native-source.sh").read_text()
    system_packages = (ROOT / "scripts/install-system-packages.sh").read_text()
    pyproject = (ROOT / "pyproject.toml").read_text()
    base = lock["system_abi"]["base_index_digest"]
    snapshot = lock["system_abi"]["ubuntu_snapshot"]
    require(docker, base, "Dockerfile")
    require(apptainer, base, "apptainer.def")
    require(docker, snapshot, "Dockerfile")
    require(apptainer, snapshot, "apptainer.def")
    for token in (
        "DVCS_APT_OBJECT_RETRIES", "DVCS_APT_NETWORK_TIMEOUT", "apt_retry",
        "attempt * retry_delay_base", "DVCS_APT_RETRY_ATTEMPTS",
    ):
        require(system_packages, token, "install-system-packages.sh")
    require(docker, "dvcs-install-system-packages", "Dockerfile")
    require(apptainer, "install-system-packages.sh", "apptainer.def")
    for name in ("partons", "apfelxx", "elementary_utils", "numa"):
        commit = lock["native_sources"][name]["commit"]
        require(docker, commit, "Dockerfile")
        require(native, commit, "build-native-source.sh")
    for archive in ("gsl",):
        checksum = lock["system_abi"][archive]["sha256"]
        require(docker, checksum, "Dockerfile")
        require(native, checksum, "build-native-source.sh")
    lhapdf = lock["native_sources"]["lhapdf"]["sha256"]
    require(docker, lhapdf, "Dockerfile")
    require(native, lhapdf, "build-native-source.sh")
    for package, version in lock["python"].items():
        if package in {
            "cpu_wheel_index", "cuda_wheel_index", "cuda_runtime",
            "packaging_tools", "transitive_constraints", "python",
        } or not isinstance(version, str):
            continue
        normalized = "PyYAML" if package == "PyYAML" else package
        require(pyproject, f'"{normalized}=={version}"', "pyproject.toml")
    if "latest" in re.sub(r"#.*", "", docker).lower():
        raise RuntimeError("Dockerfile contains floating latest")
    required_tests = (
        "partons_bridge --self-test", "partons_bridge --capabilities",
        "tests.test_architecture_contracts", "dvcs-infer --help", "set -eu",
        "if test -w /cache/partons-logs", "native self-test deferred",
        "chmod 1777 /cache/partons-logs",
    )
    for token in required_tests:
        require(apptainer, token, "apptainer.def %test")
    report = {
        "schema_version": 1, "status": "consistent",
        "authority": "provenance/dependencies.lock.json",
        "base_image_digest": base, "ubuntu_snapshot": snapshot,
        "native_commits_checked": 4, "archive_checksums_checked": 2,
        "docker_apptainer_shared_base": True,
        "docker_apptainer_shared_apt_retry_policy": True,
        "apptainer_smoke_contract_complete": True,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
