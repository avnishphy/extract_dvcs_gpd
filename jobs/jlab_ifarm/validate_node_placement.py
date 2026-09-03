#!/usr/bin/env python3
"""Discover stable JLab node-family features without guessing constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess


FAMILY = re.compile(r"(?:^|,)(farm(?:19|23|25))(?:,|$)")


def discover(slurm_config: Path) -> dict[str, list[str]]:
    families: dict[str, set[str]] = {name: set() for name in ("farm19", "farm23", "farm25")}
    source = str(slurm_config)
    text = slurm_config.read_text(encoding="utf-8") if slurm_config.is_file() else ""
    if not text:
        process = subprocess.run(
            ["scontrol", "show", "nodes", "--json"], text=True,
            capture_output=True, check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(
                "cannot resolve node-family features from Slurm controller or "
                f"{slurm_config}; run `scontrol show nodes -o | grep -E "
                "'AvailableFeatures=.*farm(19|23|25)'` on ifarm"
            )
        source = "scontrol show nodes --json"
        payload = json.loads(process.stdout)
        for node in payload.get("nodes", []):
            features = node.get("available_features", [])
            for family in families:
                if family in features:
                    families[family].add(str(node.get("name", "unknown")))
    else:
        current_host = None
        for line in text.splitlines():
            host_match = re.search(r"NodeName=([^ \\]+)", line)
            if host_match:
                current_host = host_match.group(1)
            feature_match = re.search(r"Feature=([^ \\]+)", line)
            if not feature_match or current_host is None:
                continue
            host, features = current_host, feature_match.group(1)
            for family in families:
                if family in features.split(","):
                    families[family].add(host)
    return {
        "schema_version": 1, "source": source,
        "families": {name: sorted(hosts) for name, hosts in families.items()},
        "constraints": {name: f"el9,{name}" for name in families},
        "production_policy": {
            "corpus_worker": "el9,farm25", "explicit_fallback": "el9,farm19",
            "excluded": "farm23", "worker_cpus": 16,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slurm-config", type=Path, default=Path("/etc/slurm/slurm.conf"))
    parser.add_argument("--require", choices=("farm19", "farm23", "farm25"), action="append", default=[])
    args = parser.parse_args()
    report = discover(args.slurm_config)
    missing = [name for name in args.require if not report["families"][name]]
    if missing:
        raise SystemExit(f"requested node families cannot be resolved: {missing}")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
