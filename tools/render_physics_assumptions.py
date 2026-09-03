#!/usr/bin/env python3
"""Validate the canonical physics-assumption registry and render its table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STATUSES = {
    "enforced_by_construction",
    "implemented_in_native_backend",
    "included_in_likelihood",
    "included_as_nuisance",
    "imposed_as_prior",
    "imposed_as_soft_penalty",
    "diagnostic_only",
    "assumed",
    "omitted",
    "deferred",
    "unknown",
}
FIELDS = {
    "name",
    "configured_value",
    "source",
    "scope",
    "status",
    "enforcement_mechanism",
    "validation_test",
    "validity_domain",
    "corpus_identity_effect",
    "claim_implication",
}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def load_registry(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("assumption registry schema_version must be 1")
    assumptions = value.get("assumptions")
    if not isinstance(assumptions, list) or not assumptions:
        raise ValueError("assumption registry needs a nonempty assumptions list")
    names: set[str] = set()
    for index, assumption in enumerate(assumptions):
        if not isinstance(assumption, dict) or set(assumption) != FIELDS:
            raise ValueError(
                f"assumption {index} key mismatch: "
                f"missing={sorted(FIELDS - set(assumption))}, "
                f"unknown={sorted(set(assumption) - FIELDS)}"
            )
        name = assumption["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"assumption {index} needs a unique nonempty name")
        names.add(name)
        if assumption["status"] not in STATUSES:
            raise ValueError(f"assumption {name!r} has an invalid status")
        for field in FIELDS - {"configured_value"}:
            if not isinstance(assumption[field], str) or not assumption[field].strip():
                raise ValueError(f"assumption {name!r}.{field} must be nonempty")
    identity_payload = {
        key: item for key, item in value.items() if key != "registry_sha256"
    }
    observed = hashlib.sha256(canonical(identity_payload)).hexdigest()
    declared = value.get("registry_sha256")
    if declared not in {None, observed}:
        raise ValueError(
            f"registry_sha256 mismatch: declared={declared}, observed={observed}"
        )
    value["registry_sha256"] = observed
    return value


def text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def render(registry: dict[str, Any]) -> str:
    lines = [
        "# Production native assumptions",
        "",
        "This table is generated from `provenance/production-native-assumptions.json`.",
        "Do not edit it by hand. Unknown, omitted, and deferred entries are claim",
        "boundaries, not zero-valued physics effects.",
        "",
        f"- Campaign: `{registry['campaign_id']}`",
        f"- Registry SHA-256: `{registry['registry_sha256']}`",
        f"- Registry status: `{registry['registry_status']}`",
        f"- Intended claim: {registry['intended_claim_scope']}",
        "",
        "| Assumption | Configured value | Status | Identity effect | Claim implication |",
        "|---|---|---|---|---|",
    ]
    for item in registry["assumptions"]:
        cells = [
            item["name"],
            text(item["configured_value"]),
            item["status"],
            item["corpus_identity_effect"],
            item["claim_implication"],
        ]
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |")
    lines.extend(("", "## Sources and enforcement", ""))
    for item in registry["assumptions"]:
        lines.extend(
            (
                f"### `{item['name']}`",
                "",
                f"- Source: {item['source']}",
                f"- Scope: {item['scope']}",
                f"- Enforcement: {item['enforcement_mechanism']}",
                f"- Validation: {item['validation_test']}",
                f"- Validity domain: {item['validity_domain']}",
                "",
            )
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("registry", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    registry = load_registry(args.registry)
    rendered = render(registry)
    if args.output is None:
        print(rendered)
    elif args.check:
        if args.output.read_text(encoding="utf-8") != rendered + "\n":
            raise RuntimeError(f"generated assumption table is stale: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "assumption_count": len(registry["assumptions"]),
        "registry_sha256": registry["registry_sha256"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
