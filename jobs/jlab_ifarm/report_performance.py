#!/usr/bin/env python3
"""Aggregate reaped workflow telemetry without inventing missing fields."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any, Iterable


def node_family(host: str | None) -> str | None:
    if not host:
        return None
    match = re.match(r"(farm(?:19|23|25)|sciml[0-9]+)", host.lower())
    return match.group(1) if match else "other"


def records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    result = []
    for root in paths:
        candidates = [root] if root.is_file() else sorted(root.rglob("*.json"))
        for path in candidates:
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(value, dict) or "stage" not in value:
                continue
            item = dict(value)
            item["source_path"] = str(path)
            item["node_family"] = item.get("node_family") or node_family(item.get("hostname"))
            item.setdefault("workflow", None)
            item.setdefault("campaign", None)
            item.setdefault("submission_time", None)
            item.setdefault("start_time", None)
            item.setdefault("queue_wait_seconds", None)
            item.setdefault("attempt_number", None)
            item.setdefault("retry_or_failure_reason", None)
            result.append(item)
    return result


def finite_values(items: list[dict[str, Any]], path: tuple[str, ...]) -> list[float]:
    values = []
    for item in items:
        value: Any = item
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def aggregate(items: list[dict[str, Any]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[tuple(item.get(field) for field in fields)].append(item)
    output = []
    for key, group in sorted(groups.items(), key=lambda pair: str(pair[0])):
        runtime = finite_values(group, ("wall_seconds_sampled",))
        cpu = finite_values(group, ("cpu", "efficiency_percent"))
        rss = finite_values(group, ("memory", "max_peak_bytes"))
        queue = finite_values(group, ("queue_wait_seconds",))
        output.append({
            **dict(zip(fields, key)), "record_count": len(group),
            "runtime_seconds_mean": statistics.fmean(runtime) if runtime else None,
            "runtime_seconds_min": min(runtime) if runtime else None,
            "runtime_seconds_max": max(runtime) if runtime else None,
            "queue_wait_seconds_mean": statistics.fmean(queue) if queue else None,
            "cpu_efficiency_percent_mean": statistics.fmean(cpu) if cpu else None,
            "maximum_rss_bytes": max(rss) if rss else None,
            "missing_queue_wait_count": sum(item["queue_wait_seconds"] is None for item in group),
        })
    return output


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-supplied-corpus-aggregates", action="store_true")
    args = parser.parse_args()
    items = records(args.paths)
    report = {
        "schema_version": 1, "record_count": len(items),
        "grouped_by_workflow_stage_family_host_campaign_status": aggregate(
            items, ("workflow", "stage", "node_family", "hostname", "campaign", "status")
        ),
        "supplied_corpus_worker_aggregates": (
            {
                "farm25": {"completed_jobs": 8, "mean_hours": 8.39, "range_hours": [7.96, 9.17]},
                "farm19": {"completed_jobs": 9, "mean_hours": 12.74, "range_hours": [11.34, 13.81]},
                "farm23": {"completed_jobs": 13, "mean_hours": 17.89, "range_hours": [14.74, 19.97]},
                "provenance": "user-supplied aggregates; no per-job rows synthesized",
            } if args.include_supplied_corpus_aggregates else None
        ),
    }
    if args.output:
        atomic_write(args.output, report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
