#!/usr/bin/env python3
"""Sample one SWIF2 job cgroup and its allocated GPUs with low overhead."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import threading
import time
from typing import Any


def read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def cgroup_v2() -> Path | None:
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            hierarchy, controllers, relative = line.split(":", 2)
            if hierarchy == "0" and controllers == "":
                path = Path("/sys/fs/cgroup") / relative.lstrip("/")
                return path if path.is_dir() else None
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def key_values(path: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            key, value = line.split()[:2]
            result[key] = int(value)
    except (FileNotFoundError, PermissionError, ValueError):
        return {}
    return result


def io_totals(path: Path) -> dict[str, int]:
    totals = {"read_bytes": 0, "write_bytes": 0, "read_ios": 0, "write_ios": 0}
    mapping = {"rbytes": "read_bytes", "wbytes": "write_bytes", "rios": "read_ios", "wios": "write_ios"}
    found = False
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            for item in line.split()[1:]:
                key, value = item.split("=", 1)
                if key in mapping:
                    totals[mapping[key]] += int(value)
                    found = True
    except (FileNotFoundError, PermissionError, ValueError):
        return {}
    return totals if found else {}


def cgroup_sample(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"version": None}
    cpu = key_values(path / "cpu.stat")
    result: dict[str, Any] = {
        "version": 2,
        "path": str(path),
        "memory_current_bytes": read_int(path / "memory.current"),
        "memory_peak_bytes": read_int(path / "memory.peak"),
        "cpu_usage_usec": cpu.get("usage_usec"),
        "cpu_user_usec": cpu.get("user_usec"),
        "cpu_system_usec": cpu.get("system_usec"),
        "io": io_totals(path / "io.stat"),
    }
    return result


GPU_FIELDS = (
    "index", "uuid", "name", "utilization.gpu", "utilization.memory",
    "memory.used", "memory.total", "power.draw",
)


def number(value: str) -> float | None:
    value = value.strip()
    if value in {"", "N/A", "[N/A]", "Not Supported"}:
        return None
    # ``nounits`` is requested from nvidia-smi, but older drivers and test
    # fixtures can still include a separated or attached unit.  Accept one
    # finite leading numeric token and reject trailing non-unit garbage.
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"(?:\s*(?:%|MiB|GiB|W))?",
        value,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    try:
        parsed = float(match.group(1))
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def node_family(hostname: str) -> str:
    match = re.match(r"(farm(?:19|23|25)|sciml[0-9]+)", hostname.lower())
    return match.group(1) if match else "other"


def gpu_sample(enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {"status": "not_requested", "devices": []}
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "unavailable", "devices": [], "error": "nvidia-smi not found"}
    query = ",".join(GPU_FIELDS)
    command = [executable]
    allocated = os.environ.get("SLURM_STEP_GPUS") or os.environ.get("SLURM_JOB_GPUS")
    if allocated:
        command.append(f"--id={allocated}")
    command.extend([f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    try:
        completed = subprocess.run(
            command,
            text=True, capture_output=True, timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "error", "devices": [], "error": str(exc)[-500:]}
    if completed.returncode != 0:
        return {
            "status": "error", "devices": [],
            "error": completed.stderr.strip()[-500:],
        }
    devices = []
    for row in csv.reader(completed.stdout.splitlines()):
        if len(row) != len(GPU_FIELDS):
            continue
        devices.append({
            "index": row[0].strip(), "uuid": row[1].strip(),
            "name": row[2].strip(),
            "gpu_utilization_percent": number(row[3]),
            "memory_utilization_percent": number(row[4]),
            "memory_used_mib": number(row[5]),
            "memory_total_mib": number(row[6]),
            "power_watts": number(row[7]),
        })
    return {"status": "ok" if devices else "empty", "devices": devices}


def sample(started: float, cgroup: Path | None, accelerator: str) -> dict[str, Any]:
    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epoch_seconds": time.time(),
        "elapsed_seconds": time.monotonic() - started,
        "cgroup": cgroup_sample(cgroup),
        "gpu": gpu_sample(accelerator == "cuda"),
    }


def delta(first: dict[str, Any], last: dict[str, Any], key: str) -> int | None:
    left, right = first.get(key), last.get(key)
    if not isinstance(left, int) or not isinstance(right, int):
        return None
    return max(0, right - left)


def slurm_memory_bytes(value: str | None) -> int | None:
    if not value:
        return None
    text = value.strip().upper()
    scale = 1024**2  # Slurm's unsuffixed memory environment values are MiB.
    if text[-1:] in {"K", "M", "G", "T"}:
        suffix, text = text[-1], text[:-1]
        scale = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[suffix]
    try:
        return int(text) * scale
    except ValueError:
        return None


def aggregate(
    samples: list[dict[str, Any]], args: argparse.Namespace, *, complete: bool = False,
) -> dict[str, Any]:
    first, last = samples[0], samples[-1]
    wall = max(0.0, float(last["elapsed_seconds"]) - float(first["elapsed_seconds"]))
    cgroups = [item["cgroup"] for item in samples]
    cpu_usec = delta(cgroups[0], cgroups[-1], "cpu_usage_usec")
    allocated_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", len(os.sched_getaffinity(0))))
    cpu_efficiency = None
    if cpu_usec is not None and wall > 0 and allocated_cpus > 0:
        cpu_efficiency = 100.0 * (cpu_usec / 1_000_000.0) / (wall * allocated_cpus)
    memory_current = [item.get("memory_current_bytes") for item in cgroups]
    memory_peak = [item.get("memory_peak_bytes") for item in cgroups]
    max_current = max((v for v in memory_current if isinstance(v, int)), default=None)
    max_peak = max((v for v in memory_peak if isinstance(v, int)), default=None)
    requested_memory = slurm_memory_bytes(os.environ.get("SLURM_MEM_PER_NODE"))
    if requested_memory is None:
        per_cpu = slurm_memory_bytes(os.environ.get("SLURM_MEM_PER_CPU"))
        requested_memory = per_cpu * allocated_cpus if per_cpu is not None else None
    observed_peak = max(
        (v for v in (max_current, max_peak) if isinstance(v, int)), default=None
    )
    memory_efficiency = None
    if observed_peak is not None and requested_memory:
        memory_efficiency = 100.0 * observed_peak / requested_memory
    io_first, io_last = cgroups[0].get("io", {}), cgroups[-1].get("io", {})

    gpu_records: dict[str, list[dict[str, Any]]] = {}
    gpu_errors = 0
    for item in samples:
        gpu = item["gpu"]
        if gpu.get("status") in {"error", "unavailable"}:
            gpu_errors += 1
        for device in gpu.get("devices", []):
            gpu_records.setdefault(device["uuid"], []).append(device)

    def values(records: list[dict[str, Any]], key: str) -> list[float]:
        return [float(item[key]) for item in records if isinstance(item.get(key), (int, float))]

    gpu_summary = []
    for uuid, records in sorted(gpu_records.items()):
        utilization = values(records, "gpu_utilization_percent")
        memory = values(records, "memory_used_mib")
        power = values(records, "power_watts")
        total = values(records, "memory_total_mib")
        gpu_summary.append({
            "uuid": uuid,
            "index": records[0]["index"],
            "name": records[0]["name"],
            "sample_count": len(records),
            "mean_gpu_utilization_percent": statistics.fmean(utilization) if utilization else None,
            "max_gpu_utilization_percent": max(utilization) if utilization else None,
            "max_memory_used_mib": max(memory) if memory else None,
            "memory_total_mib": max(total) if total else None,
            "mean_power_watts": statistics.fmean(power) if power else None,
            "max_power_watts": max(power) if power else None,
        })

    return {
        "schema_version": 1,
        "status": "complete" if complete else "running",
        "stage": args.stage,
        "workflow": args.workflow,
        "campaign": args.campaign,
        "accelerator": args.accelerator,
        "hostname": platform.node(),
        "node_family": node_family(platform.node()),
        "submission_time": os.environ.get("SWIF_JOB_SUBMISSION_TIME"),
        "start_time": os.environ.get("SWIF_JOB_START_TIME"),
        "queue_wait_seconds": None,
        "attempt_number": args.attempt_number,
        "retry_or_failure_reason": args.retry_or_failure_reason,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "swif_job_id": os.environ.get("SWIF_JOB_ID"),
        "swif_job_attempt_id": os.environ.get("SWIF_JOB_ATTEMPT_ID"),
        "sample_interval_seconds": args.interval,
        "sample_count": len(samples),
        "wall_seconds_sampled": wall,
        "throughput": {
            "shard_count": args.shard_count,
            "group_count": args.group_count,
            "shards_per_hour": (
                args.shard_count * 3600.0 / wall
                if args.shard_count is not None and wall > 0 else None
            ),
            "groups_per_hour": (
                args.group_count * 3600.0 / wall
                if args.group_count is not None and wall > 0 else None
            ),
        },
        "allocation": {
            "cpus": allocated_cpus,
            "affinity_cpus": len(os.sched_getaffinity(0)),
            "memory_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
            "memory_per_cpu": os.environ.get("SLURM_MEM_PER_CPU"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
        },
        "cpu": {
            "usage_seconds": cpu_usec / 1_000_000.0 if cpu_usec is not None else None,
            "efficiency_percent": cpu_efficiency,
        },
        "memory": {
            "requested_bytes": requested_memory,
            "max_current_bytes": max_current,
            "max_peak_bytes": max_peak,
            "efficiency_percent": memory_efficiency,
        },
        "io": {
            key: delta(io_first, io_last, key)
            for key in ("read_bytes", "write_bytes", "read_ios", "write_ios")
        },
        "gpu": {"query_error_samples": gpu_errors, "devices": gpu_summary},
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--samples", required=True)
    result.add_argument("--summary", required=True)
    result.add_argument("--stage", required=True)
    result.add_argument("--accelerator", choices=("cpu", "cuda"), required=True)
    result.add_argument("--interval", type=float, default=30.0)
    result.add_argument("--workflow")
    result.add_argument("--campaign")
    result.add_argument("--attempt-number", type=int)
    result.add_argument("--retry-or-failure-reason")
    result.add_argument("--shard-count", type=int)
    result.add_argument("--group-count", type=int)
    result.add_argument("--once", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if not math.isfinite(args.interval) or args.interval < 1.0:
        raise SystemExit("interval must be at least one second")
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    started = time.monotonic()
    cgroup = cgroup_v2()
    records: list[dict[str, Any]] = []
    samples_path = Path(args.samples)
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", encoding="utf-8", buffering=1) as stream:
        while True:
            record = sample(started, cgroup, args.accelerator)
            records.append(record)
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            atomic_json(Path(args.summary), aggregate(records, args))
            if args.once:
                break
            if stop.wait(args.interval):
                record = sample(started, cgroup, args.accelerator)
                records.append(record)
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                atomic_json(Path(args.summary), aggregate(records, args))
                break
    atomic_json(Path(args.summary), aggregate(records, args, complete=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
