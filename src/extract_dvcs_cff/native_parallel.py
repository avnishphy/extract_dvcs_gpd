"""Affinity-aware policy for isolated native PARTONS subprocesses.

PARTONS modules are not assumed to be thread-safe.  In particular, the local
``GPDGK16`` header explicitly forbids threaded use because CLN is not
thread-safe.  This module therefore controls only the number of independent
bridge subprocesses.  It never shares a PARTONS module instance between
workers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any


@dataclass(frozen=True)
class NativeWorkerResolution:
    """Auditable resolution of a user worker request against CPU affinity."""

    requested: int | str
    affinity_cpus: tuple[int, ...]
    resolved_workers: int
    task_count: int | None
    isolation: str = "one_partons_bridge_subprocess_per_active_worker"

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["affinity_cpus"] = list(self.affinity_cpus)
        return value


def available_affinity_cpus() -> tuple[int, ...]:
    """Return interactive affinity or the active Slurm job allocation."""

    if hasattr(os, "sched_getaffinity"):
        cpus = tuple(sorted(os.sched_getaffinity(0)))
    else:  # pragma: no cover - Linux/WSL is the verified deployment target.
        cpus = tuple(range(os.cpu_count() or 1))
    if not cpus:
        raise RuntimeError("the process has no schedulable CPUs")
    slurm_limit = (
        os.environ.get("SLURM_CPUS_PER_TASK")
        if os.environ.get("SLURM_JOB_ID")
        else None
    )
    if slurm_limit is not None:
        try:
            allocated = int(slurm_limit)
        except ValueError as exc:
            raise RuntimeError("SLURM_CPUS_PER_TASK must be a positive integer") from exc
        if allocated < 1:
            raise RuntimeError("SLURM_CPUS_PER_TASK must be a positive integer")
        cpus = cpus[:allocated]
    return cpus


def resolve_native_workers(
    requested: int | str,
    *,
    task_count: int | None = None,
) -> NativeWorkerResolution:
    """Resolve ``all_available`` or an integer without exceeding affinity.

    ``task_count`` may reduce the active pool because creating idle processes
    cannot accelerate a smaller number of independent native batches.
    """

    cpus = available_affinity_cpus()
    if requested == "all_available":
        desired = len(cpus)
    elif isinstance(requested, int) and not isinstance(requested, bool):
        if not 1 <= requested <= 256:
            raise ValueError("runtime.native_workers must be in [1,256]")
        desired = requested
    else:
        raise ValueError(
            "runtime.native_workers must be an integer or 'all_available'"
        )
    if task_count is not None:
        if not isinstance(task_count, int) or isinstance(task_count, bool):
            raise ValueError("task_count must be an integer")
        if task_count < 0:
            raise ValueError("task_count cannot be negative")
        desired = min(desired, max(1, task_count))
    return NativeWorkerResolution(
        requested=requested,
        affinity_cpus=cpus,
        resolved_workers=min(desired, len(cpus)),
        task_count=task_count,
    )
