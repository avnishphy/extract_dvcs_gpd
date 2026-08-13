"""Auditable accelerator selection for Stage 10 neural inference.

The exact PARTONS simulator remains a CPU native executable.  This module is
used only by PyTorch/sbi computations.  ``auto`` deliberately asks PyTorch,
rather than inspecting device files or guessing from ``nvidia-smi``: a GPU is
usable only when the installed Torch build can initialize CUDA.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any

import torch


@dataclass(frozen=True)
class DeviceResolution:
    """Resolved execution device and enough provenance to audit the choice."""

    requested: str
    resolved: str
    torch_version: str
    torch_cuda_build: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    fallback_reason: str | None
    local_rank: int
    world_size: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable provenance record."""

        return asdict(self)


def resolve_torch_device(requested: str = "auto") -> DeviceResolution:
    """Resolve ``auto``, ``cpu``, or ``cuda`` without silent fallback.

    ``auto`` is the user default and therefore uses CUDA whenever the current
    Torch runtime reports it as available.  ``cuda`` is an assertion: it raises
    when no CUDA runtime is usable.  This distinction prevents a requested GPU
    validation campaign from being mislabeled as a CPU result.
    """

    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("accelerator must be one of: auto, cpu, cuda")
    cuda_build = torch.version.cuda
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    if requested == "cuda" and not available:
        build_note = (
            "this Torch build has no CUDA runtime"
            if cuda_build is None
            else "Torch cannot initialize a CUDA device"
        )
        raise RuntimeError(f"accelerator=cuda requested, but {build_note}")

    use_cuda = available and requested != "cpu"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if local_rank < 0 or world_size < 1 or local_rank >= world_size:
        raise RuntimeError("invalid torchrun rank environment")
    if world_size > 1 and not use_cuda:
        raise RuntimeError("multi-process neural training requires CUDA")
    if use_cuda:
        if local_rank >= count:
            raise RuntimeError("LOCAL_RANK exceeds CUDA_VISIBLE_DEVICES")
        if world_size > 1:
            torch.cuda.set_device(local_rank)
    resolved = (
        (f"cuda:{local_rank}" if world_size > 1 else "cuda")
        if use_cuda else "cpu"
    )
    name = torch.cuda.get_device_name(local_rank) if use_cuda else None
    fallback = None
    if requested == "auto" and not use_cuda:
        fallback = (
            "installed Torch build is CPU-only"
            if cuda_build is None
            else "Torch reported no usable CUDA device"
        )
    return DeviceResolution(
        requested=requested,
        resolved=resolved,
        torch_version=str(torch.__version__),
        torch_cuda_build=cuda_build,
        cuda_available=available,
        cuda_device_count=count,
        cuda_device_name=name,
        fallback_reason=fallback,
        local_rank=local_rank,
        world_size=world_size,
    )


def initialize_distributed_training(resolution: DeviceResolution) -> tuple[int, int]:
    """Initialize one NCCL rank per visible GPU for ensemble sharding."""

    rank = int(os.environ.get("RANK", "0"))
    world_size = resolution.world_size
    if world_size == 1:
        return rank, world_size
    if resolution.resolved == "cpu":
        raise RuntimeError("distributed training cannot use CPU")
    if world_size > resolution.cuda_device_count:
        raise RuntimeError("torchrun requested more ranks than visible GPUs")
    if not torch.distributed.is_nccl_available():
        raise RuntimeError("NCCL is unavailable in this PyTorch runtime")
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size


def finish_distributed_training() -> None:
    """Synchronize and close an initialized training process group."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def cuda_runtime_metrics(resolution: DeviceResolution) -> dict[str, Any]:
    """Capture synchronized CUDA memory metrics, or an explicit CPU record."""

    if not resolution.resolved.startswith("cuda"):
        return {
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        }
    torch.cuda.synchronize()
    return {
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
