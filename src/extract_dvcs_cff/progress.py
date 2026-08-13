"""Terminal progress utilities for the public pseudodata workflow.

Progress is written exclusively to stderr.  This preserves stdout for the
stable JSON result emitted by ``dvcs-infer`` and lets shell pipelines parse the
result even while an interactive user sees live progress.  Non-interactive
runs disable bars automatically unless a test or caller explicitly enables
them.
"""

from __future__ import annotations

import sys
from typing import Iterable, TypeVar

from tqdm.auto import tqdm


T = TypeVar("T")


def progress_enabled(requested: bool | None) -> bool:
    """Resolve ``None`` to interactive stderr and preserve explicit choices."""

    return sys.stderr.isatty() if requested is None else bool(requested)


def progress_bar(
    *,
    total: int,
    description: str,
    enabled: bool | None,
    leave: bool = True,
    unit: str = "step",
) -> tqdm:
    """Create a consistently formatted progress bar on stderr."""

    return tqdm(
        total=total,
        desc=description,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
        disable=not progress_enabled(enabled),
        file=sys.stderr,
        mininterval=0.1,
        maxinterval=1.0,
        miniters=1,
    )


def progress_iter(
    values: Iterable[T],
    *,
    total: int,
    description: str,
    enabled: bool | None,
    unit: str,
    leave: bool = False,
) -> Iterable[T]:
    """Wrap an iterable without materializing it or changing its values."""

    return tqdm(
        values,
        total=total,
        desc=description,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
        disable=not progress_enabled(enabled),
        file=sys.stderr,
        mininterval=0.1,
        maxinterval=1.0,
        miniters=1,
    )
