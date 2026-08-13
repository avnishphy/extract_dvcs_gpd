#!/usr/bin/env python3
"""Conflict-safe import of the approved extract_dvcs_cff runtime surface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM = ROOT.parent / "extract_dvcs_cff"
APPROVED = ROOT / "provenance" / "upstream-runtime-files.txt"
STATE = ROOT / "provenance" / "upstream-import.json"
REJECT_PARTS = {".git", "__pycache__", ".pytest_cache", "build", "artifacts", "workspaces"}
REJECT_SUFFIXES = {".pyc", ".pyo"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def approved_files(upstream: Path) -> list[tuple[Path, Path]]:
    result: list[tuple[Path, Path]] = []
    for raw in APPROVED.read_text(encoding="utf-8").splitlines():
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        source = upstream / item
        if not source.exists():
            raise RuntimeError(f"approved upstream path is missing: {item}")
        candidates = [source] if source.is_file() else sorted(source.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(upstream)
            if REJECT_PARTS.intersection(relative.parts) or candidate.suffix in REJECT_SUFFIXES:
                if source.is_file():
                    raise RuntimeError(
                        f"generated/development path is explicitly approved: {relative}"
                    )
                continue
            result.append((candidate, ROOT / relative))
    destinations = [destination for _, destination in result]
    if len(destinations) != len(set(destinations)):
        raise RuntimeError("approved manifest expands to duplicate paths")
    return result


def git(upstream: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(upstream), *arguments], text=True
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="import an explicitly recorded working-tree snapshot",
    )
    args = parser.parse_args()
    upstream = args.upstream.resolve()
    revision = git(upstream, "rev-parse", "HEAD")
    dirty_paths = git(upstream, "status", "--porcelain", "--untracked-files=all").splitlines()
    if dirty_paths and not args.allow_dirty:
        raise RuntimeError(
            "upstream has uncommitted content; review it and rerun with --allow-dirty"
        )
    previous = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    previous_files = previous.get("files", {})
    entries: dict[str, dict[str, str]] = {}
    changes: list[str] = []
    conflicts: list[str] = []
    for source, destination in approved_files(upstream):
        relative = str(destination.relative_to(ROOT))
        source_hash = digest(source)
        old = previous_files.get(relative, {})
        current_hash = digest(destination) if destination.is_file() else None
        if current_hash != source_hash:
            changes.append(relative)
        if destination.exists() and old and current_hash != old.get("imported_sha256"):
            conflicts.append(relative)
        entries[relative] = {
            "upstream_sha256": source_hash,
            "imported_sha256": source_hash,
        }
    removed = sorted(set(previous_files) - set(entries))
    for relative in removed:
        destination = ROOT / relative
        if destination.exists() and digest(destination) != previous_files[relative]["imported_sha256"]:
            conflicts.append(relative)
    if conflicts:
        print("conflicts (local file changed since last import):", file=sys.stderr)
        for item in sorted(conflicts):
            print(f"  {item}", file=sys.stderr)
        return 3
    if args.check:
        print(json.dumps({"revision": revision, "changes": changes, "removed": removed}, indent=2))
        return 1 if changes or removed else 0
    for source, destination in approved_files(upstream):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for relative in removed:
        destination = ROOT / relative
        if destination.exists():
            destination.unlink()
    snapshot = hashlib.sha256()
    for relative, value in sorted(entries.items()):
        snapshot.update(f"{relative}\0{value['upstream_sha256']}\n".encode())
    state = {
        "schema_version": 1,
        "upstream_url": git(upstream, "remote", "get-url", "origin"),
        "upstream_commit": revision,
        "upstream_dirty": bool(dirty_paths),
        "upstream_dirty_status": dirty_paths,
        "approved_manifest": str(APPROVED.relative_to(ROOT)),
        "runtime_snapshot_sha256": snapshot.hexdigest(),
        "files": entries,
        "required_post_import_tests": [
            "./tests/run.sh static",
            "./tests/run.sh native",
            "./tests/run.sh quick",
        ],
    }
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"imported {len(entries)} approved files from {revision}")
    if dirty_paths:
        print(f"recorded {len(dirty_paths)} upstream dirty paths")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
