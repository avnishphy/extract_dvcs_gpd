#!/usr/bin/env python3
"""Verify the frozen presentation inputs; never update them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
manifest = json.loads((ROOT / "snapshot_manifest.json").read_text(encoding="utf-8"))
for record in manifest["assets"]:
    path = ROOT / record["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise SystemExit(f"snapshot hash mismatch: {record['path']}")
print(f"verified {len(manifest['assets'])} immutable presentation assets")
