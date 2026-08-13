#!/usr/bin/env python3
"""Copy exact dependency license texts from clean evidence trees."""
from pathlib import Path
import argparse
import shutil

ROOT = Path(__file__).resolve().parents[1]
MAPPING = {
    "PARTONS-GPL-3.0.txt": "partons/LICENSE",
    "APFELXX-GPL-3.0.txt": "apfelxx/LICENSE",
    "ElementaryUtils-Apache-2.0.txt": "elementary-utils/LICENSE",
    "NumA-GPL-3.0.txt": "numa/LICENSE",
    "LHAPDF-GPL-3.0.txt": "LHAPDF-6.5.6/COPYING",
    "gpddatabase-GPL-3.0.txt": "hadronic_physics/gpddatabase/LICENSE",
}

parser = argparse.ArgumentParser()
parser.add_argument("--sources-root", type=Path, required=True)
args = parser.parse_args()
output = ROOT / "licenses"
output.mkdir(exist_ok=True)
for name, relative in MAPPING.items():
    source = args.sources_root / relative
    if not source.is_file():
        raise SystemExit(f"missing license evidence: {source}")
    shutil.copyfile(source, output / name)
print(f"copied {len(MAPPING)} dependency license texts")
