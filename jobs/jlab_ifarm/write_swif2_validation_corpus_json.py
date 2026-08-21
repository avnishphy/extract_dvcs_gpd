#!/usr/bin/env python3
"""Write the importable parallel SWIF2 validation-corpus workflow."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[2]
if len(sys.argv) > 2:
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} [OUTPUT.json]")
OUTPUT = (
    Path(sys.argv[1]).resolve()
    if len(sys.argv) == 2
    else Path(__file__).with_name(
        "swif-staged-partons-corpus-validation-16384-parallel.json"
    )
)
RESULT_ROOT = REPOSITORY / "results/swif_outputs/partons-corpus-validation-16384"
IMAGE_NAME = "extract-dvcs-gpd-jlab_ifarm-0.2.0-validation.sif"
IMAGE = REPOSITORY / f".dvcs/{IMAGE_NAME}"
DATABASE = REPOSITORY / ".dvcs/swif_staging/gpddatabase-v1.1.3.tar.gz"
EXPERIMENT = REPOSITORY / "experiment.json"
WORKER = REPOSITORY / "jobs/jlab_ifarm/run_staged_partons_corpus_chunk.sh"
MERGER = REPOSITORY / "jobs/jlab_ifarm/run_staged_partons_corpus_merge.sh"


def transfer(local: str, remote: Path) -> dict[str, str]:
    return {"local": local, "remote": str(remote)}


def worker_job(batch: int) -> dict[str, object]:
    name = f"partons-validation-batch-{batch:02d}"
    batch_root = RESULT_ROOT / f"batch-{batch:02d}"
    return {
        "name": name,
        "account": "hallc",
        "partition": "production",
        "constraint": "el9",
        "command": [f"/bin/bash run_staged_partons_corpus_chunk.sh {batch}"],
        "cpu_cores": 16,
        "ram_bytes": 32000000000,
        "disk_bytes": 32000000000,
        "disk_bytes_type": "scratch",
        "time_secs": 43200,
        "batch_flags": ["--nodes=1", "--ntasks=1", "--cpus-per-task=16"],
        "inputs": [
            transfer(IMAGE_NAME, IMAGE),
            transfer("gpddatabase-v1.1.3.tar.gz", DATABASE),
            transfer("experiment.json", EXPERIMENT),
            transfer("run_staged_partons_corpus_chunk.sh", WORKER),
        ],
        "outputs": [
            transfer(f"corpus-batch-{batch}.tar.gz", batch_root / f"corpus-batch-{batch}.tar.gz"),
            transfer(
                f"corpus-batch-{batch}-summary.json",
                batch_root / f"corpus-batch-{batch}-summary.json",
            ),
        ],
        "stdout": str(REPOSITORY / f"results/swif_stdout/{name}.out"),
        "stderr": str(REPOSITORY / f"results/swif_stderr/{name}.err"),
    }


def merge_job(workers: list[dict[str, object]]) -> dict[str, object]:
    inputs = [
        transfer(IMAGE_NAME, IMAGE),
        transfer("run_staged_partons_corpus_merge.sh", MERGER),
    ]
    for batch in range(1, 33):
        inputs.append(transfer(
            f"corpus-batch-{batch}.tar.gz",
            RESULT_ROOT / f"batch-{batch:02d}/corpus-batch-{batch}.tar.gz",
        ))
    return {
        "name": "partons-validation-merge",
        "antecedents": [str(job["name"]) for job in workers],
        "account": "hallc",
        "partition": "production",
        "constraint": "el9",
        "command": ["/bin/bash run_staged_partons_corpus_merge.sh"],
        "cpu_cores": 4,
        "ram_bytes": 16000000000,
        "disk_bytes": 32000000000,
        "disk_bytes_type": "scratch",
        "time_secs": 43200,
        "batch_flags": ["--nodes=1", "--ntasks=1", "--cpus-per-task=4"],
        "inputs": inputs,
        "outputs": [
            transfer(
                "partons-corpus-validation-16384.tar.gz",
                RESULT_ROOT / "final/partons-corpus-validation-16384.tar.gz",
            ),
            transfer(
                "partons-corpus-validation-16384-summary.json",
                RESULT_ROOT / "final/partons-corpus-validation-16384-summary.json",
            ),
        ],
        "stdout": str(REPOSITORY / "results/swif_stdout/partons-validation-merge.out"),
        "stderr": str(REPOSITORY / "results/swif_stderr/partons-validation-merge.err"),
    }


workers = [worker_job(batch) for batch in range(1, 33)]
workflow = {
    "name": "extract-dvcs-gpd-partons-validation-16384-parallel",
    "jobs": [*workers, merge_job(workers)],
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
print(OUTPUT)
