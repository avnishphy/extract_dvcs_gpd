#!/usr/bin/env python3
"""Tiny real-SBI smoke for CPU-resident data and minibatch device transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from extract_dvcs_cff.inference.stage10 import (
    STAGE10_GLOBAL_FEATURE_NAMES,
    STAGE10_POINT_FEATURE_NAMES,
)
from extract_dvcs_cff.workflows.pseudodata import sha256, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    root = Path(tempfile.mkdtemp(prefix="dvcs-sbi-minibatch-smoke-"))
    config = json.loads(
        (repository / "configs/inference/stage10_pseudodata_workflow.json")
        .read_text(encoding="utf-8")
    )
    config["runtime"].update(accelerator=args.accelerator, cpu_threads=1)
    config["profiles"]["quick"].update(
        native_parameter_count=6,
        noise_replicates_per_parameter=2,
        max_num_epochs=1,
        stop_after_epochs=1,
        ensemble_seeds=[123],
        active_ensemble_member_count=1,
    )
    config["network"].update(
        point_hidden=8,
        point_layers=1,
        dataset_hidden=8,
        dataset_layers=1,
        embedding_features=8,
        flow_hidden_features=8,
        num_transforms=2,
        num_bins=4,
        training_batch_size=3,
    )
    configuration = root / "config.json"
    configuration.write_text(json.dumps(config), encoding="utf-8")
    generated = root / "generated"
    generated.mkdir()
    count = 12
    width = (
        len(config["kinematics"])
        * len(config["observables"])
        * len(STAGE10_POINT_FEATURE_NAMES)
        + len(STAGE10_GLOBAL_FEATURE_NAMES)
    )
    rng = np.random.default_rng(7)
    arrays = {
        "theta_latent": rng.normal(size=(count, 82)).astype("float32"),
        "contexts": rng.normal(size=(count, width)).astype("float32"),
        "train_indices": np.arange(0, 8, dtype="int64"),
        "validation_indices": np.arange(8, 10, dtype="int64"),
        "test_indices": np.arange(10, 12, dtype="int64"),
    }
    manifest: dict[str, object] = {"arrays": {}}
    records = manifest["arrays"]
    assert isinstance(records, dict)
    for name, value in arrays.items():
        path = generated / f"{name}.npy"
        np.save(path, value, allow_pickle=False)
        records[name] = {
            "sha256": sha256(path),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    (generated / "array_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    result = train_model(
        configuration_path=configuration,
        workspace=root,
        profile="quick",
        evaluate_test=False,
        show_progress=False,
    )
    assert result["candidate_ensemble_seeds"] == [123]
    print(f"CPU-resident SBI minibatch {args.accelerator} smoke: PASS")


if __name__ == "__main__":
    main()
