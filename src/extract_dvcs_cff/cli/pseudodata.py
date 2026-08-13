"""Console entry point for the Stage 10 pseudodata usability release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extract_dvcs_cff.workflows.pseudodata import (
    compare_posteriors,
    doctor,
    evaluate_model,
    generate_pseudodata,
    run_workflow,
    train_model,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dvcs-pseudodata",
        description=(
            "Generate exact-native DVCS pseudodata, train the DeepSets/NPE "
            "posterior, report train/test scores, and compare with "
            "conventional Bayesian importance sampling."
        ),
    )
    parser.add_argument(
        "--bridge",
        type=Path,
        default=Path("build/stage10_native/cpp/partons_bridge/partons_bridge"),
        help="versioned native PARTONS bridge executable",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/inference/stage10_pseudodata_workflow.json"
        ),
        help="strict Stage 10 JSON configuration",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("runs/pseudodata_quickstart"),
        help="new or exactly matching restart workspace",
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "validation"),
        default="quick",
        help="quick user smoke run or frozen validation campaign",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("doctor", help="verify native and neural runtime")
    generate = subcommands.add_parser(
        "generate", help="generate exact-native simulations and pseudodata"
    )
    generate.add_argument(
        "--force-native",
        action="store_true",
        help="explicitly recompute instead of using an exact cache hit",
    )
    subcommands.add_parser("train", help="train/restart the NPE ensemble")
    subcommands.add_parser(
        "evaluate", help="report held-out, coverage, and predictive scores"
    )
    subcommands.add_parser(
        "compare", help="compare neural and conventional posteriors"
    )
    subcommands.add_parser(
        "run", help="execute the complete ordered workflow"
    )
    return parser


def main() -> int:
    """Dispatch one user command and emit its machine-readable summary."""

    args = _parser().parse_args()
    common = {
        "configuration_path": args.config,
        "workspace": args.workspace,
        "profile": args.profile,
    }
    if args.command == "doctor":
        result = doctor(bridge=args.bridge, configuration_path=args.config)
    elif args.command == "generate":
        result = generate_pseudodata(
            bridge=args.bridge,
            force_native=args.force_native,
            **common,
        )
    elif args.command == "train":
        result = train_model(**common)
    elif args.command == "evaluate":
        result = evaluate_model(bridge=args.bridge, **common)
    elif args.command == "compare":
        result = compare_posteriors(**common)
    else:
        result = run_workflow(bridge=args.bridge, **common)
    print(json.dumps(result, indent=4, sort_keys=True))
    if isinstance(result, dict) and result.get("passed") is False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
