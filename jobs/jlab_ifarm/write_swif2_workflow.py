#!/usr/bin/env python3
"""Generate staged, importable JLab SWIF2 workflows.

All job inputs use content-derived local names.  SWIF2 may cache staged inputs
by local name, so stable generic names are unsafe after a file changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any


NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
ANALYSIS_ORDER = (
    "selection", "materialize", "optimize", "train", "evaluate",
    "compare", "holdout", "plot",
)
GPU_STAGES = {"optimize", "train", "evaluate"}
CORPUS_STAGES = {"selection", "materialize"}

ANALYSIS_DEFAULTS = {
    "selection": (1, "4G", "8G", "30min", 0, "production"),
    "materialize": (2, "20G", "48G", "4h", 0, "production"),
    "optimize": (8, "32G", "48G", "12h", 1, "gpu"),
    "train": (4, "32G", "48G", "12h", 1, "gpu"),
    "evaluate": (8, "16G", "48G", "12h", 1, "gpu"),
    "compare": (2, "8G", "32G", "4h", 0, "production"),
    "holdout": (16, "16G", "48G", "12h", 0, "production"),
    "plot": (2, "8G", "32G", "1h", 0, "production"),
}


def fail(message: str) -> None:
    raise SystemExit(message)


def checked_file(value: str, label: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        fail(f"{label} must be a real file: {path}")
    return path


def checked_name(value: str, label: str) -> str:
    if not NAME.fullmatch(value):
        fail(f"{label} must contain only letters, digits, dot, underscore, or hyphen")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def staged(path: Path, label: str) -> tuple[str, dict[str, str]]:
    suffixes = "".join(path.suffixes)
    local = f"{label}-{sha256(path)[:16]}{suffixes}"
    return local, {"local": local, "remote": str(path)}


def size_bytes(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([KMGT]?)B?", value.strip(), re.I)
    if not match:
        fail(f"invalid byte size: {value}")
    scale = {"": 1, "K": 1000, "M": 1000**2, "G": 1000**3, "T": 1000**4}
    return int(match.group(1)) * scale[match.group(2).upper()]


def time_seconds(value: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)(min|m|h|d|s)", value.strip(), re.I)
    if not match:
        fail(f"invalid duration: {value}; use integer s, min, h, or d")
    scale = {"s": 1, "m": 60, "min": 60, "h": 3600, "d": 86400}
    return int(match.group(1)) * scale[match.group(2).lower()]


def env_resource(stage: str, field: str, default: Any) -> Any:
    value = os.environ.get(f"SWIF_{stage.upper()}_{field}")
    return default if value in (None, "") else value


def resource(stage: str) -> dict[str, Any]:
    cores, ram, disk, duration, gpus, partition = ANALYSIS_DEFAULTS[stage]
    cores = int(env_resource(stage, "CORES", cores))
    gpus = int(env_resource(stage, "GPUS", gpus))
    if cores < 1 or gpus < 0:
        fail(f"invalid {stage} core/GPU request")
    if stage in GPU_STAGES and gpus < 1:
        fail(f"{stage} requires at least one GPU")
    return {
        "cpu_cores": cores,
        "ram_bytes": size_bytes(str(env_resource(stage, "RAM", ram))),
        "disk_bytes": size_bytes(str(env_resource(stage, "DISK", disk))),
        "time_secs": time_seconds(str(env_resource(stage, "TIME", duration))),
        "gpus": gpus,
        "partition": str(env_resource(stage, "PARTITION", partition)),
    }


def transfer(local: str, remote: Path, *, optional: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {"local": local, "remote": str(remote)}
    if optional:
        record["optional"] = True
    return record


def command(*values: str) -> list[str]:
    return [shlex.join(values)]


def job_base(
    *, name: str, account: str, constraint: str, resources: dict[str, Any],
    log_root: Path, workflow: str, stage: str,
) -> dict[str, Any]:
    cores = int(resources["cpu_cores"])
    flags = ["--nodes=1", "--ntasks=1", f"--cpus-per-task={cores}"]
    gpus = int(resources["gpus"])
    if gpus:
        # SWIF2 adds scratch as --gres=disk:...; another --gres can overwrite
        # rather than combine with it. Use Slurm's independent GPU TRES flag.
        flags.append(f"--gpus={gpus}")
    return {
        "name": name,
        "account": account,
        "partition": resources["partition"],
        "constraint": constraint,
        "cpu_cores": cores,
        "ram_bytes": int(resources["ram_bytes"]),
        "disk_bytes": int(resources["disk_bytes"]),
        "disk_bytes_type": "scratch",
        "time_secs": int(resources["time_secs"]),
        "batch_flags": flags,
        "tags": [
            {"name": "dvcs-workflow", "value": workflow},
            {"name": "dvcs-stage", "value": stage},
        ],
        "stdout": str(log_root / f"{name}.out"),
        "stderr": str(log_root / f"{name}.err"),
    }


def select_stages(args: argparse.Namespace) -> list[str]:
    order = list(ANALYSIS_ORDER)
    if not args.include_optimize:
        order.remove("optimize")
    if args.only:
        return [checked_name(args.only, "stage")] if args.only in order else fail(
            f"--only must be one of: {', '.join(order)}"
        )
    if args.from_stage not in order or args.through_stage not in order:
        fail(f"stage bounds must be in: {', '.join(order)}")
    start = order.index(args.from_stage)
    stop = order.index(args.through_stage)
    if start > stop:
        fail("--from occurs after --through")
    return order[start:stop + 1]


def analysis_workflow(args: argparse.Namespace) -> dict[str, Any]:
    workflow = checked_name(args.workflow, "workflow")
    project = checked_name(args.project, "project")
    corpus = checked_name(args.corpus, "corpus")
    selection = checked_name(args.selection, "selection")
    profile = checked_name(args.profile, "profile")
    image = checked_file(args.image, "image")
    database = checked_file(args.database, "database archive")
    corpus_archive = checked_file(args.corpus_archive, "corpus archive")
    initial_project = checked_file(args.project_archive, "project archive")
    runner = checked_file(args.runner, "analysis runner")
    collector = checked_file(args.collector, "performance collector")
    result_root = Path(args.result_root).expanduser().resolve()
    log_root = Path(args.log_root).expanduser().resolve()
    stages = select_stages(args)

    image_name, image_input = staged(image, "dvcs-image")
    database_name, database_input = staged(database, "gpddatabase")
    corpus_name, corpus_input = staged(corpus_archive, "corpus")
    runner_name, runner_input = staged(runner, "analysis-runner")
    collector_name, collector_input = staged(collector, "performance-collector")
    project_name, project_input = staged(initial_project, "project-state")
    prior_job: str | None = None
    jobs: list[dict[str, Any]] = []
    state_remote = initial_project
    state_local = project_name

    for index, stage in enumerate(stages, 1):
        name = f"dvcs-{stage}"
        resources = resource(stage)
        output_local = f"project-after-{stage}.tar"
        summary_local = f"summary-{stage}.json"
        performance_local = f"performance-{stage}.json"
        samples_local = f"performance-{stage}.jsonl"
        output_remote = result_root / "state" / f"{workflow}-{output_local}"
        summary_remote = result_root / "summaries" / f"{workflow}-{summary_local}"
        job = job_base(
            name=name, account=args.account, constraint=args.constraint,
            resources=resources, log_root=log_root, workflow=workflow, stage=stage,
        )
        if prior_job:
            job["antecedents"] = [prior_job]
        inputs = [image_input, database_input, runner_input, collector_input]
        if index == 1:
            inputs.append(project_input)
        else:
            inputs.append(transfer(state_local, state_remote))
        if stage in CORPUS_STAGES:
            inputs.append(corpus_input)
        job["inputs"] = inputs
        job["outputs"] = [
            transfer(output_local, output_remote),
            transfer(summary_local, summary_remote),
            transfer(
                performance_local,
                result_root / "performance" / f"{workflow}-{performance_local}",
            ),
            transfer(
                samples_local,
                result_root / "performance" / f"{workflow}-{samples_local}",
            ),
        ]
        accelerator = "cuda" if stage in GPU_STAGES else "cpu"
        job["command"] = command(
            "/bin/bash", runner_name, stage, project, profile, corpus,
            selection, image_name, database_name,
            corpus_name if stage in CORPUS_STAGES else "none",
            state_local, output_local, summary_local, accelerator,
            str(args.heartbeat_seconds),
            collector_name, performance_local, samples_local,
            str(args.performance_interval_seconds),
        )
        jobs.append(job)
        prior_job = name
        state_remote = output_remote
        state_local = output_local

    return {
        "name": workflow,
        "site_name": args.site,
        "max_problems": 1,
        "max_dispatched": max(1, int(args.max_dispatched)),
        "jobs": jobs,
    }


def corpus_workflow(args: argparse.Namespace) -> dict[str, Any]:
    workflow = checked_name(args.workflow, "workflow")
    project = checked_name(args.project, "project")
    corpus = checked_name(args.corpus, "corpus")
    profile = checked_name(args.profile, "profile")
    image = checked_file(args.image, "image")
    database = checked_file(args.database, "database archive")
    experiment = checked_file(args.experiment, "experiment")
    worker = checked_file(args.worker, "corpus worker")
    merger = checked_file(args.merger, "corpus merger")
    collector = checked_file(args.collector, "performance collector")
    config = json.loads(experiment.read_text(encoding="utf-8"))
    try:
        parameter_count = int(config["inference"]["profiles"][profile]["native_parameter_count"])
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"cannot read profile parameter count: {exc}")
    shard_size = int(args.shard_size)
    shards_per_worker = int(args.shards_per_worker)
    if parameter_count < 1 or shard_size < 1 or shards_per_worker < 1:
        fail("corpus counts must be positive")
    shard_count = math.ceil(parameter_count / shard_size)
    worker_count = math.ceil(shard_count / shards_per_worker)
    result_root = Path(args.result_root).expanduser().resolve()
    log_root = Path(args.log_root).expanduser().resolve()

    image_name, image_input = staged(image, "dvcs-image")
    database_name, database_input = staged(database, "gpddatabase")
    experiment_name, experiment_input = staged(experiment, "experiment")
    worker_name, worker_input = staged(worker, "corpus-worker")
    merger_name, merger_input = staged(merger, "corpus-merger")
    collector_name, collector_input = staged(collector, "performance-collector")
    worker_resources = {
        "cpu_cores": int(env_resource("corpus_worker", "CORES", 16)),
        "ram_bytes": size_bytes(str(env_resource("corpus_worker", "RAM", "32G"))),
        "disk_bytes": size_bytes(str(env_resource("corpus_worker", "DISK", "32G"))),
        "time_secs": time_seconds(str(env_resource("corpus_worker", "TIME", "12h"))),
        "gpus": 0,
        "partition": str(env_resource("corpus_worker", "PARTITION", "production")),
    }
    merge_resources = {
        "cpu_cores": int(env_resource("corpus_merge", "CORES", 4)),
        "ram_bytes": size_bytes(str(env_resource("corpus_merge", "RAM", "16G"))),
        "disk_bytes": size_bytes(str(env_resource("corpus_merge", "DISK", "64G"))),
        "time_secs": time_seconds(str(env_resource("corpus_merge", "TIME", "12h"))),
        "gpus": 0,
        "partition": str(env_resource("corpus_merge", "PARTITION", "production")),
    }
    if worker_resources["cpu_cores"] < 1 or merge_resources["cpu_cores"] < 1:
        fail("corpus worker/merge core requests must be positive")
    jobs: list[dict[str, Any]] = []
    batch_outputs: list[tuple[str, Path]] = []
    for batch in range(worker_count):
        start = batch * shards_per_worker
        count = min(shards_per_worker, shard_count - start)
        name = f"dvcs-corpus-{batch:03d}"
        archive_local = f"corpus-batch-{batch:03d}.tar.gz"
        summary_local = f"corpus-batch-{batch:03d}.json"
        performance_local = f"performance-corpus-{batch:03d}.json"
        samples_local = f"performance-corpus-{batch:03d}.jsonl"
        archive_remote = result_root / "batches" / archive_local
        summary_remote = result_root / "summaries" / summary_local
        job = job_base(
            name=name, account=args.account, constraint=args.constraint,
            resources=worker_resources, log_root=log_root,
            workflow=workflow, stage="corpus-worker",
        )
        job["inputs"] = [
            image_input, database_input, experiment_input, worker_input,
            collector_input,
        ]
        job["outputs"] = [
            transfer(archive_local, archive_remote),
            transfer(summary_local, summary_remote),
            transfer(performance_local, result_root / "performance" / performance_local),
            transfer(samples_local, result_root / "performance" / samples_local),
        ]
        job["command"] = command(
            "/bin/bash", worker_name, project, profile, corpus, image_name,
            database_name, experiment_name, str(shard_size), str(start),
            str(count), archive_local, summary_local,
            str(args.heartbeat_seconds),
            collector_name, performance_local, samples_local,
            str(args.performance_interval_seconds),
        )
        jobs.append(job)
        batch_outputs.append((archive_local, archive_remote))

    final_local = f"{corpus}.tar.gz"
    final_remote = result_root / "final" / final_local
    final_summary = f"{corpus}-summary.json"
    merge_performance = "performance-corpus-merge.json"
    merge_samples = "performance-corpus-merge.jsonl"
    merge = job_base(
        name="dvcs-corpus-merge", account=args.account,
        constraint=args.constraint, resources=merge_resources,
        log_root=log_root, workflow=workflow, stage="corpus-merge",
    )
    merge["antecedents"] = [str(job["name"]) for job in jobs]
    merge["inputs"] = [image_input, merger_input, collector_input] + [
        transfer(local, remote) for local, remote in batch_outputs
    ]
    merge["outputs"] = [
        transfer(final_local, final_remote),
        transfer(final_summary, result_root / "summaries" / final_summary),
        transfer(merge_performance, result_root / "performance" / merge_performance),
        transfer(merge_samples, result_root / "performance" / merge_samples),
    ]
    merge["command"] = command(
        "/bin/bash", merger_name, corpus, image_name, final_local,
        final_summary, str(args.heartbeat_seconds),
        collector_name, merge_performance, merge_samples,
        str(args.performance_interval_seconds),
        *[local for local, _ in batch_outputs],
    )
    jobs.append(merge)
    return {
        "name": workflow,
        "site_name": args.site,
        "max_problems": 1,
        "max_dispatched": max(1, int(args.max_dispatched)),
        "jobs": jobs,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workflow", required=True)
    common.add_argument("--account", default=os.environ.get("JLAB_ACCOUNT", ""))
    common.add_argument("--site", default=os.environ.get("SWIF_SITE_NAME", "jlab/enp"))
    common.add_argument("--constraint", default=os.environ.get("SWIF_CONSTRAINT", "el9"))
    common.add_argument("--result-root", required=True)
    common.add_argument("--log-root", required=True)
    common.add_argument("--image", required=True)
    common.add_argument("--database", required=True)
    common.add_argument("--output", required=True)
    common.add_argument("--max-dispatched", type=int, default=64)
    common.add_argument(
        "--collector",
        default=str(Path(__file__).with_name("collect_performance_metrics.py")),
    )
    common.add_argument("--performance-interval-seconds", type=int, default=30)
    commands = result.add_subparsers(dest="mode", required=True)

    analysis = commands.add_parser("analysis", parents=[common])
    analysis.add_argument("--project", required=True)
    analysis.add_argument("--profile", default="validation")
    analysis.add_argument("--corpus", required=True)
    analysis.add_argument("--selection", required=True)
    analysis.add_argument("--corpus-archive", required=True)
    analysis.add_argument("--project-archive", required=True)
    analysis.add_argument(
        "--runner", default=str(Path(__file__).with_name("run_swif2_analysis_stage.sh"))
    )
    analysis.add_argument("--include-optimize", action="store_true")
    analysis.add_argument("--from", dest="from_stage", default="selection")
    analysis.add_argument("--through", dest="through_stage", default="plot")
    analysis.add_argument("--only")
    analysis.add_argument("--heartbeat-seconds", type=int, default=300)

    corpus = commands.add_parser("corpus", parents=[common])
    corpus.add_argument("--project", required=True)
    corpus.add_argument("--profile", default="validation")
    corpus.add_argument("--corpus", required=True)
    corpus.add_argument("--experiment", required=True)
    corpus.add_argument("--shard-size", type=int, default=16)
    corpus.add_argument("--shards-per-worker", type=int, default=32)
    corpus.add_argument("--heartbeat-seconds", type=int, default=300)
    corpus.add_argument(
        "--worker", default=str(Path(__file__).with_name("run_swif2_corpus_worker.sh"))
    )
    corpus.add_argument(
        "--merger", default=str(Path(__file__).with_name("run_swif2_corpus_merge.sh"))
    )
    return result


def main() -> int:
    args = parser().parse_args()
    if not args.account or args.account.startswith("REQUIRED_"):
        fail("set a valid JLab account")
    if getattr(args, "heartbeat_seconds", 0) < 0:
        fail("heartbeat seconds must be nonnegative")
    if args.performance_interval_seconds < 1:
        fail("performance interval seconds must be positive")
    payload = analysis_workflow(args) if args.mode == "analysis" else corpus_workflow(args)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
