# JLab SWIF2 workflows

SWIF2 is the supported interface for sustained ifarm/farm work. It manages the
underlying Slurm jobs, dependencies, retries, and file movement. Direct
`.sbatch` scripts remain low-level diagnostics; do not use them for a normal
analysis campaign.

## One-time setup

Install on ifarm, create the project and corpus selection, then copy and edit
the ignored resource file:

```bash
cp jobs/jlab_ifarm/resources.env.example jobs/jlab_ifarm/resources.env
swif2 list -display json
```

Set at least `JLAB_ACCOUNT`, `DVCS_PROJECT`, `DVCS_CORPUS`, the DVCS paths, and
`SWIF_CORPUS_ARCHIVE`. `swif2 list` checks that the JLab SciComp certificate is
usable.

## Submit analysis stages

First create and validate everything without contacting SWIF2:

```bash
./dvcs farm-submit --project PROJECT --corpus CORPUS --selection baseline --profile validation --corpus-archive /absolute/corpus.tar.gz --from train --through plot --dry-run
```

Remove `--dry-run` to import and start the workflow. A unique workflow name is
generated, or set one with `--workflow NAME`. Useful controls are:

```text
--from STAGE --through STAGE   contiguous portion of the analysis
--only STAGE                   one stage only
--include-optimize             add optimize before train
--import-only                  import without starting
```

Analysis stage order is `selection`, optional `optimize`, `train`, `evaluate`,
`compare`, `holdout`, `plot`. GPU requests and Apptainer `--nv` passthrough are
automatic for optimize, train, and evaluate. The remaining stages are CPU
jobs. Each job is tagged with the workflow and stage names.

On an ifarm login node, direct `optimize`, `train`, and `evaluate` commands are
rejected with a pointer to `farm-submit`. `doctor` may still request a short
interactive GPU allocation because it is a bounded environment check.

## Data movement and outputs

Submission packages these immutable inputs under `.dvcs/swif2_inputs/`:

- installed SIF;
- GPD database archive;
- project-state archive;
- portable corpus archive.

Names include content hashes so SWIF2 cannot reuse stale content under an old
logical name. Each job receives them through SWIF2, runs from disposable
`$SWIF_JOB_WORK_DIR`, and performs active PARTONS/Torch I/O on node-local
storage. A successful stage returns:

- the next project-state archive below `SWIF_OUTPUT_ROOT/WORKFLOW/state/`;
- a stage summary JSON below `SWIF_OUTPUT_ROOT/WORKFLOW/summaries/`;
- aggregate performance JSON and raw JSONL samples below
  `SWIF_OUTPUT_ROOT/WORKFLOW/performance/`;
- stdout/stderr under `SWIF_LOG_ROOT/WORKFLOW/` (normally `/farm_out`).

The output archive of one stage is the declared input of its successor. Failed
or missing output therefore cannot silently release downstream analysis.

## Resources

Defaults live in `jobs/jlab_ifarm/resources.env.example`. Override a stage in
the ignored `resources.env`, for example:

```bash
SWIF_TRAIN_CORES=8
SWIF_TRAIN_RAM=24G
SWIF_TRAIN_TIME=12h
SWIF_TRAIN_GPUS=1
SWIF_METRICS_INTERVAL_SECONDS=30
```

The 24G initial training request accompanies the streaming materializer, which
eliminates the previous duplicate 5.4G context allocation that caused a 16G
OOM. After one successful representative job, tune RAM and wall time from
measured peak use. CPU count controls Torch threads inside the allocation;
requesting unused CPUs only delays dispatch.

Performance collection is required for every stage. The host-side sampler
reads the job's Slurm cgroup for CPU time, current/peak memory, and block I/O.
GPU jobs also query `nvidia-smi` for utilization, VRAM, and power. It writes an
atomic rolling summary used by heartbeats plus the complete time series. The
30-second default is small relative to PARTONS/training work; increase it only
for unusually short or overhead-sensitive tests.

Use several successful representative jobs before changing requests:

- CPU efficiency is cgroup CPU seconds divided by allocated core-walltime;
  reduce cores when sustained efficiency is low and elapsed time does not
  improve.
- Set RAM above the largest observed cgroup peak, retaining a measured safety
  margin for workload variation.
- Compare GPU mean utilization, maximum VRAM, and training throughput before
  changing GPU count or CPU feeder resources.
- Set walltime above the slowest comparable successful attempt, not merely the
  mean.

`SWIF_MAX_CONCURRENT` limits running jobs and `SWIF_MAX_DISPATCHED` limits
dispatch pressure. Scientific dependencies still prevent incompatible stages
from running together.

## Parallel corpus generation

The corpus submitter can split any project parameter list into non-overlapping
workers and a verified merge job. Validate before import:

```bash
./dvcs farm-corpus-submit --project PROJECT --corpus CORPUS --profile validation --shard-size 16 --shards-per-worker 32 --dry-run
```

Remove `--dry-run` to start. The lower-level JSON interface is
`python3 jobs/jlab_ifarm/write_swif2_workflow.py corpus --help`.

Each worker receives a deterministic shard range, an isolated node-local
native-work directory, and writes an immutable partial archive. The merge job
rejects identity mismatches, overlaps, missing shards, or wrong parameter
indices before exporting a normal portable corpus. Choose `--shard-size` and
`--shards-per-worker` from a measured PARTONS benchmark; do not request one
oversized multi-node job.

## Monitor and recover

```bash
swif2 status WORKFLOW -jobs -transfers -storage -display json
swif2 diagnose WORKFLOW
swif2 retry-jobs WORKFLOW -problems
```

The wrappers emit periodic `[dvcs-swif2]` heartbeats with cumulative CPU
efficiency, memory, and—on GPU stages—GPU utilization/VRAM. Training also
reports materialization/model progress. Corroborate the final JSON with
`seff SLURM_JOB_ID` or `sacct`; the stage summary records that ID. Because
every completed stage is reaped as an archive, a later workflow can resume
with `--from STAGE` from the appropriate project archive.

See the [JLab SWIF2 guide](https://scicomp.jlab.org/docs/swif2),
[SWIF command reference](https://scicomp.jlab.org/cli/swif.html), and
[data-movement reference](https://scicomp.jlab.org/cli/data.html).
