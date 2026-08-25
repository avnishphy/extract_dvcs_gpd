# JLab SWIF2 workflows

SWIF2 is the supported interface for sustained ifarm/farm work. It manages the
underlying Slurm jobs, dependencies, retries, and file movement. Direct
`.sbatch` scripts remain low-level diagnostics; do not use them for a normal
analysis campaign.

## One-time setup

Install on ifarm, then copy and edit the ignored resource file:

```bash
cp jobs/jlab_ifarm/resources.env.example jobs/jlab_ifarm/resources.env
vi jobs/jlab_ifarm/resources.env
set -a; source jobs/jlab_ifarm/resources.env; set +a
swif2 list -display json
```

Set at least `JLAB_ACCOUNT`, `DVCS_PROJECT`, `DVCS_CORPUS`, and the DVCS/SWIF
paths. Set `SWIF_CORPUS_ARCHIVE` after corpus generation, or pass
`--corpus-archive` explicitly. `swif2 list` checks that the JLab SciComp
certificate is usable.

Use absolute paths. `SWIF_OUTPUT_ROOT` is persistent scientific storage;
`SWIF_LOG_ROOT` is scheduler logging, normally `/farm_out/$USER/dvcs`. Do not
point either at node-local scratch. The ignored resource file is read at
submission time and is not required on compute nodes.

## Complete first campaign

Create and inspect the scientific identities on ifarm:

```bash
./dvcs init PROJECT
./dvcs show PROJECT
./dvcs doctor PROJECT
./dvcs corpus-create PROJECT CORPUS --profile validation --shard-size 16
./dvcs corpus-plan PROJECT CORPUS
```

Generate the corpus with independent shard-range workers and a verified merge:

```bash
./dvcs farm-corpus-submit --project PROJECT --corpus CORPUS --profile validation --shard-size 16 --shards-per-worker 32 --workflow PROJECT-corpus-v1 --dry-run
./dvcs farm-corpus-submit --project PROJECT --corpus CORPUS --profile validation --shard-size 16 --shards-per-worker 32 --workflow PROJECT-corpus-v1
```

After all transfers finish, the merged archive is
`$SWIF_OUTPUT_ROOT/PROJECT-corpus-v1/final/CORPUS.tar.gz`. Submit selection,
training, evaluation, comparison, holdout, and plots:

```bash
./dvcs farm-submit --project PROJECT --corpus CORPUS --selection baseline --profile validation --corpus-archive "$SWIF_OUTPUT_ROOT/PROJECT-corpus-v1/final/CORPUS.tar.gz" --from selection --through plot --workflow PROJECT-analysis-v1 --dry-run
./dvcs farm-submit --project PROJECT --corpus CORPUS --selection baseline --profile validation --corpus-archive "$SWIF_OUTPUT_ROOT/PROJECT-corpus-v1/final/CORPUS.tar.gz" --from selection --through plot --workflow PROJECT-analysis-v1
```

The submission commands import and start automatically. `--import-only` stops
after import; start that workflow later with `swif2 run WORKFLOW
-maxconcurrent N`. The concurrency limit is a ceiling, not a CPU request.
Analysis dependencies make the ordinary stage chain effectively serial, while
independent corpus workers can run concurrently.

## Submit analysis stages

First create and validate everything without contacting SWIF2:

```bash
./dvcs farm-submit --project PROJECT --corpus CORPUS --selection baseline --profile validation --corpus-archive /absolute/verified-corpus.tar.gz --from selection --through plot --dry-run
```

Remove `--dry-run` to import and start the workflow. A unique workflow name is
generated, or set one with `--workflow NAME`. Useful controls are:

```text
--from STAGE --through STAGE   contiguous portion of the analysis
--only STAGE                   one stage only
--include-optimize             add optimize before train
--import-only                  import without starting
```

Analysis stage order is `selection`, CPU `materialize`, optional `optimize`,
`train`, `evaluate`, `compare`, `holdout`, `plot`. Materialization uses two
CPU workers and no GPU. GPU requests and Apptainer `--nv` passthrough are
automatic only for optimize, train, and evaluate. Each job is tagged with the
workflow and stage names.

Managed GPU jobs use Slurm `--gpus=N`, not `--gres=gpu:N`. SWIF2 independently
adds scratch as `--gres=disk:...`; a second `--gres` can replace the GPU request.
Submission preflight rejects this inconsistent form.

On an ifarm login node, direct `optimize`, `train`, and `evaluate` commands are
rejected with a pointer to `farm-submit`. `doctor` may still request a short
interactive GPU allocation because it is a bounded environment check.

Start at `selection` when the named selection is absent from the staged
project. Start at `materialize` when the selection exists but its deterministic
arrays do not. Start at `train` only when complete materialized arrays already
exist. `evaluate` consumes the saved trained result contract; do not combine
artifacts from a changed experiment, profile, image, or bridge.

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

The source workspace is not updated in place. The final analysis project is
the last archive, for example
`state/PROJECT-analysis-v1-project-after-plot.tar`. Preserve the archive rather
than copying isolated result files out of it. To inspect it without replacing
an existing project:

```bash
mkdir -p /path/to/inspection-directory
tar -tf "$SWIF_OUTPUT_ROOT/PROJECT-analysis-v1/state/PROJECT-analysis-v1-project-after-plot.tar" | head
tar -xf "$SWIF_OUTPUT_ROOT/PROJECT-analysis-v1/state/PROJECT-analysis-v1-project-after-plot.tar" -C /path/to/inspection-directory
```

## Resources

Defaults live in `jobs/jlab_ifarm/resources.env.example`. Override a stage in
the ignored `resources.env`, for example:

```bash
SWIF_MATERIALIZE_CORES=2
SWIF_MATERIALIZE_RAM=20G
SWIF_MATERIALIZE_TIME=4h
SWIF_TRAIN_CORES=4
SWIF_TRAIN_RAM=32G
SWIF_TRAIN_TIME=12h
SWIF_TRAIN_GPUS=1
SWIF_METRICS_INTERVAL_SECONDS=30
```

The CPU materialize job precomputes experiment-constant covariance terms once,
then assigns deterministic parameter groups across two measured-default
workers. Its 6+ GB state serialization is mostly memory/I/O-bound, so more
cores can reduce CPU efficiency without reducing wall time. Its state archive
is reaped before GPU allocation begins. The 32G train request
covers CPU memory mapping plus temporary SBI indexing copies; complete contexts
stay on CPU and only minibatches move to CUDA. Tune both stages independently
from successful telemetry.

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

`shard-size` is the number of accepted native parameter vectors stored in one
atomic shard. `shards-per-worker` is the number of non-overlapping shards one
farm job generates sequentially. Their product is the maximum vectors assigned
to one worker job; it does not merge shards or change corpus identity.

## Reusing an existing corpus

Do not regenerate a compatible expensive corpus. Obtain its portable archive,
source `experiment.json`, SIF/bridge identity, and checksums. Initialize a new
project, copy the source experiment, change `experiment.name` to the new
project directory, then run `show`, local `corpus-import`, `corpus-plan`, and
deep verification. Only after full compatibility should the same archive be
passed to `farm-submit`. Exact commands and the compatibility boundary are in
[Corpus and data selection](CORPUS_AND_DATA_SELECTION.md#using-another-users-corpus).

## Monitor and recover

```bash
swif2 status WORKFLOW -jobs -transfers -storage -display json
swif2 diagnose WORKFLOW
swif2 retry-jobs WORKFLOW -problems PROBLEM_CLASS
```

Use `swif2 show-job WORKFLOW -name dvcs-train -display json` for one stage.
The returned attempt contains the underlying Slurm job ID. `UNDISPATCHED`
means its walltime has not started; an antecedent dependency can make that
normal. After Slurm exits, the stage is not complete until SWIF2 reaps all
declared outputs.

The wrappers emit periodic `[dvcs-swif2]` heartbeats with cumulative CPU
efficiency, memory, and—on GPU stages—GPU utilization/VRAM. Materialization
reports parameter progress; training reports completed models. Corroborate with
`seff SLURM_JOB_ID` or `sacct`; the stage summary records that ID. Because
every completed stage is reaped as an archive, a later workflow can resume
with `--from STAGE` from the appropriate project archive.

For a transient site or transfer problem, diagnose before retrying. For OOM or
timeout, use the performance record and stage summary to choose a new request;
do not repeatedly retry an unchanged insufficient allocation. Stop an entire
campaign with:

```bash
swif2 cancel WORKFLOW
```

Use `swif2 cancel WORKFLOW -delete` only if the workflow record may also be
removed and the name reused. Already reaped outputs remain on persistent
storage. Do not manage the underlying Slurm jobs separately from SWIF2.

## Tune later submissions from measurements

Each stage writes `performance-*.json` and `performance-*.jsonl`. The aggregate
record reports allocated cores, cgroup CPU efficiency, current/peak memory,
block I/O, hostname, and job IDs; GPU records add utilization, VRAM, and power.
Compare several successful jobs of the same stage and input scale. Set RAM
above the largest measured peak with a workload-justified margin, reduce cores
when low efficiency does not buy lower elapsed time, and size walltime above
the slowest comparable success. A running `seff` report can show zero because
accounting is incomplete; use framework heartbeats during execution and final
telemetry after completion.

See the [JLab SWIF2 guide](https://scicomp.jlab.org/docs/swif2),
[SWIF command reference](https://scicomp.jlab.org/cli/swif.html), and
[data-movement reference](https://scicomp.jlab.org/cli/data.html).
