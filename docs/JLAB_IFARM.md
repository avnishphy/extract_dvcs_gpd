# JLab ifarm and farm guide

The measured corpus-worker means were 8.39 h on farm25 (8 jobs), 12.74 h on
farm19 (9), and 17.89 h on farm23 (13). The resulting stage-specific policy is
farm25 strict, farm19 explicit fallback, and farm23 excluded. Mount the master
corpus read-only in every downstream architecture job.

## Execution model

Use an ifarm login host for installation, editing, project setup, and short
checks. Use managed SWIF2 workflows for sustained PARTONS generation, neural
training, evaluation, comparison, holdout, and plotting. SWIF2 dispatches the
underlying JLab Slurm jobs; users should not manually maintain dependency job
IDs for a normal campaign.

The framework uses Apptainer as a process container, not a virtual machine or
background service. A command starts a normal process in the SIF environment;
when the command/job exits, there is no container daemon to kill.

## Install and configure

```bash
git clone <repository-url> extract_dvcs_gpd
cd extract_dvcs_gpd
./install.sh --profile jlab_ifarm --accelerator auto
cp jobs/jlab_ifarm/resources.env.example jobs/jlab_ifarm/resources.env
vi jobs/jlab_ifarm/resources.env
set -a; source jobs/jlab_ifarm/resources.env; set +a
```

Use that ordinary installer command for initial setup, routine verification,
and recovery. It reuses the accepted SIF and does not rebuild merely because it
was rerun. `--source-build` is a maintainer operation for image-impacting
changes; use the [image update and recovery runbook](IMAGE_UPDATE_RUNBOOK.md)
so the current production image remains available until its replacement is
fully tested.

No Docker daemon is used on farm nodes. One CUDA-capable SIF serves CPU and GPU
jobs. CPU jobs omit `--nv`; allocated GPU jobs set `DVCS_ACCELERATOR=cuda`, add
`--nv`, and forward the scheduler-provided `CUDA_VISIBLE_DEVICES` through
Apptainer `--cleanenv`.

Set the project/account/path variables and `SWIF_CORPUS_ARCHIVE` in the ignored
`resources.env`. Never commit this file. SWIF variables control managed jobs:

| Variable | Meaning |
|---|---|
| `JLAB_ACCOUNT` | Valid JLab farm account, for example `hallc`. |
| `DVCS_PROJECT`, `DVCS_CORPUS`, `DVCS_SELECTION` | Existing analysis identities. |
| `DVCS_WORKSPACE`, `DVCS_RESULTS`, `DVCS_CACHE`, `DVCS_DATABASE` | Persistent host paths. |
| `SWIF_CORPUS_ARCHIVE` | Portable, deeply verified corpus archive. |
| `SWIF_OUTPUT_ROOT` | Reaped project archives and summaries. |
| `SWIF_LOG_ROOT` | stdout/stderr root, normally below `/farm_out/$USER`. |
| `SWIF_MAX_CONCURRENT`, `SWIF_MAX_DISPATCHED` | Workflow pressure limits. |
| `SWIF_STAGE_{CORES,RAM,DISK,TIME,GPUS}` | Per-stage requests. |

The older `JLAB_*` resource values serve only the retained direct-Slurm
diagnostic scripts.

For a first campaign, set every placeholder path to an absolute user-owned
location, set `JLAB_ACCOUNT` to an account shown by `sacctmgr show user
"$USER" withassoc`, and set `SWIF_LOG_ROOT` to
`/farm_out/$USER/dvcs`. `DVCS_PROJECT`, `DVCS_CORPUS`, and `DVCS_SELECTION`
are defaults only; explicit `farm-*` command arguments take precedence.
`resources.env` is shell syntax: write `NAME=value`, with no spaces around
`=`. Validate it without submitting:

```bash
bash -n jobs/jlab_ifarm/resources.env
test -d "$DVCS_WORKSPACE" && test -d "$DVCS_RESULTS"
swif2 list -display json
```

Do not commit `resources.env`; it contains user/site paths and campaign
settings.

## Interactive setup and checks

```bash
./dvcs init PROJECT
./dvcs show PROJECT
./dvcs doctor PROJECT
source .dvcs/install.env
cp configs/examples/master_corpus_gpd_truth_smoke_v1.json \
  "$DVCS_WORKSPACE/PROJECT/gpd_truth.json"
# Replace the smoke table with the reviewed production coordinates.
./dvcs corpus-create PROJECT CORPUS --profile validation --shard-size 16
./dvcs corpus-plan PROJECT CORPUS
```

Without a scheduler allocation, native CPU commands see the process’s full
host affinity (`all_available`); the framework does not request or invent a CPU
limit. Inside a farm job it respects allocated cores.

On ifarm, `doctor` may request a short Slurm GPU allocation. A successful GPU
doctor reports the allocated compute hostname, requested/resolved CUDA device,
visible-device count/name, PyTorch CUDA build, and CPU affinity. The allocation
ends when doctor exits.

Sustained `optimize`, `train`, and `evaluate` are not launched interactively on
ifarm. They point to `./dvcs farm-submit`, which gives them explicit resources,
retries, logs, node-local I/O, and reaped outputs.

For post-training validation, pass one content-hashed design with
`farm-submit --holdout-design FILE`. Submit same-kinematic and fresh-kinematic
holdouts as separate workflows from the same compare archive; see
[JLAB_SWIF2.md](JLAB_SWIF2.md).

## Submit the managed workflow

For a new corpus, first validate and then submit the parallel corpus workflow:

```bash
./dvcs farm-corpus-submit --project PROJECT --corpus CORPUS --profile validation --shard-size 16 --shards-per-worker 32 --workflow PROJECT-corpus-v1 --dry-run
./dvcs farm-corpus-submit --project PROJECT --corpus CORPUS --profile validation --shard-size 16 --shards-per-worker 32 --workflow PROJECT-corpus-v1
```

Wait until the merge and output transfer finish. The analysis input is then
`$SWIF_OUTPUT_ROOT/PROJECT-corpus-v1/final/CORPUS.tar.gz`. Validate analysis:

```bash
./dvcs farm-submit --project PROJECT --corpus CORPUS --selection baseline --profile validation --corpus-archive "$SWIF_OUTPUT_ROOT/PROJECT-corpus-v1/final/CORPUS.tar.gz" --from selection --through plot --workflow PROJECT-analysis-v1 --dry-run
./dvcs farm-submit --project PROJECT --corpus CORPUS --selection baseline --profile validation --corpus-archive "$SWIF_OUTPUT_ROOT/PROJECT-corpus-v1/final/CORPUS.tar.gz" --from selection --through plot --workflow PROJECT-analysis-v1
```

If selection already exists, start at `materialize`; start at `train` only when
materialized arrays already exist. Otherwise start at `selection`. Use
`--only STAGE`, or `--from` and `--through`,
to control the analysis range. Add `--include-optimize` only when an Optuna
study is required. Full details, monitoring, recovery, external-corpus reuse,
parallel corpus generation, and file movement are in
[JLab SWIF2 workflows](JLAB_SWIF2.md).

## CPUs, GPUs, and memory

- CPU counts are requests, not automatic speedups. The application uses the
  allocated affinity, but speed is limited by ready independent work and
  serial PARTONS regions.
- GPU stages request a GPU explicitly. Slurm restricts visibility and
  Apptainer exposes only the allocated devices.
- Materialize requests 2 CPUs, no GPU, and 20G RAM. V6 telemetry showed an
  8-CPU request was only 17.8% efficient because constant covariance work is
  precomputed and 6+ GB state serialization is mostly memory/I/O-bound.
- Train initially requests 4 CPUs, 1 GPU, and 32G RAM. Complete generated
  arrays remain memory-mapped on CPU; only minibatches enter GPU memory. Measure
  representative jobs before tuning RAM or batch size.
- Evaluate uses 1 GPU for saved-model scoring. The separate
  `exact-reevaluate` stage uses CPUs for PARTONS and requests no GPU.
- Large corpus production must use independent shard-range workers plus a
  verified merge, not one multi-node process or an oversized CPU request.

Current site availability can change. Check:

```bash
sinfo -o '%P %G %c %m %l %f'
sacctmgr list user name="$USER" withassoc
```

JLab’s [GPU access guide](https://scicomp.jlab.org/docs/Access_GPUs) explains
the allocation contract and `nvidia-smi` checks.

## Storage, logs, and progress

Persistent corpus/project data belongs on an appropriate JLab project or work
filesystem. Active farm I/O is staged to `$SWIF_JOB_WORK_DIR`; that directory
is disposable. SWIF2 must reap every required scientific output before a job
is considered usable.

Scheduler logs normally go below `/farm_out/$USER/dvcs/WORKFLOW/`. Scientific
state archives, summaries, and performance telemetry go below the `state/`,
`summaries/`, and `performance/` directories under
`SWIF_OUTPUT_ROOT/WORKFLOW/`. The wrappers emit five-minute heartbeats by
default. They include cumulative CPU/memory metrics; GPU stages also report
utilization and VRAM when `nvidia-smi` is available.

Monitor with:

```bash
swif2 status WORKFLOW -jobs -transfers -storage -display json
swif2 diagnose WORKFLOW
swif2 retry-jobs WORKFLOW -problems PROBLEM_CLASS
```

`RUNNING` means the payload is executing; `UNDISPATCHED`/pending work has not
started its walltime. A stage may finish in Slurm before SWIF2 finishes reaping
its declared outputs, so wait for transfer completion before consuming files.
The batch launcher intentionally uses `--no-progress`; terminal-style bars are
replaced by `[dvcs-swif2]` heartbeat lines in stdout.

Cancel all running and pending work with `swif2 cancel WORKFLOW`. Add `-delete`
only when the workflow record may also be removed and its name reused. Cancel
does not delete already reaped scientific outputs. Never cancel underlying
Slurm jobs alone for a managed campaign, because SWIF2 owns their state.

## Direct Slurm diagnostics

The `.sbatch` files and `submit_workflow.sh` remain for low-level scheduler and
container diagnosis. They are not the authoritative production interface and
may not implement SWIF2 staging/reaping guarantees. A direct diagnostic job can
still be inspected with `squeue`, `sacct`, `sstat`, and `seff` after it ends.

References: [JLab SWIF2](https://scicomp.jlab.org/docs/swif2),
[SWIF commands](https://scicomp.jlab.org/cli/swif.html),
[data movement](https://scicomp.jlab.org/cli/data.html), and
[Slurm commands](https://scicomp.jlab.org/docs/farm_slurm_commands).
