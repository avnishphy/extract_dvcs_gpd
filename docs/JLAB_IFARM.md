# JLab ifarm and farm guide

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
```

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

## Interactive setup and checks

```bash
./dvcs init PROJECT
./dvcs show PROJECT
./dvcs doctor PROJECT
./dvcs corpus-create PROJECT CORPUS --profile validation --shard-size 256
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

## Submit the managed workflow

Validate locally first:

```bash
./dvcs farm-submit --project PROJECT --corpus CORPUS --selection baseline --profile validation --corpus-archive /absolute/corpus.tar.gz --from train --through plot --dry-run
```

Remove `--dry-run` to import and start. Use `--only STAGE`, or `--from` and
`--through`, to control the analysis range. Add `--include-optimize` only when
an Optuna study is actually required. Full details, monitoring, recovery,
parallel corpus generation, and file movement are in
[JLab SWIF2 workflows](JLAB_SWIF2.md).

## CPUs, GPUs, and memory

- CPU counts are requests, not automatic speedups. The application uses the
  allocated affinity, but speed is limited by ready independent work and
  serial PARTONS regions.
- GPU stages request a GPU explicitly. Slurm restricts visibility and
  Apptainer exposes only the allocated devices.
- Train initially requests 8 CPUs, 1 GPU, and 24G RAM. The streaming
  materializer removes the previous duplicate 5.4G context array; measure one
  representative job before reducing RAM further.
- Evaluate uses 1 GPU plus CPUs for exact PARTONS work. CPU-only stages should
  not request GPU resources.
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
swif2 retry-jobs WORKFLOW -problems
```

## Direct Slurm diagnostics

The `.sbatch` files and `submit_workflow.sh` remain for low-level scheduler and
container diagnosis. They are not the authoritative production interface and
may not implement SWIF2 staging/reaping guarantees. A direct diagnostic job can
still be inspected with `squeue`, `sacct`, `sstat`, and `seff` after it ends.

References: [JLab SWIF2](https://scicomp.jlab.org/docs/swif2),
[SWIF commands](https://scicomp.jlab.org/cli/swif.html),
[data movement](https://scicomp.jlab.org/cli/data.html), and
[Slurm commands](https://scicomp.jlab.org/docs/farm_slurm_commands).
