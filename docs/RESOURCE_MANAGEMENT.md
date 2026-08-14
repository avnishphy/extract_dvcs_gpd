# Resource management

## Two different compute workloads

The framework has two resource regimes:

- native PARTONS generation/reevaluation is CPU process work;
- NPE training, scoring, and sampling is PyTorch work that can use CPU or CUDA.

These must not be conflated. A CUDA allocation does not make PARTONS run on a
GPU, and `native_workers` does not create PyTorch data-parallel ranks.

## CPU allocation discovery

Inside the runtime, available CPUs come from `os.sched_getaffinity(0)` when
supported. This reflects cpusets and scheduler/container restrictions more
reliably than the host's total CPU count. In an interactive session without
`SLURM_JOB_ID`, `native_workers` defaults to `all_available` and Torch defaults
to the full affinity-visible count. The launcher does not request or reserve
CPUs.

Only an actual Slurm job activates scheduler CPU limiting. When
`SLURM_JOB_ID` and `SLURM_CPUS_PER_TASK` are set, the usable count is the
smaller of affinity size and that positive integer. CPU requests live only in
the supplied `sbatch` templates or an explicit user `salloc`/`srun` command.

Invalid zero/non-numeric Slurm CPU values fail before computation. The runtime
records affinity CPUs, Slurm request, and selected CPU-thread count.

## Native PARTONS workers

`inference.runtime.native_workers` or `DVCS_NATIVE_WORKERS` accepts a positive
integer or `all_available`. Resolution is capped by usable affinity CPUs and
ready task count.

Each worker launches independent bridge processes. No PARTONS instance or
module object is shared between workers. Two parameter vectors are normally
evaluated in one atomic bridge batch. Generation submits at least 64 candidate
parameters per wave and scales larger waves to provide one ready task per
usable worker. A 256-CPU allocation therefore releases 512 parameters as 256
two-parameter tasks when that much work remains. The final partial wave and
small evaluation phases necessarily use fewer workers than allocated.

Higher worker counts help only when enough independent tasks exist and memory,
I/O, and native initialization overhead remain acceptable. Use timings in
`generation_metrics.json` and evaluation parallel records rather than assuming
linear scaling.

On JLab, direct ifarm development/testing therefore uses every affinity-visible
logical CPU by default. For sustained work through Slurm, treat
`SLURM_CPUS_PER_TASK` plus the affinity mask as the allocation.

## Preventing oversubscription

The entrypoint defaults these to `DVCS_MATH_THREADS`, normally 1:

```text
OMP_NUM_THREADS
OPENBLAS_NUM_THREADS
MKL_NUM_THREADS
NUMEXPR_NUM_THREADS
VECLIB_MAXIMUM_THREADS
```

If 16 native workers each start 16 BLAS threads, a nominal 16-CPU allocation
could create 256 runnable threads. Keep math threads at one unless native
profiling proves a different worker/thread decomposition is beneficial, and
always satisfy:

```text
native_workers * math_threads <= allocated CPUs
```

Torch CPU threads are controlled separately by `cpu_threads` or
`DVCS_CPU_THREADS` and bounded by allocation.

## Local CPU examples

Use all affinity-visible native workers and eight Torch CPU threads:

```bash
DVCS_NATIVE_WORKERS=all_available \
DVCS_CPU_THREADS=8 \
./dvcs corpus-generate study study-corpus
```

Limit native pressure for a shared workstation:

```bash
DVCS_NATIVE_WORKERS=4 ./dvcs corpus-generate study study-corpus
```

For Podman/Docker the launcher supplies the current process affinity as a
container CPU set. Restrict the launcher process itself with your OS/container
mechanism when a smaller allocation is desired.

## Accelerator policy

`DVCS_ACCELERATOR` and the experiment runtime accept:

| Value | Behavior |
|---|---|
| `cpu` | Force neural work to CPU even if CUDA is visible. |
| `cuda` | Require PyTorch CUDA initialization; fail closed if unavailable. |
| `auto` | Use CUDA when PyTorch can initialize it, otherwise record CPU fallback and reason. |

An environment override takes precedence for that run and is recorded. Use
explicit `cuda` for a campaign that must use a GPU; `auto` is suitable when a
recorded CPU fallback is acceptable.

## CUDA visibility and device selection

`CUDA_VISIBLE_DEVICES` is authoritative. The runtime does not reach outside
that list. On a single GPU it uses the ordinary `cuda` device. Under
`torchrun`, each process validates `LOCAL_RANK`, sets its local device, and
requires `WORLD_SIZE` not to exceed visible devices.

Device provenance includes Torch version, compiled CUDA runtime, CUDA
availability, device count/name, requested/resolved policy, local rank, and
world size. Explicit CUDA acceptance also needs a real tensor/training smoke;
seeing a device node or successful `nvidia-smi` is insufficient.

## Multi-GPU training

When more than one usable GPU is visible and the command is `train`, the
entrypoint runs:

```text
python -m torch.distributed.run --standalone --nproc-per-node=GPU_COUNT ...
```

The workflow initializes NCCL, assigns candidate ensemble seeds by rank, and
trains each member independently. All ranks synchronize; rank 0 aggregates
member records and performs the unchanged internal-validation selection.

Properties of this design:

- it uses every allocated visible GPU when enough ensemble members exist;
- deterministic seed identity does not depend on rank assignment;
- checkpoints remain ordinary single-member state dictionaries;
- it does not split one flow model across devices;
- an allocation with more GPUs than candidate seeds fails rather than idling
  devices silently.

Hardware validation must compare checkpoint/state hashes where expected and
posterior score/distribution behavior with a same-seed one-GPU run. NCCL
startup success alone does not establish statistical equivalence.

## Multi-GPU Optuna

For `optimize`, the entrypoint divides requested trials as evenly as possible
across visible devices. Each process sees one device and runs a one-process
trial stream. All processes coordinate through the same SQLite-backed Optuna
study; the database timeout is configured for concurrent access.

Before trials start, corpus-backed realization publication is protected by a
project/profile process lock. The first process materializes the deterministic
arrays; other GPU workers verify matching corpus, selection, configuration,
bridge, and complete-file identity and reuse them. This prevents concurrent
manifest publication and avoids rebuilding the same tensor set per GPU.

Rank logs and a combined exit code are retained under results provenance. A
worker failure fails the parent action. Concurrent-study behavior remains a
hardware/site acceptance item.

## Phase-by-phase recommendations

| Phase | Primary resource | Notes |
|---|---|---|
| `corpus-generate` | CPU + memory/storage | Many isolated native workers; atomically publishes reusable shards and consolidated evidence. |
| `train` | GPU preferred | CPU fallback supported; candidate members parallelize across GPUs. |
| `optimize` | GPU preferred | Independent trials; persistent shared study. |
| `evaluate` | mixed | Neural coverage/sampling plus CPU exact reevaluation. |
| `compare` | CPU/memory | Importance calculations on saved exact corpus. |
| `holdout` | CPU + memory | Fresh native truth and exact DD prediction checks. |
| `plot` | small CPU/memory | Saved artifacts only. |

This is why the JLab templates separate jobs instead of requesting a GPU for
the entire workflow.

## Resource provenance and audit

Inspect:

- `results/doctor.json` for resolved policy;
- `$DVCS_WORKSPACE/.corpora/CORPUS/corpus.json` for shard/native identity;
- `generated/generation_metrics.json` for worker/cache/timing details;
- `evaluation/evaluation_metrics.json` for each exact parallel phase;
- `training/training_summary.json` for device and peak-memory data;
- `/results/provenance/runtime*.json` for container-level allocation;
- `DVCS_RESULTS/slurm/...` for scheduler environment/hardware/job records.

Report requested, allocated, resolved, and actually used resources separately.
