# JLab ifarm and farm guide

## Execution model

Use an ifarm host for installation, editing, short verification, and Slurm
submission. Run sustained native generation, neural training, evaluation, and
holdout work on allocated farm nodes. The public templates use `production`
for long CPU work and `gpu` for neural GPU work.

Those names and account requirements were checked against JLab SciComp
documentation on 2026-08-14. Site configuration can change; confirm it before
the first campaign:

```bash
sinfo -o '%P %G %c %m %l %f'
sacctmgr list user name="$USER" withassoc
```

JLab references: [partitions/resources](https://scicomp.jlab.org/docs/node/644),
[accessing GPUs](https://scicomp.jlab.org/docs/Access_GPUs),
[GPU batch jobs](https://scicomp.jlab.org/docs/node/631), [Slurm FAQ](https://scicomp.jlab.org/docs/farm_slurm_faq),
and [Slurm commands](https://scicomp.jlab.org/docs/farm_slurm_commands).

## Install on ifarm

```bash
git clone <repository-url> extract_dvcs_gpd
cd extract_dvcs_gpd
./install.sh --profile jlab_ifarm --accelerator auto
```

No Docker daemon is used on farm nodes. The profile builds or pulls one
CUDA-capable Apptainer SIF. CPU jobs run it without `--nv`; GPU jobs request an
allocation, set `DVCS_ACCELERATOR=cuda`, and run it with `--nv`.

If fakeroot builds are unavailable or too expensive on ifarm, use an approved
published digest/SIF produced elsewhere. Do not request privileged package
installation through this project.

## Configure site-specific resources

Installation creates an untracked local file if absent:

```bash
cp -n jobs/jlab_ifarm/resources.env.example jobs/jlab_ifarm/resources.env
vi jobs/jlab_ifarm/resources.env
```

Set every field:

| Variable | Meaning |
|---|---|
| `JLAB_ACCOUNT` | Your valid Slurm project account; placeholder values are rejected. |
| `DVCS_PROJECT` | Existing project name initialized through `./dvcs init`. |
| `DVCS_CORPUS` | Existing corpus name created and planned before submission. |
| `DVCS_SELECTION` | New immutable selection name created after corpus generation. |
| `DVCS_REPOSITORY` | Absolute path to this distribution checkout. |
| `DVCS_WORKSPACE` | Large writable project workspace root. |
| `DVCS_RESULTS` | Writable runtime/Slurm provenance root. |
| `DVCS_CACHE` | Large reusable cache containing LHAPDF data. |
| `DVCS_DATABASE` | Parent of the clean pinned database checkout. |
| `JLAB_CPU_PARTITION` | Current CPU partition, normally `production`. |
| `JLAB_GPU_PARTITION` | Current GPU partition, normally `gpu`. |

Do not commit `resources.env`; it may expose paths and account information.

## Choose storage

Native caches, generated arrays, checkpoints, and plots can be large. Choose a
JLab user/group/project filesystem intended for the relevant lifetime and
throughput, rather than blindly using home. Confirm quota, backup policy,
purge policy, and compute-node visibility with current site documentation or
support.

Use the same absolute mounts for every step in one project. Moving only part
of a result tree breaks content contracts and hashes. The database is mounted
read-only even when its host parent is user-owned.

Reusable corpora live under `DVCS_WORKSPACE/.corpora/` on the host and portable
exports under `DVCS_WORKSPACE/.corpus_exports/`. Native request evidence is
consolidated per shard, but a large campaign still needs measured quota,
throughput, inode, backup, and purge behavior. Do not place the only copy on
node-local scratch.

## Initialize and verify interactively

Short commands only:

```bash
./dvcs init ifarm-acceptance
./dvcs show ifarm-acceptance
apptainer inspect .dvcs/*.sif
./dvcs doctor ifarm-acceptance
./dvcs corpus-create ifarm-acceptance ifarm-corpus \
  --profile validation --shard-size 256
./dvcs corpus-plan ifarm-acceptance ifarm-corpus
```

On an ifarm login node, `doctor` asks whether to validate through a temporary
Slurm GPU allocation. Answering yes requests one GPU, reruns `doctor` with
required CUDA, prints the allocated compute hostname, and releases the job on
completion. Answering no performs the ordinary CPU login-node check.

## Included templates

| File | Default resources | Role |
|---|---|---|
| `generate.sbatch` | production, 16 CPUs, 64 GiB, 24 h | CPU PARTONS corpus generation. |
| `selection.sbatch` | production, 1 CPU, 4 GiB, 30 min | Freeze grouped train/validation/outer-test ownership. |
| `optimize_gpu.sbatch` | gpu, 1 GPU, 8 CPUs, 64 GiB, 12 h | Optional Optuna study. |
| `train_gpu.sbatch` | gpu, 1 GPU, 8 CPUs, 64 GiB, 12 h | NPE ensemble training. |
| `evaluate.sbatch` | gpu, 1 GPU, 8 CPUs, 64 GiB, 12 h | Match GPU training for neural coverage plus exact reevaluation. |
| `compare.sbatch` | production, 4 CPUs, 32 GiB, 4 h | Conventional exact-bank comparison. |
| `holdout.sbatch` | production, 16 CPUs, 64 GiB, 12 h | Named native-model holdout after closure. |
| `plot.sbatch` | production, 2 CPUs, 8 GiB, 1 h | Saved-result plotting. |

Defaults are starting requests, not measured recommendations for every site or
study. Edit explicit `#SBATCH` CPU, memory, GPU, and wall-time directives after
examining a representative run. Keep `--nodes=1` and `--ntasks=1` unless the
implementation is deliberately extended; native parallelism is within one
node, and multi-GPU neural execution assumes one node.

The templates contain an account placeholder for readability, while the
submission wrapper overrides it with `JLAB_ACCOUNT` from `resources.env`.
It also overrides partition with `JLAB_CPU_PARTITION` or
`JLAB_GPU_PARTITION`, exports the original job directory so Slurm's spool copy
can find `resources.env`, and assigns absolute stdout/stderr paths.

## Workflow dependency graph

Run:

```bash
jobs/jlab_ifarm/submit_workflow.sh
```

The submitted graph is:

```text
generate
  -> selection
      -> optimize
      -> train
          -> evaluate
              -> compare
                  -> holdout
                      -> plot
```

Every edge uses `afterok`, so a failed prerequisite prevents scientifically
invalid downstream execution. Optimization is included in this full template
chain. If you intend to use the frozen default architecture instead, submit
`generate.sbatch`, `selection.sbatch`, then `train_gpu.sbatch` with `afterok`
dependencies, and continue the same evaluate/compare/holdout/plot ordering.
Do not simply delete the optimize job while leaving train dependent on its
missing ID.

The Optuna job records recommendations but deliberately does not rewrite
`experiment.json`; the following train job therefore uses the project's
already frozen architecture. To promote an Optuna result, create a new project,
apply the chosen neural controls before its selection/training campaign, and
reuse the corpus only if compatibility checks pass.

Holdout requires evaluation and comparison gates to pass, which is why it
depends on `compare`, not merely `train`.

## Test submissions before spending allocation

```bash
sbatch --test-only --account="$JLAB_ACCOUNT" jobs/jlab_ifarm/generate.sbatch
sbatch --test-only --account="$JLAB_ACCOUNT" jobs/jlab_ifarm/selection.sbatch
sbatch --test-only --account="$JLAB_ACCOUNT" jobs/jlab_ifarm/optimize_gpu.sbatch
sbatch --test-only --account="$JLAB_ACCOUNT" jobs/jlab_ifarm/train_gpu.sbatch
sbatch --test-only --account="$JLAB_ACCOUNT" jobs/jlab_ifarm/evaluate.sbatch
sbatch --test-only --account="$JLAB_ACCOUNT" jobs/jlab_ifarm/compare.sbatch
sbatch --test-only --account="$JLAB_ACCOUNT" jobs/jlab_ifarm/holdout.sbatch
sbatch --test-only --account="$JLAB_ACCOUNT" jobs/jlab_ifarm/plot.sbatch
```

`--test-only` validates scheduler acceptance, not application behavior.

## CPU resource propagation

Corpus generation/holdout export `DVCS_NATIVE_WORKERS=all_available`. Inside the
container, a direct interactive invocation uses the full process affinity and
does not make any Slurm CPU request. When `SLURM_JOB_ID` is present, the worker
resolver intersects the process affinity mask with `SLURM_CPUS_PER_TASK`.
Every worker owns one isolated PARTONS subprocess.
Generation waves scale to keep that resolved worker pool busy whenever enough
independent parameter batches remain.
Numerical-library thread counts default to one, preventing `CPUs × BLAS
threads` oversubscription.

Do not set `native_workers` above the allocation expecting more performance;
the resolver caps it. Do not request `--ntasks=16` for this single-process
orchestrator; request one task with `--cpus-per-task=16`.

`nproc` may show every CPU on a shared ifarm login node. The framework uses
that full affinity-visible count by default for direct interactive testing.
No CPUs are reserved in this mode, so other users can contend for them. To use
more than a submitted template's default allocation, increase its
`#SBATCH --cpus-per-task` request to a count supported by the current
`production` partition and size memory accordingly. The runtime will use all
allocated CPUs, subject only to the amount of ready native work. Larger
requests can restrict eligible nodes and increase queue time.

## GPU resource propagation

JLab requires GPU work to be run through a Slurm GPU allocation. Its
[GPU access guide](https://scicomp.jlab.org/docs/Access_GPUs) demonstrates
requesting the `gpu` partition and GPU resources, then checking the allocation
with `nvidia-smi` and `CUDA_VISIBLE_DEVICES`. The supplied templates request
one GPU generically with `--gres=gpu:1`; consult `sinfo -o '%P %G %f'` before
selecting a site-specific GPU type.

Slurm sets `CUDA_VISIBLE_DEVICES` to the allocated devices. The launcher adds
Apptainer `--nv` only for an allocated CUDA job and explicitly forwards that
variable through `--cleanenv`. This is important because `--nv` supplies the
host driver libraries and devices, while `CUDA_VISIBLE_DEVICES` keeps CUDA
applications restricted to the scheduler-assigned set. The entrypoint asks
PyTorch to initialize CUDA and fails if explicit `cuda` is unusable.

For the automatic interactive smoke test after installation:

```bash
./dvcs doctor ifarm-acceptance
# Answer yes at the GPU prompt.
```

For automation, use
`DVCS_IFARM_GPU_DOCTOR=yes ./dvcs doctor ifarm-acceptance`. Successful output
must show an allocated `sciml` hostname, requested accelerator `cuda`, at least
one visible GPU, and the container CUDA runtime. The `srun` allocation ends
when `doctor` exits. Device detection alone does not accept training.

Interactive training on an ifarm login host uses the same opt-in flow:

```bash
./dvcs train PROJECT --profile quick --corpus CORPUS --selection SELECTION
# Answer yes to request the default 1 GPU, 8 CPUs, 64 GiB, 12 h allocation.
```

For automation, set `DVCS_IFARM_GPU_TRAIN=yes`. Answering no leaves the normal
CPU training path unchanged. The supplied `train_gpu.sbatch` remains the
recommended unattended production path and never prompts inside its existing
Slurm allocation.

Interactive `evaluate` reads the saved training summary, reports whether the
project trained on CPU or GPU, and recommends the same device. Accepting the
default uses a matching GPU allocation when needed; declining cancels to
protect the result contract. Set `DVCS_IFARM_EVALUATE_MATCH_TRAINING=yes` for
automation. The supplied workflow submits `evaluate.sbatch` to the GPU
partition so it matches `train_gpu.sbatch`.

With more than one requested/visible GPU, training starts one NCCL rank per
device and shards ensemble seeds. Ensure the selected profile has at least as
many candidate seeds as GPUs. Multi-GPU Optuna starts one independent trial
worker per visible GPU against a shared study.

The templates conservatively request one GPU because the packaging machine
could not validate multi-GPU equivalence. Increase `--gres` only as part of an
explicit site acceptance campaign.

## Logs and provenance

Slurm's direct output/error files are written beneath
`jobs/jlab_ifarm/logs/`. The submission wrapper writes a timestamped map of
step names to job IDs.

To audit one job independently:

```bash
squeue -j JOB_ID -o '%.18i %.12P %.24j %.10T %.10M %.10l %R'
sacct -j JOB_ID --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,ReqMem,NodeList
tail -f jobs/jlab_ifarm/logs/JOB_NAME-JOB_ID.out
tail -f jobs/jlab_ifarm/logs/JOB_NAME-JOB_ID.err
```

`squeue` shows only queued/running jobs. A job that disappears must be checked
with `sacct`. For submissions made directly with `sbatch` instead of the
wrapper, `#SBATCH --output=logs/...` is relative to the directory from which
`sbatch` was invoked; prefer the wrapper's absolute paths.

Each step additionally writes:

```text
DVCS_RESULTS/slurm/JOB_ID/STEP/
  environment.txt
  job.txt
  lscpu.txt
  nvidia-smi.txt
  distribution-commit.txt
  images.lock.json
  stdout.log
  stderr.log
  exit-code.txt
```

Container runtime/rank resource records are under
`DVCS_RESULTS/provenance/`. Preserve both locations with scientific results.

## Monitor and diagnose

```bash
squeue -u "$USER"
scontrol show job JOB_ID
sacct --format=JobID%18,State,ExitCode,Partition,AllocCPUS,ReqMem,Elapsed -j JOB_IDS
```

An `afterok`-dependent job may remain pending with a dependency reason when an
upstream job fails. Inspect the upstream `exit-code.txt`, stderr, and Slurm
record. Correct the cause and resubmit from the failed step with appropriate
dependencies; complete corpus shards/checkpoints remain resumable.

For out-of-memory or timeout failures, use `sacct`/site tools to measure peak
use before changing resources. Do not alter scientific profile counts merely
to hide an infrastructure failure without recording a new experiment.

## JLab acceptance

JLab support is not accepted from syntax alone. On site, record:

- exact SIF SHA-256 and OCI source digest;
- Apptainer version;
- NVIDIA driver, GPU model/count, and PyTorch CUDA runtime;
- observed `CUDA_VISIBLE_DEVICES` and NCCL topology;
- Slurm job IDs, accounts, partitions, resource requests, states, and exits;
- selected filesystem paths and quota/purge behavior;
- bridge self-test and full CPU/GPU workflow outcomes;
- one-GPU versus multi-GPU deterministic/statistical comparison when used.

Use the executable sequence in [Acceptance procedures](ACCEPTANCE.md).
