# JLab SWIF2 workflow submission

SWIF2 is a workflow manager layered over JLab's Slurm farm. It does not remove
the Slurm runtime contract: each task still receives `SLURM_*` allocation
variables and runs the same Apptainer launcher as the direct `sbatch` templates.
It is preferable when workflow-level queueing, retries, dependencies, and data
handling are more useful than manually managing Slurm job IDs.

The adapter is intentionally separate from `submit_workflow.sh`; direct Slurm
submission remains supported for debugging and small controlled campaigns.

## Prerequisites

Complete the normal ifarm installation and project/corpus setup in
[JLab ifarm](JLAB_IFARM.md). SWIF2 also requires a valid JLab Scientific
Computing certificate. Check it before building a workflow:

```bash
swif2 list -display json
```

An expired certificate must be renewed through JLab SciComp; no local script
can substitute for it.

## Create and import a smoke workflow

For the first submission, use an importable JSON file containing exactly one
CPU `doctor` job. It validates SWIF2 dispatch, Apptainer, persistent mounts,
and the PARTONS bridge without generating a corpus or reserving a GPU.

Initialize the project, then set its name, account, and paths in
`jobs/jlab_ifarm/resources.env`:

```bash
./dvcs init swif-smoke
# Set DVCS_PROJECT=swif-smoke and JLAB_ACCOUNT in resources.env.
jobs/jlab_ifarm/write_swif2_smoke_json.sh swif-smoke.json dvcs-swif-smoke
swif2 import -file swif-smoke.json
```

The JSON declares one core, 4 GiB RAM, 2 GiB of job-local scratch, and a
30-minute limit. It intentionally writes application provenance to the
persistent `DVCS_RESULTS` path rather than relying on disposable SWIF staging.

## Full managed workflow

After the smoke job passes, `submit_swif2_workflow.sh` remains available to
create and start the eight-step workflow through the SWIF2 CLI. Its dependency
chain uses antecedents and phases:

```text
generate -> selection -> optimize -> train -> evaluate -> compare -> holdout -> plot
```

`plot` waits for both `compare` and `holdout`. A failure leaves downstream work
unreleased. Inspect or retry it through SWIF2 rather than creating an unrelated
Slurm dependency chain.

For a manually authored workflow JSON, each normal job command should be the
absolute path to `jobs/jlab_ifarm/run_swif2_step.sh` followed by one of these
step names: `generate`, `selection`, `optimize`, `train`, `evaluate`,
`compare`, `holdout`, or `plot`. The wrapper reads the persistent paths and
project/corpus/selection names from `resources.env`, sets the same CPU/CUDA
environment as the corresponding `.sbatch` template, and delegates to
`run_step.sh`. That keeps the actual `./dvcs` launch and Apptainer mount
contract unchanged.

```bash
swif2 status dvcs-my-study-001 -display json
swif2 diagnose dvcs-my-study-001
swif2 retry-jobs dvcs-my-study-001 -problems
```

## Resource requests

The wrapper preserves the measured-starting values from the direct Slurm
templates: 16 CPU cores/64 GiB for native generation and holdout; one GPU,
8 CPU cores, and 64 GiB for optimize/train/evaluate. SWIF2 improves workflow
management, but cannot make an oversized resource request schedulable faster.

Tune one step only after inspecting prior job accounting. Add an override to
the ignored `resources.env`, for example:

```bash
SWIF_TRAIN_RAM=48G
SWIF_TRAIN_TIME=8h
SWIF_COMPARE_CORES=2
```

Supported step names are `GENERATE`, `SELECTION`, `OPTIMIZE`, `TRAIN`,
`EVALUATE`, `COMPARE`, `HOLDOUT`, and `PLOT`; each accepts `_CORES`, `_RAM`,
or `_TIME`. Do not reduce GPU, memory, or wall time blindly: compare observed
peak memory and elapsed time first. `SWIF_MAX_CONCURRENT` defaults to one to
avoid advancing dependent scientific stages concurrently.

All computational outputs and provenance remain in the persistent paths from
`resources.env`. The per-attempt SWIF staging directory is disposable and must
not become the only copy of scientific output.

## Parallel PARTONS corpus production

The validation profile contains 16,384 native parameter vectors, or 1,024
shards of 16 vectors. `swif-staged-partons-corpus-validation-16384-parallel-v2.json`
defines 32 independent CPU workers. Worker N concurrently generates the
disjoint 32-shard range beginning at `(N - 1) * 32`, for 512 vectors per
worker. Each worker returns a deeply verified partial archive through SWIF2
file movement.

The merge job has all 32 workers as antecedents. It imports their archives
into node-local scratch, rejects identity mismatches, overlaps, or incomplete
coverage, sorts the 1,024 shard records, deep-verifies all 16,384 group
indices, and returns the normal portable corpus archive. Selection and
training continue to reject individual partial batches. Import manually:

```bash
python3 jobs/jlab_ifarm/write_swif2_validation_corpus_json.py
swif2 import -file \
  jobs/jlab_ifarm/swif-staged-partons-corpus-validation-16384-parallel-v2.json
```

Each PARTONS worker requests 16 CPUs, 32 GB RAM, 32 GB scratch, and 12 hours.
The non-PARTONS merge requests 4 CPUs, 16 GB RAM, 32 GB scratch, and 12 hours.
These scratch requests include the 6.5 GB staged SIF with ample headroom: the
measured 64-vector checkpoint is 2.96 MB compressed and 4.12 MB unpacked, so
the 16,384-vector final checkpoint is expected to be about 0.8 GB compressed.
The merge consumes its node-local source corpora to avoid duplicate shard
trees. The workers have no antecedents and may run simultaneously.

## References

- [JLab SWIF2 workflow documentation](https://scicomp.jlab.org/docs/swif2)
- [JLab SWIF2 command reference](https://scicomp.jlab.org/cli/swif.html)
- [JLab Slurm batch system guidance](https://scicomp.jlab.org/docs/farm_slurm_batch)
