# Troubleshooting

## Start with evidence

For every failure, retain the exact command, exit code, stdout/stderr, project
and profile, distribution commit, image/SIF identity, `doctor` result, and
resource allocation. Avoid deleting caches or changing configuration before
capturing evidence; many errors are deliberately diagnostic.

Run these inexpensive checks first:

```bash
git status --short
./tests/run.sh static
./dvcs doctor PROJECT
./dvcs partons-bridge --capabilities
./dvcs partons-bridge --self-test
```

## Installation failures

### First five-minute installation triage

Do not immediately restart with `--source-build`. Capture the current state:

```bash
./install.sh --profile jlab_ifarm --accelerator auto --dry-run
cat .dvcs/install.lock.d/owner 2>/dev/null || true
ls -lt .dvcs/build-logs/ 2>/dev/null
find .dvcs -maxdepth 1 -name '*.sif' -o -name '*.sif.partial'
pgrep -af 'install.sh|apptainer.*build' || true
```

The dry run says whether the next ordinary invocation will verify a final SIF,
resume a completed partial, pull, or build. Preserve the newest log, partial,
final SIF, and lock owner before changing anything. For an intentional image
change, follow the [image update and recovery runbook](IMAGE_UPDATE_RUNBOOK.md).

### No Podman or Docker

The local profile found no rootless container engine. Install/configure one
through your system administrator or distribution outside this installer, or
set `DVCS_ENGINE=podman|docker` to an existing engine. The project will not
install host packages for you.

### Rootless engine permission error

Confirm ordinary rootless test containers work, the daemon/socket belongs to
your user, and the selected workspace paths are accessible. Do not solve this
by running the entire workflow with sudo: generated files would become
root-owned and the non-root contract would be untested.

### Source download/checksum failure

Distinguish network/TLS failure from an actual checksum mismatch. A checksum
mismatch is fail-closed: preserve the object and lock, verify the upstream
release independently, and update the lock only through a reviewed dependency
change. Never disable checksum verification for convenience.

If Apptainer reports `Failed to connect to ftp.gnu.org`, update the checkout
and rerun the identical install command. Current installer prefetches verified
`gsl-2.8.tar.gz` on the host using GNU/kernel mirrors and binds it read-only
into the build. Confirm the cache before retrying:

```bash
find "${DVCS_WHEELHOUSE:-${TMPDIR:-/tmp}/extract-dvcs-gpd-wheels-$UID}" -name 'gsl-2.8.tar.gz' -o -name 'LHAPDF-6.5.6.tar.gz'
```

Do not manually place an unverified archive there: checksum failure deletes
the partial download and fails closed.

An HTTP `503 Service Unavailable` from `snapshot.ubuntu.com` is different: it
is a transient mirror response, not a checksum or dependency-resolution
failure. The installer probes the pinned snapshot before a first source build;
APT then uses three bounded per-object and transaction retries. Build output is
retained in `.dvcs/build-logs/`.

Rerun the ordinary `install.sh` command without `--source-build`. If Apptainer had
already produced a complete tag-matched `.sif.partial`, the installer verifies
that SIF and resumes the test/promotion phase instead of rebuilding it. A
corrupt or mismatched partial is preserved for diagnosis and requires an
explicit `--source-build` to replace. Do not add `--fix-missing`, change the
snapshot timestamp, or substitute unpinned packages.

If no complete partial exists, the ordinary command keeps production on the
current final SIF. Retry an intentional source build only after the snapshot is
available and only when the update still requires a new image.

### Interrupted install

Rerun the same `install.sh` command. Images/data/environment state use partial
names and atomic moves. If an incomplete final LHAPDF directory exists, point
`DVCS_CACHE` at a fresh user-owned location or remove the incomplete directory
yourself only after confirming its exact path and contents.

Installation is serialized with the atomic directory `.dvcs/install.lock.d`,
which works on the JLab NFSv3 project filesystem. Its `owner` file records the
host, PID, start time, and command when a genuinely concurrent installer is
running. The older `.dvcs/install.lock` advisory-lock file is ignored because
NFS could retain that lock after its process exited.

Read `.dvcs/install.lock.d/owner` and verify the recorded host/PID plus
`pgrep -af 'install.sh|apptainer.*build'`. Wait if either process is live. Only
after proving that all related processes are gone, move the lock to a dated
`.stale.*` name as documented in the image-update runbook. Moving preserves
evidence; deleting an unexplained lock can start two concurrent builds.

### Build-time PARTONS logger permission failure

If an older definition reports `LoggerManager::update` cannot open a file below
`/cache/partons-logs` while running the automatic `%test`, update the checkout
before rebuilding. Apptainer mounts the root filesystem read-only for its
automatic build test, so directory mode alone cannot make that path writable.
Current definitions explicitly defer only the native self-test in that phase.
The installer then reruns `%test` with the user-owned cache bound at `/cache`;
the native test is mandatory there, and `set -eu` prevents an abort from being
masked by any later successful test.

The line `native self-test deferred until writable /cache is bound` is expected
only in the automatic build test. It must not appear as the outcome of the
installer's later writable-cache test.

### Database revision mismatch or dirty checkout

The installer refuses to reset it. Preserve that checkout and set
`DVCS_DATABASE` to a separate parent where a clean locked checkout can be
installed. Do not package local uncommitted database changes.

## Launcher and mount failures

### `not installed; run ./install.sh`

`.dvcs/install.env` is absent from this checkout. Run installation here; state
from a different clone is intentionally not discovered globally.

### Workspace/result/cache permission denied

Check host ownership, parent execute permissions, filesystem ACLs, quota, and
whether the engine mapped the invoking UID/GID. On SELinux systems confirm an
approved labeling policy. The launcher does not chmod shared project storage.

### Project not found or invalid name

Use `./dvcs list`. Names must identify direct real directories below the
workspace root and cannot contain traversal or symlink escapes.

If `show` says `experiment.name must equal the project directory`, edit only
the top-level `experiment.name` so it exactly matches the name passed to
`init`. Renaming a directory or copying an experiment does not rewrite this
identity automatically.

## Native bridge failures

### PARTONS configuration or schema missing

Use the installed bridge through `./dvcs`. `partons.properties`,
`logger.properties`, and `xmlSchema.xsd` must remain beside the executable.
Do not copy only `partons_bridge` to another directory.

### Shared library not found

Inspect the installed binary and image rather than setting a development-tree
`LD_LIBRARY_PATH`:

```bash
readelf -d /opt/dvcs/bin/partons_bridge | grep -E 'RPATH|RUNPATH'
ldd /opt/dvcs/bin/partons_bridge
```

The expected installed RUNPATH is `$ORIGIN/../lib`. A missing dependency means
an incomplete/wrong image or installation.

### Native request failed

Find the corresponding content-addressed cache directory and inspect
`request.json`, `metadata.json`, `stderr.txt`, `stdout.raw.txt`, and
`exit_code.txt`. Do not overwrite `response.json`. Correct invalid parameters,
kinematics, missing data, or backend installation and rerun.

### Generation makes little progress

The main count is accepted parameters, not attempts. Inspect rejected/invalid
counts and `invalid_simulation_map.json`. A difficult support region may need
many deterministic replacements; a systematic zero-acceptance first wave
should abort. Increasing workers does not repair invalid physics.

### Unphysical fixed-target kinematics

Every point must satisfy `0 < Q²/(2 M_p E xB) < 1`. For legacy/database-backed
provenance create a new project to receive the current selector. For intentional
manual points set `kinematics_source` to exactly `{"mode":"manual"}` and
correct the coupled values. `--force-native` cannot bypass this gate.

## Result-contract failures

### `train must first materialize a named corpus and selection`

The selected profile lacks `workspace_contract.json`. First create or import
and verify a compatible corpus, create a selection, then run:

```bash
./dvcs materialize PROJECT --profile PROFILE --corpus CORPUS \
  --selection SELECTION --workers all_available
./dvcs train PROJECT --profile PROFILE
```

One-command local `train --corpus ... --selection ...` remains supported.
Evaluation and later actions consume the materialized result contract.

### Corpus compatibility or verification failure

Run `corpus-plan PROJECT CORPUS`, then `corpus-verify CORPUS --deep`. A changed
prior, physics setting, kinematic point, parameter count/seed, or bridge binary
requires a new corpus. A newly requested admitted observable may be appended
with `corpus-generate`; existing core and observable shards are never edited.
Do not repair a manifest or rename shard files manually.

### Selection already exists or does not match

Selections are immutable because they freeze train/internal-validation/outer-
test ownership. Reuse the exact selection or create another name. If its
profile, group count, or corpus core identity differs, create a compatible new
selection; never edit group lists or move replicas across roles.

### Corpus generation was interrupted

Rerun the same `corpus-generate PROJECT CORPUS` command. Complete hashed shards
are reused and unfinished `.partial` files are not accepted. Inspect the plan
and manifest before removing any work. For quota or inode failures, move the
whole corpus through verified export/import or configure a larger
`DVCS_WORKSPACE` mount and import it there; do not copy a subset of shards.

### `result contract mismatch`

The experiment's scientific configuration, profile, bridge, corpus, selection,
or materialized arrays differ from the saved result lineage. CPU/GPU and
allocated-thread changes alone are permitted and recorded separately. Create
a new project and compatible selection/realization for a scientific change.
Do not copy/edit the contract or force old arrays through a changed method.

### Checkpoint does not resume

Only complete checkpoints matching data/config/member hashes resume. Inspect
the member metrics and training logs. A partially trained ensemble member must
resume on its original accelerator; start a new study to change that training
policy. CPU allocations for later non-training stages may differ.

## CUDA failures

### Explicit CUDA request fails

Check, in order:

1. scheduler/host allocation and `CUDA_VISIBLE_DEVICES`;
2. `nvidia-smi -L` in the host/allocation;
3. Docker toolkit, Podman CDI, or Apptainer `--nv` passthrough;
4. device visibility inside the container;
5. `torch.version.cuda`, `torch.cuda.is_available()`, and device count;
6. host-driver compatibility with CUDA 12.6 wheels;
7. a real tensor/autograd/training smoke.

Do not change required `cuda` to `auto` and report the CPU fallback as GPU
success.

### Multi-GPU rank error

Ensure `CUDA_VISIBLE_DEVICES` contains the allocated device list, `WORLD_SIZE`
matches it, and the profile has at least as many candidate ensemble seeds as
GPUs. Check per-rank runtime records and NCCL diagnostics. Do not launch a
multi-node job; the implemented topology is single-node.

### Multi-GPU Optuna SQLite locking

Preserve every rank log and study file. Confirm all workers share the same
writable filesystem and are from one job, and that no stale unrelated study is
being reused. Persistent repeated locking requires site/runtime investigation,
not deletion of completed trial evidence.

## Neural/statistical failures

### Training NLL is poor or unstable

Inspect train versus internal-validation history, finite inputs, context
normalization, corpus size, architecture, learning rate, and seed-to-seed
variation. A better train loss with worse validation is overfitting. Do not
tune against the outer test or named holdouts.

### Conventional ESS is low

Importance comparison is under-resolved. Wasserstein and width ratios may be
unreliable even if the NPE trained normally. Increase an appropriately designed
exact bank in a new project or improve the conventional proposal; do not lower
the ESS gate post hoc.

### Coverage or predictive gates fail

Treat it as a scientific/method result. Inspect coordinate patterns, Monte
Carlo standard errors, split integrity, invalid reevaluations, predictive
pulls, and ensemble variation. Do not move thresholds to transform failure
into success.

### Posterior does not peak at injected truth

This is not automatically a bug in an underidentified 82-dimensional inverse
problem. Examine credible widths, correlations, repeated coverage, predictive
agreement, and conventional comparison. A falsely narrow wrong posterior is
more concerning than a broad posterior containing truth.

## JLab/SWIF2 failures

### `doctor` reports CPU and no GPU

An ifarm login host normally exposes no allocated GPU. Accept the doctor's GPU
allocation prompt, or set `DVCS_IFARM_GPU_DOCTOR=yes` noninteractively. A
successful rerun must show a compute-node hostname, requested/resolved CUDA,
and at least one visible device. Choosing no correctly diagnoses the current
CPU-only login process; it does not predict a managed GPU job.

### GPU stage exits 69 with no visible allocation

Check `scontrol show job SLURM_ID` or `sacct` for `gres/gpu` in requested and
allocated TRES. SWIF2 GPU jobs must use `--gpus=N`; `--gres=gpu:N` can be
overwritten when SWIF2 adds its own `--gres=disk:...` scratch request. Do not
retry unchanged. Regenerate the workflow with the corrected submitter.

### Training fails with `CUBLAS_STATUS_ALLOC_FAILED`

Compare final cgroup RAM and GPU VRAM telemetry. Current training keeps full
generated arrays memory-mapped on CPU and passes `data_device="cpu"` to SBI so
only minibatches enter CUDA. Use a current image and the 32G initial train RAM
request. An older image that copies complete contexts to CUDA can fill a 24G
GPU before cuBLAS initializes; increasing host RAM alone cannot fix it.

### Apptainer reports `squashfuse_ll` could not read the SIF

Compare local and staged sizes/hashes. V9 local image was 6,936,043,520 bytes
with SHA-256 prefix `8e1b48654be8f243`; SWIF input `117184533` was truncated to
3,098,542,080 bytes with prefix `921c62fd0ed52ef6`. Its SIF descriptor still
expected the full filesystem, so squashfuse correctly refused it.

Current startup verifies staged SIF hash before mounting. For this recovery we
reuse V8's already verified v3 SIF and cached SWIF input; only the separately
staged runner changes. Do not retry V9, whose cached v4 input is truncated.
Check yourself with:

```bash
stat -c '%s' .dvcs/IMAGE.sif
sha256sum .dvcs/IMAGE.sif
stat -Lc '%s %N' /lustre/enp/swif2/jobs/$USER/dvcs-compare/ATTEMPT/dvcs-image-*
sha256sum /lustre/enp/swif2/jobs/$USER/dvcs-compare/ATTEMPT/dvcs-image-*
```

### Invalid account or account/partition combination

Set `JLAB_ACCOUNT` in untracked `resources.env`; verify associations with
`sacctmgr` or JLab support. Confirm current partitions with `sinfo`.

### Dependent jobs never run

Run `swif2 status WORKFLOW -display json` and `swif2 diagnose WORKFLOW`.
Antecedents correctly prevent downstream work after a failed prerequisite.
Fix the cause, then retry only the diagnosed transient problem class with
`swif2 retry-jobs WORKFLOW -problems PROBLEM_CLASS`.

### A job disappeared from `squeue`

`squeue` shows only pending/running Slurm jobs, not complete or failed attempts.
For managed work, use `swif2 status WORKFLOW -jobs -transfers -storage -display
json`, then `swif2 show-job WORKFLOW -name JOB_NAME -display json`. Use the
reported Slurm ID with `sacct` after the attempt ends.

### stdout/stderr is empty or has no progress bar

Batch jobs have no interactive terminal, so the framework uses `--no-progress`
and writes periodic `[dvcs-swif2]` heartbeat lines instead of `tqdm` bars. A job
can also be staging, queued, or waiting on an antecedent before its payload
starts. Check SWIF2 transfers/status and read logs below
`SWIF_LOG_ROOT/WORKFLOW/`; do not look under the legacy
`jobs/jlab_ifarm/logs/` path.

### I cannot find completed outputs

The source workspace is intentionally unchanged. Wait for SWIF2 output
transfers, then inspect `SWIF_OUTPUT_ROOT/WORKFLOW/state/`, `summaries/`, and
`performance/`. Corpus workflows place their complete portable archive under
`SWIF_OUTPUT_ROOT/WORKFLOW/final/`. Farm scratch is disposable and `/farm_out`
contains logs, not authoritative scientific state.

### Batch job cannot find `resources.env`

Normal campaigns should use `./dvcs farm-submit`, which content-addresses and
stages its wrapper instead of sourcing `resources.env` on a compute node. This
error belongs to the retained direct-Slurm diagnostic path.

### PARTONS cannot open `/tmp/..._partons.log`

Older images use the compute node's shared `/tmp`, while PARTONS names its log
by date. Another user can therefore own that day's file. The bridge now writes
disabled-backend logger output beneath the user-owned
`DVCS_CACHE/partons-logs` mount. Rebuild the image after updating, then
resubmit; failed native cache entries are never reused as successful results.

### Job timeout or out of memory

Inspect the stage's reaped `performance-*.json`, summary, heartbeat, and final
`sacct` record. Increase the stage-specific SWIF RAM/walltime request or reduce
justified compute controls, then create a new workflow name. Preserve the last
reaped project-state archive and corpus; retrying an unchanged insufficient
allocation is not recovery. Do not run the workload interactively on ifarm to
evade scheduler limits.

## Plot and real-data diagnostic failures

### Holdout reports `/workspace/configs` is missing

Older images inferred the repository root by walking upward from the staged
project's `.engine/workflow.json`. On SWIF2 that incorrectly produced
`/workspace`, while canonical release configurations live under
`/opt/dvcs/app/configs`.

Current source passes the canonical holdout-manifest path explicitly. The
SWIF2 runner also copies the image-embedded manifest into the legacy lookup
location before running an older image. The manifest's internal SHA-256 check
still runs, so this compatibility step does not change kinematics or physics.
Resume from the last reaped compare archive with a new workflow name and
`--from holdout --through plot`.

### Holdout reports that token count or masked count is incompatible

An older checkpoint accepts only its training width. For Josh this is
`96 kinematics x 6 observables = 576 tokens = 10,377 context features`.
A same- or fresh-design 96-point manifest is width-compatible. A 12/30/60
design requires schema-9 masked materialization and retraining; padding an old
checkpoint would fabricate an unsupported inference contract. Inspect
`generated/observation_design_manifest.json` and the holdout design's
`relationship_to_training` before retrying.

If logs say `staged image-embedded holdout manifest` despite passing
`--holdout-design`, SWIF2 reused an older runner cached under a mutable remote
path. Current submission publishes runner, collector, and design at immutable
content-addressed remote paths. Submit a new workflow name; retrying the old
workflow preserves its stale input identity.

### VGG99 fails while GK11/GK16/GK19 succeed

VGG99 uniquely requires `MSTW2008nlo68cl`. SWIF2 compute nodes use an empty
node-local cache, so the host installation alone is insufficient. Current
submission content-addresses the verified set, stages it only into holdout,
and checks required members before launching Apptainer. The stdout must contain
`staged LHAPDF set MSTW2008nlo68cl`. Resume from the compare archive with a new
workflow name; no retraining is required.

### Compare reports a zero-width conventional stress direction

This is conventional prior-importance collapse, not a PARTONS exception and
not evidence that neural uncertainty is zero. Josh V7 measured:

- 16,384 native vectors x 8 nuisance draws = 131,072 proposals;
- 16,384 posterior resamples;
- ESS 1.00000231, below frozen minimum 80;
- largest normalized weight 0.999998843775;
- one unique resampled proposal/native vector;
- `H_u_normalization` q05=q50=q95=3.219311164780655.

One proposal carried virtually all exact likelihood in 82 dimensions (80
physics + 2 nuisance). Its empirical 90% width was zero, so old code stopped:
`conventional H_u_normalization stress direction has zero width`.

Current code writes failure code
`conventional_prior_importance_sampling_collapsed`, null Wasserstein/width
metrics, audit arrays, and `passed: false`. Holdout is diagnostic-only and
cannot enable robustness; plots label conventional reference unavailable.
Evaluation stays valid. We do not add jitter, substitute epsilon, or lower ESS.
A future adaptive exact-likelihood sampler is needed for a valid reference.

The CLI returns process success after this complete record is written while
retaining `scientific_passed: false`. This distinction lets SWIF2 reap the
state and run diagnostic-only successors. Older code returned exit 1 for the
scientific failure, so SWIF2 incorrectly reported `SLURM_FAILED` and blocked
the workflow even though `comparison_metrics.json` was complete.

Josh V8 also measured 8,190,926,848 peak bytes against 8,192,524,288 requested
(99.98%). Compare now requests 12 GB: 46.5% above this one observed peak and
about 68.3% expected memory efficiency. Re-measure before reducing it.

Check yourself:

```bash
jq '{status,passed,effective_sample_size,conventional_reference_available,conventional_reference_diagnostics}' workspace/PROJECT/results/PROFILE/comparison/comparison_metrics.json
```

Resume from last `project-after-exact-reevaluate.tar` with `farm-submit
--project-archive ... --from compare --through plot`; see
[JLab SWIF2 workflows](JLAB_SWIF2.md#resume-from-a-completed-stage).

### Plot says an input is missing/incompatible

Complete generation, training, evaluation, and comparison for the same
project/profile. Plotting intentionally refuses partial or hash-mismatched
inputs and does not rerun computation.

### Real-data comparison disabled/skipped

Read `real_data_mapping_readiness.json` and the diagnostic's skipped reasons.
The database may be absent, mapping unsupported, or required convention/unit
audits incomplete. This is a safety boundary; do not substitute fixtures or
enable a fit silently.
