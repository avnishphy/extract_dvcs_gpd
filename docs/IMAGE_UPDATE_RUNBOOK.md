# Image update and recovery runbook

This page is the operational authority for changing or rebuilding a container
image. Normal users should run the ordinary installer command and should not
force a source build. Maintainers should use this runbook so a failed update
does not displace the known-good production SIF.

## Decide whether a rebuild is required

Run from the repository root. Start by identifying what changed:

```bash
git status --short
git diff --name-only OLD_COMMIT..HEAD
./install.sh --profile jlab_ifarm --accelerator auto --dry-run
```

Replace `OLD_COMMIT` with the commit used by the current production image. The
dry run is read-only and reports whether the installer would verify, resume,
pull, or build an image.

| Situation | Supported action |
|---|---|
| No image-impacting source changed | Run the ordinary installer; it verifies and reuses the current SIF. |
| Only documentation or site resource defaults changed | Run static tests and the ordinary installer; do not rebuild merely to refresh timestamps. |
| `src/`, `cpp/`, `pyproject.toml`, a container definition, an in-image build/runtime script, or a dependency lock changed | Run one intentional `--source-build` after the pre-update checks below. |
| A complete definition-matched `.sif.partial` exists after interruption and no other in-image source changed afterward | Run the ordinary installer without `--source-build`; it verifies the image labels/embedded definition and resumes testing/promotion. |
| A source build failed before producing a complete SIF | Keep using the current final SIF. Diagnose the retained log and retry `--source-build` only after the cause is resolved. |
| An approved immutable registry digest is available | Run the ordinary installer and let it pull by digest. Do not replace the digest with a mutable tag. |

If there is doubt about whether a changed file is copied into or executed by
the image, treat it as image-impacting. A normal installer run deliberately
does not claim that an existing final SIF contains unversioned development
changes; maintainers must request a source build for those changes.

Partial verification checks image labels and the embedded Apptainer definition;
it does not prove that every copied source file matches a later checkout. If
`src/`, `cpp/`, or another in-image input changed after the partial was built,
preserve that candidate for evidence and perform a new controlled source build.

## Preserve the known-good state

Before an intentional update, verify the current production installation and
record its identity:

```bash
./install.sh --profile jlab_ifarm --accelerator auto
./tests/run.sh offline
source .dvcs/install.env
git rev-parse HEAD
sha256sum "$DVCS_IMAGE"
apptainer inspect "$DVCS_IMAGE"
```

The ordinary installer must pass before starting a replacement build. Record
the commit and SIF SHA-256 with the campaign or release notes. Existing corpus,
project, and result directories are not image rollback copies; never alter
them as part of image recovery.

For a high-value production transition, a maintainer may preserve the current
SIF under a clearly named path in `.dvcs/rollback/`. A same-filesystem hard link
preserves the old inode without duplicating the approximately 7 GB payload:

```bash
mkdir -p .dvcs/rollback
ln "$DVCS_IMAGE" ".dvcs/rollback/$(basename "$DVCS_IMAGE").before-UPDATE"
sha256sum "$DVCS_IMAGE" ".dvcs/rollback/$(basename "$DVCS_IMAGE").before-UPDATE"
```

Replace `UPDATE` with a release or date identifier. Both hashes must match.
If hard links are unavailable, check quota before making a full copy. Do not
point production at the backup by hand; restore installer state deliberately
and reverify it if rollback is required.

## Validate the change before building

```bash
./tests/run.sh static
git diff --check
git diff -- containers install.sh scripts pyproject.toml provenance docs README.md
```

`static` checks the distribution, documentation links/contracts, container
contract, Python regressions, JLab wrappers, bounded APT retry behavior,
partial-SIF resume behavior, shell syntax, and non-mutating installer dry runs.
A static pass does not prove that PARTONS compiled or that Apptainer works; the
source build and post-build checks provide that evidence.

Do not clear the wheelhouse or engine cache as a routine first step. Verified
GSL, LHAPDF, and PyTorch downloads are reusable and avoid repeating fragile
network transfers. Use a clean cache only when testing cache independence or
when a checksum check proves that a cached object is unusable.

## Run one controlled source build

On the ifarm control host:

```bash
./install.sh --profile jlab_ifarm --accelerator auto --source-build
```

The installer keeps the final SIF in place while building
`.sif.partial`. It probes the pinned Ubuntu snapshot before starting the
expensive build, uses bounded retries, records the build under
`.dvcs/build-logs/`, and atomically promotes the candidate only after all
checks pass.

Monitor from another terminal without starting a second installer:

```bash
ls -lt .dvcs/build-logs/
tail -f .dvcs/build-logs/LOG_FILE
cat .dvcs/install.lock.d/owner
```

Replace `LOG_FILE` with the newest file shown by `ls`. Long native compilation,
Python wheel installation, and SIF compression can be quiet; quiet output by
itself is not a failure. On 2026-09-01 the complete JLab build took about nine
minutes and the ordinary verified-image path about 52 seconds on `ifarm2401`.
Those measurements are diagnostic baselines, not time guarantees.

## Understand the two Apptainer test phases

Apptainer automatically executes `%test` while the build root filesystem is
read-only. The expected automatic-build sequence includes:

```text
INFO:    Running testscript
native self-test deferred until writable /cache is bound
...
OK
INFO:    Creating SIF file...
```

Directory permissions cannot make image-internal `/cache` writable in that
phase. The definition therefore runs read-only capabilities/import/contract
checks and defers only the logger-writing native self-test.

After SIF creation, `install.sh` runs a second, mandatory test with the
user-owned cache bound at `/cache`:

```bash
source .dvcs/install.env
apptainer test --bind "$DVCS_CACHE:/cache" "$DVCS_IMAGE"
```

The second phase must run the native self-test and must not print the defer
message. Both phases use fail-fast behavior so a native abort cannot be hidden
by a later successful Python test.

## Accept the new image

After a successful build, run and record:

```bash
source .dvcs/install.env
./scripts/verify-apptainer-image.sh \
  apptainer "$DVCS_IMAGE" 0.3.0 containers/apptainer.def
apptainer test --bind "$DVCS_CACHE:/cache" "$DVCS_IMAGE"
sha256sum "$DVCS_IMAGE"
./install.sh --profile jlab_ifarm --accelerator auto
./tests/run.sh offline
```

Update the version argument when the distribution version changes. Acceptance
requires all of the following:

- embedded title/version and `containers/apptainer.def` hash match;
- native self-test and 18 architecture tests pass with writable cache;
- `pip check` reports no broken requirements;
- the JLab image records a non-null PyTorch CUDA runtime;
- `.dvcs/install.env` contains the observed SIF digest;
- no `.sif.partial` or `.dvcs/install.lock.d` remains;
- the ordinary installer reuses and verifies the promoted image.

Then update the dated verification record, dependency/image locks when their
authoritative values changed, and release/campaign notes. Do not describe a
source build as accepted from SIF creation alone.

## Failure recovery without losing production

### Ubuntu snapshot or download failure

An HTTP 503 is a transient service failure, not permission to alter the pinned
snapshot. Keep the current final SIF. Inspect the newest build log and rerun the
ordinary installer first. It resumes a complete matching partial; otherwise it
verifies the current production image. Retry an intentional source build later
only if the update is still required.

Do not add `apt --fix-missing`, increase retries without a bound, change the
snapshot timestamp, disable checksums, or replace locked sources.

### Installer lock is present

Inspect the recorded owner before taking action:

```bash
cat .dvcs/install.lock.d/owner
pgrep -af 'install.sh|apptainer.*build'
```

If the owner host and PID are live, wait for that installer. If the owner is on
another host, verify it there. Only after confirming that the recorded process
and every related Apptainer build are gone, preserve and release a stale lock
with:

```bash
mv .dvcs/install.lock.d \
  ".dvcs/install.lock.d.stale.$(date -u +%Y%m%dT%H%M%SZ)"
```

Moving is recoverable and preserves the owner evidence. Never remove a lock
merely because a build is quiet.

### Candidate test fails

The final production SIF remains available because promotion occurs only after
verification. Preserve the `.sif.partial`, terminal output, and newest build
log. Correct the definition or installer contract, rerun static tests, then run
one new intentional source build. Do not rename an unverified partial into the
final path.

### PARTONS logger cannot write

If the message occurs during the automatic read-only test, confirm the current
definition prints the documented defer line. If it occurs during the external
writable-cache test, inspect ownership and write access of
`$DVCS_CACHE/partons-logs`; do not weaken or skip the native self-test.

## Update completion record

Every accepted image update should record:

- distribution commit and changed-file classification;
- exact installer command and host;
- build-log path and elapsed time;
- dependency lock and snapshot identities;
- final SIF SHA-256 and embedded-definition verification;
- native, Python, CUDA-runtime, static, and offline test results;
- whether the previous SIF was retained and where;
- any transient failures and the bounded recovery used.

This record separates an observed, accepted runtime from a candidate that only
finished compiling.
