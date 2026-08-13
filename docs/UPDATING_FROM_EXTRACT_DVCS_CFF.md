# Updating from `extract_dvcs_cff`

## Synchronization goal

The public distribution imports only runtime and user-science material from
the validated development repository. It intentionally does not mirror the
development repository. Plans, stage reports/artifacts/status, developer
workspaces, generated results, caches, local builds, credentials, and unrelated
history are out of scope.

The updater is small and auditable: `tools/sync_upstream.py` reads an approved
file list, compares recorded hashes, and refuses to overwrite locally changed
destinations.

## Recorded state

`provenance/upstream-runtime-files.txt` lists every approved relative path.
`provenance/upstream-import.json` records:

- exact upstream commit;
- whether the source worktree was dirty;
- source path used during the import;
- SHA-256 for every imported file;
- mandatory regression commands.

`provenance/distribution-modifications.md` lists packaging-owned changes that
are expected to conflict with a raw future import and therefore require manual
review.

## Pre-import review

In the upstream checkout:

```bash
git status --short
git rev-parse HEAD
git log -1 --oneline
```

Do not import an unexplained dirty worktree. Prefer a clean committed source.
If a reviewed dirty snapshot is genuinely required, record that fact and use
`--allow-dirty`; never let the flag become routine.

Review upstream changes for scientific scope, interface/schema changes,
licenses, data/access implications, and new runtime files. A file is not added
to the allowlist merely because it exists upstream.

## Conflict-checking dry run

```bash
python3 tools/sync_upstream.py \
  --upstream /path/to/extract_dvcs_cff \
  --check
```

The check reports proposed additions/updates and conflicts. A conflict means
the destination no longer matches the hash recorded at the prior import. The
tool will not silently overwrite distribution modifications.

Expected conflicts commonly include the public CLI, device/resource handling,
parallel workflow, native relocation/provenance, CMake packaging, and public
documentation command paths listed in the modifications record.

## Import and resolve

After review, run the same command without `--check`:

```bash
python3 tools/sync_upstream.py \
  --upstream /path/to/extract_dvcs_cff
```

Resolve every conflict manually. Preserve scientific definitions and accepted
thresholds unless the upstream change deliberately and evidentially changes
them. Reapply packaging behavior against the new code rather than copying an
old patch blindly.

If upstream adds a user/runtime file, amend the allowlist only after confirming
it is not development machinery and is necessary for the public runtime.

## Required follow-up

1. Update `COMPATIBILITY.json`, `VERSION`, and `CHANGELOG.md` as appropriate.
2. Update dependency locks, image definitions/locks, license texts, citations,
   and SBOM inputs for changed dependencies.
3. Update every affected documentation page and schema/CLI reference.
4. Regenerate native fixture hashes only after reviewing intentional source
   changes; never use new hashes to bless unexplained output drift.
5. Build/install the bridge from a clean prefix and inspect RUNPATH.
6. Run bridge capabilities/self-test and every retained native physics test.
7. Run targeted and complete applicable Python regressions.
8. Run real CPU generation/training workflow acceptance.
9. Run CUDA/multi-GPU, OCI, Apptainer, and JLab acceptance where applicable.
10. Run clean-clone/static/path/development-artifact checks.
11. Review `git diff` and commit the import plus compatibility evidence.

The prior import record lists mandatory baseline commands; new changes may
require more.

## Versioning guidance

- Patch release: packaging/docs/fix with unchanged public science/schema.
- Minor release: compatible public capability or workflow extension.
- Major release: incompatible experiment/result/native schema or scientific
  interpretation.

Never silently migrate an older experiment/result meaning. Add explicit
conversion with validation or reject the old schema and require a new project.

## Post-import publication

Build new images under new immutable tags, record real registry digests, emit
SBOM/attestations, and rerun installation from those digests. Do not mutate an
existing published tag or describe source-only verification as image
verification.

The unresolved application license remains a publication blocker regardless
of technical readiness.
