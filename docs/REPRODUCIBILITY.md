# Reproducibility and provenance

## Reproducibility target

The project aims to make every scientific result traceable to source,
dependencies, configuration, native calls, seeds, resources, and saved
artifacts. It distinguishes several increasingly strong claims:

1. **source reproducibility** — exact commits/files can be recovered;
2. **environment reproducibility** — image/dependency identities match;
3. **workflow reproducibility** — configuration, seeds, and split policy match;
4. **numerical reproducibility** — outputs agree within declared tolerances;
5. **bitwise reproducibility** — every byte is identical.

The first four are supported/audited to the degree documented. Direct and
transitive Python versions are constrained, but bitwise Python environment
reproduction is not claimed because wheel hashes are not locked and GPU
arithmetic may differ across hardware/runtime stacks.

## Source identity

`provenance/upstream-import.json` records:

- upstream repository path used as evidence;
- exact imported commit;
- dirty-state assertion;
- every approved runtime file and SHA-256;
- required regression commands.

`provenance/upstream-runtime-files.txt` is the allowlist.
`provenance/distribution-modifications.md` explains intentional packaging,
resource, relocation, and public-command changes relative to those hashes.

The distribution's own Git commit is recorded by containers and JLab jobs.

## Dependency identity

`provenance/dependencies.lock.json` is the machine-readable authority for:

- Git URLs and exact commits;
- release archive URLs and SHA-256;
- version/build flags;
- base-image digest and Ubuntu APT snapshot;
- Python direct versions, transitive constraint file, and wheel indexes;
- LHAPDF set/member hashes;
- installed native-model source hashes;
- database revision and access/licensing boundary.

The final image also retains a complete `dpkg-manifest.tsv`. The bridge
capabilities response records runtime-observed native/compiler versions.

## Image identity

`provenance/images.lock.json` maps variants to version tags and registry
digests. A digest, not a tag, is the immutable identity. Null digests mean no
published image has been verified and force a source build.

For an accepted run retain:

```text
registry/repository@sha256:...
SIF SHA-256, when applicable
distribution Git commit
image lock
SBOM and build attestation
dpkg/Python package manifests
```

## Configuration identity

The public `experiment.json` is translated into canonical `.engine` files.
Corpus creation hashes the canonical native configuration and exact bridge.
The corpus manifest additionally inventories every atomic native shard and
evidence archive. Selection creation binds immutable group roles to the corpus
core identity. Corpus-backed materialization then writes
`workspace_contract.json` and a realization manifest binding configuration,
bridge, corpus, selection, and every array. All later steps recompute and
compare the result contract.

This prevents accidental reuse after edits, but users must still retain the
public experiment because it explains intent and edit classifications.

## Seed lineage

Independent declared seeds control:

- native prior parameter draws, derived deterministically per corpus shard;
- nuisance/noise replicas, derived from immutable group and replica indices;
- displayed pseudodata noise;
- conventional nuisance proposals;
- posterior comparison/resampling projections;
- coverage selection (reserved by current selector behavior);
- Optuna sampling;
- every candidate neural ensemble member.

The workflow uses deterministic NumPy/Torch streams and requires
`deterministic_algorithms: true`. Changing one seed defines a new experiment
configuration and content hash.

Multi-GPU training assigns existing member seeds to ranks; the rank does not
replace the member seed. This is intended to preserve model identity across
one- and multi-GPU scheduling.

## Native determinism and cache

The bridge returns deterministic JSON for an exact request/backend. Cache
metadata hashes the request, response, and executable and records that no
surrogate was used. Retained native source fixtures are listed in
`provenance/native-fixtures.sha256` and native tests preserve their established
numerical thresholds.

`--self-test` verifies one GK16 reference at `5e-13` absolute tolerance. The
larger retained physics executables validate DD normalization/support,
evolution behavior, basis mixing, conformal reconstruction, and shadow
composition.

A corpus is reproducible only as a complete verified object: retain
`corpus.json`, all core/observable shards, and all declared evidence archives.
Use `corpus-export` for transfer; its archive is deep-verified before creation,
and import verifies the archive again. Preserve the project-local selection
JSON separately. A corpus alone does not identify the neural data split or
noise realization.

## Arrays, checkpoints, and plots

Numerical arrays are saved without pickle and inventoried by dtype, shape, and
hash. Checkpoints are tied to member seed, configuration, and training data.
Plots have a manifest containing all required input hashes and output PNG
hashes. Summaries should never be separated from their contracts/manifests.

## Resource and hardware identity

CPU affinity, Slurm CPU request, used Torch threads, native worker request,
math-library threads, accelerator policy, visible GPUs, Torch/CUDA versions,
host, rank/world size, image digest, and Git commit are written at runtime.

For GPU comparison also record model name, driver, compute capability,
interconnect/NCCL version, and deterministic-algorithm warnings. A nominally
identical CUDA wheel on different GPUs does not guarantee bitwise identity.

## Reproducing a run

1. Check out the recorded distribution commit.
2. Obtain the recorded image digest or rebuild from the dependency lock.
3. Verify the SIF/image and native bridge capabilities/self-test.
4. Restore the exact public project, verified corpus, immutable selection, and
   database/LHAPDF identities.
5. Use the recorded mount layout or an equivalent one.
6. Match accelerator, visible devices, affinity, and scheduler resources.
7. Run `doctor` and compare its resolution with the saved record.
8. Verify the corpus deeply, then run the same profile/actions with the
   recorded corpus and selection names without changing the experiment.
9. Compare contracts and hashes first, then numerical metrics/tolerances.
10. Explain any expected nondeterminism instead of silently accepting drift.

## Offline reproduction

Once image and mounted data are installed, runtime actions require no network.
Use engine network-disable options and execute bridge self-test plus a cached or
fresh local workflow. Record whether a cache was already populated: offline
execution and offline installation are different claims.

## Verification commands

```bash
./tests/run.sh static
DVCS_BRIDGE=/opt/dvcs/bin/partons_bridge ./tests/run.sh native
./tests/run.sh quick
./tests/run.sh offline
sha256sum -c provenance/native-fixtures.sha256
```

Container, CUDA, Apptainer, and JLab procedures are in
[Acceptance](ACCEPTANCE.md). Unavailable hardware is recorded as unverified,
not skipped-and-passed.

## Publishing reproducible results

At minimum archive:

- paper/analysis claim and result interpretation;
- distribution commit and upstream import record;
- dependency and image locks plus actual digest/SIF hash;
- SBOM/package manifests;
- `experiment.json`, engine/workspace contracts, and bridge capabilities;
- generated/training/evaluation/comparison and each named `holdouts/DESIGN/`
  summary/manifest;
- seeds, checkpoints, array manifests, invalid maps, and plot manifest;
- runtime/Slurm/hardware provenance;
- exact commands and exit statuses;
- known limitations and any changed validation policy.
