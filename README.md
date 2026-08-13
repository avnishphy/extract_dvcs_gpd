# extract-dvcs-gpd

Containerized, non-root distribution of the validated `extract_dvcs_cff` DVCS/GPD pseudodata and neural-posterior workflow. The C++ PARTONS bridge is the sole physics backend; Python handles orchestration and statistical inference only.

## Five-minute local start

```bash
git clone <repository-url> extract_dvcs_gpd
cd extract_dvcs_gpd
./install.sh --profile local --accelerator auto
./dvcs init first-study
./dvcs doctor first-study
./dvcs generate first-study --profile quick
./dvcs train first-study --profile quick
./dvcs evaluate first-study --profile quick
./dvcs plot first-study --profile quick
```

`install.sh` uses a digest-pinned published image only after a maintainer records its registry digest. Until then it performs the reproducible locked source build. It never invokes a host package manager. Workspaces, results, caches, and database data remain under bind-mounted user-owned directories.

For JLab:

```bash
./install.sh --profile jlab_ifarm --accelerator auto
cp jobs/jlab_ifarm/resources.env.example jobs/jlab_ifarm/resources.env
# edit paths and JLAB_ACCOUNT, then initialize a project and submit
./dvcs init first-study
jobs/jlab_ifarm/submit_workflow.sh
```

Read [Installation](docs/INSTALLATION.md), [JLab ifarm](docs/JLAB_IFARM.md), and the [User guide](docs/USER_GUIDE.md). The exact dependency pins are in [provenance/dependencies.lock.json](provenance/dependencies.lock.json). No image has been published from this local repository; null digests are never presented as verified.

## Status

- Verified upstream runtime: clean committed snapshot `7d690f69af60082d0bca8eb22a3a44144a98cd30`; uncommitted upstream and database content was not packaged.
- Verified locally: Python syntax/import-independent tests, installer dry runs, resource policy tests, manifest/path hygiene, and existing native bridge execution against the audited development libraries.
- Conditional: complete OCI source build, non-root OCI execution, CUDA wheel/runtime, Apptainer, Slurm, and JLab filesystem behavior require runtimes/hardware absent from this machine.
- Publication blocker: the upstream application/bridge has no repository-level license. Do not publish source or images until its copyright holders provide a compatible license; see [Known limitations](docs/KNOWN_LIMITATIONS.md).

## Scientific boundary

PARTONS stays CPU-native and is evaluated in isolated subprocesses. CUDA is used only by eligible PyTorch/sbi neural work. Real database measurements remain quarantined from fitting in this release; `compare-real` is diagnostic only. See [Workflow and physics](docs/WORKFLOW_AND_PHYSICS.md).
