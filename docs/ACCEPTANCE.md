# Acceptance procedures

## Local CPU

```bash
./tests/run.sh static
./install.sh --profile local --accelerator cpu
./dvcs init acceptance
./dvcs doctor acceptance
./dvcs corpus-create acceptance acceptance-corpus --profile quick
./dvcs corpus-plan acceptance acceptance-corpus
./dvcs corpus-generate acceptance acceptance-corpus
./dvcs corpus-verify acceptance-corpus --deep
./dvcs selection-create acceptance acceptance-corpus baseline --profile quick
./dvcs train acceptance --profile quick \
  --corpus acceptance-corpus --selection baseline
./dvcs evaluate acceptance --profile quick
./dvcs plot acceptance --profile quick
./tests/run.sh offline
```

## Local CUDA

```bash
./install.sh --profile local --accelerator cuda
./dvcs init acceptance-cuda
./dvcs doctor acceptance-cuda
./dvcs corpus-create acceptance-cuda acceptance-cuda-corpus --profile quick
./dvcs corpus-plan acceptance-cuda acceptance-cuda-corpus
./dvcs corpus-generate acceptance-cuda acceptance-cuda-corpus
./dvcs selection-create acceptance-cuda acceptance-cuda-corpus baseline --profile quick
./dvcs train acceptance-cuda --profile quick \
  --corpus acceptance-cuda-corpus --selection baseline
nvidia-smi
```

For multi-GPU acceptance, allocate at least two visible GPUs and use an experiment with at least as many ensemble seeds as GPUs. Compare state hashes and score distributions with a same-seed one-GPU run; do not mark statistical equivalence from device detection alone.

## JLab ifarm/farm

Run exactly these commands after replacing the repository URL and user paths:

```bash
git clone <repository-url> extract_dvcs_gpd
cd extract_dvcs_gpd
./install.sh --profile jlab_ifarm --accelerator auto
cp -n jobs/jlab_ifarm/resources.env.example jobs/jlab_ifarm/resources.env
vi jobs/jlab_ifarm/resources.env
./dvcs init ifarm-acceptance
apptainer inspect .dvcs/*.sif
./dvcs doctor ifarm-acceptance
./dvcs corpus-create ifarm-acceptance ifarm-acceptance-corpus --profile validation
./dvcs corpus-plan ifarm-acceptance ifarm-acceptance-corpus
sinfo -o '%P %G %c %m %l %f'
swif2 list -display json
./dvcs farm-corpus-submit --project ifarm-acceptance --corpus ifarm-acceptance-corpus --profile validation --dry-run
./dvcs farm-submit --project ifarm-acceptance --corpus ifarm-acceptance-corpus --selection baseline --profile validation --corpus-archive /absolute/corpus.tar.gz --dry-run
```

Remove `--dry-run` only after reviewing the generated requests and paths. Add
`--include-optimize` only when an Optuna campaign is intended.

Answer yes when the ifarm GPU-doctor prompt appears. In noninteractive
acceptance, run
`DVCS_IFARM_GPU_DOCTOR=yes ./dvcs doctor ifarm-acceptance` instead.

Before submitting the full workflow, run the allocated GPU smoke test in
[JLab ifarm and farm guide](JLAB_IFARM.md#interactive-setup-and-checks). It follows
JLab's [GPU access instructions](https://scicomp.jlab.org/docs/Access_GPUs)
and verifies `nvidia-smi`, the scheduler's `CUDA_VISIBLE_DEVICES`, Apptainer
`--nv` passthrough, and PyTorch CUDA detection inside the SIF.

After completion:

```bash
sacct --format=JobID%18,State,ExitCode,Partition,AllocCPUS,ReqMem,Elapsed -j <job-ids>
find "$DVCS_RESULTS/slurm" -type f -maxdepth 4 -print
apptainer exec --cleanenv --network none \
  --bind "$DVCS_WORKSPACE:/workspace" --bind "$DVCS_RESULTS:/results" \
  --bind "$DVCS_CACHE:/cache" --bind "$DVCS_DATABASE:/database:ro" \
  .dvcs/*.sif /opt/dvcs/bin/partons_bridge --self-test
```

Record the actual SIF checksum, OCI source digest, NVIDIA driver/GPU, CUDA runtime, job IDs, exit codes, and filesystem selected. JLab acceptance is not complete until these commands run on site.
