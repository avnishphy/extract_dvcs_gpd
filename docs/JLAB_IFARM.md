# JLab ifarm

Run installation and submission on ifarm, but run sustained work through Slurm. The templates use `production` for CPU work and `gpu` for neural work. These names and the required account directive were checked against JLab SciComp documentation on 2026-08-12; confirm them with `sinfo` because site policy changes.

```bash
./install.sh --profile jlab_ifarm --accelerator auto
vi jobs/jlab_ifarm/resources.env
./dvcs init my-study
jobs/jlab_ifarm/submit_workflow.sh
```

Generation and holdout are PARTONS-heavy CPU jobs. Training and Optuna are neural GPU jobs. Evaluation may use both. Templates specify nodes, tasks, CPUs, memory, wall time, partition, GPU count, and placeholder account. The submission wrapper supplies the account from the untracked `resources.env` and records every job ID.

Apptainer runs with `--nv` only for CUDA. It never needs a Docker daemon on a compute node. Put `DVCS_CACHE` and `DVCS_RESULTS` on a user/group project filesystem suitable for large files; do not assume another user's path. Each job retains environment, hardware, image lock/digest, Git commit, Slurm record, stdout, stderr, and exit status.

Site references: [JLab partitions and resources](https://scicomp.jlab.org/docs/node/644), [GPU jobs](https://scicomp.jlab.org/docs/node/631), and [Slurm FAQ](https://scicomp.jlab.org/docs/farm_slurm_faq).
