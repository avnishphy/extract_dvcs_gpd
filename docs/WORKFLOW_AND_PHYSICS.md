# Workflow and physics

The authoritative path is

```text
experiment JSON -> isolated C++ bridge process -> PARTONS services/modules
                -> synthetic observations -> sbi NPE -> validation/plots
```

The bridge retains the upstream definitions for the four twist-2 GPDs (H, E, \widetilde H, \widetilde E), flavor/channel parameterization, CFFs, six DVCS observables, covariance, priors, and validation gates. This packaging layer does not reproduce those equations in Python.

The NPE targets (q_\phi(\theta,\eta\mid D,m)), where (\theta) contains GPD controls, (\eta) nuisance parameters, (D) the synthetic dataset, and (m) the model/context indicators. PARTONS generation is CPU-only. CUDA accelerates eligible PyTorch training and sampling only.

Native parallelism means one independent bridge process per worker; no PARTONS object is shared. The affinity/cgroup mask and `SLURM_CPUS_PER_TASK` bound worker use. BLAS/OpenMP defaults are one thread per process to prevent oversubscription.

Verified science claims and detailed conventions remain in `user/WORKFLOW_AND_PHYSICS.md` and `user/PHYSICS_AND_RESULTS_GUIDE.md` imported by hash from the validated upstream.
