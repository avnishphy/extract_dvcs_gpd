# Resource management

The launcher intersects Linux process affinity with `SLURM_CPUS_PER_TASK`. `all_available` native workers never exceed that affinity and never share a PARTONS instance. `OMP_NUM_THREADS`, OpenBLAS, MKL, NumExpr, and vecLib default to one thread; set `DVCS_MATH_THREADS` only after accounting for worker multiplication.

`CUDA_VISIBLE_DEVICES` is authoritative. `cuda` is an assertion and fails closed. `auto` may fall back to CPU, recording the reason. Apptainer gets `--nv` only for a resolved CUDA install.

When multiple GPUs are visible for `train`, `torchrun` starts one NCCL rank per device and deterministically shards independent ensemble members across ranks. Each checkpoint remains an ordinary single-device state dict; rank 0 performs the unchanged validation-only selection after all ranks finish. Multi-GPU Optuna partitions the requested trials across one process per allocated device and coordinates through the persistent study. Both implementations are statically tested, but hardware/statistical equivalence remains unverified.
