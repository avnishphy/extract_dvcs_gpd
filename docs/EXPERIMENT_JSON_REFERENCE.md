# Experiment JSON reference

The editable schema is version 7. Major sections are `experiment`, `injected_truth`, `synthetic_dataset`, `inference`, `output_diagnostics`, and `validation_gates`. Unknown or missing keys fail closed.

Runtime controls are:

- `accelerator`: `auto`, `cpu`, or `cuda`;
- `cpu_threads`: positive integer bounded by the allocation;
- `native_workers`: positive integer or `all_available`, bounded by affinity and task count;
- `deterministic_algorithms`: required to remain true.

Scheduler/container overrides `DVCS_ACCELERATOR`, `DVCS_CPU_THREADS`, and `DVCS_NATIVE_WORKERS` are recorded and supersede the file only for that run. For the complete field-by-field scientific reference, see the imported `user/README.md` and an initialized project's generated comments/readme.
