# User guide

Initialize one isolated study, edit only its `experiment.json`, and write each profile to its own results tree:

```bash
./dvcs init study
./dvcs show study
./dvcs doctor study
./dvcs generate study --profile quick
./dvcs train study --profile quick
./dvcs optimize study --profile quick --trials 4
./dvcs evaluate study --profile quick
./dvcs compare study --profile quick
./dvcs holdout study --profile quick
./dvcs compare-real study --profile quick
./dvcs plot study --profile quick
```

Changing an experiment after results exist causes a content-contract failure; create a new study instead. Generation and evaluation use content-addressed native caches and resume complete units. Training resumes only complete, hash-valid member checkpoints. Results are under the configured workspace and result mounts; Slurm provenance is under `results/slurm/<job-id>/`.

`compare-real` does not fit measurements. Database rows select/read kinematics and remain quarantined as documented by the runtime.
