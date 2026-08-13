# Troubleshooting

- “no Podman or Docker”: install/configure a rootless engine outside this installer; it never changes the host OS.
- explicit CUDA failure: check `nvidia-smi`, `CUDA_VISIBLE_DEVICES`, container GPU passthrough, PyTorch CUDA runtime, and driver compatibility. Do not switch a required CUDA run to CPU silently.
- PARTONS configuration missing: invoke the installed bridge or `./dvcs`; do not copy the binary away from its properties/schema files.
- LHAPDF set mismatch: preserve the directory for inspection, then point `DVCS_CACHE` at a fresh location and rerun installation.
- database dirty/revision mismatch: use a separate clean checkout at the locked commit; never reset the user's checkout.
- Slurm invalid account: edit `resources.env`; verify membership with JLab support.
- checkpoint accelerator mismatch: resume with the same accelerator policy or start a new project.
