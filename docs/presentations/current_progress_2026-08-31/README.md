# Current progress snapshot — 2026-08-31

This is a one-time, immutable presentation snapshot. Its figures were copied
from the `experiment-josh-validation-same96-v22` post-plot bundle and the
same-vs-fresh holdout comparison PDF; the source result bundles are unchanged.
Nothing in CI, hooks, or the documentation build regenerates this directory.

Build and verify from this directory:

```bash
make
make verify
```

The build requires `pdflatex` and `bibtex`. `snapshot_manifest.json` records
the source hashes, asset hashes, toolchain, and unavailable expected evidence.
