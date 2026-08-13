# Dependencies

The machine-readable authority is `provenance/dependencies.lock.json`. PARTONS 5.0.0, NumA++ 5.0.0, APFEL++ source-version 4.8.0, ElementaryUtils 5.0.0, LHAPDF 6.5.6, and GSL 2.8 are source built. Boost JSON, CLN, SFML-system, Eigen, LibXml2, GCC/GFortran, CMake, and Python come from the base distribution ABI.

Only `MSTW2008nlo68cl` is required, and only for the VGG99 holdout. Its archive, metadata, and central-member hashes are locked.

`gpddatabase` v1.1.3 at commit `1e9e97f` is public and declares GPL-3.0, while its README additionally says non-profit scientific use. The installer obtains a separate clean checkout and mounts it read-only; it is not embedded in images. Dataset-level citation/redistribution terms need upstream review.

PARTONS, APFEL++, NumA++, LHAPDF, GSL, and gpddatabase are GPL-family; ElementaryUtils is Apache-2.0. Exact notices are copied from their upstream repositories under `licenses/` during release preparation. The application itself currently lacks an upstream license grant, blocking public publication.
