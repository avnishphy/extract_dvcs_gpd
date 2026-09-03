# ADR 0005: DeepSets plus MAF scope

Status: accepted, 2026-08-31.

Neural posterior estimation uses a permutation-invariant DeepSets observation
encoder and conditional MAF (`zuko_maf`). NSF/spline, Transformers, GNNs,
diffusion, and alternative-flow comparisons are out of scope. Spline-only
`num_bins` fields are rejected or explicitly migrated with a warning.
