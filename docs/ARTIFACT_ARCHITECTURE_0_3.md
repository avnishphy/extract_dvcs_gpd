# Artifact architecture and migration in 0.3

The canonical vocabulary and dependency direction are:

1. **master native physics corpus** (schema 2): immutable PARTONS generator
   coordinates, kinematics, exact CFFs/observables, required canonical GPD
   truth, native status, and provenance;
2. **selection** (schema 1): stable group identities and disjoint
   train/validation/locked-test roles;
3. **pseudodata realization** (schema 3): physical observations, covariance,
   masks, nuisance/noise draws, and seeds, with exact truth in a sealed
   evaluation sidecar;
4. **model view** (schema 1): train-only normalized DeepSets features and DD
   or neural-function latent targets;
5. **result bundle** (schema 2): source IDs, separately named NPE,
   representation, DeepSets, MAF, PARTONS, decoder, seeds, code/image identity,
   resources, and acceptance.

Changing a DeepSets/MAF/decoder/optimizer setting, target transform, seed,
noise realization, diagnostic, or plot never changes the master-corpus ID.
Changing the physical generator prior, kinematics/observables, native physics,
bridge numerical implementation, or requested GPD coordinates does.

New schema-2 corpora cannot be created without an explicit canonical GPD
coordinate request and cannot become complete without aligned, checksummed
GPD-truth shards. Historical schema-1 corpora remain readable for
`dd_deepsets_maf`. They do not
contain architecture-neutral function truth, so `neural_gpd_deepsets_maf`
fails with a precise capability error. A schema-2 manifest with a merely
planned coordinate request also fails: neural use requires complete,
checksummed truth shards and native status masks.

`materialize` is the documented compatibility wrapper for one release. It
publishes a schema-3 realization and DD schema-1 model view as hard-linked,
content-addressed views of the existing generated arrays, so multi-gigabyte
arrays are not copied. The old generated directory remains readable.

`evaluate` is now a saved-artifact-only coverage stage. Use
`exact-reevaluate` to request PARTONS posterior reevaluation explicitly.
Selection, realization, view construction, decoder/MAF work, comparison,
plotting, literature diagnostics, and presentation generation do not resolve
or launch the bridge. Set `DVCS_DISABLE_NATIVE_EXECUTION=1` to enforce this
boundary in downstream tests.

The only registered model families are `dd_deepsets_maf` and
`neural_gpd_deepsets_maf`. Raw NLL values in their different target spaces are
not ranked against one another; use common function/observable residuals,
coverage, constraint residuals, resource use, and explicitly run native
posterior checks.
