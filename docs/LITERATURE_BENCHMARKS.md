# Literature benchmarks

The versioned authority is
`configs/literature/benchmarks_v1.json`. It inventories the result-figure
families in Xu et al. (NNGPD, arXiv:2605.06994) and Moffat et al.
(arXiv:2303.12006), plus the earlier Bertone et al. shadow study. Original
arXiv records are citations; no paper figures or digitized curves are stored.

The NNGPD-inspired branch in this repository is a latent-function
representation inside DeepSets-conditioned neural posterior estimation. It is
not a reproduction of the paper's BNN objective and is not fully
model-independent. On a DD-generated master corpus it is a closure and
representation test over a DD-induced truth distribution.

`extract_dvcs_cff.literature.plot_function_closure` creates a compatible
function-space diagnostic from stored native truth and saved posterior
functions. Before any external numerical overlay,
`assert_conventions_compatible` requires exact agreement in GPD, flavor
combination, normalization, scale, scheme, order, and sign convention.

Current Stage-09 shadow checks remain fixed stress tests. They do not implement
the Moffat type-A/type-B bases, numerical null-space searches, or the paper's
complete evolution/identifiability program. The registry therefore marks those
figures as awaiting a compatible corpus or not scientifically comparable.

To emit requirements without changing a corpus configuration:

```bash
PYTHONPATH=src python -c 'from pathlib import Path; from extract_dvcs_cff.literature import write_coverage_requirements; write_coverage_requirements(registry_path=Path("configs/literature/benchmarks_v1.json"), corpus_manifest_path=Path("CORPUS/corpus.json"), output=Path("coverage_requirements.json"))'
```

The output explicitly records `corpus_configuration_modified: false`.
