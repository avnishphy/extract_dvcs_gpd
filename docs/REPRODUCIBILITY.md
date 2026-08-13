# Reproducibility

Provenance consists of the dependency lock, upstream import state, image lock/digest, distribution commit, bridge capability response, exact experiment JSON, content-addressed native cache, seed/checkpoint hashes, resource record, and hardware/environment record.

The source commits and release archives are immutable and verified. The base image is digest pinned, and APT uses Ubuntu archive snapshot `20260801T000000Z`; each image retains its complete `dpkg-manifest.tsv`. Python direct dependencies are exact-version pinned, but a published release must add wheel hashes produced by CI before claiming bit-for-bit Python environment reproduction.

After installation, run offline acceptance with `./tests/run.sh offline`. Native fixtures retain their upstream numerical tolerances; packaging does not relax them.
