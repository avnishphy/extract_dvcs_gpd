# Security and data

The installer never uses sudo or a host package manager. Containers run non-root and mount the database read-only. User workspaces are constrained to one real, non-symlink root; project names reject traversal. Dependency downloads use HTTPS and declared checksums or exact reachable Git commits.

Do not commit `resources.env`, tokens, registry credentials, experiment secrets, database modifications, generated results, or caches. CI publication uses GitHub's scoped token and produces attestations/SBOMs. Treat imported experimental data according to source licenses, collaboration policy, and citation requirements.

Images should be rebuilt regularly for security fixes; immutable old digests remain reproducible but may contain known vulnerabilities. Scan before promotion.
