# Security and data handling

## Privilege boundary

The installer never invokes `sudo`, a host package manager, or a privileged
Docker operation. Local containers run with the invoking UID/GID; Apptainer
runs as the invoking user. Dependencies are installed inside an image or
user-owned cache/prefix.

Review container definitions before building untrusted revisions: image build
steps execute downloaded source build systems. Exact commits/checksums reduce
substitution risk but do not replace code review or vulnerability scanning.

## Download integrity

- Release and data archives use HTTPS plus declared SHA-256.
- Git dependencies are checked out at exact commits.
- Published images must be pulled by registry digest, not tag alone.
- SIF files require an independently recorded SHA-256.
- SBOMs and build attestations are required for promotion where practical.

A checksum proves identity with the locked object, not that the object is free
of vulnerabilities or malicious code.

## Filesystem and project isolation

Project names are confined to one canonical workspace root and reject path
traversal/symlink escape. Workspaces, results, and caches are writable mounts;
the database is read-only. The framework does not delete user projects or
clean dirty database checkouts.

Corpus and selection names obey the same direct-child naming boundary. Corpus
creation/import refuses existing targets and symlink roots. Export deep-
verifies content and rejects symlinks. Import rejects absolute paths, parent
traversal, links, devices, FIFOs, and other special archive members before
extracting to a partial directory, then deep-verifies before publication.
Treat a portable corpus archive as untrusted until that import succeeds.

Generated results may be scientifically valuable and large. Apply site backup,
quota, retention, and access-control policy. Container isolation is not a
substitute for filesystem permissions between collaborating users.
Corpora are reusable scientific assets rather than disposable caches; preserve
their manifest, evidence archives, and selection lineage together with access
controls appropriate to the collaboration.

## Credentials and secrets

Do not commit:

- `jobs/jlab_ifarm/resources.env`;
- registry tokens or GitHub credentials;
- private repository/database credentials;
- Slurm/account secrets;
- unpublished experiment data;
- user paths/environment dumps that violate policy.

GitHub Actions publication uses the scoped repository `GITHUB_TOKEN`. The
workflow should keep least-privilege `contents: read`, `packages: write`, and
attestation permissions. Protected environments/manual approval are advised
before enabling publication.

## Database and experimental data

The installed database is a distinct pinned checkout. Initialization reads a
measurement-free catalog of kinematics and metadata. Synthetic training sets
explicit flags proving measurement values/uncertainties were not used.

The real-observable diagnostic reads an allowlisted source from the read-only
database and retains its source hash/revision. It cannot call a training,
optimization, likelihood, or posterior-update path. This software boundary
does not supersede collaboration, citation, privacy, export, or dataset license
requirements.

## Untrusted artifacts

JSON and non-pickled NumPy arrays are preferred. Neural checkpoints and SQLite
studies can still be unsafe or misleading when obtained from an untrusted
source. Do not load externally supplied checkpoints/studies into a trusted
environment without provenance and format/security review.

Never edit a native response, checkpoint metric, or validation summary to make
it pass. Preserve failures and invalid maps as evidence.

## Container hardening

Production operation should:

- run non-root;
- mount only required paths;
- keep database read-only;
- disable network after installation where possible;
- avoid host socket/device mounts except allocated GPU devices;
- scan the exact digest/SIF before promotion;
- rebuild regularly against reviewed security updates;
- retain immutable prior digests for reproducibility.

APT snapshots improve reproducibility but freeze known vulnerabilities too.
Security maintenance should advance the snapshot deliberately, rebuild, scan,
rerun regressions, and publish a new immutable version rather than mutating an
old tag.

## Incident response

If a dependency/image/cache may be compromised:

1. stop using and publishing the affected digest;
2. preserve hashes/logs without executing suspect artifacts;
3. identify source, build, registry, and affected result lineage;
4. rotate exposed credentials;
5. rebuild from independently verified sources in a clean environment;
6. rerun native/workflow regression and compare outputs;
7. issue a new version/digest and document affected prior results.
