# Updating from extract_dvcs_cff

1. Review upstream status and commit. Never import an unexplained dirty tree.
2. Run `python3 tools/sync_upstream.py --upstream /path/to/extract_dvcs_cff --check`.
3. Review every changed approved file and amend `upstream-runtime-files.txt` only for an intentional public runtime addition.
4. Import with the same command without `--check`; add `--allow-dirty` only when a reviewed snapshot must be recorded.
5. Resolve conflicts manually. The tool refuses to overwrite any destination changed since the previous import.
6. Update `COMPATIBILITY.json`, `CHANGELOG.md`, and dependency/image locks.
7. Run the mandatory static, native, and quick regression commands recorded in `upstream-import.json`.

The manifest excludes plans, stage reports/artifacts/status, developer workspaces, caches, generated results, ADR history, and credentials by construction.
