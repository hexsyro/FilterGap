# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-10-19

### Added

- Initial release.
- Account heuristics: new-account age, posting frequency, link density, low-follower/high-frequency combinations.
- Shannon-entropy screening for machine-random text.
- Duplicate detection via flags or references to an original post.
- Coordinated-campaign clustering with SimHash fingerprints and token-set Jaccard (3+ members).
- Logistic layer combining weak signals into a 0–1 bot probability.
- Tunable `--threshold` with per-finding signal breakdown.
- CLI with table/json/csv/md, --fail-on exit codes, JSON config.
- Importable library API: filtergap.run(rows, options).
