# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-11-19

### Added

- Result contract: `RESULT_SCHEMA_VERSION = "1.1"` exported from the package and stamped on every
  report as `summary["result_schema_version"]`; every run also reports `skipped` and `skip_reason`.
- First-class explanations: `summary["explanations"]` (top 50 flagged rows, with
  `explanations_truncated`), `metadata["top_signals"]` and `metadata["explanation"]` on findings,
  and an `--explain` flag for table/markdown output. JSON always includes explanations.
- Calibration against optional ground-truth columns (`is_bot`, `bot_label`, `label`, `ground_truth`):
  `n`, `brier_score`, `precision_at_threshold`, `false_discovery_rate`, and five reliability bins.
  Reports without ground truth warn that this is an uncalibrated heuristic score.
- `automation_confidence` and `harm_confidence` in explanations and finding metadata. They are
  triage signals, not verdicts: automation alone is not harm.
- Deterministic LSH + token-signature candidate generation for coordinated clustering, replacing
  the unconditional O(n^2) SimHash comparison loop, with a warning above 2000 eligible rows.
- Strict type validation for option values; invalid config values exit 2 from the CLI and degrade
  safely (with a warning) through the library API.

### Fixed

- Entropy screening no longer false-flags CJK/Cyrillic/Arabic/emoji-heavy text; it only runs on
  mostly-ASCII text and reports `summary["entropy_skipped_non_ascii"]`.
- The logistic layer now gives `verified` a negative weight (-3.0), so verified accounts are not
  flagged by that layer.
- `summary["signals"]` counts every row where a signal fired; the old flagged-only counts moved to
  `summary["signals_flagged"]`.
- `threshold` no longer uses a bare `float()` that could raise out of `run()`.
- The "no rows" warning no longer implies text-only rows are skipped; text is always checked.

### Inspiration

- [Bot-detection false-positive discipline](https://fingerprint.com/blog/bot-detection)
- ["Differentiate" good vs bad bots](https://pmc.ncbi.nlm.nih.gov/articles/PMC11958817)
- [Coordinated timing/hashtag signals](https://link.springer.com/article/10.1007/s13278-025-01557-1)
- [CLI-as-API best practices](https://hackmd.io/@arturtamborski/cli-best-practices)

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
