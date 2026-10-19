# Contributing

Thanks for considering a contribution. This is a small tool on purpose, and
that's a feature worth protecting.

## Ground rules

- **Keep it stdlib-only at runtime.** No third-party dependencies in
  `src/filtergap/`. The hashing and entropy code lives in `stats.py`.
- **A new signal needs a false-positive story.** Every rule you add must come
  with a test showing what it must *not* flag. Spam detection that convicts
  honest users is worse than no detection.
- **No silent behavior changes.** If a weight, threshold default, or
  `PATTERNS_VERSION` changes, that goes in the changelog.

## Workflow

1. Open an issue describing the gap in detection before starting work on
   anything bigger than a typo fix.
2. Fork, branch, make the change, add tests.
3. Run `python -m pytest` and make sure it's green.
4. Open a pull request that says what changed and why, in plain language.

## Tuning weights

Signal weights live in `_score_row` and `ML_WEIGHTS` in `logic.py`. They are
deliberately conservative. If you change one, explain the false-positive
trade-off in the changelog and add a test that pins both the flag and the
non-flag case.
