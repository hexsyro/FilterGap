# FilterGap

Cheap social-listening tools let bots through. FilterGap scores each mention
with layered signals so you can catch the spam your listening tool counted as
engagement.

## Why this exists

Budget-tier listening platforms count a mention as a mention. A brand-new
account posting 300 times a day with a link and two followers looks exactly
like a customer until you look closer. FilterGap looks closer: account
heuristics, text randomness, duplicate flags, and coordinated-campaign
clustering, folded into a single 0–1 confidence per mention.

## What it does

- **Account heuristics.** New-account age, posting frequency, link density, and
  low-follower/high-frequency combinations each contribute a weighted signal.
- **Entropy screening.** Machine-spun or hash-like text carries far more
  per-character entropy than human prose; FilterGap flags it.
- **Duplicate detection.** Explicit duplicate flags, or a reference to an
  original post (`duplicate_of`, `retweet`, `quote_tweet`).
- **Coordinated-campaign clustering.** SimHash fingerprints plus token-set
  Jaccard group near-identical posts across accounts, so 3+ members posting the
  same boilerplate get flagged together.
- **A logistic layer on top.** Weak signals that never fire loudly on their own
  add up, catching the borderline bots a rule list alone misses.
- **Tunable confidence.** One `--threshold` decides how aggressive you want to
  be, because "bot" is a business decision, not a fact.
- **Explanations and calibration.** Each finding carries its top weighted
  signals; if you supply ground truth, the report also shows how well the
  scores actually separated bots from humans.

## What it is not

It is not a verdict. Automation confidence and harm confidence are triage
signals, not proof: a bot can be harmless and a human can post spam. Every
finding carries its signals and explanation so a human can overrule it. A
verified, established account posting a link is not a bot, and FilterGap is
built to not pretend otherwise.

## Install

Python 3.10+, no runtime dependencies.

```bash
git clone https://github.com/hexsyro/FilterGap.git
cd FilterGap
python -m pip install .
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Usage

```bash
filtergap examples/sample.csv
```

Columns are matched loosely: text from `text`, `content`, `message`, `body`, or
`post`; account age from `account_age_days`, `account_age`, `age_days`,
`account_age_hours`, or `account_created`; frequency from `posts_per_day`,
`post_count`, `posts_24h`, `posts_per_hour`, `frequency`, or
`tweets_per_day`; links from `link_count`, `has_link`, `urls`, `links`, or
`external_links`; and more for followers, `bot_score`, and `verified`.

## Options

| Flag | What it does |
| --- | --- |
| `--threshold T` | Bot/spam confidence (0.0–1.0) at or above which a mention is flagged (default 0.5) |
| `--config FILE` | JSON config; explicit flags win |
| `--format {table,json,csv,md}` | Output format (default `table`) |
| `--output FILE` | Write to file instead of stdout |
| `--explain` | Print per-finding explanations in table/markdown output (JSON always includes them) |
| `--fail-on {none,low,medium,high}` | Exit 1 if any finding is at or above this severity |
| `--version` | Print version and exit |

Config files (JSON) accept `threshold` (number) and `explain` (boolean). Invalid
value types exit 2 with a clear message; through the library API they are
ignored with a warning instead of raising.

## Output (json)

```json
{
  "tool": "filtergap",
  "version": "0.2.0",
  "status": "success",
  "findings": [
    {
      "severity": "high",
      "field": "bot_confidence",
      "message": "Mention scored 1.00 bot/spam confidence (threshold 0.50).",
      "record_id": "b1",
      "metadata": {
        "confidence": 1.0,
        "heuristic_score": 1.0,
        "classifier_score": 0.87,
        "threshold": 0.5,
        "signals": {"new_account": 0.35, "high_frequency": 0.5, "link_heavy": 0.2},
        "top_signals": [
          {"name": "high_frequency", "value": 0.5, "contribution": 1.5},
          {"name": "new_account", "value": 0.35, "contribution": 1.05}
        ],
        "explanation": "Automation 1.00, harm 0.70; top signals: high_frequency (+1.50), new_account (+1.05). Triage signal, not a verdict.",
        "automation_confidence": 1.0,
        "harm_confidence": 0.7
      }
    }
  ],
  "summary": {
    "result_schema_version": "1.1",
    "checked": 7,
    "flagged": 4,
    "flagged_percent": 57.14,
    "avg_confidence": 0.62,
    "signals": {"new_account": 4, "coordinated_cluster": 3},
    "signals_flagged": {"new_account": 4, "coordinated_cluster": 3},
    "entropy_skipped_non_ascii": 1,
    "explanations": [
      {
        "record_id": "b1",
        "score": 1.0,
        "severity": "high",
        "top_signals": [{"name": "high_frequency", "value": 0.5, "contribution": 1.5}],
        "text": "Check out this amazing deal now now now click here",
        "automation_confidence": 1.0,
        "harm_confidence": 0.7
      }
    ],
    "explanations_truncated": 0,
    "calibration": null,
    "skipped": false,
    "skip_reason": null,
    "patterns_version": "2026.10",
    "ml_layer": true,
    "classifier_flagged": 4
  }
}
```

### Explanations

Every flagged row appears in `summary["explanations"]` with its `record_id`,
`score`, `severity`, the full `text`, and `top_signals` ranked by weighted
contribution to the logistic score. Only the first 50 are stored;
`explanations_truncated` says how many were omitted. Findings also carry
`metadata["top_signals"]` and a readable `metadata["explanation"]`. Use
`--explain` to print them in table/markdown reports; JSON always includes them.

### Calibration

If your data has a boolean or 0-1 column named `is_bot`, `bot_label`, `label`,
or `ground_truth`, the summary includes a `calibration` object with `n`,
`brier_score`, `precision_at_threshold`, `false_discovery_rate`, and five
reliability buckets comparing predicted score with the observed positive rate.
Without ground truth, `calibration` is `null` and the report warns that the
score is uncalibrated — treat it as triage only.

### Automation vs harm

`automation_confidence` is the final score (how automated the behaviour looks).
`harm_confidence` is built only from spam/harm signals — duplicate content,
link-heavy posts, coordinated clusters, and low-trust/low-engagement — capped
to 0–1. They answer different questions: a bot can be harmless, and a human can
post spam. Neither is a verdict.

## Using it in CI

```bash
filtergap mentions.csv --fail-on high || exit 1
```

Exit codes: `0` ran fine, `1` gate tripped, `2` input broken.

## Library use

```python
from filtergap import run

result = run(rows, {"threshold": 0.6})
result.summary["flagged"]
```

## Limitations, honestly

- **Signals, not proof.** A confidence score is a reason to look, not a
  conviction. The rule weights are heuristics tuned for typical spam.
- **The pattern list ages.** Spam evolves; `PATTERNS_VERSION` is stamped on
  every report so you know which detector scored it.
- **Clustering needs a crowd.** The coordinated-campaign check needs 3+ near-
  identical posts to fire; a lone bot posting boilerplate won't trip it.
- **Your columns matter.** Without account-age or frequency fields, FilterGap
  can only judge the text itself.

## Changelog

## [0.2.0] - 2026-11-19

### Added

- Result schema `1.1`: exported `RESULT_SCHEMA_VERSION`, plus `skipped`/`skip_reason` on every run.
- First-class explanations (`summary["explanations"]`, `metadata["top_signals"]`,
  `metadata["explanation"]`) and an `--explain` flag for table/markdown output.
- Calibration against optional ground-truth columns: Brier score, precision/FDR at threshold, and
  five reliability bins; a warning when no ground truth is available.
- Separate `automation_confidence` and `harm_confidence` triage scores in explanations and findings.
- Deterministic LSH/token-signature candidate generation for coordinated clustering, with a
  warning when more than 2000 text rows are eligible.
- Strict option type validation with clear CLI errors (exit 2).

### Fixed

- Entropy screening skips mostly non-Latin text and reports `entropy_skipped_non_ascii`.
- `verified` has a negative weight in the logistic layer, so verified accounts are not flagged by it.
- `summary["signals"]` counts all fired signals; the old flagged-only counts are now in
  `summary["signals_flagged"]`.
- `threshold` no longer raises out of `run()` on malformed config values.
- The no-rows warning no longer implies text-only rows are excluded from checking.

### Inspiration

- [Bot-detection false-positive discipline](https://fingerprint.com/blog/bot-detection)
- ["Differentiate" good vs bad bots](https://pmc.ncbi.nlm.nih.gov/articles/PMC11958817)
- [Coordinated timing/hashtag signals](https://link.springer.com/article/10.1007/s13278-025-01557-1)
- [CLI-as-API best practices](https://hackmd.io/@arturtamborski/cli-best-practices)

## License

MIT. See [LICENSE](LICENSE).
