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

## What it is not

It is not a verdict. Every finding carries its signals and confidence so a
human can overrule it. A verified, established account posting a link is not a
bot, and FilterGap is built to not pretend otherwise.

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
| `--fail-on {none,low,medium,high}` | Exit 1 if any finding is at or above this severity |
| `--version` | Print version and exit |

## Output (json)

```json
{
  "tool": "filtergap",
  "version": "0.1.0",
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
        "signals": {"new_account": 0.35, "high_frequency": 0.5, "link_heavy": 0.2}
      }
    }
  ],
  "summary": {
    "checked": 7,
    "flagged": 4,
    "flagged_percent": 57.14,
    "avg_confidence": 0.62,
    "signals": {"new_account": 4, "coordinated_cluster": 3},
    "patterns_version": "2026.10",
    "ml_layer": true,
    "classifier_flagged": 4
  }
}
```

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

## License

MIT. See [LICENSE](LICENSE).
