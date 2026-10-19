"""FilterGap — catch bot/spam mentions slipping through budget-tier listening tools.

Scores each mention with layered signals: account heuristics (age, frequency,
duplicate flags, link behaviour), Shannon-entropy screening for machine-random
text, and SimHash clustering that spots coordinated campaigns posting
near-identical content across accounts. Scores combine into a 0..1 confidence
per mention so a team can tune its own tolerance.
"""

from __future__ import annotations

import math
from typing import Any

from filtergap.common import ToolResult, as_bool, as_str, clamp, find_id, get, to_float, to_int, tokenize
from filtergap.stats import hamming, shannon_entropy, simhash_fingerprint

TEXT_COLUMNS = ["text", "content", "message", "body", "post"]
AGE_COLUMNS = ["account_age_days", "account_age", "age_days", "account_age_hours", "account_created"]
FREQUENCY_COLUMNS = ["posts_per_day", "post_count", "posts_24h", "posts_per_hour", "frequency", "tweets_per_day"]
LINK_COLUMNS = ["link_count", "has_link", "urls", "links", "external_links"]
FOLLOWER_COLUMNS = ["followers", "followers_count", "follower_count"]
BOT_SCORE_COLUMNS = ["bot_score", "spam_score", "bot_probability", "spam_probability"]
VERIFIED_COLUMNS = ["verified", "is_verified"]

# Bumped whenever the signal weights / rule set change, so a report can state
# exactly which detector version scored it ("ongoing updates as patterns evolve").
PATTERNS_VERSION = "2026.10"

# Linear (logistic) classifier layer on top of the rule signals: catches
# borderline accounts where no single rule fires loudly but several weak ones
# co-occur — the harder-to-catch bots the heuristic alone misses.
ML_WEIGHTS: dict[str, float] = {
    "new_account": 3.0,
    "high_frequency": 3.0,
    "duplicate_content": 3.5,
    "random_text": 2.0,
    "coordinated_cluster": 2.5,
    "link_heavy": 1.5,
    "low_trust_low_engagement": 1.5,
}
ML_BIAS = -2.6


def _apply_signal(signals: dict[str, float], name: str, value: float) -> None:
    signals[name] = round(value, 2)


def _is_duplicate(row: dict) -> bool:
    """A mention is a duplicate if flagged, or if it references an original.

    ``is_duplicate``/``duplicate`` are booleans; ``duplicate_of``/``retweet``/
    ``quote_tweet`` usually hold a URL or id of the original post, so any
    non-empty value counts.
    """
    if as_bool(get(row, ["is_duplicate", "duplicate"])):
        return True
    ref = as_str(get(row, ["duplicate_of", "retweet", "quote_tweet"]))
    return bool(ref)


class FilterGapLogic:
    """Heuristic bot/spam detector."""

    def run(self, rows: list[dict], options: dict[str, Any]) -> ToolResult:
        threshold = clamp(float(options.get("threshold", 0.5)), 0.0, 1.0)

        findings: list[dict] = []
        warnings: list[str] = []
        checked = 0
        flagged = 0
        signal_counts: dict[str, int] = {}
        confidence_sum = 0.0
        ml_flagged = 0
        coordinated = self._coordinated_members(rows)

        for pos, row in enumerate(rows):
            text = as_str(get(row, TEXT_COLUMNS))
            has_identity = get(row, AGE_COLUMNS) is not None or get(row, FREQUENCY_COLUMNS) is not None
            if not text and not has_identity:
                continue
            checked += 1

            heuristic, signals = self._score_row(row, pos in coordinated)
            classifier = self._ml_score(signals)
            confidence = clamp(max(heuristic, classifier))
            confidence_sum += confidence
            if classifier >= threshold:
                ml_flagged += 1
            if confidence < threshold:
                continue

            flagged += 1
            for signal in signals:
                signal_counts[signal] = signal_counts.get(signal, 0) + 1

            severity = "high" if confidence >= 0.7 else "medium"
            findings.append(
                {
                    "severity": severity,
                    "field": "bot_confidence",
                    "message": (
                        f"Mention scored {confidence:.2f} bot/spam confidence "
                        f"(threshold {threshold:.2f})."
                    ),
                    "record_id": find_id(row),
                    "metadata": {
                        "confidence": confidence,
                        "heuristic_score": round(heuristic, 3),
                        "classifier_score": round(classifier, 3),
                        "threshold": threshold,
                        "signals": signals,
                        "text_preview": text[:120] if text else None,
                    },
                }
            )

        if checked == 0:
            warnings.append(
                "No rows carried filterable signals (text/body plus account_age_days, "
                "posting frequency or duplicate flags)."
            )

        summary: dict[str, Any] = {
            "checked": checked,
            "flagged": flagged,
            "flagged_percent": round(flagged / checked * 100, 2) if checked else 0.0,
            "avg_confidence": round(confidence_sum / checked, 3) if checked else 0.0,
            "signals": signal_counts,
            "patterns_version": PATTERNS_VERSION,
            "ml_layer": True,
            "classifier_flagged": ml_flagged,
        }
        return ToolResult(
            status="success",
            findings=findings,
            warnings=warnings,
            summary=summary,
        )

    def _coordinated_members(self, rows: list[dict]) -> set[int]:
        """Row positions in near-duplicate text clusters (3+ members).

        Coordinated campaigns reuse boilerplate across accounts. Screening is
        SimHash Hamming distance (cheap at scale), confirmed by token-set
        Jaccard so short texts sharing a few words don't false-positive.
        """
        prints: list[tuple[int, int, set[str]]] = []
        for pos, row in enumerate(rows):
            text = as_str(get(row, TEXT_COLUMNS))
            toks = set(tokenize(text))
            if len(text) >= 20 and len(toks) >= 6:
                prints.append((pos, simhash_fingerprint(sorted(toks)), toks))
        parent = list(range(len(prints)))
        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for i in range(len(prints)):
            for j in range(i + 1, len(prints)):
                if hamming(prints[i][1], prints[j][1]) > 16:
                    continue
                a, b = prints[i][2], prints[j][2]
                if len(a & b) / len(a | b) >= 0.8:
                    parent[find(i)] = find(j)
        groups: dict[int, list[int]] = {}
        for i, (pos, _, _) in enumerate(prints):
            groups.setdefault(find(i), []).append(pos)
        return {pos for members in groups.values() if len(members) >= 3 for pos in members}

    @staticmethod
    def _ml_score(signals: dict[str, float]) -> float:
        """Logistic layer over the rule signals (0..1 bot probability).

        Correlated weak signals that no single threshold catches add up here:
        a new account that is link-heavy and posts fast crosses the line even
        when each rule fired below its own cutoff.
        """
        z = ML_BIAS
        for name, weight in ML_WEIGHTS.items():
            if name in signals:
                z += weight * min(1.0, max(0.0, signals[name]))
        return round(1.0 / (1.0 + math.exp(-z)), 3)

    def _score_row(self, row: dict, coordinated: bool = False) -> tuple[float, dict[str, float]]:
        confidence = 0.0
        signals: dict[str, float] = {}

        age_hours = to_float(get(row, ["account_age_hours"]))
        age_days = to_float(get(row, ["account_age_days", "account_age", "age_days"]))
        if age_days is None and age_hours is not None:
            age_days = age_hours / 24.0
        if age_days is not None:
            if age_days < 1:
                _apply_signal(signals, "new_account", 0.35)
            elif age_days < 7:
                _apply_signal(signals, "new_account", 0.2)
            elif age_days < 30:
                _apply_signal(signals, "new_account", 0.1)

        posts_per_day = to_float(get(row, ["posts_per_day", "post_count", "tweets_per_day"]))
        posts_24h = to_float(get(row, ["posts_24h"]))
        posts_per_hour = to_float(get(row, ["posts_per_hour"]))
        if posts_per_hour is not None and posts_per_day is None:
            posts_per_day = posts_per_hour * 24.0
        if posts_per_day is not None:
            if posts_per_day > 100:
                _apply_signal(signals, "high_frequency", 0.5)
            elif posts_per_day > 20:
                _apply_signal(signals, "high_frequency", 0.3)
            elif posts_per_day > 5:
                _apply_signal(signals, "high_frequency", 0.15)
        if posts_24h is not None and posts_24h > 50:
            _apply_signal(signals, "high_frequency", max(signals.get("high_frequency", 0.0), 0.3))

        if _is_duplicate(row):
            _apply_signal(signals, "duplicate_content", 0.45)

        text = as_str(get(row, TEXT_COLUMNS))
        if len(text) >= 20 and shannon_entropy(text) > 4.75:
            # Machine-random text (spun content, hash-like strings) carries
            # far more per-character entropy than human prose (~4.0-4.7).
            _apply_signal(signals, "random_text", 0.15)

        if coordinated:
            _apply_signal(signals, "coordinated_cluster", 0.25)

        link_count = to_int(get(row, LINK_COLUMNS))
        has_link = as_bool(get(row, ["has_link"]))
        if link_count is not None and link_count >= 2:
            _apply_signal(signals, "link_heavy", 0.2)
        elif link_count is not None and link_count >= 1:
            _apply_signal(signals, "link_heavy", 0.1)
        elif has_link:
            _apply_signal(signals, "link_heavy", 0.05)

        followers = to_float(get(row, FOLLOWER_COLUMNS))
        if followers is not None and followers < 10 and "high_frequency" in signals:
            _apply_signal(signals, "low_trust_low_engagement", 0.1)

        if as_bool(get(row, VERIFIED_COLUMNS)):
            _apply_signal(signals, "verified", -0.5)

        confidence = clamp(sum(signals.values()))

        bot_score = to_float(get(row, BOT_SCORE_COLUMNS))
        if bot_score is not None:
            confidence = max(confidence, clamp(bot_score))

        return confidence, signals


def run(rows: list[dict], options: dict[str, Any] | None = None) -> ToolResult:
    return FilterGapLogic().run(rows, options or {})
