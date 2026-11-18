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

from filtergap.common import (
    RESULT_SCHEMA_VERSION,
    ToolResult,
    as_bool,
    as_str,
    clamp,
    find_id,
    get,
    opt_bool,
    opt_float,
    to_float,
    to_int,
    tokenize,
    validate_options,
)
from filtergap.stats import (
    hamming,
    lsh_bands_for_threshold,
    lsh_candidate_pairs,
    minhash_signature,
    shannon_entropy,
    simhash_fingerprint,
)

TEXT_COLUMNS = ["text", "content", "message", "body", "post"]
AGE_COLUMNS = ["account_age_days", "account_age", "age_days", "account_age_hours", "account_created"]
FREQUENCY_COLUMNS = ["posts_per_day", "post_count", "posts_24h", "posts_per_hour", "frequency", "tweets_per_day"]
LINK_COLUMNS = ["link_count", "has_link", "urls", "links", "external_links"]
FOLLOWER_COLUMNS = ["followers", "followers_count", "follower_count"]
BOT_SCORE_COLUMNS = ["bot_score", "spam_score", "bot_probability", "spam_probability"]
VERIFIED_COLUMNS = ["verified", "is_verified"]
GROUND_TRUTH_COLUMNS = ["is_bot", "bot_label", "label", "ground_truth"]

# Bumped whenever the signal weights / rule set change, so a report can state
# exactly which detector version scored it ("ongoing updates as patterns evolve").
PATTERNS_VERSION = "2026.10"

# Linear (logistic) classifier layer on top of the rule signals: catches
# borderline accounts where no single rule fires loudly but several weak ones
# co-occur — the harder-to-catch bots the heuristic alone misses. "verified" is
# a protective signal with a negative weight so the logistic layer doesn't
# flag established accounts just for posting a link.
ML_WEIGHTS: dict[str, float] = {
    "new_account": 3.0,
    "high_frequency": 3.0,
    "duplicate_content": 3.5,
    "random_text": 2.0,
    "coordinated_cluster": 2.5,
    "link_heavy": 1.5,
    "low_trust_low_engagement": 1.5,
    "verified": -3.0,
}
ML_BIAS = -2.6

# Only these signals describe *harmful* content, as opposed to merely automated
# behaviour. They drive the separate harm_confidence triage score.
HARM_SIGNALS = frozenset({
    "duplicate_content",
    "link_heavy",
    "coordinated_cluster",
    "low_trust_low_engagement",
})

MAX_EXPLANATIONS = 50
EXPLANATION_TOP_SIGNALS = 5
CLUSTER_WARNING_ROWS = 2000
ASCII_RATIO_MIN = 0.9


def _ordered_pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


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

    def run(self, rows: list[dict] | None, options: dict[str, Any] | None = None) -> ToolResult:
        rows = list(rows) if rows is not None else []
        if not isinstance(options, dict):
            options = {}
        invalid = validate_options(options)

        threshold = clamp(opt_float(options, "threshold", 0.5), 0.0, 1.0)
        # Kept for the shared library contract; JSON output always carries the
        # explanations, table/markdown renderers honor this option.
        opt_bool(options, "explain", False)

        findings: list[dict] = []
        warnings: list[str] = []
        for error in invalid:
            warnings.append(f"Ignored invalid option: {error}")

        checked = 0
        flagged = 0
        signal_counts: dict[str, int] = {}
        flagged_signal_counts: dict[str, int] = {}
        confidence_sum = 0.0
        ml_flagged = 0
        entropy_skipped_non_ascii = 0
        explanations: list[dict[str, Any]] = []
        calibration_pairs: list[tuple[float, bool]] = []
        coordinated = self._coordinated_members(rows, warnings)

        for pos, row in enumerate(rows):
            text = as_str(get(row, TEXT_COLUMNS))
            has_identity = get(row, AGE_COLUMNS) is not None or get(row, FREQUENCY_COLUMNS) is not None
            if not text and not has_identity:
                continue
            checked += 1

            heuristic, signals, entropy_skipped = self._score_row_detailed(row, pos in coordinated)
            if entropy_skipped:
                entropy_skipped_non_ascii += 1
            classifier = self._ml_score(signals)
            confidence = clamp(max(heuristic, classifier))
            confidence_sum += confidence
            if classifier >= threshold:
                ml_flagged += 1

            # Count every signal that fired, not just the ones on flagged rows.
            for signal in signals:
                signal_counts[signal] = signal_counts.get(signal, 0) + 1

            truth = self._ground_truth(row)
            if truth is not None:
                calibration_pairs.append((confidence, truth))

            if confidence < threshold:
                continue

            flagged += 1
            for signal in signals:
                flagged_signal_counts[signal] = flagged_signal_counts.get(signal, 0) + 1

            severity = "high" if confidence >= 0.7 else "medium"
            top_signals = self._top_signals(signals)
            harm = self._harm_confidence(signals)
            automation = round(confidence, 3)
            explanation_text = self._explanation_text(top_signals, automation, harm)
            explanation: dict[str, Any] = {
                "record_id": find_id(row),
                "score": automation,
                "severity": severity,
                "top_signals": top_signals,
                "text": text,
                "automation_confidence": automation,
                "harm_confidence": harm,
            }
            if len(explanations) < MAX_EXPLANATIONS:
                explanations.append(explanation)

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
                        "top_signals": top_signals,
                        "explanation": explanation_text,
                        "automation_confidence": automation,
                        "harm_confidence": harm,
                        "text_preview": text[:120] if text else None,
                    },
                }
            )

        if checked == 0:
            warnings.append("No rows contained text or account-age/posting-frequency fields to check.")

        if calibration_pairs:
            calibration: dict[str, Any] | None = self._calibration(calibration_pairs, threshold)
        else:
            calibration = None
            warnings.append("uncalibrated heuristic score — treat as triage only")

        if not rows:
            skipped, skip_reason = True, "no rows provided"
        elif checked == 0:
            skipped, skip_reason = True, "no rows contained filterable text or account fields"
        else:
            skipped, skip_reason = False, None

        summary: dict[str, Any] = {
            "result_schema_version": RESULT_SCHEMA_VERSION,
            "checked": checked,
            "flagged": flagged,
            "flagged_percent": round(flagged / checked * 100, 2) if checked else 0.0,
            "avg_confidence": round(confidence_sum / checked, 3) if checked else 0.0,
            "signals": signal_counts,
            "signals_flagged": flagged_signal_counts,
            "entropy_skipped_non_ascii": entropy_skipped_non_ascii,
            "explanations": explanations,
            "explanations_truncated": flagged - len(explanations),
            "calibration": calibration,
            "skipped": skipped,
            "skip_reason": skip_reason,
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

    @staticmethod
    def _candidate_pairs(prints: list[tuple[int, int, set[str]]]) -> set[tuple[int, int]]:
        """Generate a bounded candidate set instead of comparing every pair.

        Identical token sets are chained (linear), everything else is indexed
        with the deterministic MinHash/LSH helpers from ``stats.py`` plus a
        cheap "three smallest tokens" signature. Large buckets are compared
        with a small neighbour window so a flood of identical boilerplate
        cannot reintroduce O(n^2) work.
        """
        candidates: set[tuple[int, int]] = set()
        if not prints:
            return candidates

        by_tokens: dict[frozenset[str], list[int]] = {}
        for idx, (_, _, toks) in enumerate(prints):
            by_tokens.setdefault(frozenset(toks), []).append(idx)

        reps: list[int] = []
        for members in by_tokens.values():
            if len(members) >= 3:
                # Chain the whole duplicate group, then screen one representative.
                for k in range(len(members) - 1):
                    candidates.add(_ordered_pair(members[k], members[k + 1]))
                reps.append(members[0])
            else:
                reps.extend(members)
        reps.sort()
        if len(reps) < 2:
            return candidates

        signatures = [minhash_signature(prints[i][2], num_perm=64) for i in reps]
        bands, rows_per_band = lsh_bands_for_threshold(0.8, num_perm=64)

        buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}
        for local, sig in enumerate(signatures):
            for band in range(bands):
                key = (band, tuple(sig[band * rows_per_band : (band + 1) * rows_per_band]))
                buckets.setdefault(key, []).append(local)

        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) <= 16:
                small_pairs = lsh_candidate_pairs([signatures[m] for m in members], 1, rows_per_band)
                for a, b in small_pairs:
                    candidates.add(_ordered_pair(members[a], members[b]))
                continue
            ordered = sorted(members, key=lambda m: signatures[m])
            for k in range(len(ordered) - 1):
                candidates.add(_ordered_pair(reps[ordered[k]], reps[ordered[k + 1]]))
            for k in range(len(ordered) - 2):
                candidates.add(_ordered_pair(reps[ordered[k]], reps[ordered[k + 2]]))

        token_buckets: dict[tuple[str, ...], list[int]] = {}
        for idx in reps:
            token_buckets.setdefault(tuple(sorted(prints[idx][2])[:3]), []).append(idx)
        for members in token_buckets.values():
            if len(members) < 2:
                continue
            ordered = sorted(members, key=lambda m: tuple(sorted(prints[m][2])))
            for k in range(len(ordered) - 1):
                candidates.add(_ordered_pair(ordered[k], ordered[k + 1]))
            for k in range(len(ordered) - 2):
                candidates.add(_ordered_pair(ordered[k], ordered[k + 2]))
        return candidates

    def _coordinated_members(self, rows: list[dict], warnings: list[str] | None = None) -> set[int]:
        """Row positions in near-duplicate text clusters (3+ members).

        Coordinated campaigns reuse boilerplate across accounts. Candidate
        pairs are generated with LSH/token signatures, then confirmed by
        SimHash Hamming distance and token-set Jaccard so short texts sharing
        a few words don't false-positive.
        """
        prints: list[tuple[int, int, set[str]]] = []
        for pos, row in enumerate(rows):
            text = as_str(get(row, TEXT_COLUMNS))
            toks = set(tokenize(text))
            if len(text) >= 20 and len(toks) >= 6:
                prints.append((pos, simhash_fingerprint(sorted(toks)), toks))

        if len(prints) > CLUSTER_WARNING_ROWS and warnings is not None:
            warnings.append(
                f"Coordinated clustering screened {len(prints)} eligible rows; "
                "candidate generation limits the comparison set."
            )

        parent = list(range(len(prints)))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, j in sorted(self._candidate_pairs(prints)):
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
        when each rule fired below its own cutoff. Protective signals such as
        ``verified`` carry negative weights and use their magnitude as
        presence, so they actively push the probability down.
        """
        z = ML_BIAS
        for name, weight in ML_WEIGHTS.items():
            if name in signals:
                z += weight * min(1.0, abs(signals[name]))
        return round(1.0 / (1.0 + math.exp(-z)), 3)

    @staticmethod
    def _top_signals(signals: dict[str, float]) -> list[dict[str, Any]]:
        """Rank fired signals by their weighted contribution to the score."""
        ranked = sorted(
            signals.items(),
            key=lambda item: (-abs(ML_WEIGHTS.get(item[0], 1.0) * min(1.0, abs(item[1]))), item[0]),
        )
        return [
            {
                "name": name,
                "value": value,
                "contribution": round(ML_WEIGHTS.get(name, 1.0) * min(1.0, abs(value)), 3),
            }
            for name, value in ranked[:EXPLANATION_TOP_SIGNALS]
        ]

    @staticmethod
    def _harm_confidence(signals: dict[str, float]) -> float:
        """Score from spam/harm signals only; automation alone is not harm."""
        return round(clamp(sum(signals.get(name, 0.0) for name in HARM_SIGNALS)), 3)

    @staticmethod
    def _explanation_text(top_signals: list[dict[str, Any]], automation: float, harm: float) -> str:
        parts = [f"{item['name']} ({item['contribution']:+.2f})" for item in top_signals] or ["no strong signals"]
        return (
            f"Automation {automation:.2f}, harm {harm:.2f}; "
            f"top signals: {', '.join(parts)}. Triage signal, not a verdict."
        )

    @staticmethod
    def _ground_truth(row: dict) -> bool | None:
        """Read a strict boolean/0-1 ground-truth cell, if the row has one."""
        value = get(row, GROUND_TRUTH_COLUMNS)
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
            return None
        text = as_str(value).lower()
        if text in {"1", "true", "yes", "bot", "spam"}:
            return True
        if text in {"0", "false", "no", "human", "ham"}:
            return False
        return None

    @staticmethod
    def _calibration(pairs: list[tuple[float, bool]], threshold: float) -> dict[str, Any]:
        """Brier score, precision/FDR at threshold, and 5 reliability bins."""
        n = len(pairs)
        brier = sum((score - int(label)) ** 2 for score, label in pairs) / n
        tp = sum(1 for score, label in pairs if score >= threshold and label)
        fp = sum(1 for score, label in pairs if score >= threshold and not label)
        predicted = tp + fp
        grouped: list[list[tuple[float, bool]]] = [[] for _ in range(5)]
        for score, label in pairs:
            grouped[min(4, int(score * 5))].append((score, label))

        buckets: list[dict[str, Any]] = []
        for index, members in enumerate(grouped):
            size = len(members)
            buckets.append(
                {
                    "bin": f"{index / 5:.1f}-{(index + 1) / 5:.1f}",
                    "n": size,
                    "avg_predicted": round(sum(s for s, _ in members) / size, 3) if size else 0.0,
                    "observed_positive_rate": round(sum(1 for _, label in members if label) / size, 3)
                    if size
                    else 0.0,
                }
            )
        return {
            "n": n,
            "brier_score": round(brier, 4),
            "precision_at_threshold": round(tp / predicted, 3) if predicted else 0.0,
            "false_discovery_rate": round(fp / predicted, 3) if predicted else 0.0,
            "reliability_buckets": buckets,
        }

    @staticmethod
    def _mostly_ascii(text: str) -> bool:
        """Guard the Latin-script entropy heuristic against script bias."""
        if not text:
            return True
        return sum(1 for ch in text if ord(ch) < 128) / len(text) > ASCII_RATIO_MIN

    def _score_row(self, row: dict, coordinated: bool = False) -> tuple[float, dict[str, float]]:
        """Backwards-compatible wrapper around the detailed scorer."""
        confidence, signals, _ = self._score_row_detailed(row, coordinated)
        return confidence, signals

    def _score_row_detailed(
        self, row: dict, coordinated: bool = False
    ) -> tuple[float, dict[str, float], bool]:
        confidence = 0.0
        signals: dict[str, float] = {}
        entropy_skipped = False

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
        if len(text) >= 20:
            if self._mostly_ascii(text):
                # Machine-random ASCII text (spun content, hashes) carries far
                # more per-character entropy than human prose (~4.0-4.7).
                if shannon_entropy(text) > 4.75:
                    _apply_signal(signals, "random_text", 0.15)
            else:
                # CJK/Cyrillic/Arabic/emoji-heavy text has a different entropy
                # baseline; the Latin-script threshold would false-positive.
                entropy_skipped = True

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

        return confidence, signals, entropy_skipped


def run(rows: list[dict] | None, options: dict[str, Any] | None = None) -> ToolResult:
    """Library entry point. Malformed option values never escape as exceptions."""
    return FilterGapLogic().run(rows, options or {})
