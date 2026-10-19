"""Tests for FilterGap bot/spam detection."""

from __future__ import annotations

from filtergap.logic import FilterGapLogic

logic = FilterGapLogic()

RANDOM_TEXT = "a7bQ2xZ9mK4pL8wR3tY6uI0oP5nM1cV7bN4jH2gF9dS3"
COORDINATED_TEXT = "Buy cheap followers now visit our website for the best deals anywhere"


def test_obvious_bot_flagged():
    rows = [
        {
            "id": "b1",
            "text": "Check out this amazing deal now now now click here",
            "account_age_days": 0.5,
            "posts_per_day": 250,
            "link_count": 3,
            "followers": 2,
        }
    ]
    result = logic.run(rows, {"threshold": 0.5})
    assert result.summary["flagged"] == 1
    assert result.findings[0]["severity"] == "high"


def test_clean_account_not_flagged():
    rows = [
        {
            "id": "c1",
            "text": "Really enjoying the new update, the dashboard feels much faster.",
            "account_age_days": 900,
            "posts_per_day": 2,
            "followers": 1200,
        }
    ]
    result = logic.run(rows, {"threshold": 0.5})
    assert result.findings == []
    assert result.summary["flagged"] == 0


def test_rows_without_signals_skipped():
    result = logic.run([{"foo": "bar"}], {})
    assert result.summary["checked"] == 0
    assert any("No rows" in w for w in result.warnings)


def test_new_account_signal():
    rows = [{"id": "n1", "text": "hello world everyone here today", "account_age_days": 0.2}]
    result = logic.run(rows, {"threshold": 0.1})
    assert result.findings[0]["metadata"]["signals"]["new_account"] == 0.35


def test_high_frequency_signal():
    rows = [{"id": "f1", "text": "posting a lot today about things", "posts_per_day": 250}]
    result = logic.run(rows, {"threshold": 0.1})
    assert result.findings[0]["metadata"]["signals"]["high_frequency"] == 0.5


def test_duplicate_content_signal():
    rows = [{"id": "d1", "text": "same text copied again", "is_duplicate": True}]
    result = logic.run(rows, {"threshold": 0.1})
    assert result.findings[0]["metadata"]["signals"]["duplicate_content"] == 0.45


def test_random_text_signal():
    rows = [{"id": "r1", "text": RANDOM_TEXT}]
    result = logic.run(rows, {"threshold": 0.05})
    assert "random_text" in result.findings[0]["metadata"]["signals"]


def test_coordinated_cluster_signal():
    rows = [{"id": f"c{i}", "text": COORDINATED_TEXT} for i in range(3)]
    result = logic.run(rows, {"threshold": 0.1})
    assert result.summary["flagged"] == 3
    for finding in result.findings:
        assert finding["metadata"]["signals"]["coordinated_cluster"] == 0.25


def test_verified_reduces_confidence():
    base = {
        "id": "v1",
        "text": "some ordinary looking post text here",
        "account_age_days": 0.2,
        "verified": True,
    }
    result = logic.run([base], {"threshold": 0.05})
    signals = result.findings[0]["metadata"]["signals"]
    assert signals["verified"] == -0.5


def test_classifier_score_present():
    rows = [
        {
            "id": "m1",
            "text": "generic promotional post that many accounts repeat",
            "account_age_days": 0.5,
            "posts_per_day": 250,
            "link_count": 3,
        }
    ]
    result = logic.run(rows, {"threshold": 0.5})
    assert result.findings[0]["metadata"]["classifier_score"] > 0
    assert result.summary["ml_layer"] is True


def test_bot_score_column_overrides():
    rows = [{"id": "s1", "text": "looks harmless honestly", "bot_score": 0.95}]
    result = logic.run(rows, {"threshold": 0.5})
    assert result.findings
    assert result.findings[0]["metadata"]["confidence"] == 0.95


def test_threshold_configurable():
    rows = [
        {
            "id": "t1",
            "text": "borderline promotional content repeated across accounts",
            "account_age_days": 0.5,
            "posts_per_day": 250,
        }
    ]
    high = logic.run(rows, {"threshold": 0.99})
    low = logic.run(rows, {"threshold": 0.1})
    assert high.summary["flagged"] == 0
    assert low.summary["flagged"] == 1


def test_summary_fields():
    result = logic.run([], {})
    assert result.summary["patterns_version"]
    assert result.summary["flagged_percent"] == 0.0
