"""Tests for the 0.2.0 result-schema, explanation, calibration and safety upgrades."""

from __future__ import annotations

import json

import pytest

import filtergap
from filtergap import RESULT_SCHEMA_VERSION, ToolResult, __version__
from filtergap.cli import main
from filtergap.logic import ML_WEIGHTS, FilterGapLogic, run

COORDINATED_TEXT = "Buy cheap followers now visit our website for the best deals anywhere"
CJK_TEXT = "这是一个用于测试的中文句子重复很多次以增加长度和字符数量，用来验证非拉丁文本不会被误判。" * 2


def _csv(tmp_path, content: str):
    path = tmp_path / "data.csv"
    path.write_text(content, encoding="utf-8")
    return path


BOT_CSV = (
    "id,text,account_age_days,posts_per_day,link_count,followers\n"
    'b1,"Check out this amazing deal now now now click here",0.5,250,3,2\n'
)


def test_public_api_exports():
    assert __version__ == "0.2.0"
    assert RESULT_SCHEMA_VERSION == "1.1"
    assert ToolResult().status == "success"
    assert {"RESULT_SCHEMA_VERSION", "ToolResult"}.issubset(filtergap.__all__)


def test_result_schema_metadata_and_skipped():
    normal = run([{"id": "a", "text": "an ordinary sentence for checking", "account_age_days": 900}], {})
    assert normal.summary["result_schema_version"] == "1.1"
    assert normal.summary["skipped"] is False
    assert normal.summary["skip_reason"] is None

    empty = run([], {})
    assert empty.summary["skipped"] is True
    assert empty.summary["skip_reason"] == "no rows provided"

    unusable = run([{"foo": "bar"}], {})
    assert unusable.summary["skipped"] is True
    assert unusable.summary["skip_reason"]
    assert unusable.summary["checked"] == 0


def test_malformed_options_never_raise_from_run():
    result = run(
        [{"id": "a", "text": "some text for checking", "account_age_days": 0.2}],
        {"threshold": "abc", "explain": "maybe"},
    )
    assert result.status == "success"
    assert any("Ignored invalid option" in warning for warning in result.warnings)

    result = run([{"id": "a", "text": "some text"}], None)
    assert result.summary["checked"] == 1


def test_cli_invalid_config_value_types_exit_two(tmp_path, capsys):
    data = _csv(tmp_path, BOT_CSV)
    for value in ("abc", ["high"], None, True):
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"threshold": value}), encoding="utf-8")
        assert main([str(data), "--config", str(config)]) == 2
        error = capsys.readouterr().err
        assert "threshold" in error
        assert "invalid config" in error

    config.write_text(json.dumps({"explain": "perhaps"}), encoding="utf-8")
    assert main([str(data), "--config", str(config)]) == 2
    assert "explain" in capsys.readouterr().err


def test_non_latin_text_skips_latin_entropy_heuristic():
    logic = FilterGapLogic()
    _, signals, skipped = logic._score_row_detailed({"text": CJK_TEXT})
    assert skipped is True
    assert "random_text" not in signals

    result = run([{"id": "cjk", "text": CJK_TEXT, "account_age_days": 900}], {})
    assert result.summary["entropy_skipped_non_ascii"] == 1
    assert "random_text" not in result.summary["signals"]


def test_ascii_entropy_still_applies():
    random_text = "a7bQ2xZ9mK4pL8wR3tY6uI0oP5nM1cV7bN4jH2gF9dS3"
    result = run([{"id": "ascii", "text": random_text}], {"threshold": 0.05})
    assert result.summary["entropy_skipped_non_ascii"] == 0
    assert "random_text" in result.summary["signals"]


def test_clustering_warning_and_determinism():
    rows = [{"id": f"c{i}", "text": COORDINATED_TEXT} for i in range(2001)]
    first = run(rows, {"threshold": 0.1})
    second = run(rows, {"threshold": 0.1})
    assert any("2001 eligible rows" in warning for warning in first.warnings)
    assert first.summary["flagged"] == 2001
    assert first.summary == second.summary
    assert first.findings == second.findings


def test_signal_counts_cover_unflagged_rows():
    rows = [
        {"id": "low", "text": "ordinary text for checking", "account_age_days": 0.2},
        {"id": "high", "text": "ordinary text for checking", "account_age_days": 0.2, "bot_score": 0.9},
    ]
    result = run(rows, {"threshold": 0.5})
    assert result.summary["signals"]["new_account"] == 2
    assert result.summary["signals_flagged"]["new_account"] == 1


def test_no_rows_warning_no_longer_claims_text_only_rows_are_ignored():
    result = run([{"foo": "bar"}], {})
    warning = next(warning for warning in result.warnings if warning.startswith("No rows"))
    assert "text" in warning
    assert "duplicate" not in warning.lower()


def test_verified_has_negative_logistic_weight():
    base = {
        "text": "generic promotional post that many accounts repeat",
        "account_age_days": 0.5,
        "posts_per_day": 250,
        "link_count": 3,
    }
    plain = run([{**base, "id": "u"}], {"threshold": 0.01})
    verified = run([{**base, "id": "v", "verified": True}], {"threshold": 0.01})
    assert ML_WEIGHTS["verified"] == -3.0
    plain_score = plain.findings[0]["metadata"]["classifier_score"]
    verified_score = verified.findings[0]["metadata"]["classifier_score"]
    assert verified_score < plain_score
    assert verified_score < 0.5


def test_explanations_structure_and_truncation():
    rows = [
        {"id": f"r{i}", "text": f"spam duplicate message number {i}", "bot_score": 0.9, "link_count": 2}
        for i in range(55)
    ]
    result = run(rows, {"threshold": 0.5})
    explanations = result.summary["explanations"]
    assert len(explanations) == 50
    assert result.summary["explanations_truncated"] == 5

    first = explanations[0]
    assert {"record_id", "score", "severity", "top_signals", "text"}.issubset(first)
    assert first["score"] == 0.9
    assert first["severity"] == "high"
    assert first["text"]
    assert first["top_signals"]
    assert set(first["top_signals"][0]) == {"name", "value", "contribution"}

    metadata = result.findings[0]["metadata"]
    assert metadata["top_signals"] == first["top_signals"]
    assert isinstance(metadata["explanation"], str)
    assert "triage" in metadata["explanation"].lower()


def test_calibration_with_ground_truth():
    rows = [
        {"id": "a", "text": "promo text one here", "bot_score": 0.9, "is_bot": True},
        {"id": "b", "text": "promo text two here", "bot_score": 0.8, "bot_label": 1},
        {"id": "c", "text": "promo text three here", "bot_score": 0.2, "label": "human"},
        {"id": "d", "text": "promo text four here", "bot_score": 0.1, "ground_truth": 0},
    ]
    result = run(rows, {"threshold": 0.5})
    calibration = result.summary["calibration"]
    assert calibration["n"] == 4
    assert calibration["brier_score"] == pytest.approx(0.025)
    assert calibration["precision_at_threshold"] == 1.0
    assert calibration["false_discovery_rate"] == 0.0
    assert len(calibration["reliability_buckets"]) == 5
    assert sum(bucket["n"] for bucket in calibration["reliability_buckets"]) == 4
    assert all("observed_positive_rate" in bucket for bucket in calibration["reliability_buckets"])
    assert not any("uncalibrated" in warning for warning in result.warnings)


def test_calibration_absent_warns():
    result = run([{"text": "plain text here", "account_age_days": 900}], {})
    assert result.summary["calibration"] is None
    assert "uncalibrated heuristic score — treat as triage only" in result.warnings


def test_automation_and_harm_confidences_are_separate():
    automation_only = run(
        [{"id": "a", "text": "plain text here", "account_age_days": 0.2, "posts_per_day": 250}],
        {"threshold": 0.01},
    )
    harmful = run(
        [{"id": "s", "text": "plain text here", "is_duplicate": True, "link_count": 3}],
        {"threshold": 0.01},
    )
    automated_meta = automation_only.findings[0]["metadata"]
    harmful_meta = harmful.findings[0]["metadata"]
    assert automated_meta["automation_confidence"] > 0
    assert automated_meta["harm_confidence"] == 0.0
    assert harmful_meta["harm_confidence"] > 0
    assert harmful_meta["automation_confidence"] == harmful_meta["confidence"]

    explanation = automation_only.summary["explanations"][0]
    assert explanation["automation_confidence"] == automated_meta["automation_confidence"]
    assert explanation["harm_confidence"] == automated_meta["harm_confidence"]


def test_cli_explain_flag_and_json_always_includes_explanations(tmp_path, capsys):
    data = _csv(tmp_path, BOT_CSV)

    assert main([str(data), "--format", "md", "--explain"]) == 0
    markdown = capsys.readouterr().out
    assert "## Explanations" in markdown

    assert main([str(data), "--format", "table"]) == 0
    table = capsys.readouterr().out
    assert "Explanations:" not in table

    assert main([str(data), "--format", "table", "--explain"]) == 0
    assert "Explanations:" in capsys.readouterr().out

    assert main([str(data), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == "0.2.0"
    assert payload["summary"]["result_schema_version"] == "1.1"
    assert payload["summary"]["explanations"]
