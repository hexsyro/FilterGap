"""CLI tests for FilterGap."""

from __future__ import annotations

import csv
import json

from filtergap.cli import main


def write_csv(tmp_path, content: str):
    p = tmp_path / "data.csv"
    p.write_text(content, encoding="utf-8")
    return p


BOT_CSV = (
    "id,text,account_age_days,posts_per_day,link_count,followers\n"
    'b1,"Check out this amazing deal now now now click here",0.5,250,3,2\n'
)

MODERATE_CSV = (
    "id,text,account_age_days,posts_per_day\n"
    'm1,"posting a lot today about all sorts of things",0.5,50\n'
)


def test_json_output(tmp_path, capsys):
    p = write_csv(tmp_path, BOT_CSV)
    assert main([str(p), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "filtergap"
    assert payload["findings"]


def test_fail_on_high(tmp_path):
    p = write_csv(tmp_path, BOT_CSV)
    assert main([str(p), "--fail-on", "high"]) == 1
    assert main([str(p), "--fail-on", "none"]) == 0


def test_csv_output(tmp_path, capsys):
    p = write_csv(tmp_path, BOT_CSV)
    assert main([str(p), "--format", "csv"]) == 0
    rows = list(csv.DictReader(capsys.readouterr().out.splitlines()))
    assert rows


def test_high_threshold_suppresses(tmp_path, capsys):
    p = write_csv(tmp_path, MODERATE_CSV)
    assert main([str(p), "--threshold", "0.7", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["findings"] == []


def test_config_file(tmp_path, capsys):
    p = write_csv(tmp_path, MODERATE_CSV)
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"threshold": 0.7}), encoding="utf-8")
    assert main([str(p), "--config", str(cfg), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["findings"] == []


def test_missing_file_returns_two(tmp_path):
    assert main([str(tmp_path / "nope.csv")]) == 2


def test_version(capsys):
    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert "filtergap" in capsys.readouterr().out
