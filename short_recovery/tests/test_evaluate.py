import json

from short_recovery.evaluate import scan_jsonl


def test_local_cap_is_censored_not_hard_failure(tmp_path):
    source = tmp_path / "pred.jsonl"
    source.write_text(json.dumps({"evaluation_scope": "local_injected_256",
                                  "finish_reason": "length", "raw_ids": list(range(256)),
                                  "raw_text": "valid", "parser_ok": True}) + "\n")
    result = scan_jsonl(source, tmp_path / "out.json")
    assert result["records"][0]["horizon_censored"] is True
    assert result["hard_failures"] == 0


def test_full_meeting_cap_is_hard_failure(tmp_path):
    source = tmp_path / "pred.jsonl"
    source.write_text(json.dumps({"evaluation_scope": "full_meeting",
                                  "finish_reason": "length", "raw_ids": [1, 2],
                                  "raw_text": "valid", "parser_ok": True}) + "\n")
    result = scan_jsonl(source, tmp_path / "out.json")
    assert result["hard_failures"] == 1


def test_not_applicable_and_execution_failure_have_separate_denominators(tmp_path):
    source = tmp_path / "pred.jsonl"
    source.write_text("\n".join([
        json.dumps({"status": "not_applicable", "case": "substitute"}),
        json.dumps({"status": "error", "case": "repeat2", "error": "oom"}),
    ]) + "\n")
    result = scan_jsonl(source, tmp_path / "out.json")
    assert result["not_applicable"] == 1
    assert result["execution_failures"] == 1
    assert result["evaluated_rows"] == 0
    assert result["status"] == "complete_with_errors"
