import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit import (
    _comparison,
    _match_record,
    import_metrics,
)


def test_match_record_does_not_fallback_after_key_miss():
    with pytest.raises(ValueError, match="no JSON record matched"):
        _match_record([{"key": "other", "model": "base"}], "wanted")


def test_metrics_requires_unique_key_and_model(tmp_path: Path):
    metrics_path = tmp_path / "per_record_metrics.json"
    metrics_path.write_text(json.dumps([
        {"key": "k", "model": "sft", "text": {"CER": 0.2}},
        {"key": "k", "model": "sft", "text": {"CER": 0.3}},
    ]), encoding="utf-8")
    prediction = {"key": "k", "model": "sft"}
    result = import_metrics(metrics_path, prediction)
    assert result["available"] is False
    assert result["matching_rows"] == 2

    missing = import_metrics(metrics_path, {"key": "missing", "model": "sft"})
    assert missing["available"] is False
    assert "no unique metric row" in missing["reason"]


def test_comparison_flags_key_dtype_and_actual_cap_mismatch():
    def row(key, audio, dtype, cap):
        return {
            "key": key,
            "path_fields": {"audio": audio},
            "prompt": {"prompt_ids_sha256": "p", "prompt_len": 10},
            "decoding": {
                "temperature": 0,
                "dtype": dtype,
                "actual_max_new_tokens_cap": cap,
            },
            "metrics": {"available": False},
        }

    result = _comparison({
        "base": row("k", "a.wav", "bfloat16", 65536),
        "c": row("other", "a.wav", "float16", 32768),
    })
    assert result["same_key"] is False
    assert result["same_dtype"] is False
    assert result["same_actual_max_new_tokens_cap"] is False
    assert result["same_decoding"] is False
    assert result["all_input_checks_pass"] is False


def test_metrics_model_specific_files_can_be_distinct(tmp_path: Path):
    base_metrics = tmp_path / "base.json"
    c_metrics = tmp_path / "c.json"
    base_metrics.write_text(json.dumps([
        {"key": "k", "model": "base", "text": {"CER": 0.1}},
    ]), encoding="utf-8")
    c_metrics.write_text(json.dumps([
        {"key": "k", "model": "sft", "text": {"CER": 0.4}},
    ]), encoding="utf-8")
    base = import_metrics(base_metrics, {"key": "k", "model": "base"})
    c = import_metrics(c_metrics, {"key": "k", "model": "sft"})
    assert base["available"] is True and base["CER"] == 0.1
    assert c["available"] is True and c["CER"] == 0.4
