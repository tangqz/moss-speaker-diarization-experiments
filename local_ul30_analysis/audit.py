"""Small CPU-only audit for the local UL30 problem-meeting experiment.

The command consumes one JSON prediction per model and one reference JSON.  It
deliberately reports signals and context; it does not decide that every text
repeat is pathological.  No model, tokenizer, GPU, or external metric service
is required.  Existing CER/cpCER records can be imported with ``--metrics``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "local-ul30-audit-v1"
DEFAULT_SINGLE_MIN = 32
DEFAULT_PERIODIC_MIN = 64
DEFAULT_MAX_PERIOD = 16


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def _candidate_records(value: Any) -> list[Mapping[str, Any]]:
    """Extract records from the common single/list/dictionary JSON shapes."""
    if isinstance(value, list):
        return [item for item in value if _is_mapping(item)]
    if not _is_mapping(value):
        return []
    if isinstance(value.get("predictions"), list):
        return [item for item in value["predictions"] if _is_mapping(item)]
    if "generated_ids" in value or "segments" in value:
        return [value]
    # references.json and some manifest files are keyed by the record key.
    records: list[Mapping[str, Any]] = []
    for key, item in value.items():
        if _is_mapping(item):
            if "key" not in item:
                item = dict(item)
                item["key"] = key
            records.append(item)
    return records


def _match_record(records: Iterable[Mapping[str, Any]], key: str | None,
                  *, model: str | None = None) -> Mapping[str, Any]:
    records = list(records)
    if key is not None:
        records = [r for r in records if r.get("key") == key]
        if not records:
            raise ValueError(f"no JSON record matched key={key!r} model={model!r}")
    if model is not None:
        records = [r for r in records if str(r.get("model", "")).lower() == model.lower()]
        if not records:
            raise ValueError(f"no JSON record matched key={key!r} model={model!r}")
    if not records:
        raise ValueError(f"no JSON record matched key={key!r} model={model!r}")
    if len(records) > 1:
        labels = [str(r.get("key", r.get("session_id", "?"))) for r in records[:5]]
        raise ValueError(f"multiple records matched key={key!r}: {labels}")
    return records[0]


def load_record(path: Path, key: str | None, *, model: str | None = None) -> Mapping[str, Any]:
    return _match_record(_candidate_records(read_json(path)), key, model=model)


def select_reference(path: Path, key: str) -> Mapping[str, Any]:
    record = _match_record(_candidate_records(read_json(path)), key)
    if "segments" not in record:
        raise ValueError(f"reference record {key!r} has no segments")
    return record


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _segments(value: Mapping[str, Any]) -> list[tuple[float, float]]:
    result = []
    for item in value.get("segments", []) or []:
        if not _is_mapping(item):
            continue
        if not (_finite_number(item.get("start")) and _finite_number(item.get("end"))):
            continue
        start, end = float(item["start"]), float(item["end"])
        if end > start:
            result.append((start, end))
    return result


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    merged: list[list[float]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def interval_coverage(reference: Mapping[str, Any], prediction: Mapping[str, Any]) -> dict[str, Any]:
    """Measure time coverage using unions of reference and predicted segments.

    This is deliberately a time-only diagnostic.  It says nothing about
    whether text, speaker labels, or timestamps are correct.
    """
    ref_union = merge_intervals(_segments(reference))
    pred_union = merge_intervals(_segments(prediction))
    uncovered: list[tuple[float, float]] = []
    for ref_start, ref_end in ref_union:
        cursor = ref_start
        for pred_start, pred_end in pred_union:
            if pred_end <= cursor:
                continue
            if pred_start >= ref_end:
                break
            if pred_start > cursor:
                uncovered.append((cursor, min(pred_start, ref_end)))
            cursor = max(cursor, min(pred_end, ref_end))
            if cursor >= ref_end:
                break
        if cursor < ref_end:
            uncovered.append((cursor, ref_end))
    reference_seconds = sum(end - start for start, end in ref_union)
    uncovered_seconds = sum(end - start for start, end in uncovered)
    return {
        "method": "union_of_reference_speech_intervals_minus_union_of_prediction_segments",
        "reference_speech_seconds_union": reference_seconds,
        "predicted_segment_seconds_union": sum(end - start for start, end in pred_union),
        "uncovered_seconds_union": uncovered_seconds,
        "uncovered_ratio_union": uncovered_seconds / reference_seconds if reference_seconds else None,
        "reference_interval_count": len(ref_union),
        "prediction_interval_count": len(pred_union),
        "uncovered_intervals": [
            {"start": start, "end": end, "seconds": end - start}
            for start, end in uncovered
        ],
        "note": "time coverage only; this is not a text-correctness or diarization score",
    }


def longest_identical_run(ids: Sequence[int], context: int = 12) -> dict[str, Any] | None:
    values = [int(x) for x in ids]
    if not values:
        return None
    best_start, best_end = 0, 1
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            if index - start > best_end - best_start:
                best_start, best_end = start, index
            start = index
    return {
        "start": best_start,
        "end_exclusive": best_end,
        "length": best_end - best_start,
        "token": values[best_start],
        "prefix_token_ids": values[max(0, best_start - context):best_start],
        "suffix_token_ids": values[best_end:min(len(values), best_end + context)],
        "context_note": "token IDs are reported because no tokenizer/model is loaded by this CPU audit",
    }


def periodic_runs(ids: Sequence[int], *, min_single: int = DEFAULT_SINGLE_MIN,
                  max_period: int = DEFAULT_MAX_PERIOD,
                  min_periodic: int = DEFAULT_PERIODIC_MIN) -> list[dict[str, int]]:
    """Mirror short_recovery.core.periodic_runs without importing torch."""
    values = [int(x) for x in ids]
    found: list[dict[str, int]] = []
    n = len(values)
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[j] == values[i]:
            j += 1
        if j - i >= min_single:
            found.append({"start": i, "end": j, "period": 1, "length": j - i})
        i = j
    for period in range(2, max_period + 1):
        i = 0
        while i + min_periodic <= n:
            j = i + period
            while j < n and values[j] == values[i + (j - i) % period]:
                j += 1
            if j - i >= min_periodic:
                found.append({"start": i, "end": j, "period": period, "length": j - i})
                i = j
            else:
                i += 1
    found.sort(key=lambda item: (item["start"], item["period"], -item["length"]))
    dedup: list[dict[str, int]] = []
    for item in found:
        if any(item["start"] >= previous["start"] and item["end"] <= previous["end"]
               for previous in dedup):
            continue
        dedup.append(item)
    return dedup


def _natural_repeat_candidates(prediction: Mapping[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    """Return text repeats for human review, without calling them errors."""
    candidates: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(prediction.get("segments", []) or []):
        if not _is_mapping(segment):
            continue
        text = str(segment.get("text", ""))
        if len(text) < 4:
            continue
        # Character n-grams are useful for Chinese outputs and avoid requiring
        # a tokenizer.  This intentionally reports candidates only.
        for width in range(2, min(10, len(text) // 2 + 1)):
            best_count = 1
            for start in range(0, len(text) - width * 2 + 1):
                phrase = text[start:start + width]
                count = 1
                cursor = start + width
                while text[cursor:cursor + width] == phrase:
                    count += 1
                    cursor += width
                if count > best_count:
                    best_count = count
                    if count >= 2:
                        candidates.append({
                            "segment_index": segment_index,
                            "start_char": start,
                            "width": width,
                            "repetitions": count,
                            "phrase": phrase,
                            "segment_start": segment.get("start"),
                            "segment_end": segment.get("end"),
                            "speaker": segment.get("speaker"),
                            "candidate_only": True,
                            "note": "may be natural speech repetition; inspect context before interpretation",
                        })
    candidates.sort(key=lambda item: (-item["repetitions"] * item["width"], item["segment_index"]))
    return candidates[:limit]


def _normalise_decoding(prediction: Mapping[str, Any]) -> dict[str, Any] | None:
    decoding = prediction.get("decoding")
    if not isinstance(decoding, Mapping):
        return None
    selected = (
        "temperature", "top_p", "top_k", "repetition_penalty", "presence_penalty",
        "frequency_penalty", "max_context", "max_new_tokens_cap", "eos_token_id",
    )
    result = {key: decoding.get(key) for key in selected if key in decoding}
    # The top-level field is the effective cap recorded by vLLM.  Keep it
    # separate from the protocol's requested cap so a mismatch is visible.
    result["actual_max_new_tokens_cap"] = prediction.get("max_new_tokens")
    result["dtype"] = prediction.get("dtype", decoding.get("dtype"))
    return result


def _text_metrics(value: Mapping[str, Any]) -> dict[str, Any] | None:
    text = value.get("text") if isinstance(value.get("text"), Mapping) else value
    if not isinstance(text, Mapping):
        return None
    names = (
        "reference_characters", "character_errors", "cp_character_errors", "substitutions",
        "deletions", "insertions", "CER", "cpCER", "DeltaCP",
    )
    found = {name: text[name] for name in names if name in text}
    return found or None


def import_metrics(metrics_path: Path | None, prediction: Mapping[str, Any]) -> dict[str, Any]:
    """Import an existing per-record metric row if available."""
    key = prediction.get("key")
    model = str(prediction.get("model", "")).lower()
    candidates: list[Mapping[str, Any]] = []
    source = None
    if metrics_path:
        source = str(metrics_path)
        payload = read_json(metrics_path)
        if isinstance(payload, list):
            candidates = [item for item in payload if _is_mapping(item)]
        elif _is_mapping(payload):
            if isinstance(payload.get("per_record_metrics"), list):
                candidates = [item for item in payload["per_record_metrics"] if _is_mapping(item)]
            elif "key" in payload:
                candidates = [payload]
    if not candidates:
        embedded = prediction.get("metrics")
        if isinstance(embedded, Mapping):
            found = _text_metrics(embedded)
            if found:
                return {"available": True, "source": "prediction.metrics", **found}
        found = _text_metrics(prediction)
        if found:
            return {"available": True, "source": "prediction", **found}
    matches = [item for item in candidates if item.get("key") == key]
    if model:
        matches = [item for item in matches if str(item.get("model", "")).lower() == model]
    if not matches:
        return {"available": False, "source": source,
                "reason": f"no unique metric row for key={key!r}, model={model!r}"}
    if len(matches) > 1:
        return {"available": False, "source": source,
                "reason": f"multiple metric rows for key={key!r}, model={model!r}",
                "matching_rows": len(matches)}
    found = _text_metrics(matches[0])
    if not found:
        return {"available": False, "source": source, "reason": "matching row has no text metrics"}
    return {"available": True, "source": source, **found}


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def audit_prediction(prediction: Mapping[str, Any], reference: Mapping[str, Any],
                     metrics_path: Path | None) -> dict[str, Any]:
    generated_ids = prediction.get("generated_ids") or []
    longest = longest_identical_run(generated_ids)
    loops = periodic_runs(generated_ids)
    predicted_segments = _segments(prediction)
    reference_segments = _segments(reference)
    maximum_segment_end = max((end for _, end in predicted_segments), default=0.0)
    reference_maximum_end = max((end for _, end in reference_segments), default=0.0)
    premature_segment_end = bool(
        reference_maximum_end
        and maximum_segment_end < reference_maximum_end - max(30.0, reference_maximum_end * 0.05)
    )
    return {
        "key": prediction.get("key"),
        "model": prediction.get("model"),
        "path_fields": {
            "audio": prediction.get("audio"),
            "session_id": prediction.get("session_id"),
            "backend": prediction.get("backend"),
            "model_path": prediction.get("model_path"),
        },
        "prompt": {
            "prompt_ids_sha256": prediction.get("prompt_ids_sha256"),
            "prompt_len": prediction.get("prompt_len"),
            "prompt_sha256": _hash_json(prediction.get("prompt")) if prediction.get("prompt") is not None else None,
            "official_prompt_ids_checked": prediction.get("official_prompt_ids_checked"),
        },
        "decoding": _normalise_decoding(prediction),
        "output": {
            "generated_tokens": prediction.get("generated_tokens", len(generated_ids)),
            "generated_ids_count": len(generated_ids),
            "ended_eos": prediction.get("ended_eos"),
            "truncated": prediction.get("truncated"),
            "max_new_tokens": prediction.get("max_new_tokens"),
            "finish_reason": prediction.get("finish_reason"),
            "stop_reason": prediction.get("stop_reason"),
            "parse_empty": prediction.get("parse_empty"),
            "major_parse_loss": (prediction.get("diagnostics") or {}).get("major_parse_loss")
                if isinstance(prediction.get("diagnostics"), Mapping) else None,
            "segment_count": len(prediction.get("segments", []) or []),
            "maximum_segment_end": maximum_segment_end,
            "reference_maximum_end": reference_maximum_end,
            "potential_premature_segment_end": premature_segment_end,
            "last_segment": (prediction.get("segments") or [])[-1]
                if prediction.get("segments") else None,
        },
        "repetition": {
            "longest_identical_run": longest,
            "periodic_runs": loops,
            "raw_token_repeat_signal": bool(loops),
            "natural_text_repeat_candidates": _natural_repeat_candidates(prediction),
            "note": "raw-token signals and text candidates require human/context review",
        },
        "time_coverage": interval_coverage(reference, prediction),
        "metrics": import_metrics(metrics_path, prediction),
    }


def _same(values: Sequence[Any]) -> bool:
    return len(values) <= 1 or all(value == values[0] for value in values[1:])


def _present_and_same(values: Sequence[Any]) -> bool:
    return bool(values) and all(value is not None for value in values) and _same(values)


def _decoding_present_and_same(values: Sequence[Any]) -> bool:
    required = ("dtype", "actual_max_new_tokens_cap")
    return (bool(values)
            and all(isinstance(value, Mapping)
                    and all(value.get(field) is not None for field in required)
                    for value in values)
            and _same(values))


def _comparison(records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    keys = list(records)
    prompts = {name: (records[name].get("prompt") or {}).get("prompt_ids_sha256") for name in keys}
    decodings = {name: records[name].get("decoding") for name in keys}
    prompt_lens = {name: (records[name].get("prompt") or {}).get("prompt_len") for name in keys}
    audio = {name: (records[name].get("path_fields") or {}).get("audio") for name in keys}
    same_prompt = _present_and_same(list(prompts.values()))
    same_decoding = _decoding_present_and_same(list(decodings.values()))
    same_prompt_len = _present_and_same(list(prompt_lens.values()))
    same_audio = _present_and_same(list(audio.values()))
    metrics_delta: dict[str, Any] = {}
    base_metrics = records.get("base", {}).get("metrics", {})
    if base_metrics.get("available"):
        for name, row in records.items():
            if name == "base" or not row.get("metrics", {}).get("available"):
                continue
            current = row["metrics"]
            metrics_delta[name] = {
                field: current[field] - base_metrics[field]
                for field in ("CER", "cpCER", "DeltaCP")
                if field in current and field in base_metrics
            }
    return {
        "same_key": _present_and_same([records[name].get("key") for name in keys]),
        "same_audio": same_audio,
        "same_prompt_ids_sha256": same_prompt,
        "same_prompt_len": same_prompt_len,
        "same_decoding": same_decoding,
        "same_dtype": _present_and_same([
            (decodings[name] or {}).get("dtype") for name in keys
        ]),
        "same_actual_max_new_tokens_cap": _present_and_same([
            (decodings[name] or {}).get("actual_max_new_tokens_cap") for name in keys
        ]),
        "prompt_ids_sha256": prompts,
        "decoding": decodings,
        "prompt_len": prompt_lens,
        "audio": audio,
        "comparable_model_count": len(keys),
        "all_input_checks_pass": len(keys) < 2 or all((
            _present_and_same([records[name].get("key") for name in keys]),
            same_prompt, same_decoding, same_prompt_len, same_audio,
        )),
        "metrics_delta_vs_base": metrics_delta,
        "note": "matching prompt hash and decoding fields support input parity; they do not prove audio/data parity beyond recorded fields",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths: dict[str, Path | None] = {
        "base": Path(args.base),
        "b": Path(args.b) if args.b else None,
        "c": Path(args.c) if args.c else None,
    }
    base = load_record(paths["base"], args.key, model="base" if not args.key else None)
    key = args.key or str(base.get("key"))
    reference = select_reference(Path(args.reference), key)
    records: dict[str, Mapping[str, Any]] = {"base": base}
    for name in ("b", "c"):
        if paths[name] is not None:
            records[name] = load_record(paths[name], key)
    fallback_metrics = Path(args.metrics) if args.metrics else None
    metrics_paths: dict[str, Path | None] = {
        "base": Path(args.base_metrics) if args.base_metrics else fallback_metrics,
        "b": Path(args.b_metrics) if args.b_metrics else fallback_metrics,
        "c": Path(args.c_metrics) if args.c_metrics else fallback_metrics,
    }
    audited = {
        name: audit_prediction(prediction, reference, metrics_paths[name])
        for name, prediction in records.items()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_scope": "R8005_M8009 local UL30 known-problem sample",
        "cpu_only": True,
        "inputs": {
            "base": str(paths["base"]),
            "b": str(paths["b"]) if paths["b"] else None,
            "c": str(paths["c"]) if paths["c"] else None,
            "reference": str(args.reference),
            "metrics_fallback": str(fallback_metrics) if fallback_metrics else None,
            "metrics_by_model": {
                name: str(path) if path else None for name, path in metrics_paths.items()
            },
            "key": key,
        },
        "reference": {
            "key": reference.get("key", key),
            "duration": reference.get("duration"),
            "segment_count": len(reference.get("segments", []) or []),
        },
        "input_consistency": _comparison(audited),
        "models": audited,
        "interpretation": {
            "raw_token_loops": "Signals are reported for review; a natural repeated phrase is not automatically a failure.",
            "time_coverage": "Uncovered seconds/ratio are reference-speech interval coverage only and do not imply text correctness.",
            "acceptance": "Apply the UL30 design's human-reviewed pathology, EOS, truncation, parse, and integrity gates after inspecting this report.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base prediction JSON")
    parser.add_argument("--reference", required=True, help="Reference JSON or references.json")
    parser.add_argument("--b", help="B/control prediction JSON")
    parser.add_argument("--c", help="C/local-UL prediction JSON")
    parser.add_argument("--metrics", help="Optional per_record_metrics.json for CER/cpCER import")
    parser.add_argument("--base-metrics", help="Base per_record_metrics.json; overrides --metrics")
    parser.add_argument("--b-metrics", help="B per_record_metrics.json; overrides --metrics")
    parser.add_argument("--c-metrics", help="C per_record_metrics.json; overrides --metrics")
    parser.add_argument("--key", help="Record key, inferred from Base when omitted")
    parser.add_argument("--out", help="Write report JSON here; stdout is always emitted")
    args = parser.parse_args()
    try:
        report = run(args)
    except Exception as exc:  # concise machine-readable failure for batch use
        print(json.dumps({"schema_version": SCHEMA_VERSION, "audit_error": type(exc).__name__,
                          "message": str(exc)}, ensure_ascii=False))
        return 2
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
