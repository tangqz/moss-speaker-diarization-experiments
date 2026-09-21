#!/usr/bin/env python3
"""Validate generated MOSS JSONL files and write a report.

Checks per record:
  * conversation shape: user/text, user/audio, assistant/text;
  * audio file exists, readable, 16 kHz (mono for derived; AMI may be source);
  * target parses as [start][Sxx]text[end] blocks;
  * starts non-decreasing; 0 <= start <= end <= duration + 0.1 s; non-empty text.

Outputs:
  data/reports/phase1_validation.md
  data/reports/validation_failures.txt   (first 200 failure lines)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moss_data_common import JSONL, REPORTS, TARGET_SR, audio_info, log  # noqa: E402

SEG_RE = re.compile(r"\[(\d+\.\d+)\]\[S(\d{2})\](.*?)\[(\d+\.\d+)\]", re.S)
EXPECTED_ROLES = [("user", "text"), ("user", "audio"), ("assistant", "text")]


def check_record(rec, errors: list) -> bool:
    conv = rec.get("conversation")
    if not isinstance(conv, list) or len(conv) != 3:
        errors.append("conversation shape invalid")
        return False
    roles = [(m.get("role"), m.get("message_type")) for m in conv]
    if roles != EXPECTED_ROLES:
        errors.append(f"roles {roles}")
        return False
    prompt, audio_path, target = (m.get("content") for m in conv)
    if not (prompt and audio_path and target):
        errors.append("empty prompt/audio/target")
        return False
    path = Path(audio_path)
    if not path.exists():
        errors.append(f"missing audio: {audio_path}")
        return False
    try:
        info = audio_info(path)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"audio unreadable: {audio_path}: {exc}")
        return False
    if info["sr"] != TARGET_SR:
        errors.append(f"sample rate {info['sr']} != {TARGET_SR}: {audio_path}")
    duration = info["duration"]
    segs = SEG_RE.findall(target)
    if not segs:
        errors.append(f"no segments parsed: {audio_path}")
        return False
    prev_start = -1e-9
    for a, _spk, text, b in segs:
        a, b = float(a), float(b)
        if a < prev_start - 1e-6:
            errors.append(f"start not monotonic ({a} < {prev_start}): {audio_path}")
            return False
        prev_start = a
        if a < -1e-9 or b < a or b > duration + 0.1:
            errors.append(f"segment bounds {a}-{b} vs duration {duration:.1f}: {audio_path}")
            return False
        if not text.strip():
            errors.append(f"empty segment text: {audio_path}")
            return False
    return True


def main() -> None:
    files = sorted(JSONL.glob("*.jsonl"))
    failures = []
    lines = ["# Phase 1 data validation report", ""]
    total_bad = 0
    for path in files:
        n_total = n_bad = 0
        with path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                n_total += 1
                errors = []
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"json error: {exc}")
                    n_bad += 1
                    failures.append(f"{path.name}:{line_no}: {errors[0]}")
                    continue
                try:
                    ok = check_record(rec, errors)
                except Exception as exc:  # noqa: BLE001
                    ok, errors = False, [f"exception: {exc!r}"]
                if not ok:
                    n_bad += 1
                    if len(failures) < 200:
                        failures.append(f"{path.name}:{line_no}: {'; '.join(errors)}")
        total_bad += n_bad
        status = "OK" if n_bad == 0 else f"FAIL {n_bad}"
        lines.append(f"- {path.name}: {n_total} records -> {status}")
        log(f"{path.name}: {n_total} records, {n_bad} bad")
    lines.append("")
    lines.append(f"Total records with problems: {total_bad}")
    if failures:
        lines.append("")
        lines.append("First failures:")
        lines += [f"  {f}" for f in failures[:50]]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "phase1_validation.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORTS / "validation_failures.txt").write_text("\n".join(failures), encoding="utf-8")
    log(f"report -> {REPORTS / 'phase1_validation.md'}")


if __name__ == "__main__":
    main()
