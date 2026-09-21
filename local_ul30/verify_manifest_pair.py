#!/usr/bin/env python3
"""Verify B/C share the same source/event order and only differ in arm/loss path."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery", required=True, type=Path)
    parser.add_argument("--ul", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    left = [json.loads(line) for line in args.recovery.read_text(encoding="utf-8").splitlines() if line.strip()]
    right = [json.loads(line) for line in args.ul.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(left) != len(right) or len(left) != 240:
        raise RuntimeError(f"expected 240 matching rows, got {len(left)} and {len(right)}")
    rows = []
    for index, (a, b) in enumerate(zip(left, right)):
        ma = a["chat_template_kwargs"]["recovery_meta"]
        mb = b["chat_template_kwargs"]["recovery_meta"]
        aa, bb = dict(ma), dict(mb)
        aa.pop("arm", None)
        bb.pop("arm", None)
        if digest(aa) != digest(bb):
            raise RuntimeError(f"B/C event mismatch at row {index}")
        if ma.get("arm") != "recovery" or mb.get("arm") != "ul":
            raise RuntimeError(f"arm mismatch at row {index}")
        if ma.get("view") == "aux":
            expected_repeat = 2 if ma["source_slot"] < 2 else 3
            if ma.get("event_kind") != "repeat_previous" or ma.get("repeat_count") != expected_repeat:
                raise RuntimeError(f"repeat protocol mismatch at row {index}")
        rows.append({"row": index, "event_digest": digest(aa), "source_slot": ma["source_slot"],
                     "optimizer_step": ma["optimizer_step"], "view": ma["view"]})
    result = {"status": "pass", "rows": len(rows), "event_digest_sha256": hashlib.sha256(
        "".join(row["event_digest"] for row in rows).encode()).hexdigest(), "events": rows}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rows": result["rows"],
                      "event_digest_sha256": result["event_digest_sha256"]}))


if __name__ == "__main__":
    main()
