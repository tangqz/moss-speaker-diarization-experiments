#!/usr/bin/env python3
"""Build mixed training/validation JSONL + smoke subsets + dataset statistics.

Outputs (under data/moss_jsonl/):
    train_mix.jsonl    AliMeeting train + AISHELL-4 train + AMI train (shuffled, seed 0)
    val_mix.jsonl      AliMeeting eval (dev) + AMI dev (shuffled)
    smoke_train.jsonl / smoke_val.jsonl / smoke_test.jsonl
Statistics -> data/reports/data_stats.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moss_data_common import JSONL, MANIFESTS, REPORTS, log  # noqa: E402

TRAIN_FILES = ["alimeeting_train.jsonl", "aishell4_train.jsonl", "ami_train.jsonl"]
VAL_FILES = ["alimeeting_dev.jsonl", "ami_dev.jsonl"]
TEST_FILES = ["alimeeting_test.jsonl", "aishell4_test.jsonl", "ami_test.jsonl"]


def load(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def dump(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    rng = random.Random(0)

    train = []
    for name in TRAIN_FILES:
        rows = load(JSONL / name)
        train += rows
        log(f"train part {name}: {len(rows)}")
    rng.shuffle(train)
    dump(JSONL / "train_mix.jsonl", train)
    log(f"train_mix.jsonl: {len(train)} records")

    val = []
    for name in VAL_FILES:
        rows = load(JSONL / name)
        val += rows
        log(f"val part {name}: {len(rows)}")
    rng.shuffle(val)
    dump(JSONL / "val_mix.jsonl", val)
    log(f"val_mix.jsonl: {len(val)} records")

    for n_per, files, out in ((2, TRAIN_FILES, "smoke_train.jsonl"),
                              (1, VAL_FILES, "smoke_val.jsonl"),
                              (2, TEST_FILES, "smoke_test.jsonl")):
        rows = []
        for name in files:
            rows += load(JSONL / name)[:n_per]
        dump(JSONL / out, rows)
        log(f"{out}: {len(rows)} records")

    stats = {}
    for man in sorted(MANIFESTS.glob("*.jsonl")):
        rows = load(man)
        speech = sum(r["speech_time"] for r in rows)
        overlap = sum(r["overlap_time"] for r in rows)
        stats[man.stem] = {
            "sessions": len(rows),
            "hours": round(sum(r["duration"] for r in rows) / 3600.0, 2),
            "segments": sum(r["n_segments"] for r in rows),
            "speakers_min": min((r["n_speakers"] for r in rows), default=0),
            "speakers_max": max((r["n_speakers"] for r in rows), default=0),
            "overlap_ratio": round(overlap / speech, 4) if speech else 0.0,
        }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "data_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"stats -> {REPORTS / 'data_stats.json'}")


if __name__ == "__main__":
    main()
