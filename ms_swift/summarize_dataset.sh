#!/usr/bin/env bash
set -euo pipefail
/work/qt28/moss/envs/ms-swift-20260914/bin/python - <<'PY'
import collections
import json
from pathlib import Path

for split in ("train", "dev"):
    path = Path(f"/work/qt28/moss/dkucc/ms_swift_20260914/data/{split}.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    print(split, len(rows), sorted(rows[0]))
    print(json.dumps(rows[0], ensure_ascii=True)[:2000])
    counts = collections.Counter()
    for row in rows:
        text = json.dumps(row, ensure_ascii=True).lower()
        for dataset in ("alimeeting", "aishell4", "ami"):
            if dataset in text:
                counts[dataset] += 1
                break
        else:
            counts["unknown"] += 1
    print("counts", json.dumps(counts, sort_keys=True))
PY
