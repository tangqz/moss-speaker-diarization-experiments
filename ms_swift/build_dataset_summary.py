"""Create an auditable count and duration summary for the final report."""
import collections
import json
from pathlib import Path
import statistics
import sys

import soundfile as sf

TASK = Path("/work/qt28/moss/dkucc/ms_swift_20260914")
OLD_EVAL = Path("/work/qt28/moss/results/eval-62994-63363")


def dataset_from_path(path):
    low = str(path).lower()
    for name in ("alimeeting", "aishell4", "ami"):
        if name in low:
            return name
    raise ValueError(f"Unknown dataset path: {path}")


def stats(durations):
    return {
        "records": len(durations),
        "hours": sum(durations) / 3600,
        "median_minutes": statistics.median(durations) / 60,
        "min_minutes": min(durations) / 60,
        "max_minutes": max(durations) / 60,
    }


def jsonl_split(name):
    rows = [json.loads(line) for line in (TASK / f"data/{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    grouped = collections.defaultdict(list)
    for row in rows:
        assert len(row["audios"]) == 1
        audio = row["audios"][0]
        grouped[dataset_from_path(audio)].append(sf.info(audio).duration)
    return {dataset: stats(values) for dataset, values in sorted(grouped.items())}


def test_split():
    grouped = collections.defaultdict(list)
    for row in json.loads((OLD_EVAL / "inputs.json").read_text(encoding="utf-8")):
        if row["split"] == "test":
            grouped[row["dataset"]].append(row["duration"])
    return {dataset: stats(values) for dataset, values in sorted(grouped.items())}


result = {"train": jsonl_split("train"), "dev": jsonl_split("dev"), "test": test_split()}
for split in result.values():
    split["all"] = {
        "records": sum(item["records"] for item in split.values()),
        "hours": sum(item["hours"] for item in split.values()),
    }
Path(sys.argv[1]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=True, indent=2))
