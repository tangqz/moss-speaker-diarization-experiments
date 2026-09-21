"""Prepare the gated full Test folder without reusing greedy predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TRAIN_RUN = Path("/work/qt28/moss/results/ms-swift-63643")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.sweep / "selected_config.json")
    if not config:
        raise RuntimeError("single-sample gate did not select a decoding config")
    source = TRAIN_RUN / "evaluations/test-150"
    inputs = read(source / "inputs.json")
    references = read(source / "references.json")
    if len(inputs) != 56:
        raise RuntimeError(f"expected 56 Test meetings, got {len(inputs)}")
    counts = {}
    for item in inputs:
        counts[item["dataset"]] = counts.get(item["dataset"], 0) + 1
    if counts != {"alimeeting": 20, "aishell4": 20, "ami": 16}:
        raise RuntimeError(f"unexpected Test composition: {counts}")
    write(args.run / "inputs.json", inputs)
    write(args.run / "references.json", references)
    write(args.run / "selected_config.json", config)
    write(
        args.run / "protocol.json",
        {
            "scope": "full_test_engineering_reevaluation_after_known_failure_selection",
            "models": {
                "base": "/work/qt28/moss/models/MOSS-Transcribe-Diarize",
                "sft": str(TRAIN_RUN / "vllm_exports/checkpoint-150"),
            },
            "datasets": counts,
            "decoding": config,
            "both_models_use_identical_decoding": True,
            "predictions_reused": False,
            "test_used_for_decoding_selection": True,
            "blind_test_claim_allowed": False,
            "historical_greedy_results_preserved": str(source),
            "metrics": "CER, cpCER, DeltaCP, DER collars 0 and 0.25, RTF, truncation and raw-token diagnostics",
        },
    )
    print(json.dumps({"run": str(args.run), "config": config, "counts": counts}))


if __name__ == "__main__":
    main()
