"""Create single-sample sweep folders and immutable protocol metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/work/qt28/moss")
TRAIN_RUN = ROOT / "results/ms-swift-63643"
TARGET_KEY = "alimeeting/test/R8005_M8009"
CONFIG_NAMES = ["A0_seed0", "A1_seed0", "A2_seed0", "A3_seed0"] + [
    f"{family}_seed{seed}" for family in ("B1", "B2") for seed in (0, 1, 2)
]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    recovery = TRAIN_RUN / "evaluations/recovery-diagnostics-d0-d1"
    item = read(recovery / "input.json")
    references = read(TRAIN_RUN / "evaluations/test-150/references.json")
    reference = references[TARGET_KEY]
    for name in CONFIG_NAMES:
        run = args.out / "runs" / name
        write(run / "inputs.json", [item])
        write(run / "references.json", {TARGET_KEY: reference})
    write(
        args.out / "protocol.json",
        {
            "purpose": "decode intervention before retraining",
            "target": TARGET_KEY,
            "known_test_diagnostic": True,
            "not_blind_selection": True,
            "models": {
                "base": str(ROOT / "models/MOSS-Transcribe-Diarize"),
                "sft": str(TRAIN_RUN / "vllm_exports/checkpoint-150"),
            },
            "groups": {
                "A0": {"repetition_penalty": 1.0, "temperature": 0.0, "seeds": [0]},
                "A1": {"repetition_penalty": 1.02, "temperature": 0.0, "seeds": [0]},
                "A2": {"repetition_penalty": 1.05, "temperature": 0.0, "seeds": [0]},
                "A3": {"repetition_penalty": 1.10, "temperature": 0.0, "seeds": [0]},
                "B1": {"repetition_penalty": 1.0, "temperature": 0.2, "seeds": [0, 1, 2]},
                "B2": {"repetition_penalty": 1.0, "temperature": 0.4, "seeds": [0, 1, 2]},
            },
            "scope": "full problematic meeting only; fixed-prefix backend replay omitted by user priority",
            "full_meeting_max_new_tokens": 40000,
            "unchanged": {
                "top_p": 1,
                "top_k": -1,
                "min_p": 0,
                "presence_penalty": 0,
                "frequency_penalty": 0,
            },
            "selection_priority": ["A1", "A2", "A3", "B1", "B2"],
            "execution_order": "penalty-only first; temperature fallback is submitted only if A1/A2/A3 all fail",
            "sampling_rule": "all three fixed seeds must pass; full Test uses predeclared seed 0",
            "full_sample_gate": {
                "ended_eos": True,
                "truncated": False,
                "last_parsed_end_fraction": 0.95,
                "max_identical_token_run": 255,
                "per_model_metric_regression_max_pp": 1.0,
            },
        },
    )
    print(json.dumps({"out": str(args.out), "configs": CONFIG_NAMES}))


if __name__ == "__main__":
    main()
