#!/usr/bin/env python3
"""Freeze the 150-step sparse config at or below the Dev calibration bounds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    proposed = load_config(args.proposed)
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    recommendation = calibration.get("recommendation", {})
    if (calibration.get("schema_version") != "short-recovery-dev-calibration-v2"
            or calibration.get("status") != "pass"
            or calibration.get("dev_sha256") != proposed.dev_sha256
            or calibration.get("initial_checkpoint") != proposed.model_path):
        raise RuntimeError("calibration does not match the proposed run")

    payload = json.loads(args.proposed.read_text(encoding="utf-8"))
    payload["model_history_probability"] = min(
        float(payload["model_history_probability"]),
        float(recommendation["model_history_probability_max"]))
    payload["sampling_probability"] = min(
        float(payload["sampling_probability"]),
        float(recommendation["sampling_probability_max"]))
    payload["repeat_probability"] = min(
        float(payload["repeat_probability"]),
        float(recommendation["repeat_probability_max"]))
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frozen = load_config(args.output)
    print(json.dumps({
        "status": "pass",
        "config": str(args.output),
        "config_sha256": frozen.digest,
        "model_history_probability": frozen.model_history_probability,
        "sampling_probability": frozen.sampling_probability,
        "repeat_probability": frozen.repeat_probability,
    }, indent=2))


if __name__ == "__main__":
    main()
