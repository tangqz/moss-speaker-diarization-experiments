#!/usr/bin/env python3
"""Executable GPU gate result shell; every stage remains pending until real evidence exists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

import torch
from .preflight import validate_stage_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[f"G{i}" for i in range(6)], required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--result-json", type=Path,
                        help="Stage-specific harness result containing all DESIGN acceptance checks.")
    parser.add_argument("--gate", required=True, type=Path,
                        help="Bound pending gate whose source/plan/model/tokenizer identity is copied.")
    parser.add_argument("--exec-argv", nargs=argparse.REMAINDER,
                        help="Exact stage harness argv; it must freshly create --result-json.")
    args = parser.parse_args()
    command = " ".join(sys.argv)
    gate = json.loads(args.gate.read_text())
    if not gate.get("bound"):
        raise RuntimeError("GPU checks require a fully bound gate")
    payload = {"schema_version": "short-recovery-gpu-result-v1", "stage": args.stage,
               "status": "pending", "command": command, "cuda_available": torch.cuda.is_available(),
               "gpu_count": torch.cuda.device_count(), "torch": torch.__version__,
               "python": platform.python_version(), "bindings": gate["bindings"],
               "reason": "no stage-specific executed result was supplied"}
    if args.exec_argv:
        if args.result_json is None:
            raise RuntimeError("--exec-argv requires --result-json")
        args.result_json.unlink(missing_ok=True)
        completed = subprocess.run(args.exec_argv, check=False)
        if completed.returncode != 0 or not args.result_json.is_file():
            raise RuntimeError(f"stage harness failed or emitted no fresh result: rc={completed.returncode}")
        result = json.loads(args.result_json.read_text())
        validate_stage_result(result, args.stage, gate["bindings"])
        payload.update(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
