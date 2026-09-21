"""Prepare and summarize single-meeting vLLM replays across saved checkpoints."""
import json
from pathlib import Path


RUN = Path("/work/qt28/moss/results/ms-swift-63643")
SOURCE = RUN / "source" / "evaluation"
TARGET_KEY = "alimeeting/test/R8005_M8009"
STEPS = (30, 60, 90, 120)
OUT = RUN / "evaluations" / "loop-onset-R8005_M8009"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare():
    items = read(RUN / "evaluations" / "test-150" / "inputs.json")
    item = next(x for x in items if x["key"] == TARGET_KEY)
    item = dict(item, rank=0)
    for step in STEPS:
        folder = OUT / f"step-{step}"
        write(folder / "inputs.json", [item])
        manifest = read(RUN / "evaluations" / f"dev-{step}" / "model_manifest.json")
        write(folder / "model_manifest.json", manifest)
        write(folder / "protocol.json", {
            "scope": "single_meeting_loop_onset_diagnostic",
            "target_key": TARGET_KEY,
            "checkpoint_step": step,
            "checkpoint": str(RUN / "vllm_exports" / f"checkpoint-{step}"),
            "generator": str(SOURCE / "vllm_generate.py"),
            "decoding": {
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "repetition_penalty": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "max_context": 131072,
                "max_new_tokens_cap": 65536,
            },
        })


def summarize():
    rows = []
    for step in STEPS:
        path = OUT / f"step-{step}" / "predictions" / "sft" / f"{TARGET_KEY}.json"
        record = read(path)
        diag = record.get("diagnostics", {})
        longest = diag.get("longest_identical_run") or {}
        rows.append({
            "step": step,
            "status": record.get("status"),
            "generated_tokens": record.get("generated_tokens"),
            "ended_eos": record.get("ended_eos"),
            "truncated": record.get("truncated"),
            "finish_reason": record.get("finish_reason"),
            "stop_reason": record.get("stop_reason"),
            "longest_identical_run_start": longest.get("start"),
            "longest_identical_run_count": longest.get("count"),
            "longest_identical_token_id": longest.get("token"),
            "parsed_segments": len(record.get("segments", [])),
            "last_parsed_end_sec": (record.get("segments") or [{}])[-1].get("end_time"),
            "raw_text_tail": record.get("raw_text", "")[-300:],
            "e2e_seconds": record.get("e2e_seconds"),
            "prediction_path": str(path),
        })
    write(OUT / "summary.json", {
        "target_key": TARGET_KEY,
        "steps": list(STEPS),
        "rows": rows,
        "criterion": "loop if truncated and longest identical token run is anomalously long",
    })
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "summarize"))
    args = parser.parse_args()
    prepare() if args.mode == "prepare" else summarize()
