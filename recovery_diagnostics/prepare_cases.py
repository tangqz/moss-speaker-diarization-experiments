"""Build auditable fixed-prefix replay cases from existing evaluation evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGET_KEY = "alimeeting/test/R8005_M8009"
REPEAT_TOKEN_ID = 47815
DEFAULT_RUN = Path("/work/qt28/moss/results/ms-swift-63643")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def target_item(records: list[dict]) -> dict:
    return next(dict(item) for item in records if item["key"] == TARGET_KEY)


def longest_run(record: dict) -> dict:
    run = record.get("diagnostics", {}).get("longest_identical_run") or {}
    if int(run.get("token", -1)) != REPEAT_TOKEN_ID:
        raise ValueError(f"expected repeat token {REPEAT_TOKEN_ID}, got {run}")
    return {"start": int(run["start"]), "count": int(run["count"]), "token": int(run["token"])}


def ids_before_eos(record: dict) -> list[int]:
    ids = [int(value) for value in record["generated_ids"]]
    if ids and ids[-1] == 151645:
        ids.pop()
    return ids


def gold_prefix(reference: dict, anchor_seconds: float) -> str:
    # Keep complete reference segments only. Overlap is legal, so selection is
    # by end time rather than requiring non-overlap or strict monotonicity.
    selected = [segment for segment in reference["segments"] if float(segment["end"]) <= anchor_seconds]
    if not selected:
        raise ValueError("gold prefix is empty")
    return "".join(
        f'[{float(s["start"]):.2f}][{s["speaker"]}]' f'{s["text"]}[{float(s["end"]):.2f}]'
        for s in selected
    )


def make_case(
    name: str,
    source: str,
    *,
    prefix_ids=None,
    prefix_text=None,
    append_repeat_token_count: int = 0,
    note: str,
) -> dict:
    if (prefix_ids is None) == (prefix_text is None):
        raise ValueError("exactly one prefix representation is required")
    row = {"name": name, "source": source, "note": note}
    if prefix_ids is not None:
        row["prefix_ids"] = [int(value) for value in prefix_ids]
    else:
        row["prefix_text"] = prefix_text
    row["append_repeat_token_count"] = int(append_repeat_token_count)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--anchor-seconds", type=float, default=980.0)
    args = parser.parse_args()

    run = args.run
    out = args.out or run / "evaluations" / "recovery-diagnostics-d0-d1"
    test = run / "evaluations" / "test-150"
    input_item = target_item(read(test / "inputs.json"))
    input_item["rank"] = 0
    references = read(test / "references.json")
    reference = references[TARGET_KEY]

    base = read(test / "predictions" / "base" / f"{TARGET_KEY}.json")
    step150 = read(test / "predictions" / "sft" / f"{TARGET_KEY}.json")
    loop_root = run / "evaluations" / "loop-onset-R8005_M8009"
    if not loop_root.exists():
        # Downloaded local evidence stores `raw/` and the loop replay as siblings.
        loop_root = run.parent / "loop-onset-R8005_M8009"
    step60 = read(loop_root / "step-60" / "predictions" / "sft" / f"{TARGET_KEY}.json")
    step90 = read(loop_root / "step-90" / "predictions" / "sft" / f"{TARGET_KEY}.json")

    records = {"base": base, "step60": step60, "step90": step90, "step150": step150}
    runs = {name: longest_run(record) for name, record in records.items()}
    generated = {name: ids_before_eos(record) for name, record in records.items()}
    gold = gold_prefix(reference, args.anchor_seconds)

    cases: list[dict] = []
    for repeat_count in (0, 8, 32, 64, 128):
        cases.append(
            make_case(
                f"gold_r{repeat_count}",
                "gold",
                prefix_text=gold,
                append_repeat_token_count=repeat_count,
                note=f"correct reference through {args.anchor_seconds:.2f}s plus token 47815 repeated {repeat_count} times",
            )
        )

    base_start, base_count = runs["base"]["start"], runs["base"]["count"]
    cases.extend(
        [
            make_case(
                "base_pre_loop",
                "base_generation",
                prefix_ids=generated["base"][:base_start],
                note="Base history immediately before its recovered repeat run",
            ),
            make_case(
                "base_at_exit",
                "base_generation",
                prefix_ids=generated["base"][: base_start + base_count],
                note="Base history after its full repeat run and immediately before the observed exit token",
            ),
        ]
    )
    for label in ("step60", "step90", "step150"):
        start = runs[label]["start"]
        ids = generated[label]
        cases.extend(
            [
                make_case(
                    f"{label}_pre_loop",
                    f"{label}_generation",
                    prefix_ids=ids[:start],
                    note=f"{label} history immediately before the repeated token",
                ),
                make_case(
                    f"{label}_bad_r64",
                    f"{label}_generation",
                    prefix_ids=ids[:start] + [REPEAT_TOKEN_ID] * 64,
                    note=f"{label} pre-loop history with a controlled 64-token bad suffix",
                ),
            ]
        )

    models = {
        "base": "/work/qt28/moss/models/MOSS-Transcribe-Diarize",
        "step60": str(run / "vllm_exports" / "checkpoint-60"),
        "step90": str(run / "vllm_exports" / "checkpoint-90"),
        "step150": str(run / "vllm_exports" / "checkpoint-150"),
    }
    manifest = {
        "protocol_version": 1,
        "purpose": "D0 backend/precision parity and D1 fixed-prefix recovery diagnosis",
        "selection_prohibition": "Known Test failure; diagnostic use only, never checkpoint selection or training input",
        "target_key": TARGET_KEY,
        "repeat_token_id": REPEAT_TOKEN_ID,
        "anchor_seconds": args.anchor_seconds,
        "models": models,
        "source_runs": runs,
        "cases": cases,
        "d0_cases": ["gold_r64", "base_at_exit", "step90_bad_r64", "step150_bad_r64"],
        "d1_cases": [case["name"] for case in cases],
        "continuation_tokens": 256,
        "decoding": {
            "temperature": 0,
            "repetition_penalty": 1,
            "no_repeat_ngram_size": 0,
            "eos_token_id": 151645,
            "pad_token_id": 151643,
        },
        "step0_status": "unavailable: no zero-update checkpoint was saved; do not substitute an unverified roundtrip",
    }
    write(out / "input.json", input_item)
    write(out / "cases.json", manifest)
    print(json.dumps({"out": str(out), "cases": len(cases), "models": list(models)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
