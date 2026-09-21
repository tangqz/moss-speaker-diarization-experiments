"""Merge D0/D1 outputs into compact machine-readable and human-readable reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = ("base", "step60", "step90", "step150")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def first_divergence(left: list[int], right: list[int]):
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return {"index": index, "left": a, "right": b}
    if len(left) != len(right):
        return {
            "index": min(len(left), len(right)),
            "left": left[min(len(left), len(right))] if len(left) > len(right) else None,
            "right": right[min(len(left), len(right))] if len(right) > len(left) else None,
        }
    return None


def maybe(path: Path):
    return read(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment
    spec = read(root / "cases.json")

    d0 = []
    for model in MODELS:
        for case in spec["d0_cases"]:
            hf = maybe(root / "hf" / "bf16" / model / f"{case}.json")
            vllm = maybe(root / "vllm" / "bf16" / model / f"{case}.json")
            fp32 = maybe(root / "hf" / "fp32" / model / f"{case}.json")
            row = {"model": model, "case": case}
            if hf and hf.get("status") == "ok":
                row["hf_bf16_status"] = "ok"
                row["hf_bf16_generated_tokens"] = hf["generated_tokens"]
            else:
                row["hf_bf16_status"] = (hf or {}).get("status", "missing")
            if vllm and vllm.get("status") == "ok":
                first = vllm["runs"][0]
                row.update(
                    vllm_status="ok",
                    vllm_deterministic=vllm["deterministic_repeat"],
                    hf_vllm_first_divergence=(
                        first_divergence(hf["generated_ids"], first["generated_ids"])
                        if hf and hf.get("status") == "ok"
                        else None
                    ),
                )
            else:
                row["vllm_status"] = (vllm or {}).get("status", "missing")
            if fp32 and fp32.get("status") == "ok":
                row.update(
                    hf_fp32_status="ok",
                    bf16_fp32_first_divergence=(
                        first_divergence(hf["generated_ids"], fp32["generated_ids"])
                        if hf and hf.get("status") == "ok"
                        else None
                    ),
                )
            elif fp32:
                row["hf_fp32_status"] = fp32.get("status")
                row["hf_fp32_error"] = fp32.get("error")
            else:
                row["hf_fp32_status"] = "not_run"
            d0.append(row)

    d1 = []
    for model in MODELS:
        for case in spec["d1_cases"]:
            result = maybe(root / "hf" / "bf16" / model / f"{case}.json")
            if result is None:
                d1.append({"model": model, "case": case, "status": "missing"})
                continue
            if result.get("status") != "ok":
                d1.append(
                    {
                        "model": model,
                        "case": case,
                        "status": "error",
                        "error_type": result.get("error_type"),
                        "error": result.get("error"),
                    }
                )
                continue
            first_margin = result["margin_trajectory"][0]["repeat_margin"] if result["margin_trajectory"] else None
            escaped_within_budget = bool(
                result["prefix_ends_in_repeat"]
                and result["generated_tokens"]
                and result["leading_repeat_tokens"] < result["generated_tokens"]
            )
            d1.append(
                {
                    "model": model,
                    "case": case,
                    "status": "ok",
                    "prefix_ends_in_repeat": result["prefix_ends_in_repeat"],
                    "first_repeat_margin": first_margin,
                    "leading_repeat_tokens": result["leading_repeat_tokens"],
                    "escaped_repeat": escaped_within_budget,
                    "escaped_immediately": bool(
                        result["prefix_ends_in_repeat"]
                        and result["generated_tokens"]
                        and result["leading_repeat_tokens"] == 0
                    ),
                    "timestamp_advanced": result["timestamp_advanced"],
                    "last_parsed_end_before": result["last_parsed_end_before_replay"],
                    "last_parsed_end_after": result["last_parsed_end_after_replay"],
                    "ended_eos": result["ended_eos"],
                    "generated_tokens": result["generated_tokens"],
                }
            )

    summary = {
        "protocol_version": spec["protocol_version"],
        "target_key": spec["target_key"],
        "selection_prohibition": spec["selection_prohibition"],
        "step0_status": spec["step0_status"],
        "d0": d0,
        "d1": d1,
        "interpretation_boundary": (
            "Escaping token 47815 is necessary but not sufficient for transcription recovery; "
            "timestamp_advanced and decoded continuation must also be inspected."
        ),
    }
    write(root / "summary.json", summary)

    lines = [
        "# D0/D1 fixed-prefix recovery diagnostic",
        "",
        f'- Target: `{spec["target_key"]}` (known Test diagnostic only; prohibited for checkpoint selection/training)',
        f'- Step 0: {spec["step0_status"]}',
        "",
        "## D0 backend and precision parity",
        "",
        "| Model | Case | HF BF16 | vLLM BF16 | vLLM repeat | HF-vLLM first divergence | HF FP32 | BF16-FP32 first divergence |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in d0:
        lines.append(
            f'| {row["model"]} | {row["case"]} | {row["hf_bf16_status"]} | '
            f'{row["vllm_status"]} | {row.get("vllm_deterministic")} | '
            f'{json.dumps(row.get("hf_vllm_first_divergence"), ensure_ascii=False)} | '
            f'{row["hf_fp32_status"]} | '
            f'{json.dumps(row.get("bf16_fp32_first_divergence"), ensure_ascii=False)} |'
        )
    lines.extend(
        [
            "",
            "## D1 fixed-prefix recovery",
            "",
            "| Model | Case | First repeat margin | Leading repeats | Escaped | Timestamp advanced | EOS | Status |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in d1:
        lines.append(
            f'| {row["model"]} | {row["case"]} | {row.get("first_repeat_margin")} | '
            f'{row.get("leading_repeat_tokens")} | {row.get("escaped_repeat")} | '
            f'{row.get("timestamp_advanced")} | {row.get("ended_eos")} | {row["status"]} |'
        )
    lines.extend(
        [
            "",
            "Escaping `外` is not by itself successful recovery. Inspect decoded continuations and require valid forward transcription progress.",
            "",
        ]
    )
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(root / "summary.json"), "d0_rows": len(d0), "d1_rows": len(d1)}))


if __name__ == "__main__":
    main()
