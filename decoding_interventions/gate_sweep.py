"""Apply predeclared structural and Base-regression gates to the problem sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TARGET_KEY = "alimeeting/test/R8005_M8009"
GROUPS = {
    "A0": ["A0_seed0"],
    "A1": ["A1_seed0"],
    "A2": ["A2_seed0"],
    "A3": ["A3_seed0"],
    "B1": ["B1_seed0", "B1_seed1", "B1_seed2"],
    "B2": ["B2_seed0", "B2_seed1", "B2_seed2"],
}


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def structural(record: dict) -> tuple[bool, list[str]]:
    failures = []
    if record.get("status") != "ok":
        failures.append("status")
    if not record.get("ended_eos"):
        failures.append("no_eos")
    if record.get("truncated"):
        failures.append("truncated")
    if record.get("parse_empty"):
        failures.append("parse_empty")
    segments = record.get("segments") or []
    last_end = float(segments[-1]["end"]) if segments else 0.0
    if last_end < 0.95 * float(record["duration"]):
        failures.append("coverage_below_95pct")
    longest = record.get("diagnostics", {}).get("longest_identical_run", {}).get("count", 10**9)
    if int(longest) >= 256:
        failures.append("identical_run_at_least_256")
    return not failures, failures


def metric_triplet(summary: dict, label: str) -> dict:
    group = summary["groups"][f"{label}/alimeeting/test"]
    return {
        "CER": float(group["CER"]),
        "cpCER": float(group["cpCER"]),
        "DER025": float(group["DER_collar_0.25"]["DER"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=("A1", "A2", "A3", "B1", "B2"), required=True)
    args = parser.parse_args()
    protocol = read(args.out / "protocol.json")
    baseline_summary = read(args.out / "runs/A0_seed0/metrics_summary.json")
    baseline = {label: metric_triplet(baseline_summary, label) for label in ("base", "sft")}
    rows = []
    eligible_groups = []
    for group in args.groups:
        names = GROUPS[group]
        group_ok = True
        group_rows = []
        for name in names:
            summary = read(args.out / "runs" / name / "metrics_summary.json")
            metrics = {label: metric_triplet(summary, label) for label in ("base", "sft")}
            result = {"config": name, "metrics": metrics, "models": {}}
            for label in ("base", "sft"):
                record = read(args.out / "runs" / name / "predictions" / label / f"{TARGET_KEY}.json")
                ok, failures = structural(record)
                result["models"][label] = {
                    "structural_pass": ok,
                    "failures": failures,
                    "generated_tokens": record["generated_tokens"],
                    "last_end": (record.get("segments") or [{}])[-1].get("end"),
                    "longest_identical_run": record["diagnostics"]["longest_identical_run"],
                }
                group_ok = group_ok and ok
            regressions = {
                label: {metric: 100 * (metrics[label][metric] - baseline[label][metric]) for metric in baseline[label]}
                for label in ("base", "sft")
            }
            metric_guards = {
                label: all(delta <= 1.0 for delta in regressions[label].values()) for label in ("base", "sft")
            }
            result["metric_regression_pp"] = regressions
            result["metric_guard"] = metric_guards
            group_ok = group_ok and all(metric_guards.values())

            group_rows.append(result)
            rows.append(result)
        if group_ok:
            eligible_groups.append(group)

    selected = next((name for name in protocol["selection_priority"] if name in eligible_groups), None)
    config = protocol["groups"].get(selected) if selected else None
    selected_config = None
    if config:
        selected_config = {
            "name": selected,
            "repetition_penalty": config["repetition_penalty"],
            "temperature": config["temperature"],
            "seed": 0,
            "top_p": 1,
            "top_k": -1,
            "min_p": 0,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "selected_on_known_test_diagnostic": True,
            "blind_test_claim_allowed": False,
        }
    outcome = {
        "complete": True,
        "evaluated_groups": args.groups,
        "eligible_groups": eligible_groups,
        "selected": selected,
        "selected_config": selected_config,
        "full_test_authorized_by_gate": selected is not None,
        "rows": rows,
    }
    write(args.out / "sweep_outcome.json", outcome)
    write(args.out / "selected_config.json", selected_config)
    print(json.dumps(outcome, ensure_ascii=True))
    if selected is None:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
