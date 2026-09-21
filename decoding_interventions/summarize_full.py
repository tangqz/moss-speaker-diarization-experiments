"""Compare the gated full Test score with the preserved historical greedy score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TRAIN_RUN = Path("/work/qt28/moss/results/ms-swift-63643")


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact(summary: dict) -> dict:
    result = {}
    for key, group in summary["groups"].items():
        result[key] = {
            "CER": group["CER"],
            "cpCER": group["cpCER"],
            "DeltaCP": group["DeltaCP"],
            "DER025": group["DER_collar_0.25"]["DER"],
            "truncated": group["truncated"],
            "empty_predictions": group["empty_predictions"],
            "RTF": group["RTF"],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    current = read(args.run / "metrics_summary.json")
    historical = read(TRAIN_RUN / "evaluations/test-150/metrics_summary.json")
    current_compact = compact(current)
    historical_compact = compact(historical)
    deltas = {}
    for key, values in current_compact.items():
        deltas[key] = {
            metric: values[metric] - historical_compact[key][metric]
            for metric in ("CER", "cpCER", "DeltaCP", "DER025", "RTF")
        }
    report = {
        "complete": current["complete"],
        "selected_config": read(args.run / "selected_config.json"),
        "new": current_compact,
        "historical_greedy": historical_compact,
        "delta_new_minus_greedy": deltas,
        "evidence_boundary": "Decoding config was selected using known Test case R8005_M8009; this is an engineering re-evaluation, not a fresh blind Test.",
    }
    write(args.run / "comparison_with_greedy.json", report)
    lines = [
        "# Full Test decoding-intervention re-evaluation",
        "",
        f'- Selected config: `{json.dumps(report["selected_config"], ensure_ascii=False)}`',
        "- Both Base and Step 150 were regenerated from scratch under the same config.",
        "- This is not a blind Test because R8005_M8009 was used to select the decoding config.",
        "",
        "| Group | CER | cpCER | DER@0.25 | Truncated | Delta CER vs greedy (pp) | Delta cpCER (pp) | Delta DER (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, values in current_compact.items():
        delta = deltas[key]
        lines.append(
            f'| {key} | {100*values["CER"]:.3f}% | {100*values["cpCER"]:.3f}% | '
            f'{100*values["DER025"]:.3f}% | {values["truncated"]} | {100*delta["CER"]:+.3f} | '
            f'{100*delta["cpCER"]:+.3f} | {100*delta["DER025"]:+.3f} |'
        )
    (args.run / "comparison_with_greedy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"complete": current["complete"], "report": str(args.run / "comparison_with_greedy.json")}))


if __name__ == "__main__":
    main()
