import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
STEPS = (5, 10, 15, 20, 30, 40)


def load(step):
    rows = json.loads((HERE / f"per_record_metrics_step{step}.json").read_text(encoding="utf-8"))
    return {row["key"]: row for row in rows if row["model"] == "sft"}


records = {step: load(step) for step in STEPS}
keys = {dataset: sorted(k for k in records[20] if k.startswith(dataset + "/"))
        for dataset in ("alimeeting", "ami")}
assert all(set(records[step]) == set(records[20]) for step in STEPS)


def score(step, sampled):
    dataset_terms = []
    for dataset, chosen in sampled.items():
        rows = [records[step][key] for key in chosen]
        ref_chars = sum(row["text"]["reference_characters"] for row in rows)
        cp_errors = sum(row["text"]["cp_character_errors"] for row in rows)
        der_errors = sum(
            row["diarization"]["DER_collar_0.25"][name]
            for row in rows for name in ("confusion", "missed detection", "false alarm")
        )
        der_total = sum(row["diarization"]["DER_collar_0.25"]["total"] for row in rows)
        dataset_terms.append((100 * cp_errors / ref_chars + 100 * der_errors / der_total) / 2)
    return sum(dataset_terms) / len(dataset_terms)


observed_sample = keys
observed = {step: score(step, observed_sample) for step in STEPS}

rng = np.random.default_rng(20260914)
n = 50000
boot = {step: np.empty(n) for step in STEPS}
for i in range(n):
    sampled = {
        dataset: [dataset_keys[j] for j in rng.integers(0, len(dataset_keys), len(dataset_keys))]
        for dataset, dataset_keys in keys.items()
    }
    for step in STEPS:
        boot[step][i] = score(step, sampled)

comparisons = {}
for other in (5, 10, 15, 30, 40):
    diff = boot[20] - boot[other]
    comparisons[f"20_minus_{other}"] = {
        "observed": observed[20] - observed[other],
        "ci95": np.quantile(diff, [0.025, 0.975]).tolist(),
        "probability_step20_better": float(np.mean(diff < 0)),
    }

matrix = np.column_stack([boot[step] for step in STEPS])
winner = np.argmin(matrix, axis=1)
winner_probability = {str(step): float(np.mean(winner == i)) for i, step in enumerate(STEPS)}

leave_one_out = []
all_keys = keys["alimeeting"] + keys["ami"]
for omitted in all_keys:
    sampled = {dataset: [k for k in dataset_keys if k != omitted]
               for dataset, dataset_keys in keys.items()}
    scores = {step: score(step, sampled) for step in STEPS}
    leave_one_out.append({"omitted": omitted, "winner": min(scores, key=scores.get), "scores": scores})

result = {
    "method": "paired meeting bootstrap, resampled within dataset; lower score is better",
    "iterations": n,
    "meeting_counts": {dataset: len(dataset_keys) for dataset, dataset_keys in keys.items()},
    "observed_scores": observed,
    "comparisons": comparisons,
    "winner_probability": winner_probability,
    "leave_one_out_step20_wins": sum(row["winner"] == 20 for row in leave_one_out),
    "leave_one_out_total": len(leave_one_out),
    "leave_one_out_non20": [row for row in leave_one_out if row["winner"] != 20],
}
(HERE / "checkpoint_bootstrap_result.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
