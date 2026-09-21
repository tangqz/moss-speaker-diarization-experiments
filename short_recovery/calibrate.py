#!/usr/bin/env python3
"""Bounded Dev26 greedy-token calibration for sparse recovery v2."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path

import torch

from .config import load_config
from .core import (argmax_smallest, body_spans, candidate_set, eligible_anchors,
                   legal_body_ids, tempered_weights)
from .injected_eval import _device_batch, _load_model, _official_batch
from .prepare import _encode_official, file_sha256, load_official, parse_source, read_jsonl, target_encoding
from .preflight import experiment_source_hash, tree_hash


def _uniform(values, limit=64):
    values = sorted(set(values))
    if len(values) <= limit:
        return values
    return [values[min(len(values) - 1, int((i + 0.5) * len(values) / limit))] for i in range(limit)]


def _wilson(errors, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    p = errors / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def run(config_path: Path, output: Path):
    config = load_config(config_path)
    dev_path = Path(config.dev_manifest)
    if file_sha256(dev_path) != config.dev_sha256:
        raise RuntimeError("Dev hash mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Dev calibration requires exactly one allocated GPU")
    device = torch.device("cuda", 0); torch.cuda.set_device(0)
    processor, collator = load_official(config)
    legal = set(legal_body_ids(processor.tokenizer))
    tokenizer_hash = hashlib.sha256(processor.tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    model = _load_model(config, config.model_path, device)
    rows = read_jsonl(dev_path)
    if len(rows) != 26:
        raise RuntimeError(f"Dev calibration requires exactly 26 meetings, got {len(rows)}")
    totals = {"all_body": {"n": 0, "wrong": 0, "legal_wrong": 0, "illegal": 0},
              "eligible": {"n": 0, "wrong": 0, "legal_wrong": 0,
                           "illegal": 0, "temperature_error_sum": 0.0,
                           "low_mass": 0}}
    per_meeting = []
    for row in rows:
        prompt, audio, target = parse_source(row, dev_path.parent)
        batch = _official_batch(collator, prompt, audio, target)
        z, labels, target_ids, offsets, target_start = _encode_official(
            processor, collator, prompt, audio, target)
        spans = body_spans(target); starts = [lo for lo, _ in spans]
        all_body = []
        for rel, (a, b) in enumerate(offsets):
            si = bisect_right(starts, a) - 1
            absolute = target_start + rel
            if a < b and si >= 0 and b <= spans[si][1] and absolute > 0:
                all_body.append(absolute)
        eligible = list(eligible_anchors(
            z=z, labels=labels, target_offsets=offsets, target_start=target_start, spans=spans,
            tokenizer=processor.tokenizer, k=config.k, precomputed_legal_ids=legal))
        all_sample = _uniform(all_body); eligible_sample = _uniform(eligible)
        if not all_sample or not eligible_sample:
            raise RuntimeError("each Dev meeting must provide all-body and eligible calibration positions")
        union = sorted(set(all_sample + eligible_sample))
        model_inputs = _device_batch(batch, device)
        positions = torch.tensor([x - 1 for x in union], device=device, dtype=torch.long)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**model_inputs, use_cache=False, logits_to_keep=positions).logits[0].float().cpu()
        by_anchor = {anchor: logits[i].tolist() for i, anchor in enumerate(union)}
        item = {"source_key": hashlib.sha256((audio + "\x1f" + target).encode()).hexdigest(),
                "all_body_population": len(all_body), "eligible_population": len(eligible),
                "all_body_sample": len(all_sample), "eligible_sample": len(eligible_sample),
                "all_body_wrong": 0, "eligible_wrong": 0, "eligible_illegal": 0}
        for group_name, sample in (("all_body", all_sample), ("eligible", eligible_sample)):
            for anchor in sample:
                greedy = argmax_smallest(by_anchor[anchor]); gold = z[anchor]
                wrong = greedy != gold; illegal = greedy not in legal
                legal_wrong = wrong and not illegal
                totals[group_name]["n"] += 1; totals[group_name]["wrong"] += int(wrong)
                totals[group_name]["legal_wrong"] += int(legal_wrong)
                totals[group_name]["illegal"] += int(illegal)
                item[f"{group_name}_wrong"] = item.get(f"{group_name}_wrong", 0) + int(wrong)
                if group_name == "eligible":
                    item["eligible_illegal"] += int(illegal)
                    candidates = candidate_set(by_anchor[anchor], gold, legal, include_gold=True)
                    if candidates.candidate_mass < 0.5:
                        totals["eligible"]["low_mass"] += 1
                    else:
                        weights = tempered_weights(candidates.probs, config.sampling_temperature)
                        qgold = (weights[candidates.ids.index(gold)] / sum(weights)
                                 if gold in candidates.ids else 0.0)
                        totals["eligible"]["temperature_error_sum"] += 1 - qgold
        per_meeting.append(item)
    all_e0 = totals["all_body"]["wrong"] / max(1, totals["all_body"]["n"])
    e0 = totals["eligible"]["legal_wrong"] / totals["eligible"]["n"]
    e_t = totals["eligible"]["temperature_error_sum"] / totals["eligible"]["n"]
    epsilon = min(0.02, config.sampling_probability)
    e_mix = (1 - epsilon) * e0 + epsilon * e_t
    eta = min(1.0, (e0 + 0.02) / e_mix) if e_mix > 0 else 1.0
    repeat = min(0.05, e0)
    payload = {"schema_version": "short-recovery-dev-calibration-v2", "status": "pass",
               "protocol_version": config.protocol_version, "dev_sha256": config.dev_sha256,
               "initial_checkpoint": config.model_path, "tokenizer_sha256": tokenizer_hash,
               "model_tree_sha256": tree_hash(Path(config.model_path)),
               "experiment_source_sha256": experiment_source_hash(),
               "sampling_temperature": config.sampling_temperature,
               "all_body": dict(totals["all_body"], error_rate=all_e0,
                                wilson95=_wilson(totals["all_body"]["wrong"], totals["all_body"]["n"])),
               "eligible": dict(totals["eligible"], raw_error_rate=(
                                    totals["eligible"]["wrong"] / totals["eligible"]["n"]),
                                legal_error_rate=e0, expected_temperature_error=e_t,
                                legal_error_wilson95=_wilson(
                                    totals["eligible"]["legal_wrong"], totals["eligible"]["n"])),
               "mixture_error_at_eta1": e_mix, "recommendation": {
                   "model_history_probability_max": eta, "sampling_probability_max": epsilon,
                   "repeat_probability_max": repeat}, "per_meeting": per_meeting,
               "meeting_count": len(rows), "records_complete": len(per_meeting) == len(rows),
               "note": ("Test CER is context only; probabilities use Dev token probes. Nominal Wilson "
                        "intervals ignore within-meeting dependence and are descriptive only; inference "
                        "requires meeting-level resampling."),
               "limitation": "Per-corpus calibration aggregation is not implemented in this entrypoint."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path); parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(); print(json.dumps(run(args.config, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
