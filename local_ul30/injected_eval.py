#!/usr/bin/env python3
"""Build the fixed Dev injection library and run TF plus true free recovery."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import re
import sys
import traceback
import unicodedata

import torch

from .config import load_config
from .core import (IGNORE_INDEX, apply_event, body_spans, eligible_anchors,
                   legal_body_ids, sparse_substitution, stable_seed)
from .prepare import (_encode_official, file_sha256, load_official, parse_source,
                      read_jsonl)


def _device_batch(batch, device, *, include_labels=False):
    skip = set() if include_labels else {"labels"}
    return {k: v.to(device) for k, v in batch.items()
            if k not in skip and isinstance(v, torch.Tensor)}


def _load_model(config, checkpoint: str, device):
    sys.path.insert(0, config.official_source)
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True, dtype=torch.float32,
        attn_implementation="sdpa").to(device)
    model.tie_weights()
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    return model.eval()


def _official_batch(collator, prompt, audio, target):
    return collator([{"prompt": prompt.replace("<audio>", "").strip(),
                      "audio": audio, "target": target}])


def _tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _normalize_body(text):
    text = re.sub(r"\[(?:S\d+|\d+(?:\.\d+)?)\]", "", text)
    return "".join(ch for ch in unicodedata.normalize("NFKC", text).casefold()
                   if not ch.isspace() and unicodedata.category(ch)[0] not in "PSZC")


def _edit_ops(reference, prediction):
    """Return deterministic Levenshtein S/D/I counts from reference to prediction."""
    # Each cell stores (total, substitutions, deletions, insertions); tuple
    # ordering makes ties stable without claiming a unique linguistic alignment.
    prev = [(j, 0, 0, j) for j in range(len(prediction) + 1)]
    for i, ref in enumerate(reference, 1):
        cur = [(i, 0, i, 0)]
        for j, pred in enumerate(prediction, 1):
            if ref == pred:
                cur.append(prev[j - 1])
            else:
                sub = (prev[j - 1][0] + 1, prev[j - 1][1] + 1,
                       prev[j - 1][2], prev[j - 1][3])
                delete = (prev[j][0] + 1, prev[j][1], prev[j][2] + 1, prev[j][3])
                insert = (cur[-1][0] + 1, cur[-1][1], cur[-1][2], cur[-1][3] + 1)
                cur.append(min(sub, delete, insert))
        prev = cur
    total, substitutions, deletions, insertions = prev[-1]
    return {"distance": total, "substitutions": substitutions,
            "deletions": deletions, "insertions": insertions}


def _candidate_probe(model, batch, anchor, device):
    prefix = _device_batch(batch, device)
    prefix["input_ids"] = prefix["input_ids"][:, :anchor]
    prefix["attention_mask"] = prefix["attention_mask"][:, :anchor]
    prefix["position_ids"] = torch.arange(anchor, device=device).unsqueeze(0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**prefix, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
    return logits.tolist()


def build_library(config_path: Path, output: Path):
    config = load_config(config_path)
    if file_sha256(Path(config.dev_manifest)) != config.dev_sha256:
        raise RuntimeError("Dev manifest hash mismatch")
    rows = read_jsonl(Path(config.dev_manifest))
    if len(rows) != 26:
        raise RuntimeError("fixed library requires all 26 Dev meetings")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("fixed-library construction requires one explicitly allocated GPU")
    torch.cuda.set_device(0); device = torch.device("cuda", 0)
    processor, collator = load_official(config)
    legal = set(legal_body_ids(processor.tokenizer))
    model = _load_model(config, config.model_path, device)
    sources, cases = [], []
    for source_index, row in enumerate(rows):
        prompt, audio, target = parse_source(row, Path(config.dev_manifest).parent)
        batch = _official_batch(collator, prompt, audio, target)
        z, labels, target_ids, offsets, target_start = _encode_official(
            processor, collator, prompt, audio, target)
        anchors = list(eligible_anchors(z=z, labels=labels, target_offsets=offsets,
                                        target_start=target_start, spans=body_spans(target),
                                        tokenizer=processor.tokenizer, k=config.k,
                                        precomputed_legal_ids=legal))
        selected = []
        for progress in (0.25, 0.50, 0.75):
            wanted = target_start + progress * len(target_ids)
            available = [x for x in anchors if x not in selected]
            if available:
                selected.append(min(available, key=lambda x: (abs(x - wanted), x)))
        source_key = hashlib.sha256((audio + "\x1f" + target).encode()).hexdigest()
        sources.append({"source_index": source_index, "source_key": source_key,
                        "audio": audio, "prompt": prompt, "target": target,
                        "clean_ids": z, "clean_ids_sha256": hashlib.sha256(
                            bytes().join(int(x).to_bytes(4, "little", signed=True) for x in z)).hexdigest(),
                        "feature_sha256": {key: _tensor_sha(batch[key]) for key in
                                            ("input_features", "audio_feature_lengths", "audio_chunk_mapping")},
                        "anchors_found": len(anchors), "selected_anchors": selected})
        for anchor_index, anchor in enumerate(selected):
            gold = z[anchor]
            logits = _candidate_probe(model, batch, anchor, device)
            seed_parts = (config.seed, source_key, anchor_index, anchor, "fixed-dev-sub")
            decision = sparse_substitution(
                logits, gold=gold, legal_ids=legal,
                history_seed=stable_seed(*seed_parts, "model-history"),
                branch_seed=stable_seed(*seed_parts, "temperature-branch"),
                sample_seed=stable_seed(*seed_parts, "temperature-sample"),
                model_history_probability=config.model_history_probability,
                sampling_probability=config.sampling_probability,
                temperature=config.sampling_temperature,
                minimum_candidate_mass=0.5)
            replacement = decision.replacement
            base = {"source_index": source_index, "source_key": source_key,
                    "anchor_index": anchor_index, "anchor_abs": anchor, "query_abs": anchor - 1,
                    "k": config.k, "gold_id": gold,
                    "history_triggered": decision.history_triggered,
                    "temperature_branch": decision.temperature_branch,
                    "argmax_id": decision.argmax_id, "sample_equals_gold": decision.sample_equals_gold,
                    "candidate_mass": (decision.candidates.candidate_mass
                                       if decision.candidates is not None else None)}
            cases.append(dict(base, case="clean", applicable=True, replacement_id=None, repeat_count=0))
            cases.append(dict(base, case="substitute", applicable=decision.changed,
                              replacement_id=replacement, repeat_count=0,
                              not_applicable_reason=decision.reason))
            cases.append(dict(base, case="repeat2", applicable=True, replacement_id=None, repeat_count=1))
            cases.append(dict(base, case="repeat3", applicable=True, replacement_id=None, repeat_count=2))
    payload = {"schema_version": "short-recovery-fixed-dev-v2-sparse", "status": "complete",
               "evaluation_identity": hashlib.sha256(json.dumps({
                   "protocol": "short-recovery-fixed-dev-v2-sparse", "dev": config.dev_sha256,
                   "checkpoint": config.model_path, "tokenizer": hashlib.sha256(
                       processor.tokenizer.backend_tokenizer.to_str().encode()).hexdigest(), "k": config.k,
                   "model_history_probability": config.model_history_probability,
                   "sampling_probability": config.sampling_probability,
                   "sampling_temperature": config.sampling_temperature
               }, sort_keys=True).encode()).hexdigest(),
               "initial_checkpoint": config.model_path,
               "initial_checkpoint_role": "common checkpoint-150 weights", "dev_sha256": config.dev_sha256,
               "tokenizer_sha256": hashlib.sha256(processor.tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
               "sources": sources, "cases": cases, "case_count": len(cases),
               "max_possible_cases": 26 * 3 * 4}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    return payload


def _case_transform(z, case):
    kind = case["case"]
    if kind == "clean":
        return apply_event(z, anchor=case["anchor_abs"], k=case["k"], kind="clean_aux",
                           clean_scale=1, aux_scale=1)
    if kind == "substitute":
        return apply_event(z, anchor=case["anchor_abs"], k=case["k"], kind="substitute",
                           replacement=case["replacement_id"], clean_scale=1, aux_scale=1)
    return apply_event(z, anchor=case["anchor_abs"], k=case["k"], kind="repeat_previous",
                       repeat_count=case["repeat_count"], clean_scale=1, aux_scale=1)


def run_library(config_path: Path, library_path: Path, checkpoint: str, output: Path):
    config = load_config(config_path)
    library = json.loads(library_path.read_text())
    if (library.get("schema_version") != "short-recovery-fixed-dev-v2-sparse"
            or library.get("status") != "complete"
            or library.get("dev_sha256") != config.dev_sha256
            or library.get("initial_checkpoint") != config.model_path):
        raise RuntimeError("fixed library/config identity mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("injected evaluation requires one explicitly allocated GPU")
    torch.cuda.set_device(0); device = torch.device("cuda", 0)
    processor, collator = load_official(config)
    tokenizer_sha256 = hashlib.sha256(
        processor.tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    if tokenizer_sha256 != library.get("tokenizer_sha256"):
        raise RuntimeError("fixed library tokenizer hash mismatch")
    model = _load_model(config, checkpoint, device)
    source_map = {x["source_index"]: x for x in library["sources"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    with output.open("w", encoding="utf-8") as handle:
        for case in library["cases"]:
            record = dict(case, checkpoint=checkpoint, evaluation_scope="local_injected_256")
            if not case["applicable"]:
                record["status"] = "not_applicable"
                handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush(); continue
            try:
                source = source_map[case["source_index"]]
                batch = _official_batch(collator, source["prompt"], source["audio"], source["target"])
                if batch["input_ids"][0].tolist() != source["clean_ids"]:
                    raise RuntimeError("official re-collation changed clean token IDs")
                for key, digest in source["feature_sha256"].items():
                    if _tensor_sha(batch[key]) != digest:
                        raise RuntimeError(f"official re-collation changed {key}")
                z = source["clean_ids"]
                transformed = _case_transform(z, case)
                full = _device_batch(batch, device)
                full_ids = torch.tensor(transformed.input_ids, device=device).unsqueeze(0)
                full["input_ids"] = full_ids
                full["attention_mask"] = torch.ones_like(full_ids)
                full["position_ids"] = torch.arange(full_ids.shape[1], device=device).unsqueeze(0)
                indices = torch.arange(transformed.recovery_start - 1,
                                       transformed.recovery_end - 1, device=device)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(**full, use_cache=False, logits_to_keep=indices).logits[0].float()
                targets = torch.tensor(transformed.labels[transformed.recovery_start:transformed.recovery_end],
                                       device=device)
                losses = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
                top1 = logits.argmax(-1)
                tf = {str(n): {"ce": float(losses[:n].mean()),
                               "top1": float((top1[:n] == targets[:n]).float().mean())}
                      for n in (1, 4, 8, 16)}
                prefix_end = transformed.recovery_start
                prefix = _device_batch(batch, device)
                prefix_ids = full_ids[:, :prefix_end]
                prefix["input_ids"] = prefix_ids
                prefix["attention_mask"] = torch.ones_like(prefix_ids)
                prefix["position_ids"] = torch.arange(prefix_end, device=device).unsqueeze(0)
                model.config.use_cache = True; model.config.text_config.use_cache = True
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    generated = model.generate(
                        **prefix, max_new_tokens=256, do_sample=False, num_beams=1, use_cache=True,
                        logits_to_keep=1, repetition_penalty=1.0, no_repeat_ngram_size=0,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id)
                ids = generated[0, prefix_end:].tolist()
                truth16 = list(transformed.labels[transformed.recovery_start:transformed.recovery_end])
                truth256 = z[case["anchor_abs"] + 1:case["anchor_abs"] + 1 + 256]
                raw_text = processor.tokenizer.decode(ids, skip_special_tokens=False)
                truth_text = processor.tokenizer.decode(truth256, skip_special_tokens=False)
                pred_body_text = processor.tokenizer.decode(ids, skip_special_tokens=True)
                truth_body_text = processor.tokenizer.decode(truth256, skip_special_tokens=True)
                pred_norm, truth_norm = _normalize_body(pred_body_text), _normalize_body(truth_body_text)
                edit_ops = _edit_ops(truth_norm, pred_norm)
                pred_times = [float(x) for x in re.findall(r"\[(\d+(?:\.\d+)?)\]", raw_text)]
                truth_times = [float(x) for x in re.findall(r"\[(\d+(?:\.\d+)?)\]", truth_text)]
                record.update(status="ok", teacher_forced=tf, raw_ids=ids, raw_text=raw_text,
                              finish_reason="eos" if ids and ids[-1] == processor.tokenizer.eos_token_id else "length",
                              horizon_censored=not (ids and ids[-1] == processor.tokenizer.eos_token_id),
                              exact={str(n): ids[:n] == truth16[:n] for n in (1, 4, 8, 16)},
                              normalized_edit_distance=edit_ops["distance"],
                              local_body_edit_ops=edit_ops,
                              normalized_pred_chars=len(pred_norm), normalized_truth_chars=len(truth_norm),
                              last_pred_timestamp=max(pred_times) if pred_times else None,
                              last_truth_timestamp=max(truth_times) if truth_times else None)
            except Exception as exc:
                errors += 1
                record.update(status="error", error_type=type(exc).__name__, error=str(exc),
                              traceback=traceback.format_exc())
            handle.write(json.dumps(record, ensure_ascii=False) + "\n"); handle.flush()
    return {"status": "complete_with_errors" if errors else "complete",
            "errors": errors, "predictions": str(output),
            "limitation": "Local 256-token body alignment includes window-boundary effects and is not full-meeting CER."}


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-library")
    build.add_argument("--config", required=True, type=Path); build.add_argument("--output", required=True, type=Path)
    run = sub.add_parser("run-library")
    run.add_argument("--config", required=True, type=Path); run.add_argument("--library", required=True, type=Path)
    run.add_argument("--checkpoint", required=True); run.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = build_library(args.config, args.output) if args.command == "build-library" else run_library(
        args.config, args.library, args.checkpoint, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") == "complete_with_errors":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
