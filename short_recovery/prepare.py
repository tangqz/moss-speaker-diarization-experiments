#!/usr/bin/env python3
"""Validate canonical data/tokenization and create a fully ordered training manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from bisect import bisect_right
import sys
from typing import Any

from .config import load_config
from .core import (IGNORE_INDEX, body_spans, choose_quartile_anchor, eligible_anchors,
                   legal_body_ids, loss_budget, sha256_bytes, stable_seed)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                row["_line_no"] = line_no
                rows.append(row)
    return rows


def parse_source(row: dict[str, Any], root: Path) -> tuple[str, str, str]:
    messages = row.get("messages")
    audios = row.get("audios")
    if (not isinstance(messages, list) or len(messages) != 2
            or [m.get("role") for m in messages] != ["user", "assistant"]):
        raise ValueError(f"line {row.get('_line_no')}: expected user/assistant messages")
    if not isinstance(audios, list) or len(audios) != 1:
        raise ValueError(f"line {row.get('_line_no')}: expected one complete audio")
    prompt, target = messages[0].get("content"), messages[1].get("content")
    if not all(isinstance(x, str) and x.strip() for x in (prompt, target, audios[0])):
        raise ValueError(f"line {row.get('_line_no')}: empty prompt, target, or audio")
    audio = Path(audios[0]).expanduser()
    if not audio.is_absolute():
        audio = (root / audio).resolve()
    return prompt.strip(), str(audio), target.strip()


class TokenizersAdapter:
    """Small compatibility layer used only for local tokenizer qualification."""
    def __init__(self, tokenizer):
        self.inner = tokenizer
        vocab = tokenizer.get_vocab()
        self._size = max(vocab.values()) + 1
        self.all_special_ids = tuple(sorted(
            int(token_id) for token_id, token in tokenizer.get_added_tokens_decoder().items()
            if token.special))

    def __len__(self):
        return self._size

    def encode(self, text, add_special_tokens=False):
        return self.inner.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids, **_kwargs):
        return self.inner.decode(ids, skip_special_tokens=False)


def target_encoding(tokenizer, target: str) -> tuple[list[int], list[tuple[int, int]]]:
    if isinstance(tokenizer, TokenizersAdapter):
        encoded = tokenizer.inner.encode(target, add_special_tokens=False)
        return encoded.ids, encoded.offsets
    encoded = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    return list(encoded["input_ids"]), [tuple(x) for x in encoded["offset_mapping"]]


def tokenizer_audit(manifest: Path, tokenizer_json: Path, output: Path) -> dict[str, Any]:
    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(str(tokenizer_json))
    tokenizer = TokenizersAdapter(raw)
    rows = read_jsonl(manifest)
    span_count = token_count = legal_count = eligible_like = 0
    invalid_rows = []
    legal = set(legal_body_ids(tokenizer))
    for row in rows:
        try:
            _prompt, _audio, target = parse_source(row, manifest.parent)
            ids, offsets = target_encoding(tokenizer, target)
            spans = body_spans(target)
            if not spans:
                raise ValueError("no legal numeric timestamp/speaker/body/end span")
            span_count += len(spans)
            token_count += len(ids)
            span_starts = [lo for lo, _ in spans]
            body_flags = []
            for a, b in offsets:
                span_index = bisect_right(span_starts, a) - 1
                body_flags.append(a < b and span_index >= 0 and b <= spans[span_index][1])
            legal_count += sum(flag and token in legal for flag, token in zip(body_flags, ids))
            for i in range(1, max(1, len(ids) - 16)):
                if (body_flags[i] and ids[i] in legal and ids[i - 1] != ids[i] != ids[i + 1]
                        and sum(body_flags[i + 1:i + 17]) >= 4):
                    eligible_like += 1
        except Exception as exc:
            invalid_rows.append({"line": row.get("_line_no"), "reason": str(exc)})
    result = {
        "schema_version": "short-recovery-tokenizer-audit-v1",
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "tokenizer": str(tokenizer_json.resolve()),
        "tokenizer_sha256": file_sha256(tokenizer_json),
        "rows": len(rows), "body_spans": span_count, "target_tokens": token_count,
        "legal_body_tokens": legal_count, "eligible_like_positions": eligible_like,
        "invalid_rows": invalid_rows, "status": "pass" if not invalid_rows and eligible_like else "fail",
        "limitation": "Tokenizer-only qualification; official audio collator and absolute z coordinates are not exercised."
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def load_official(config):
    sys.path.insert(0, config.official_source)
    from finetune import DataCollator
    from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor
    processor = MossTranscribeDiarizeProcessor.from_pretrained(config.processor_path, trust_remote_code=True)
    return processor, DataCollator(processor, config.max_length)


def _encode_official(processor, collator, prompt: str, audio: str, target: str):
    batch = collator([{"prompt": prompt.replace("<audio>", "").strip(), "audio": audio, "target": target}])
    z = batch["input_ids"][0].tolist()
    labels = batch["labels"][0].tolist()
    if labels[-1] != processor.tokenizer.eos_token_id:
        raise ValueError("official collator did not preserve target EOS")
    target_ids, offsets = target_encoding(processor.tokenizer, target)
    target_start = next((i for i, x in enumerate(labels) if x != IGNORE_INDEX), None)
    if target_start is None or z[target_start:target_start + len(target_ids)] != target_ids:
        raise ValueError("standalone target tokenizer IDs do not align with official collator IDs")
    return z, labels, target_ids, offsets, target_start


def build_manifest(config_path: Path, output_dir: Path, source_limit: int | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    train_path = Path(config.train_manifest)
    if file_sha256(train_path) != config.train_sha256:
        raise ValueError("canonical train manifest SHA-256 mismatch")
    rows = read_jsonl(train_path)
    if len(rows) != 536:
        raise ValueError(f"expected 536 train rows, got {len(rows)}")
    dev_path = Path(config.dev_manifest)
    if file_sha256(dev_path) != config.dev_sha256:
        raise ValueError("canonical dev manifest SHA-256 mismatch")
    dev_rows = read_jsonl(dev_path)
    if len(dev_rows) != 26:
        raise ValueError(f"expected 26 dev rows, got {len(dev_rows)}")

    def identity(row, root):
        _prompt, audio, target = parse_source(row, root)
        normalized = os.path.normcase(os.path.realpath(audio))
        parts = Path(audio).parts
        corpus = next((parts[i + 1].lower() for i, part in enumerate(parts[:-1])
                       if part.lower() == "external"), "unknown")
        recording = Path(audio).stem.lower()
        return normalized, f"{corpus}:{recording}", sha256_bytes(target.encode()), audio

    train_identities = [identity(row, train_path.parent) for row in rows]
    dev_identities = [identity(row, dev_path.parent) for row in dev_rows]
    for field, label in [(0, "canonical audio path"), (1, "corpus+recording ID")]:
        overlap = set(x[field] for x in train_identities) & set(x[field] for x in dev_identities)
        if overlap:
            raise ValueError(f"train/dev overlap by {label}: {sorted(overlap)[:5]}")
    known_test = {"r8005_m8009"}
    contaminated = [x[1] for x in train_identities + dev_identities
                    if any(key in x[1] for key in known_test)]
    if contaminated:
        raise ValueError(f"known-test recording entered train/dev: {contaminated}")

    audio_hash_cache = {}
    for _path_id, _recording, _target_hash, audio in train_identities + dev_identities:
        if audio not in audio_hash_cache:
            audio_path = Path(audio)
            if not audio_path.is_file():
                raise FileNotFoundError(f"audio required for split-isolation hash is missing: {audio}")
            audio_hash_cache[audio] = file_sha256(audio_path)
    train_audio_hashes = {audio_hash_cache[x[3]] for x in train_identities}
    dev_audio_hashes = {audio_hash_cache[x[3]] for x in dev_identities}
    if train_audio_hashes & dev_audio_hashes:
        raise ValueError("train/dev overlap by audio SHA-256")
    processor, collator = load_official(config)
    legal_ids = legal_body_ids(processor.tokenizer)
    legal_id_set = set(legal_ids)
    needed = config.optimizer_updates * config.source_slots_per_update
    if source_limit is not None:
        needed = min(needed, source_limit)
    # A 150-update run consumes 600 source slots, so it crosses the 536-row
    # boundary.  Preserve the seed-0 first-epoch order used by the pilot and
    # build each later epoch from a fresh deterministic seed.
    chosen = []
    source_epoch = 0
    while len(chosen) < needed:
        order = list(range(len(rows)))
        random.Random(config.seed + source_epoch).shuffle(order)
        chosen.extend(order[:needed - len(chosen)])
        source_epoch += 1
    prepared = []
    eligibility = []
    for occurrence, index in enumerate(chosen):
        row = rows[index]
        prompt, audio, target = parse_source(row, train_path.parent)
        source_key = hashlib.sha256((os.path.normcase(os.path.normpath(audio)) + "\x1f" + target).encode()).hexdigest()
        try:
            z, labels, target_ids, offsets, target_start = _encode_official(processor, collator, prompt, audio, target)
            anchors = eligible_anchors(z=z, labels=labels, target_offsets=offsets,
                                      target_start=target_start, spans=body_spans(target),
                                      tokenizer=processor.tokenizer, k=config.k,
                                      precomputed_legal_ids=legal_id_set)
            anchor, quartile, quartile_fallback = choose_quartile_anchor(
                anchors, target_start=target_start, target_token_count=len(target_ids), seed=config.seed,
                source_key=source_key, occurrence=occurrence)
            span_starts = [lo for lo, _ in body_spans(target)]
            spans = body_spans(target)
            body_flags = []
            for aa, bb in offsets:
                si = bisect_right(span_starts, aa) - 1
                body_flags.append(aa < bb and si >= 0 and bb <= spans[si][1])
            rel_anchor = anchor - target_start
            body_label_count = sum(body_flags[rel_anchor + 1:rel_anchor + 1 + config.k])
            prepared.append({"source": (prompt, audio, target), "source_key": source_key,
                             "source_target_hash": sha256_bytes(target.encode()), "z": z, "labels": labels,
                             "anchor": anchor, "quartile": quartile, "quartile_fallback": quartile_fallback,
                             "clean_count": sum(x != IGNORE_INDEX for x in labels), "raw_length": len(z),
            "original_source_index": index, "source_epoch": occurrence // len(rows),
            "recording_id": train_identities[index][1],
                             "audio_sha256": audio_hash_cache[audio], "body_label_count": body_label_count})
            prepared[-1]["original_body_token_count"] = sum(body_flags)
            eligibility.append({"occurrence": occurrence, "source_key": source_key, "eligible": len(anchors),
                                "anchor": anchor, "quartile": quartile, "quartile_fallback": quartile_fallback})
        except Exception as exc:
            eligibility.append({"occurrence": occurrence, "source_key": source_key, "eligible": 0,
                                "reason": str(exc)})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "eligibility.json").write_text(json.dumps(eligibility, indent=2) + "\n")
    if len(prepared) != len(chosen):
        raise RuntimeError("at least one planned source lacks a valid anchor; no all-matched manifest emitted")

    expanded = []
    updates = []
    for update, start in enumerate(range(0, len(prepared), 4)):
        group = prepared[start:start + 4]
        if len(group) != 4:
            raise RuntimeError("partial update is forbidden")
        nc = sum(x["clean_count"] for x in group)
        na = 0 if config.arm == "ce" else 4 * config.k
        budget = loss_budget(nc, na, update, ce_only=config.arm == "ce",
                             maximum=config.lambda_max, warmup_updates=config.lambda_warmup_updates)
        mix_sub_slots = set()
        mix_repeat_counts = {}
        error_repeat_counts = {}
        if config.arm == "mix":
            mix_sub_slots = set(sorted(range(4), key=lambda s: stable_seed(config.seed, update, s, "mix"))[:2])
            repeat_slots = sorted(set(range(4)) - mix_sub_slots,
                                  key=lambda s: stable_seed(config.seed, update, s, "repeat-order"))
            mix_repeat_counts = {repeat_slots[0]: 1, repeat_slots[1]: 2}
        elif config.arm == "sub_repeat":
            # The sampled error occupies the original anchor. Insert two or
            # three copies so the erroneous token occurs three/four times in
            # total. Balance both lengths within every optimizer update while
            # letting the new experiment seed choose which source slots get each.
            ordered = sorted(range(4), key=lambda s: stable_seed(
                config.seed, update, s, "error-repeat-length"))
            error_repeat_counts = {slot: (2 if i < 2 else 3)
                                   for i, slot in enumerate(ordered)}
        for slot, item in enumerate(group):
            prompt, audio, target = item["source"]
            kinds = ["clean"]
            if config.arm != "ce":
                if config.arm == "clean_aux":
                    kinds.append("clean_aux")
                elif config.arm == "sub":
                    kinds.append("substitute")
                elif config.arm == "sub_repeat":
                    kinds.append("substitute_then_repeat")
                else:
                    kinds.append("substitute" if slot in mix_sub_slots else "repeat_previous")
            for view, kind in enumerate(kinds):
                repeat_count = (error_repeat_counts.get(slot, 0)
                                if config.arm == "sub_repeat"
                                else mix_repeat_counts.get(slot, 0))
                meta = {
                    "schema_version": "short-recovery-row-v2-sparse", "arm": config.arm,
                    "optimizer_step": update, "source_slot": slot, "occurrence": start + slot,
                    "view": "clean" if kind == "clean" else "aux", "event_kind": kind,
                    "source_key": item["source_key"], "source_target_hash": item["source_target_hash"],
                    "split": "train", "recording_id": item["recording_id"],
                    "audio_sha256": item["audio_sha256"],
                    "original_source_index": item["original_source_index"],
                    "source_epoch": item["source_epoch"],
                    "anchor_abs": item["anchor"], "query_abs": item["anchor"] - 1,
                    "k": config.k, "repeat_count": repeat_count, "raw_length": item["raw_length"],
                    "clean_count": item["clean_count"], "planned_nc": budget.nc,
                    "planned_na": budget.na, "planned_n": budget.total,
                    "lambda": budget.lambda_value, "clean_scale": budget.clean_scale,
                    "aux_scale": budget.aux_scale, "quartile": item["quartile"],
                    "quartile_fallback": item["quartile_fallback"],
                    "body_label_count": item["body_label_count"],
                    "original_body_token_count": item["original_body_token_count"],
                    "model_history_probability": config.model_history_probability,
                    "sampling_probability": config.sampling_probability,
                    "sampling_temperature": config.sampling_temperature,
                    "repeat_probability": config.repeat_probability
                }
                expanded.append({"messages": [{"role": "user", "content": prompt},
                                               {"role": "assistant", "content": target}],
                                 "audios": [audio], "chat_template_kwargs": {"recovery_meta": meta}})
        updates.append({"optimizer_step": update, "nc": budget.nc, "na": budget.na,
                        "n": budget.total, "lambda": budget.lambda_value,
                        "clean_scale": budget.clean_scale, "aux_scale": budget.aux_scale})
    manifest = output_dir / f"train.{config.arm}.jsonl"
    manifest.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n"
                                for x in expanded), encoding="utf-8")
    (output_dir / "updates.json").write_text(json.dumps(updates, indent=2) + "\n")
    (output_dir / "legal_body_ids.json").write_text(json.dumps(list(legal_ids)) + "\n")
    tokenizer_payload = processor.tokenizer.backend_tokenizer.to_str().encode("utf-8")
    special_payload = json.dumps(sorted(processor.tokenizer.all_special_ids), separators=(",", ":")).encode()
    legal_payload = json.dumps(list(legal_ids), separators=(",", ":")).encode("utf-8")
    targets_payload = "".join(parse_source(row, train_path.parent)[2] + "\n" for row in rows).encode("utf-8")
    summary = {"schema_version": "short-recovery-manifest-v2-sparse", "arm": config.arm,
               "source_rows": len(prepared), "expanded_rows": len(expanded),
               "manifest": str(manifest), "manifest_sha256": file_sha256(manifest),
               "config_sha256": config.digest, "tokenizer_sha256": sha256_bytes(tokenizer_payload),
               "special_ids_sha256": sha256_bytes(special_payload),
               "all_train_targets_sha256": sha256_bytes(targets_payload),
               "legal_body_ids": len(legal_ids), "legal_body_ids_sha256": sha256_bytes(legal_payload),
               "train_audio_identity_count": len(train_audio_hashes),
               "dev_audio_identity_count": len(dev_audio_hashes), "split_isolation": "pass",
               "status": "pass"}
    (output_dir / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("tokenizer-audit")
    audit.add_argument("--manifest", required=True, type=Path)
    audit.add_argument("--tokenizer-json", required=True, type=Path)
    audit.add_argument("--output", required=True, type=Path)
    plan = sub.add_parser("manifest")
    plan.add_argument("--config", required=True, type=Path)
    plan.add_argument("--output-dir", required=True, type=Path)
    plan.add_argument("--source-limit", type=int)
    args = parser.parse_args()
    result = (tokenizer_audit(args.manifest, args.tokenizer_json, args.output)
              if args.command == "tokenizer-audit"
              else build_manifest(args.config, args.output_dir, args.source_limit))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
