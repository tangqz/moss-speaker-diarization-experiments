"""Ignore selected target positions in native CE, preserving all input tokens."""
import hashlib
import json
import os
from pathlib import Path


def digest_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_ids(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def compact_ranges(selected):
    ranges = []
    for i, flag in enumerate(selected):
        if flag:
            if ranges and ranges[-1][1] == i:
                ranges[-1][1] = i + 1
            else:
                ranges.append([i, i + 1])
    return ranges


def token_mask_from_char_spans(target, tokenizer, char_spans, eos_token):
    encoded = tokenizer(target + eos_token, add_special_tokens=False,
                        return_offsets_mapping=True)
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    assert ids[-1] == tokenizer.eos_token_id
    selected = []
    cursor = 0
    for i, (start, end) in enumerate(offsets):
        while cursor < len(char_spans) and char_spans[cursor][1] <= start:
            cursor += 1
        # A BPE token crossing a selected boundary is indivisible: ignore it.
        hit = (cursor < len(char_spans) and start < char_spans[cursor][1]
               and end > char_spans[cursor][0] and end > start)
        selected.append(bool(hit and i != len(ids)-1))
    assert not selected[-1]
    assert 0 < sum(not v for v in selected[:-1]), "Would train only EOS"
    return dict(target_sha256=digest_text(target), target_ids_sha256=digest_ids(ids),
                target_tokens_with_eos=len(ids), ignored_tokens=sum(selected),
                retained_tokens=len(ids)-sum(selected), ignore_ranges=compact_ranges(selected))


class LossMaskIndex:
    def __init__(self, path):
        self.path = Path(path)
        self.plan = json.loads(self.path.read_text(encoding="utf-8"))
        if not self.plan.get("ready_for_training"):
            raise RuntimeError("Overlap mask plan has not passed its data/semantic gate")
        self.records = self.plan["records"]
        self.dev_audio = set(self.plan["unmasked_dev_audio"])
        assert set(self.records).isdisjoint(self.dev_audio)

    def apply(self, audio_path, target, input_ids, labels):
        # This is only label masking. Context representations can still receive
        # gradients through losses at other positions; no hidden states are detached.
        import torch
        key = str(audio_path)
        if key in self.dev_audio:
            return labels
        if key not in self.records:
            raise ValueError(f"Audio missing from both train mask and dev allowlist: {key}")
        entry = self.records[key]
        if digest_text(target) != entry["target_sha256"]:
            raise ValueError("Training target changed after the mask was prepared")
        assert labels.ndim == input_ids.ndim == 1
        supervised = labels.ne(-100).nonzero().flatten()
        if len(supervised) != entry["target_tokens_with_eos"]:
            raise ValueError("Official collator target length differs from mask indexing")
        first = int(supervised[0])
        assert torch.equal(supervised, torch.arange(first, len(labels), device=labels.device))
        target_ids = input_ids[first:].tolist()
        if digest_ids(target_ids) != entry["target_ids_sha256"]:
            raise ValueError("Mask tokens do not match the official collator target tokens")
        assert torch.equal(labels[first:], input_ids[first:])
        masked = labels.clone()
        for start, end in entry["ignore_ranges"]:
            assert 0 <= start < end < len(target_ids), "EOS must remain supervised"
            masked[first+start:first+end] = -100
        assert int(masked[first:].eq(-100).sum()) == entry["ignored_tokens"]
        assert masked[-1] == labels[-1] and bool(masked[:first].eq(-100).all())
        return masked


_INDEX = None


def apply_training_mask(audio_path, target, input_ids, labels):
    global _INDEX
    path = os.environ.get("MOSS_OVERLAP_MASK_PLAN")
    if not path:
        raise RuntimeError("Overlap ablation requires an explicit, validated mask plan")
    if _INDEX is None:
        _INDEX = LossMaskIndex(path)
    return _INDEX.apply(audio_path, target, input_ids, labels)
