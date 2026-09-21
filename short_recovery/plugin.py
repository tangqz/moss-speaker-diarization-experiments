"""Independent MS-Swift template plugin for short-recovery rows."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from swift.template import TemplateMeta, register_template

from .core import IGNORE_INDEX, clean_view

TEMPLATE_TYPE = "moss_short_recovery_v2_sparse"


def _load_baseline():
    existing = sys.modules.get("moss_plugin")
    if existing is not None and hasattr(existing, "MossTemplate"):
        return existing
    path = Path(os.environ.get(
        "MOSS_BASELINE_PLUGIN", "/work/qt28/moss/dkucc/ms_swift_20260914/moss_plugin.py"))
    if not path.is_file():
        raise RuntimeError(f"baseline MOSS plugin missing: {path}")
    spec = importlib.util.spec_from_file_location("moss_plugin", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["moss_plugin"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_baseline = _load_baseline()
MossTemplate = _baseline.MossTemplate


class RecoveryMossTemplate(MossTemplate):
    """Preserve official MOSS collation and add only labels/scales/metadata."""

    def _encode(self, inputs):
        if not getattr(self, "_recovery_tokenizer_verified", False):
            observed = hashlib.sha256(self.tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
            if observed != os.environ.get("RECOVERY_TOKENIZER_SHA256"):
                raise ValueError("runtime tokenizer differs from frozen recovery manifest tokenizer")
            self._recovery_tokenizer_verified = True
        encoded = super()._encode(inputs)
        meta = inputs.chat_template_kwargs.get("recovery_meta")
        if not isinstance(meta, dict) or meta.get("schema_version") != "short-recovery-row-v2-sparse":
            raise ValueError("missing or invalid recovery_meta in chat_template_kwargs")
        z = encoded["input_ids"]
        official_labels = encoded.get("labels")
        if official_labels is None or len(z) != len(official_labels):
            raise ValueError("training row lacks aligned official labels")
        if len(z) != int(meta["raw_length"]):
            raise ValueError("official collator length differs from frozen manifest")
        if int(official_labels[-1]) != int(self.tokenizer.eos_token_id):
            raise ValueError("official target EOS missing")
        view = meta["view"]
        if view == "clean":
            result = clean_view(z, official_labels, float(meta["clean_scale"]))
        elif view == "aux":
            anchor, k = int(meta["anchor_abs"]), int(meta["k"])
            if anchor + 1 + k > len(z):
                raise ValueError("auxiliary recovery range exceeds clean sequence")
            labels = [IGNORE_INDEX] * len(z)
            labels[anchor + 1:anchor + 1 + k] = z[anchor + 1:anchor + 1 + k]
            scales = [0.0] * len(z)
            scales[anchor + 1:anchor + 1 + k] = [float(meta["aux_scale"])] * k
            result = type("Encoded", (), {"input_ids": tuple(z), "labels": tuple(labels),
                                            "loss_scale": tuple(scales)})()
        else:
            raise ValueError(f"unknown view: {view}")
        encoded["input_ids"] = list(result.input_ids)
        encoded["labels"] = list(result.labels)
        encoded["loss_scale"] = list(result.loss_scale)
        encoded["_recovery_meta_json"] = json.dumps(meta, sort_keys=True, separators=(",", ":"))
        return encoded

    def _data_collator(self, batch, *, padding_to=None):
        collator_batch = [dict(row) for row in batch]
        metadata = [row.pop("_recovery_meta_json") for row in collator_batch]
        before_lengths = [len(row["input_ids"]) for row in collator_batch]
        encoded = super()._data_collator(collator_batch, padding_to=padding_to)
        # The baseline implicit-causal branch pads input_ids/labels after the
        # generic Template collator.  Mirror that right padding for loss_scale.
        if "loss_scale" in encoded and encoded["loss_scale"].shape[-1] < encoded["input_ids"].shape[-1]:
            extra = encoded["input_ids"].shape[-1] - encoded["loss_scale"].shape[-1]
            encoded["loss_scale"] = torch.nn.functional.pad(encoded["loss_scale"], (0, extra), value=0.0)
        if encoded["input_ids"].shape[0] != 1 or len(metadata) != 1:
            raise ValueError("v1 only supports one complete meeting per microbatch")
        encoded["recovery_meta_json"] = metadata
        encoded["recovery_unpadded_length"] = before_lengths
        return encoded


register_template(TemplateMeta(
    template_type=TEMPLATE_TYPE, prefix=[], prompt=["{{QUERY}}"], chat_sep=None,
    suffix=[["eos_token_id"]], template_cls=RecoveryMossTemplate))
