"""Temporary launcher that reports the exact distributed manifest length mismatch."""
from __future__ import annotations

import hashlib
import json
import os

from short_recovery import plugin


_original_encode = plugin.RecoveryMossTemplate._encode


def _diagnostic_encode(self, inputs):
    try:
        return _original_encode(self, inputs)
    except ValueError as exc:
        if "official collator length differs" not in str(exc):
            raise
        meta = inputs.chat_template_kwargs.get("recovery_meta", {})
        baseline = plugin.MossTemplate._encode(self, inputs)
        messages = inputs.messages
        target = messages[1].get("content", "") if len(messages) > 1 else ""
        report = {
            "rank": os.environ.get("RANK"),
            "local_rank": os.environ.get("LOCAL_RANK"),
            "expected_raw_length": meta.get("raw_length"),
            "observed_raw_length_second_pass": len(baseline["input_ids"]),
            "occurrence": meta.get("occurrence"),
            "view": meta.get("view"),
            "source_key": meta.get("source_key"),
            "stored_target_sha256": meta.get("source_target_hash"),
            "observed_target_sha256": hashlib.sha256(target.strip().encode()).hexdigest(),
            "audio": inputs.audios,
            "message_roles": [message.get("role") for message in messages],
            "prompt_prefix": messages[0].get("content", "")[:120] if messages else None,
        }
        print("SHORT_RECOVERY_LENGTH_DIAGNOSTIC=" + json.dumps(report, ensure_ascii=False), flush=True)
        raise


plugin.RecoveryMossTemplate._encode = _diagnostic_encode

from short_recovery.train import main


if __name__ == "__main__":
    main()
