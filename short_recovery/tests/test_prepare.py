import json
from pathlib import Path

from tokenizers import Tokenizer

from short_recovery.prepare import TokenizersAdapter, parse_source, target_encoding
from short_recovery.core import body_spans, legal_token_id


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "analysis" / "20260915-label-overlap"


def test_real_canonical_train_and_tokenizer_parse():
    row = json.loads((ASSETS / "train.jsonl").open(encoding="utf-8").readline())
    prompt, audio, target = parse_source(row, ASSETS)
    assert "<audio>" in prompt and audio.endswith(".wav")
    spans = body_spans(target)
    assert len(spans) > 10
    tokenizer = TokenizersAdapter(Tokenizer.from_file(str(ASSETS / "tokenizer.json")))
    ids, offsets = target_encoding(tokenizer, target)
    assert len(ids) == len(offsets) > 100
    body_id = next(token for token, (a, b) in zip(ids, offsets)
                   if any(a >= lo and b <= hi and a < b for lo, hi in spans))
    assert legal_token_id(tokenizer, body_id, tokenizer.all_special_ids)

