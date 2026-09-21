import collections
import hashlib
import json
from pathlib import Path
import re
import sys
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8')
segment = re.compile(r'\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]', re.S)
swift_bytes = (HERE / 'train.jsonl').read_bytes()
original_bytes = (HERE / 'train_original.jsonl').read_bytes()
swift = [json.loads(line) for line in swift_bytes.decode('utf-8', errors='strict').splitlines()]
original = [json.loads(line) for line in original_bytes.decode('utf-8', errors='strict').splitlines()]
assert len(swift) == len(original) == 536
tokenizer = Tokenizer.from_file(str(HERE / 'tokenizer.json'))
stats = collections.defaultdict(collections.Counter)
issues = []
for index, (new, old) in enumerate(zip(swift, original)):
    conversation = old['conversation']
    audio = next(r['content'] for r in conversation if r.get('message_type') == 'audio')
    prompt = next(r['content'] for r in conversation
                  if r['role'] == 'user' and r.get('message_type') == 'text')
    target = next(r['content'] for r in conversation if r['role'] == 'assistant')
    corpus = next(c for c in ['aishell4', 'alimeeting', 'ami'] if '/' + c + '/' in audio)
    s = stats[corpus]
    s['records'] += 1
    exact = (new['audios'] == [audio] and
             new['messages'][0]['content'] == '<audio>\n' + prompt and
             new['messages'][1]['content'] == target)
    s['exact_audio_prompt_target_preserved'] += int(exact)
    if not exact:
        issues.append(dict(index=index, audio=audio, issue='conversion_mismatch'))
    s['replacement_characters'] += target.count('\ufffd')
    s['null_characters'] += target.count('\0')
    matches = list(segment.finditer(target))
    remainder = segment.sub('', target).strip()
    s['segments'] += len(matches)
    s['fully_parseable_records'] += int(not remainder)
    s['empty_targets'] += int(not target.strip())
    backwards = sum(float(b[1]) < float(a[1]) for a, b in zip(matches, matches[1:]))
    bad = sum(float(m[4]) < float(m[1]) for m in matches)
    zero = sum(float(m[4]) == float(m[1]) for m in matches)
    s['start_time_regressions'] += backwards
    s['negative_durations'] += bad
    s['zero_durations'] += zero
    if remainder or backwards or bad:
        issues.append(dict(index=index, audio=audio, issue='target_structure',
                           unparsed_characters=len(remainder), start_regressions=backwards,
                           negative_durations=bad))
    ids = tokenizer.encode(target, add_special_tokens=False).ids
    decoded = tokenizer.decode(ids, skip_special_tokens=False)
    s['tokenizer_roundtrip_exact_records'] += int(decoded == target)
    s['target_tokens'] += len(ids)
    s['text_characters'] += sum(len(m[3]) for m in matches)
    s['target_characters'] += len(target)
    if decoded != target:
        issues.append(dict(index=index, audio=audio, issue='tokenizer_roundtrip_mismatch'))
    if (index + 1) % 100 == 0:
        print(f'LABEL_AUDIT {index+1}/536', flush=True)
result = dict(
    scope='All 536 current MS-Swift training records',
    swift_sha256=hashlib.sha256(swift_bytes).hexdigest(),
    original_sha256=hashlib.sha256(original_bytes).hexdigest(),
    tokenizer_sha256=hashlib.sha256((HERE / 'tokenizer.json').read_bytes()).hexdigest(),
    corpora=dict(stats), issues=issues,
    semantic_label_accuracy_verified=False,
    audio_feature_encoding_recomputed=False,
    note='UTF-8, conversion, syntax and tokenizer roundtrip checks do not prove audio/text alignment or annotation correctness.')
(HERE / 'label_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
