"""Freeze held-out inputs and references without exposing targets to inference."""
import argparse
import collections
import dataclasses
import hashlib
import json
from pathlib import Path
import sys

import soundfile as sf

ROOT = Path('/work/qt28/moss')
sys.path.insert(0, str(ROOT / 'MOSS-Transcribe-Diarize'))
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import build_transcription_messages
from moss_transcribe_diarize.processing_moss_transcribe_diarize import (
    MossTranscribeDiarizeProcessor, _compute_audio_token_length,
)


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    args.run.mkdir(parents=True, exist_ok=True)
    processor = MossTranscribeDiarizeProcessor.from_pretrained(
        ROOT / 'models/MOSS-Transcribe-Diarize', trust_remote_code=True)
    folder = ROOT / 'data/moss_jsonl'
    train = rows(folder / 'train_mix.jsonl')
    train_paths = {str(Path(c['content']).resolve()) for r in train for c in r['conversation']
                   if c['message_type'] == 'audio'}
    expected = {'alimeeting_dev': 8, 'ami_dev': 18, 'alimeeting_test': 20,
                'aishell4_test': 20, 'ami_test': 16}
    result, references, checksums, seen = [], {}, {}, set()
    for name, expected_count in expected.items():
        dataset, split = name.rsplit('_', 1)
        source = folder / (name + '.jsonl')
        records = rows(source)
        manifests = rows(ROOT / 'data/manifests' / (name + '.jsonl'))
        by_path = {str(Path(r['audio_path']).resolve()): r for r in manifests}
        assert len(records) == len(manifests) == expected_count
        train_ids = {r['session_id'] for r in rows(ROOT / 'data/manifests' / (dataset + '_train.jsonl'))}
        checksums[str(source)] = sha(source)
        for r in records:
            conversation = r['conversation']
            audio = [c['content'] for c in conversation if c['message_type'] == 'audio']
            assert len(audio) == 1
            audio = str(Path(audio[0]).resolve())
            meta = by_path[audio]
            assert meta['session_id'] not in train_ids
            assert audio not in train_paths and audio not in seen
            seen.add(audio)
            info = sf.info(audio)
            assert info.samplerate == 16000 and info.channels == 1 and info.frames > 0
            prompt = '\n'.join(c['content'] for c in conversation
                               if c['role'] == 'user' and c['message_type'] == 'text')
            target = ''.join(c['content'] for c in conversation if c['role'] == 'assistant')
            segments = [dataclasses.asdict(s) for s in parse_transcript(target)]
            assert segments and all(s['end'] > s['start'] >= 0 for s in segments)
            sid = meta['session_id']
            key = f'{dataset}/{split}/{sid}'
            refdir = ROOT / 'data/refs' / dataset / split
            rttm, uem = refdir / (sid + '.rttm'), refdir / (sid + '.uem')
            assert rttm.is_file() and uem.is_file(), key
            ref_lines = [l.split() for l in rttm.read_text().splitlines() if l.strip()]
            uem_lines = [l.split() for l in uem.read_text().splitlines() if l.strip()]
            assert ref_lines and uem_lines and all(l[1] == sid for l in ref_lines)
            assert all(l[0] == sid for l in uem_lines)
            chunk = int(processor.feature_extractor.n_samples)
            audio_tokens = sum(_compute_audio_token_length(min(chunk, info.frames - start),
                processor.feature_extractor, processor.audio_merge_size)
                for start in range(0, info.frames, chunk))
            prompt_text = processor.apply_chat_template(build_transcription_messages(audio, prompt),
                tokenize=False, add_generation_prompt=True)
            prompt_len = len(processor.expand_audio_token(prompt_text, audio_tokens, 131072))
            assert prompt_len < 131072
            result.append(dict(key=key, dataset=dataset, split=split, session_id=sid,
                audio=audio, prompt=prompt, duration=info.duration, prompt_len=prompt_len))
            references[key] = dict(segments=segments, rttm=rttm.read_text(), uem=uem.read_text(),
                source_target=target, rttm_sha256=sha(rttm), uem_sha256=sha(uem))
    # Greedy duration balancing is independent of labels and model outputs.
    for split in ['dev', 'test']:
        loads = [0.] * 4
        for record in sorted([r for r in result if r['split'] == split], key=lambda r: -r['duration']):
            rank = min(range(4), key=lambda i: (loads[i], i))
            record['rank'] = rank
            loads[rank] += record['duration']
    result.sort(key=lambda r: (r['split'] != 'dev', r['duration'], r['key']))
    (args.run / 'inputs.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    (args.run / 'references.json').write_text(json.dumps(references, ensure_ascii=False))
    models = {'base': ROOT / 'models/MOSS-Transcribe-Diarize',
              'sft': ROOT / 'checkpoints/full-attention-ddp4-62994'}
    model_hashes = {}
    for label, path in models.items():
        weights = sorted(path.glob('*.safetensors'))
        assert weights, path
        index = path / 'model.safetensors.index.json'
        if index.is_file():
            names = set(json.loads(index.read_text())['weight_map'].values())
            assert names == {p.name for p in weights}
        model_hashes[label] = {p.name: sha(p) for p in weights}
    summary = dict(records=len(result), expected=expected,
        hours=sum(r['duration'] for r in result) / 3600,
        prompt_tokens_min=min(r['prompt_len'] for r in result),
        prompt_tokens_max=max(r['prompt_len'] for r in result), train_overlap=0,
        train_sha256=sha(folder / 'train_mix.jsonl'), source_hashes=checksums,
        model_sha256=model_hashes, input_sha256=sha(args.run / 'inputs.json'),
        references_sha256=sha(args.run / 'references.json'))
    (args.run / 'data_audit.json').write_text(json.dumps(summary, indent=2))
    print('EVAL_INPUT_AUDIT_PASS ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
