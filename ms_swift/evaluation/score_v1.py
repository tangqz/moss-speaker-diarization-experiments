"""Explicit CER/cpCER/DER protocol; all denominators retained for micro aggregation."""
import argparse
import collections
import json
from pathlib import Path
import unicodedata

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold()
                   if not c.isspace() and unicodedata.category(c)[0] not in 'PSZC')


def char_streams(segments):
    ordered = sorted(segments, key=lambda s: (s['start'], s['end']))
    speakers = collections.defaultdict(str)
    for s in ordered:
        speakers[s['speaker']] += normalize(s['text'])
    return ''.join(normalize(s['text']) for s in ordered), dict(speakers)


def cp_distance(reference, hypothesis):
    ref, hyp = list(reference.values()), list(hypothesis.values())
    n = max(len(ref), len(hyp))
    if n == 0:
        return 0
    ref += [''] * (n-len(ref))
    hyp += [''] * (n-len(hyp))
    costs = np.array([[Levenshtein.distance(a, b) for b in hyp] for a in ref], dtype=np.int64)
    i, j = linear_sum_assignment(costs)
    return int(costs[i, j].sum())


def text_scores(reference, hypothesis):
    r, rs = char_streams(reference)
    h, hs = char_streams(hypothesis)
    errors = Levenshtein.distance(r, h)
    cp = cp_distance(rs, hs)
    edits = collections.Counter(e.tag for e in Levenshtein.editops(r, h))
    return dict(reference_characters=len(r), hypothesis_characters=len(h),
        character_errors=errors, cp_character_errors=cp,
        substitutions=edits['replace'], deletions=edits['delete'], insertions=edits['insert'],
        CER=(errors/len(r) if r else None), cpCER=(cp/len(r) if r else None),
        DeltaCP=((cp-errors)/len(r) if r else None),
        reference_speakers=len(rs), hypothesis_speakers=len(hs))


def annotation_from_segments(segments, sid):
    ann = Annotation(uri=sid)
    for i, s in enumerate(segments):
        if s['end'] > s['start'] >= 0:
            ann[Segment(s['start'], s['end']), str(i)] = s['speaker']
    return ann


def diarization_scores(ref, hyp, sid):
    reference = Annotation(uri=sid)
    for i, line in enumerate(ref['rttm'].splitlines()):
        if not line.strip():
            continue
        f = line.split()
        assert f[1] == sid
        start, length = float(f[3]), float(f[4])
        if length > 0:
            reference[Segment(start, start+length), str(i)] = f[7]
    timeline = Timeline(uri=sid)
    for line in ref['uem'].splitlines():
        if line.strip():
            f = line.split()
            assert f[0] == sid
            timeline.add(Segment(float(f[2]), float(f[3])))
    hypothesis = annotation_from_segments(hyp, sid)
    return {f'DER_collar_{collar:g}': dict(DiarizationErrorRate(collar=collar, skip_overlap=False)(
        reference, hypothesis, uem=timeline, detailed=True)) for collar in [0., .25]}


def aggregate(records, expected):
    nref = sum(r['text']['reference_characters'] for r in records)
    total_audio = sum(r['duration'] for r in records)
    total_time = sum(r['e2e_seconds'] for r in records)
    chars = sum(r['text']['character_errors'] for r in records)
    cpchars = sum(r['text']['cp_character_errors'] for r in records)
    out = dict(completed=len(records), expected=expected, coverage=len(records)/expected,
        complete=len(records)==expected, truncated=sum(r['truncated'] for r in records),
        empty_predictions=sum(r['parse_empty'] for r in records),
        reference_characters=nref, character_errors=chars, cp_character_errors=cpchars,
        CER=chars/nref if nref else None, cpCER=cpchars/nref if nref else None,
        DeltaCP=(cpchars-chars)/nref if nref else None, audio_hours=total_audio/3600,
        processing_hours=total_time/3600, RTF=total_time/total_audio if total_audio else None,
        audio_seconds_per_processing_second=total_audio/total_time if total_time else None,
        max_peak_allocated_gib=max((r['peak_allocated_gib'] for r in records
                                    if r.get('peak_allocated_gib') is not None), default=None))
    for name in ['DER_collar_0', 'DER_collar_0.25']:
        components = {k: sum(r['diarization'][name][k] for r in records)
                      for k in ['total', 'confusion', 'missed detection', 'false alarm', 'correct']}
        components['DER'] = (sum(components[k] for k in ['confusion', 'missed detection', 'false alarm'])
                             / components['total'] if components['total'] else None)
        out[name] = components
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    inputs = json.loads((args.run / 'inputs.json').read_text())
    references = json.loads((args.run / 'references.json').read_text())
    scored, errors, missing = [], [], []
    for label in ['base', 'sft']:
        for item in inputs:
            p = args.run / 'predictions' / label / (item['key'] + '.json')
            if not p.is_file():
                missing.append(f"{label}/{item['key']}")
                continue
            pred = json.loads(p.read_text())
            if pred['status'] != 'ok':
                errors.append(dict(model=label, key=item['key'], error=pred.get('error')))
                continue
            reference = references[item['key']]
            result = {k: pred[k] for k in ['key', 'dataset', 'split', 'session_id', 'model',
                'duration', 'rank', 'generated_tokens', 'prompt_len', 'e2e_seconds',
                'generation_seconds', 'rtf', 'peak_allocated_gib', 'peak_reserved_gib',
                'ended_eos', 'truncated', 'parse_empty']}
            result['text'] = text_scores(reference['segments'], pred['segments'])
            result['diarization'] = diarization_scores(reference, pred['segments'], item['session_id'])
            scored.append(result)
    groups = {}
    for label in ['base', 'sft']:
        for dataset, split in sorted({(r['dataset'], r['split']) for r in inputs}):
            selected = [r for r in scored if r['model']==label and r['dataset']==dataset and r['split']==split]
            expected = sum(r['dataset']==dataset and r['split']==split for r in inputs)
            groups[f'{label}/{dataset}/{split}'] = aggregate(selected, expected)
    comparisons = {}
    for dataset, split in sorted({(r['dataset'], r['split']) for r in inputs}):
        maps = {label: {r['key']:r for r in scored if r['model']==label and r['dataset']==dataset
                       and r['split']==split} for label in ['base','sft']}
        common = set(maps['base']) & set(maps['sft'])
        expected = sum(r['dataset']==dataset and r['split']==split for r in inputs)
        comparisons[f'{dataset}/{split}'] = {label:aggregate([maps[label][k] for k in sorted(common)],expected)
                                             for label in ['base','sft']}
    summary = dict(protocol='CER_cpCER_DER_v1', expected_predictions=2*len(inputs),
        scored_predictions=len(scored), complete=len(scored)==2*len(inputs), errors=errors,
        missing=missing, groups=groups, paired_comparisons=comparisons)
    (args.run / 'per_record_metrics.json').write_text(json.dumps(scored, indent=2))
    (args.run / 'metrics_summary.json').write_text(json.dumps(summary, indent=2))
    print('EVAL_SCORED ' + json.dumps(dict(expected=summary['expected_predictions'],
        scored=len(scored), errors=len(errors), missing=len(missing))), flush=True)
    if not summary['complete']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
