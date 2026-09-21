import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]
sys.stdout.reconfigure(encoding='utf-8')
SEG = re.compile(r'\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]', re.S)
STAMP = re.compile(r'\[(\d+(?:\.\d+)?)\]')
HEADER = re.compile(r'\[(\d+(?:\.\d+)?)\]\[(S\d+)\]')


def normalize(s):
    return ''.join(c for c in unicodedata.normalize('NFKC', s).casefold()
                   if unicodedata.category(c)[0] not in 'PSZC')


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def timeline(ref):
    events = collections.defaultdict(list)
    for line in ref['rttm'].splitlines():
        f = line.split()
        if not f:
            continue
        a, b = round(float(f[3]), 6), round(float(f[3]) + float(f[4]), 6)
        if b <= a:
            continue
        events[a].append((f[7], 1))
        events[b].append((f[7], -1))
    active, slices = collections.Counter(), []
    last = None
    for t in sorted(events):
        n = sum(count > 0 for count in active.values())
        if last is not None and t > last:
            slices.append((last, t, n))
        for speaker, delta in events[t]:
            active[speaker] += delta
        last = t
    return slices


def interval_stats(slices, a, b):
    speech = overlap = 0.
    peak = 0
    for start, end, n in slices:
        length = max(0., min(b, end) - max(a, start))
        if not length:
            continue
        speech += length * (n >= 1)
        overlap += length * (n >= 2)
        peak = max(peak, n)
    return dict(start=a, end=b, speech_seconds=speech, overlap_seconds=overlap,
                overlap_fraction_of_wall_time=overlap/(b-a) if b>a else None,
                overlap_fraction_of_speech=overlap/speech if speech else None,
                peak_speakers=peak)


def detect_loop(text):
    # Detect a truly periodic terminal string, allowing a partial final cycle.
    tail = text[-4096:]
    for period in range(1, 129):
        if len(tail) >= 1024 and tail[period:] == tail[:-period]:
            k = len(text) - 1
            while k >= period and text[k] == text[k-period]:
                k -= 1
            start = k - period + 1
            return dict(kind='exact_periodic_tail', char_offset=start,
                        period_characters=period, repeated_unit=text[start:start+period],
                        repeated_tail_characters=len(text)-start)
    # Structured loops may advance timestamps while repeating only the utterance.
    matches = list(SEG.finditer(text))
    if matches:
        norm = normalize(matches[-1][3])
        k = len(matches)-1
        while k > 0 and normalize(matches[k-1][3]) == norm:
            k -= 1
        if len(matches)-k >= 20:
            return dict(kind='repeated_utterance_with_changing_timestamps',
                        char_offset=matches[k].start(), repeated_unit=matches[k][3],
                        repeated_segments=len(matches)-k,
                        generated_onset_timestamp=float(matches[k][1]))
    return dict(kind='unresolved', char_offset=None)


def reference_for(path, key, global_refs):
    for parent in path.parents:
        candidate = parent/'references.json'
        if candidate.is_file():
            refs = read(candidate)
            if key in refs:
                return refs[key], str(candidate)
        if parent == WORK:
            break
    return global_refs[key]


def main():
    local = read(HERE/'local_failure_inventory.json')
    paths = [WORK/row['path'] for row in local if not row.get('engineering_smoke')
             and (row.get('max_new_tokens') or 0) >= 65536]
    remote = HERE/'remote_failures'
    if remote.exists():
        paths += [p for p in remote.rglob('*.json') if 'predictions' in p.parts]
    ref_paths = [
        WORK/'dkucc/artifacts/eval-62994-63363/raw/results/eval-62994-63363/references.json',
        WORK/'dkucc/artifacts/ms-swift-63521/complete/ms-swift-63521/evaluations/test-20/references.json',
        WORK/'dkucc/artifacts/ms-swift-63521/complete/ms-swift-63521/evaluations/dev-40/references.json']
    global_refs = {}
    for path in ref_paths:
        for key, ref in read(path).items():
            global_refs[key] = (ref, str(path))
    cases, duplicates, identity = [], [], {}
    for path in paths:
        p = read(path)
        raw = p['raw_text']
        # Collapse exact artifact copies, not differing output realizations.
        ident = (p['key'], p.get('model_path'), p.get('backend') or 'hf',
                 hashlib.sha256(raw.encode()).hexdigest())
        if ident in identity:
            duplicates.append(dict(copy=str(path), retained=identity[ident]))
            continue
        identity[ident] = str(path)
        ref, ref_path = reference_for(path, p['key'], global_refs)
        slices = timeline(ref)
        loop = detect_loop(raw)
        pos = loop['char_offset']
        anchor = None
        matched = []
        if pos is not None:
            headers = list(HEADER.finditer(raw[:pos+1]))
            stamps = list(STAMP.finditer(raw[:min(len(raw), pos+150)]))
            if loop['kind'] == 'repeated_utterance_with_changing_timestamps':
                anchor = loop['generated_onset_timestamp']
                anchor_method = 'generated_start_of_first_repeated_utterance'
            elif any(c.isdigit() for c in loop.get('repeated_unit','')) and stamps:
                anchor = float(stamps[-1][1])
                anchor_method = 'repeated_generated_timestamp'
            elif headers:
                anchor = float(headers[-1][1])
                anchor_method = 'generated_start_of_unfinished_utterance'
            else:
                anchor_method = 'no_reliable_generated_time'
            if anchor is not None:
                # All possible reference utterances at this time, allowing 0.5s
                # for boundary disagreement; no word-level forced alignment.
                matched = [s for s in ref['segments']
                           if s['start']-.5 <= anchor <= s['end']+.5]
        else:
            anchor_method = 'unresolved'
        whole = interval_stats(slices, 0, p['duration'])
        row = dict(key=p['key'], model_path=p.get('model_path'), backend=p.get('backend') or 'hf',
                   source=str(path), reference_source=ref_path,
                   raw_sha256=ident[-1], generated_tokens=p.get('generated_tokens'),
                   loop=loop, anchor_seconds=anchor, anchor_method=anchor_method,
                   annotation_speakers_at_anchor=next((n for a,b,n in slices if a<=anchor<b), 0)
                     if anchor is not None else None,
                   whole_meeting=whole, reference_candidates=matched,
                   context_before_loop=raw[max(0,pos-300):pos] if pos is not None else '',
                   context_after_loop=raw[pos:pos+100] if pos is not None else '')
        if anchor is not None:
            row['window_10s'] = interval_stats(slices, max(0,anchor-5), min(p['duration'],anchor+5))
            row['reference_utterance_overlap'] = [
                dict(**s, overlap=interval_stats(slices,s['start'],s['end'])) for s in matched]
        cases.append(row)
    groups = collections.defaultdict(list)
    for row in cases:
        group = 'MS-Swift' if '/ms-swift-' in row['model_path'] else 'earlier_SFT'
        groups[group].append(row)
    summary = dict(
        unique_output_failures=len(cases), unique_meetings=len({r['key'] for r in cases}),
        duplicate_copies_removed=len(duplicates),
        groups={k:dict(cases=len(rows), unique_meetings=len({r['key'] for r in rows}),
                      resolved_anchors=sum(r['anchor_seconds'] is not None for r in rows),
                      overlap_at_generated_anchor=sum((r['annotation_speakers_at_anchor'] or 0)>=2 for r in rows))
                for k,rows in groups.items()},
        limitations=[
            'Generated timestamps are approximate anchors, not forced word alignments.',
            'One long repeated tail is one failure event; its thousands of tokens are not independent observations.',
            'The same meeting/checkpoint can have different predictions under historical inference variants.',
            'No causal enrichment test is claimed from this failure-only inventory.'])
    (HERE/'overlap_audit.json').write_text(json.dumps(dict(summary=summary,cases=cases,duplicates=duplicates),
                                                   ensure_ascii=False,indent=2),encoding='utf-8')
    with (HERE/'failure_cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['key','model_path','backend','loop_kind','repeated_unit',
                    'anchor_seconds','anchor_method','speakers_at_anchor','window10_overlap_fraction',
                    'meeting_overlap_fraction','source'])
        writer.writeheader()
        for r in cases:
            writer.writerow(dict(key=r['key'],model_path=r['model_path'],backend=r['backend'],
                loop_kind=r['loop']['kind'],repeated_unit=r['loop'].get('repeated_unit'),
                anchor_seconds=r['anchor_seconds'],anchor_method=r['anchor_method'],
                speakers_at_anchor=r['annotation_speakers_at_anchor'],
                window10_overlap_fraction=r.get('window_10s',{}).get('overlap_fraction_of_wall_time'),
                meeting_overlap_fraction=r['whole_meeting']['overlap_fraction_of_wall_time'],source=r['source']))
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    for r in cases:
        print(r['model_path'].split('/')[-2:],r['key'],r['loop']['kind'],repr(r['loop'].get('repeated_unit')),
              r['anchor_seconds'],r['annotation_speakers_at_anchor'],
              round(r.get('window_10s',{}).get('overlap_fraction_of_wall_time',0),3))


if __name__ == '__main__':
    main()
