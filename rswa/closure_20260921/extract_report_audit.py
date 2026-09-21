"""Read-only, small-file audit of checkpoint-matched sentinel coverage and failure examples."""
import collections,json
from pathlib import Path
ROOT=Path('/work/qt28/moss/results/moss-rswa-20260920')
def read(p):return json.loads(p.read_text())
sentinel=read(Path('/work/qt28/moss/dkucc/rswa_20260920/data/sentinel.json'))
keys={r['key'] for r in sentinel['inputs']}
state=read(ROOT/'state.json');rows=[]
for group,g in state['groups'].items():
    for row in g['history']:
        failed={r['key'] for r in row['failures']}
        rows.append(dict(group=group,step=row['step'],sentinel_failures=len(failed&keys),sentinel_size=6,
            other_failures=len(failed-keys),other_size=20,
            failure_corpora=dict(collections.Counter(k.split('/')[0] for k in failed))))
run=ROOT/'dev/256-seed0/step-150'
refs=read(run/'references.json');assessment=read(run/'assessment.json')
examples=[]
for failure in assessment['failures']:
    key=failure['key'];pred=read(run/'predictions/sft'/f'{key}.json');ref=refs[key]
    d=pred['diagnostics'];segments=pred['segments']
    examples.append(dict(key=key,duration=pred['duration'],generated_tokens=pred['generated_tokens'],
        ended_eos=pred['ended_eos'],truncated=pred['truncated'],parse_empty=pred['parse_empty'],
        ref_last_end=max(r['end'] for r in ref['segments']),hyp_last_end=max((r['end'] for r in segments),default=0),
        parsed_segments=len(segments),reference_text_characters=sum(len(r['text']) for r in ref['segments']),
        hypothesis_text_characters=sum(len(r['text']) for r in segments),official_normalized_characters=d['official_characters'],
        legal_scan_normalized_characters=d['legal_scan_characters'],unparsed_normalized_characters=d['unparsed_characters'],
        longest_identical_token_run=d['longest_identical_run']['count'],longest_short_period=d['longest_short_period'],
        reasons=failure['reasons']))
print(json.dumps(dict(sentinel_keys=sorted(keys),same_checkpoint_coverage=rows,r256_step150_examples=examples),ensure_ascii=False,indent=2))
