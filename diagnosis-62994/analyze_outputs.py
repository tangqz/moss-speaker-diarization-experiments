"""Post hoc diagnosis only. Original predictions and official metrics stay immutable."""
import ast
import collections
import csv
import importlib.util
import itertools
import json
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
RUN = ROOT.parent / 'artifacts/eval-62994-63363/raw/results/eval-62994-63363'
def load(p): return json.loads(p.read_text(encoding='utf-8'))
tree = ast.parse((ROOT.parent/'evaluation/score.py').read_text(encoding='utf-8'))
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ['normalize','char_streams','cp_distance','text_scores']]
exec(compile(ast.Module(body=functions,type_ignores=[]),'reviewed_text_scores','exec'))
spec=importlib.util.spec_from_file_location('diagnostic_original_parser',ROOT/'source/moss_transcribe_diarize/transcript_parser.py')
parser=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parser;spec.loader.exec_module(parser)

# Strict standalone segment grammar, with local resynchronization after invalid material.
# No invented speaker IDs, timestamps, text, or reference-driven repairs.
BLOCK=re.compile(r'\[(\d+(?:\.\d+)?)\]\s*\[(S\d+)\]([^\[\]]*)\[(\d+(?:\.\d+)?)\]')
def resync(text):
    return [dict(start=float(m[1]),end=float(m[4]),speaker=m[2],text=m[3].strip())
        for m in BLOCK.finditer(text) if float(m[4])>=float(m[1]) and m[3].strip()]

records=load(RUN/'per_record_metrics.json');refs=load(RUN/'references.json')
by={(r['model'],r['key']):r for r in records}
details=[]
for r in records:
    p=load(RUN/'predictions'/r['model']/(r['key']+'.json'))
    recovered=resync(p['raw_text']);scores=text_scores(refs[r['key']]['segments'],recovered)
    official_count=r['text']['hypothesis_characters'];recovered_count=scores['hypothesis_characters']
    row={k:r[k] for k in ['key','dataset','split','model','duration','rank','generated_tokens','truncated','parse_empty']}
    row.update(official_segments=len(p['segments']),resync_segments=len(recovered),official_chars=official_count,
        resync_chars=recovered_count,original_text_scores=r['text'],resync_text_scores=scores,
        last_parsed_time=max((s['end'] for s in p['segments']),default=0),
        last_resync_time=max((s['end'] for s in recovered),default=0),
        invalid_bracket_tokens=dict(collections.Counter(t for t in re.findall(r'\[([^\[\]]{1,32})\]',p['raw_text'])
            if not re.fullmatch(r'(?:S\d+|\d+(?:\.\d+)?)',t))),
        longest_identical_token_run=max((sum(1 for _ in g) for _,g in itertools.groupby(p['generated_ids'])),default=0),
        raw_head=p['raw_text'][:240],raw_tail=p['raw_text'][-120:])
    details.append(row)

groups={}
for group in sorted({r['dataset']+'/'+r['split'] for r in records}):
    d,s=group.split('/'); rr=[r for r in details if r['dataset']==d and r['split']==s and r['model']=='sft']
    nref=sum(r['original_text_scores']['reference_characters'] for r in rr)
    deltas=collections.defaultdict(int);buckets=collections.defaultdict(list)
    for r in rr:
        # Mutually exclusive flags derived only from generated output, not the reference or metric change.
        bucket=('truncated' if r['truncated'] else
            'major_parse_loss' if r['resync_chars']>2*max(r['official_chars'],1) and r['resync_chars']>100 else 'remaining')
        r['bucket']=bucket;buckets[bucket].append(r)
        deltas[bucket]+=r['original_text_scores']['cp_character_errors']-by['base',r['key']]['text']['cp_character_errors']
    cp_delta=sum(deltas.values())
    entry={'reference_characters':nref,'meetings':len(rr),'cpCER_delta_pp':100*cp_delta/nref,
        'cpCER_net_error_change':cp_delta,'buckets':{}}
    for bucket,rows in buckets.items():
        den=sum(r['original_text_scores']['reference_characters'] for r in rows)
        entry['buckets'][bucket]={'meetings':len(rows),'keys':[r['key'] for r in rows],
            'cpCER_contribution_pp':100*deltas[bucket]/nref,'cp_error_change':deltas[bucket],
            'base_cpCER_subset':sum(by['base',r['key']]['text']['cp_character_errors'] for r in rows)/den,
            'sft_cpCER_subset':sum(r['original_text_scores']['cp_character_errors'] for r in rows)/den}
    for model in ['base','sft']:
        selected=[r for r in details if r['dataset']==d and r['split']==s and r['model']==model]
        entry[model]={}
        for mode in ['original_text_scores','resync_text_scores']:
            entry[model][mode]={metric:sum(r[mode][field] for r in selected)/nref
                for metric,field in [('CER','character_errors'),('cpCER','cp_character_errors'),('deletions','deletions'),('insertions','insertions'),('substitutions','substitutions')]}
        entry[model]['improved_meetings_cpCER']=sum(by[model,r['key']]['text']['cpCER']<by['base',r['key']]['text']['cpCER'] for r in selected)
    groups[group]=entry

synthetic='[0.0][S01]a[1.0][1.0][1.0][S02]b[2.0][2.0][S01]c[3.0]'
synthetic_original=[dict(start=s.start,end=s.end,speaker=s.speaker,text=s.text) for s in parser.parse_transcript(synthetic)]
assert len(synthetic_original)==1 and len(resync(synthetic))==3
result={'scope':'post hoc diagnostic; formal scores unchanged; subsets are outcome-selected descriptions, not unbiased performance estimates',
    'resync_definition':'independent strict [numeric timestamp][S+digits]text[numeric timestamp] blocks; malformed text and invalid speakers are not repaired',
    'groups':groups,'records':details,'synthetic_parser_check':{'input':synthetic,'original':synthetic_original,'resync':resync(synthetic)}}
(ROOT/'output_diagnosis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
with (ROOT/'meeting_diagnosis.csv').open('w',encoding='utf-8-sig',newline='') as f:
    fields=['key','model','truncated','parse_empty','official_segments','resync_segments','official_chars','resync_chars','last_parsed_time','last_resync_time','longest_identical_token_run']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(details)
print(json.dumps({'groups':groups,'synthetic_parser_check':result['synthetic_parser_check']},ensure_ascii=False,indent=2))
