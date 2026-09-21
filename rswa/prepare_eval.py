"""Freeze evaluation inputs/sentinel using reference metadata only."""
import argparse,math,json
from pathlib import Path
from common import read,write,BASE,ROOT,OLD_EVAL,sha
from protocol import ATTENTION_BACKEND
TASK=ROOT/'dkucc/rswa_20260920'
OLD=ROOT/'dkucc/ms_swift_20260914/evaluation'

def features(item,reference):
    events=[];speakers=set()
    for line in reference['rttm'].splitlines():
        if not line.strip():continue
        row=line.split();start=float(row[3]);end=start+float(row[4])
        events.extend([(start,1),(end,-1)]);speakers.add(row[7])
    events.sort();active=0;last=0.;overlap=0.
    for t,delta in events:
        if active>=2:overlap+=t-last
        active+=delta;last=t
    return [item['duration'],len(speakers),overlap/item['duration']]

def freeze():
    inputs=read(OLD/'dev_inputs.json');references=read(OLD_EVAL/'references.json')
    assert len(inputs)==26 and all(i['key'] in references for i in inputs)
    records={r['key']:features(r,references[r['key']]) for r in inputs}
    selected=[]
    for corpus in ['alimeeting','ami']:
        rows=sorted((r for r in inputs if r['dataset']==corpus),key=lambda r:(r['duration'],r['key']))
        # Seed with shortest and longest; select a third maximally separated in
        # duration, speaker count and overlap. Outputs are never consulted.
        chosen=[rows[0],rows[-1]]
        limits=[(min(records[r['key']][j] for r in rows),max(records[r['key']][j] for r in rows)) for j in range(3)]
        def distance(a,b):
            return sum(((records[a['key']][j]-records[b['key']][j])/max(hi-lo,1e-9))**2
                       for j,(lo,hi) in enumerate(limits))
        candidates=[r for r in rows if r not in chosen]
        chosen.append(max(candidates,key=lambda r:(min(distance(r,c) for c in chosen),r['key'])))
        selected.extend(chosen)
    write(TASK/'data/sentinel.json',dict(inputs=selected,reference_features=records,
          feature_order=['duration_seconds','reference_speakers','overlap_fraction'],
          selection='per_corpus_shortest_longest_then_farthest_metadata',outputs_used=False))
    write(TASK/'data/dev_inputs.json',inputs)
    write(TASK/'data/references.json',references)
    print(json.dumps(dict(sentinel=[r['key'] for r in selected],dev=len(inputs))),flush=True)

def prepare(run,checkpoint,window,split='dev',sentinel=False):
    frozen=TASK/'data/dev_inputs.json'
    if not frozen.exists():freeze()
    if sentinel:inputs=read(TASK/'data/sentinel.json')['inputs']
    elif split=='dev':inputs=read(frozen)
    else:
        # Use the same already-established full official Test manifest.
        inputs=[r for r in read(OLD_EVAL/'inputs.json') if r['split']=='test']
        assert len(inputs)==56
    loads=[0.]*4
    for item in sorted(inputs,key=lambda r:-r['duration']):
        rank=min(range(4),key=lambda i:loads[i]);item['rank']=rank;loads[rank]+=item['duration']
    write(run/'inputs.json',inputs)
    refs=read(TASK/'data/references.json');write(run/'references.json',{i['key']:refs[i['key']] for i in inputs})
    write(run/'model_manifest.json',dict(checkpoint=str(checkpoint),window=window,
          config_sha256=sha(checkpoint/'config.json'),penalty=1.02,penalty_scope='all_generated_tokens_only',
          attention_backend=ATTENTION_BACKEND,sentinel=sentinel,split=split))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');a=ap.parse_args()
    freeze()
