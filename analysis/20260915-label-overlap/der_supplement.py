"""Re-score preserved Base/Step20 predictions, including overlap-excluded DER."""
import collections
import json
import math
from pathlib import Path
import time
from pyannote.core import Annotation, Segment, Timeline
from pyannote.metrics.diarization import DiarizationErrorRate

root = Path('/work/qt28/moss/results/ms-swift-63521/evaluations/test-20')
out = Path('/work/qt28/moss/analysis/20260915-label-overlap')
references = json.loads((root/'references.json').read_text())
inputs = json.loads((root/'inputs.json').read_text())
saved = json.loads((root/'metrics_summary.json').read_text())
records = []
started = time.time()
for i,item in enumerate(inputs):
    key, sid = item['key'], item['session_id']
    r = references[key]
    ref = Annotation(uri=sid)
    for j,line in enumerate(r['rttm'].splitlines()):
        f = line.split()
        if f and float(f[4]) > 0:
            ref[Segment(float(f[3]), float(f[3])+float(f[4])),str(j)] = f[7]
    uem = Timeline(uri=sid)
    for line in r['uem'].splitlines():
        f = line.split()
        if f:
            uem.add(Segment(float(f[2]),float(f[3])))
    for model in ['base','sft']:
        pred = json.loads((root/'predictions'/model/(key+'.json')).read_text())
        hyp = Annotation(uri=sid)
        for j,s in enumerate(pred['segments']):
            if s['end'] > s['start'] >= 0:
                hyp[Segment(s['start'],s['end']),str(j)] = s['speaker']
        result = dict(key=key,model=model,dataset=item['dataset'])
        for skip in [False,True]:
            result['exclude_overlap' if skip else 'include_overlap'] = dict(
                DiarizationErrorRate(collar=.25,skip_overlap=skip)(ref,hyp,uem=uem,detailed=True))
        records.append(result)
    if (i+1) % 10 == 0:
        print(f'DER_RESCORED {i+1}/{len(inputs)}',flush=True)
groups = {}
for corpus in sorted({i['dataset'] for i in inputs}):
    for model in ['base','sft']:
        rows = [r for r in records if r['model']==model and r['dataset']==corpus]
        group = dict(records=len(rows))
        for kind in ['include_overlap','exclude_overlap']:
            values = {name:sum(r[kind][name] for r in rows)
                      for name in ['total','correct','confusion','missed detection','false alarm']}
            values['DER'] = sum(values[k] for k in ['confusion','missed detection','false alarm'])/values['total']
            group[kind]=values
        expected=saved['groups'][f'{model}/{corpus}/test']['DER_collar_0.25']['DER']
        assert math.isclose(group['include_overlap']['DER'],expected,rel_tol=1e-9,abs_tol=1e-10)
        groups[f'{model}/{corpus}']=group
result=dict(complete=True,meetings=len(inputs),scored_predictions=len(records),collar=.25,
            original_DER_reproduced=True,wall_seconds=time.time()-started,
            inference_rerun=False,groups=groups,records=records)
(out/'der_supplement.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2),flush=True)
