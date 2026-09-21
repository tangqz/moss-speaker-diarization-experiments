#!/usr/bin/env bash
python3 - <<'PY'
import json, hashlib, subprocess
from pathlib import Path
root=Path('/work/qt28/moss')
old=root/'results/ms-swift-63521'
run=root/'results/ms-swift-63643'
task=root/'dkucc/ms_swift_20260914'
def read(p): return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608), b''): h.update(b)
 return h.hexdigest()
report={'queue':subprocess.check_output(['squeue','-u','qt28','-h','-o','%i %T %j'],text=True)}
src=old/'evaluations/test-20'; dst=run/'evaluations/test-150'
items=read(src/'inputs.json')
report['inputs_equal']=read(src/'inputs.json')==read(dst/'inputs.json')
report['refs_equal']=read(src/'references.json')==read(dst/'references.json')
m=read(src/'model_manifests.json')['base']
report['live_base_hashes_match']=all(sha(Path(m['path'])/n)==s for n,s in m['files'].items())
report['source_hashes']={}
for name in ['vllm_generate.py','common.py','diagnostics.py']:
 a=task/'evaluation'/name; b=old/'source/evaluation'/name
 report['source_hashes'][name]={'current':sha(a),'historical':sha(b) if b.exists() else None}
report['engines']={}
for label,p in [('old',src),('current',dst),('dev150',run/'evaluations/dev-150')]:
 report['engines'][label]={q.name:read(q) for q in p.glob('*engine.json')}
rows=[read(src/'predictions/base'/ (x['key']+'.json')) for x in items]
report['base_versions']=sorted(set(x['backend_version'] for x in rows))
report['base_complete']=len(rows)==56 and all(x['status']=='ok' and not x['engineering_smoke'] for x in rows)
report['current_base_records']=len(list((dst/'predictions/base').rglob('*.json')))
report['outcome']=read(run/'outcome.json')
print(json.dumps(report,indent=2))
PY
