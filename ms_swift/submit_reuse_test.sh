#!/usr/bin/env bash
python3 - <<'PY'
import hashlib, json, subprocess, shutil, py_compile
from pathlib import Path
from datetime import datetime, timezone
root=Path('/work/qt28/moss'); task=root/'dkucc/ms_swift_20260914'
run=root/'results/ms-swift-63643'; old=root/'results/ms-swift-63521'
source=old/'evaluations/test-20'; dest=run/'evaluations/test-150'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(run/'outcome.json')['selected_step']==150
assert not read(run/'outcome.json')['test_evaluated']
active=subprocess.check_output(['squeue','-u','qt28','-h','-o','%i %j'],text=True)
assert 'moss-ms-swift-final-test' not in active, active
assert not (run/'test_reuse_submission.json').exists(), 'already submitted'
checks={}
for name in ['inputs.json','references.json']:
 assert read(source/name)==read(dest/name)
 checks[name]=sha(source/name)
for name in ['vllm_generate.py','common.py','diagnostics.py']:
 a=task/'evaluation'/name; b=old/'source/evaluation'/name
 assert sha(a)==sha(b)
 checks[name]=sha(a)
for rank in range(4):
 a=read(source/f'base-worker-{rank}-engine.json'); b=read(dest/f'base-worker-{rank}-engine.json')
 for key in ['version','torch','config']: assert a[key]==b[key], key
checks['engine_version']=a['version']; checks['torch']=a['torch']
paired=[]
for p in (dest/'predictions/base').rglob('*.json'):
 a=read(p); b=read(source/'predictions/base'/p.relative_to(dest/'predictions/base'))
 paired.append({'key':a['key'],'tokens_equal':a.get('generated_ids')==b.get('generated_ids'),
                'prompt_equal':a.get('prompt_ids_sha256')==b.get('prompt_ids_sha256')})
checks['new_base_spot_check']=paired
checks['utc']=datetime.now(timezone.utc).isoformat()
(run/'base_reuse_validation.json').write_text(json.dumps(checks,indent=2))
archive=run/'interrupted-test-63680'
archive.mkdir(exist_ok=True)
if (dest/'predictions/base').exists():
 assert not (archive/'base').exists()
 shutil.move(str(dest/'predictions/base'),str(archive/'base'))
py_compile.compile(str(task/'final_test.py.new'),doraise=True)
shutil.copy2(task/'final_test.py',archive/'final_test.py')
(task/'final_test.py.new').replace(task/'final_test.py')
job=subprocess.check_output(['sbatch','--parsable',
 '--export=ALL,MOSS_RUN='+str(run)+',MOSS_REUSE_BASE='+str(source),
 str(task/'final_test.slurm')],text=True).strip().split(';')[0]
receipt={'job':int(job),'base_source':str(source),'selected_step':150,'utc':checks['utc']}
(run/'test_reuse_submission.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps({'checks':checks,'submission':receipt},indent=2))
PY
