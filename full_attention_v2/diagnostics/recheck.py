"""Inference-only checkpoint-25 comparison and checkpoint-50 repeat on TS3004c."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/work/qt28/moss')
ORIGINAL=ROOT/'results/full-attention-v2-63421'
SOURCE=ORIGINAL/'source'
sys.path.insert(0,str(SOURCE))
from common import read,write,sha,now

run=ROOT/'results'/f"full-attention-v2-diagnostic-{os.environ['SLURM_JOB_ID']}"
run.mkdir(parents=True,exist_ok=True)
(run/'logs').mkdir(exist_ok=True)
shutil.copy2(__file__,run/'recheck.py')
write(run/'status.json',dict(stage='inference_only_recheck',updated_utc=now(),training_job=63421))
try:
    manifest=read(SOURCE/'source_manifest.json')
    for name in ['generate.py','common.py','diagnostics.py','selection.py','score_v1.py']:
        assert sha(SOURCE/name)==manifest[name]
    source_eval=ORIGINAL/'evaluations/sentinel-50'
    key='ami/dev/TS3004c'
    item=next(i for i in read(source_eval/'inputs.json') if i['key']==key)
    children=[];handles=[];folders=[]
    for rank,step in enumerate([25,50]):
        checkpoint=ROOT/'checkpoints/full-attention-v2-63421'/f'checkpoint-{step}'
        assert read(checkpoint/'checkpoint_complete.json')['complete']
        folder=run/f'checkpoint-{step}';folder.mkdir(exist_ok=True);folders.append(folder)
        write(folder/'inputs.json',[dict(item,rank=rank)])
        write(folder/'references.json',{key:read(source_eval/'references.json')[key]})
        write(folder/'model_manifest.json',read(checkpoint/'v2_model_manifest.json'))
        dest=folder/'predictions/base'/f'{key}.json';dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source_eval/'predictions/base'/f'{key}.json',dest)
        write(folder/'protocol.json',dict(purpose='diagnostic_only_not_checkpoint_selection',
            source_training_job=63421,checkpoint_step=step,source_manifest=manifest,
            inference='unchanged frozen generate.py; BF16 greedy full audio; 65536 cap',
            base_prediction='Historical 63363 output reused; timing is not a contemporaneous comparison'))
        f=(run/'logs'/f'generate-{step}.log').open('w');handles.append(f)
        children.append(subprocess.Popen([sys.executable,str(SOURCE/'generate.py'),'--run',str(folder),
            '--checkpoint',str(checkpoint),'--rank',str(rank)],stdout=f,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in children]
    for f in handles:f.close()
    assert not any(codes),codes
    for folder in folders:
        with (run/'logs'/f'score-{folder.name}.log').open('w') as f:
            subprocess.run([str(ROOT/'dkucc/evaluation/metrics-env/bin/python'),str(SOURCE/'selection.py'),
                '--run',str(folder)],stdout=f,stderr=subprocess.STDOUT,check=True)
    write(run/'outcome.json',dict(status='completed',training_updates=0,
        results=[dict(step=int(p.name.split('-')[-1]),quality=read(p/'quality_status.json'),
                      metrics=read(p/'metrics_summary.json')) for p in folders],completed_utc=now()))
    write(run/'status.json',dict(stage='completed',updated_utc=now(),training_updates=0))
except Exception as exc:
    write(run/'status.json',dict(stage='failed',updated_utc=now(),error=str(exc)))
    raise
