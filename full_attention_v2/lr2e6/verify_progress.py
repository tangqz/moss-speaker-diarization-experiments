"""Read-only proof that actual low-LR updates match the prior data exposure."""
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent.parent))
import ssh_moss

REMOTE=r'''
import json,sys
from pathlib import Path
from datetime import datetime,timezone
job=int(sys.argv[1]);root=Path('/work/qt28/moss')
run=root/'results'/f'full-attention-v2-{job}'
ckpt=root/'checkpoints'/f'full-attention-v2-{job}'
old=root/'checkpoints/full-attention-v2-63421'
def read(p):return json.loads(p.read_text())
def lines(p):
    values={}
    if not p.exists():return values
    for line in p.read_text().splitlines():
        try:row=json.loads(line)
        except json.JSONDecodeError:continue
        values[row['step']]=row
    return values
records=[lines(ckpt/f'train-rank-{r}.jsonl') for r in range(4)]
common=set.intersection(*(set(rows) for rows in records))
through=max(common,default=0)
result=dict(job=job,verified_utc=datetime.now(timezone.utc).isoformat(),through_step=through,
    pipeline=read(run/'status.json') if (run/'status.json').exists() else None)
if through:
    assert common==set(range(1,through+1))
    order=read(run/'data_order.json')['epochs']
    original=[lines(old/f'train-rank-{r}.jsonl') for r in range(4)]
    steps=[]
    for step in range(1,through+1):
        epoch,offset=divmod(step-1,134)
        rows=[rr[step] for rr in records];micros=[r['microbatches'][0] for r in rows]
        tokens=sum(m['local_supervised_tokens'] for m in micros)
        for rank,(row,m) in enumerate(zip(rows,micros)):
            assert row['precision']=='FP32_parameters_gradients_Adam_BF16_autocast'
            assert abs(row['lr']-2e-6*(1-(step-1)/402))<1e-12
            assert abs(row['lr']-original[rank][step]['lr']/5)<1e-12
            assert m['indices']==[order[epoch][rank][offset]]
            for field in ['indices','sequence_tokens','local_supervised_tokens','global_supervised_tokens']:
                assert m[field]==original[rank][step]['microbatches'][0][field]
            assert m['global_supervised_tokens']==tokens
        if step==1:assert len(rows[0]['updates'])==7 and all(x['lost']==0 for x in rows[0]['updates'])
        steps.append(dict(step=step,lr=rows[0]['lr'],loss=sum(m['loss_numerator'] for m in micros)/tokens,
            supervised_tokens=tokens,peak_gpu_gib=max(r['peak_allocated_gib'] for r in rows)))
    result.update(steps=steps,matched_data_and_tokens=True,learning_rate_ratio=.2,
        precision_verified=True,first_update_lost_count=0,components_sampled=7,
        cumulative_supervised_tokens=sum(s['supervised_tokens'] for s in steps))
if (run/'device_preflight.json').exists():result['devices']=read(run/'device_preflight.json')
print(json.dumps(result))
'''

current=json.loads((HERE/'current_run.json').read_text(encoding='utf-8'))
ssh_moss.SSH[ssh_moss.SSH.index('-S')+1]=current['control_path']
result=subprocess.run(ssh_moss.SSH+['/dkucc/home/qt28/envs/moss312/bin/python - '+str(current['job'])],
    input=REMOTE.encode(),capture_output=True,check=True,timeout=60)
data=json.loads(result.stdout)
(HERE/'progress-verification.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
print(json.dumps(data))
