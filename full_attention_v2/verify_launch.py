"""Read-only verification of the actual shuffled training batches and baseline."""
import json
from pathlib import Path
import shlex
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import ssh_moss

REMOTE=r'''
import json,sys
from pathlib import Path
from datetime import datetime,timezone
job=int(sys.argv[1]);run=Path('/work/qt28/moss/results')/f'full-attention-v2-{job}'
ckpt=Path('/work/qt28/moss/checkpoints')/f'full-attention-v2-{job}'
def read(p):return json.loads(p.read_text())
def lines(p):
    result=[]
    for s in p.read_text().splitlines():
        try:result.append(json.loads(s))
        except json.JSONDecodeError:pass
    return result
rows=[lines(ckpt/f'train-rank-{i}.jsonl') for i in range(4)]
through=min(max(r['step'] for r in values) for values in rows)
assert through>=1
order=read(run/'data_order.json')
assert order['all_resume_first_batches_equal']
by_rank=[{r['step']:r for r in values} for values in rows]
trainer={r['step']:r for r in lines(ckpt/'trainer-log.jsonl') if 'grad_norm' in r}
steps=[]
for step in range(1,through+1):
    records=[rr[step] for rr in by_rank]
    epoch,offset=divmod(step-1,134)
    micros=[r['microbatches'][0] for r in records]
    for rank,m in enumerate(micros):
        assert m['indices']==[order['epochs'][epoch][rank][offset]]
    tokens=sum(m['local_supervised_tokens'] for m in micros)
    assert all(m['global_supervised_tokens']==tokens for m in micros)
    assert all(r['precision']=='FP32_parameters_gradients_Adam_BF16_autocast' for r in records)
    if step==1:
        assert len(records[0]['updates'])==7 and all(r['lost']==0 for r in records[0]['updates'])
    steps.append(dict(step=step,loss=sum(m['loss_numerator'] for m in micros)/tokens,
        supervised_tokens=tokens,lr=records[0]['lr'],grad_norm=trainer.get(step,{}).get('grad_norm'),
        peak_gpu_gib=max(r['peak_allocated_gib'] for r in records)))
fast=read(run/'evaluations/fast-0/fast_summary.json')
assert fast['complete'] and len(fast['records'])==6
for group in fast['groups'].values():assert sum(t['tokens'] for t in group['by_type'].values())==group['tokens']
print(json.dumps(dict(job=job,verified_utc=datetime.now(timezone.utc).isoformat(),through_step=through,
    actual_shuffled_order_matches=True,global_denominators_match=True,first_update_lost_count=0,
    first_update_components_sampled=7,baseline_full_audio_meetings=6,
    baseline_loss={k:v['loss'] for k,v in fast['groups'].items()},steps=steps)))
'''
current=json.loads((HERE/'current_run.json').read_text(encoding='utf-8'))
ssh_moss.SSH[ssh_moss.SSH.index('-S')+1]=current['control_path']
result=ssh_moss.run('/dkucc/home/qt28/envs/moss312/bin/python - '+str(current['job']),REMOTE.encode(),capture=True)
data=json.loads(result.stdout)
(HERE/'launch-verification.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
print(json.dumps(data))
