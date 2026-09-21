"""Read-only comparison of the recovery attempt and original failed run."""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
EXECUTION=HERE.parent
sys.path.insert(0,str(EXECUTION.parent))
import ssh_moss

REMOTE=r'''
import json,sys
from pathlib import Path
from datetime import datetime,timezone
job=int(sys.argv[1]);root=Path('/work/qt28/moss/checkpoints')/f'full-attention-v2-{job}'
old=Path('/work/qt28/moss/checkpoints/full-attention-v2-63415')
run=Path('/work/qt28/moss/results')/f'full-attention-v2-{job}'
def read(p):return json.loads(p.read_text())
def lines(p):
    result=[]
    for s in p.read_text().splitlines():
        try:result.append(json.loads(s))
        except json.JSONDecodeError:pass
    return result
newrows=[{r['step']:r for r in lines(root/f'train-rank-{i}.jsonl')} for i in range(4)]
oldrows=[{r['step']:r for r in lines(old/f'train-rank-{i}.jsonl')} for i in range(4)]
common=sorted(set.intersection(*(set(r) for r in newrows)))
assert common
host=[{r['step']:r for r in lines(root/f'host-memory-rank-{i}.jsonl')} for i in range(4)]
steps=[]
for step in common:
    newer=[r[step] for r in newrows]
    micros=[r['microbatches'][0] for r in newer]
    tokens=sum(r['local_supervised_tokens'] for r in micros)
    assert all(r['global_supervised_tokens']==tokens for r in micros)
    memory=[h[step] for h in host]
    assert all(h['rng_unchanged'] for h in memory)
    record=dict(step=step,loss=sum(m['loss_numerator'] for m in micros)/tokens,
        summed_rank_rss_before_gib=sum(h['before']['rss_gib'] for h in memory),
        summed_rank_rss_after_gib=sum(h['after']['rss_gib'] for h in memory),
        summed_pinned_owned_before_gib=sum(h['before']['pinned_owned_gib'] for h in memory),
        summed_pinned_owned_after_gib=sum(h['after']['pinned_owned_gib'] for h in memory),
        max_cleanup_seconds=max(h['cleanup_seconds'] for h in memory))
    if all(step in r for r in oldrows):
        earlier=[r[step]['microbatches'][0] for r in oldrows]
        assert [m['indices'] for m in micros]==[m['indices'] for m in earlier]
        assert [m['local_supervised_tokens'] for m in micros]==[m['local_supervised_tokens'] for m in earlier]
        record['prior_loss']=sum(m['loss_numerator'] for m in earlier)/tokens
        record['absolute_loss_difference']=abs(record['loss']-record['prior_loss'])
    steps.append(record)
checkpoints=[];weights_only=[]
for p in sorted(root.glob('checkpoint-*/checkpoint_complete.json')):
    folder=p.parent;step=int(folder.name.split('-')[1])
    required=['optimizer.pt','scheduler.pt','trainer_state.json']+[f'rng_state_{i}.pth' for i in range(4)]
    assert read(p)['complete']
    retention=read(folder/'retention.json') if (folder/'retention.json').exists() else {}
    if retention.get('full_resume_state') is False:
        assert retention['weights_preserved'] and any(folder.glob('*.safetensors'))
        weights_only.append(step)
    else:
        assert all((folder/f).is_file() for f in required)
        checkpoints.append(step)
test=read(run/'logs/host-memory-maintenance-test.log')
assert test['passed']
original=read(Path('/work/qt28/moss/results/full-attention-v2-63415/source/source_manifest.json'))
actual=read(run/'source/source_manifest.json')
critical=['train.py','common.py','memory_sft.py','design_protocol.json','preflight.json','dev_inputs.json','sentinel_manifest.json']
assert all(original[k]==actual[k] for k in critical)
print(json.dumps(dict(job=job,verified_utc=datetime.now(timezone.utc).isoformat(),through_step=max(common),
    original_numerical_source_and_protocol_hashes_match=True,cache_unit_test=test,
    saved_resume_steps=checkpoints,weights_only_steps=weights_only,steps=steps,
    passed_old_failure_step=max(common)>=16,longest_meeting_completed=max(common)>=14,
    rss_sums_note='Sum of per-rank samples taken after each update; not a simultaneous Slurm peak.')))
'''

current=json.loads((EXECUTION/'current_run.json').read_text(encoding='utf-8'))
ssh_moss.SSH[ssh_moss.SSH.index('-S')+1]=current['control_path']
raw=ssh_moss.run('/dkucc/home/qt28/envs/moss312/bin/python - '+str(current['job']),REMOTE.encode(),capture=True)
data=json.loads(raw.stdout)
(HERE/f"verification-{current['job']}.json").write_text(json.dumps(data,indent=2),encoding='utf-8')
print(json.dumps(data))
