"""Sequential training/validation phases within one Slurm allocation.

Each phase resumes the exact optimizer/scheduler/RNG checkpoint. Evaluation
loads BF16 copies after the training processes exit, preserving FP32 weights.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from common import ROOT,BASE,TRAIN,HERE,REPO,OLD_EVAL,MILESTONES,read,write,sha,emit,now

METRIC_PY=ROOT/'dkucc/evaluation/metrics-env/bin/python'


def status(run,stage,**kw):
    value=dict(stage=stage,updated_utc=now(),**kw)
    write(run/'status.json',value)
    emit('pipeline_status',**value)


def execute(run,command,name):
    log=run/'logs'/f'{name}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a') as f:
        subprocess.run([str(x) for x in command],stdout=f,stderr=subprocess.STDOUT,check=True)


def workers(run,folder,script,checkpoint,name):
    processes=[];handles=[]
    for rank in range(4):
        path=run/'logs'/f'{name}-rank-{rank}.log'
        f=path.open('a');handles.append(f)
        cmd=[sys.executable,str(HERE/script),'--run',str(folder),'--checkpoint',str(checkpoint),'--rank',str(rank)]
        processes.append(subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes]
    for f in handles:f.close()
    assert not any(codes),(name,codes)


def model_manifest(checkpoint):
    path=checkpoint/'v2_model_manifest.json'
    if path.exists():return read(path)
    record=dict(path=str(checkpoint),files={p.name:sha(p) for p in sorted(checkpoint.iterdir())
        if (p.name.startswith('model') and p.suffix in ['.safetensors','.json']) or p.suffix=='.py'},
        config_sha256=sha(checkpoint/'config.json'))
    if checkpoint!=BASE:write(path,record)
    return record


def prepare(folder,checkpoint,scope):
    all_items=read(OLD_EVAL/'inputs.json')
    if scope=='sentinel':
        keys={r['key'] for r in read(HERE/'sentinel_manifest.json')['records']}
        items=[dict(r) for r in all_items if r['key'] in keys]
        assert len(items)==6
    else:
        items=[dict(r) for r in all_items if r['split']==scope]
        assert len(items)==(26 if scope=='dev' else 56)
    frozen={r['key']:r for r in read(HERE/'dev_inputs.json')}
    for item in items:
        if item['split']=='dev':
            assert {k:v for k,v in item.items() if k!='rank'}=={k:v for k,v in frozen[item['key']].items() if k!='rank'}
    load=[0.]*4
    for item in sorted(items,key=lambda r:(-r['duration'],r['key'])):
        rank=min(range(4),key=lambda r:(load[r],r))
        item['rank']=rank;load[rank]+=item['duration']
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'inputs.json').exists():assert read(folder/'inputs.json')==items
    write(folder/'inputs.json',items)
    refs=read(OLD_EVAL/'references.json')
    write(folder/'references.json',{r['key']:refs[r['key']] for r in items})
    write(folder/'model_manifest.json',model_manifest(checkpoint))
    write(folder/'protocol.json',dict(primary='CER_cpCER_DER_v1',scope=scope,
        source_manifest=read(HERE/'source_manifest.json'),
        historical_base='63363 accuracy reused; historical timing is descriptive, not new paired hardware timing'))
    for item in items:
        src=OLD_EVAL/'predictions/base'/f"{item['key']}.json"
        dest=folder/'predictions/base'/f"{item['key']}.json"
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dest)


def fast(run,checkpoint,step):
    folder=run/'evaluations'/f'fast-{step}'
    if (folder/'fast_summary.json').is_file():return
    status(run,'fast_validation',step=step,checkpoint=str(checkpoint))
    prepare(folder,checkpoint,'sentinel')
    workers(run,folder,'fast_eval.py',checkpoint,f'fast-{step}')
    assert all(read(folder/f'fast-rank-{r}-complete.json')['complete'] for r in range(4))
    records=[read(folder/'teacher'/f"{r['key']}.json") for r in read(folder/'inputs.json')]
    assert len(records)==6
    assert len(list((folder/'prefix').rglob('*.json')))==2
    groups={}
    for dataset in ['all','alimeeting','ami']:
        rows=[r for r in records if dataset=='all' or r['dataset']==dataset]
        tokens=sum(r['supervised_tokens'] for r in rows)
        numerator=sum(r['nll_sum'] for r in rows)
        types={}
        for label in rows[0]['by_type']:
            types[label]={key:sum(r['by_type'][label][key] for r in rows)
                          for key in ['tokens','nll_sum','entropy_sum']}
        groups[dataset]=dict(tokens=tokens,nll_sum=numerator,loss=numerator/tokens,by_type=types)
    write(folder/'fast_summary.json',dict(step=step,complete=True,records=records,groups=groups))


def full(run,checkpoint,step,scope):
    folder=run/'evaluations'/f'{scope}-{step}'
    status(run,'full_generation',step=step,scope=scope,checkpoint=str(checkpoint))
    prepare(folder,checkpoint,scope)
    if not (folder/'scoring_complete.json').exists():
        workers(run,folder,'generate.py',checkpoint,f'{scope}-{step}')
        command=[METRIC_PY,HERE/'selection.py','--run',folder]
        if scope=='dev':command+=['--training-run',run,'--step',str(step)]
        execute(run,command,f'score-{scope}-{step}')
        write(folder/'scoring_complete.json',dict(complete=True,utc=now()))
    return folder


def prune_states(checkpoints):
    folders=sorted((p for p in checkpoints.glob('checkpoint-*') if
        (p/'checkpoint_complete.json').is_file()),key=lambda p:int(p.name.split('-')[-1]))
    keep={int(p.name.split('-')[-1]) for p in folders[-2:]}|set(MILESTONES)
    removed=[]
    root=checkpoints.resolve()
    for folder in folders:
        step=int(folder.name.split('-')[-1])
        if step in keep:continue
        for name in ['optimizer.pt','scheduler.pt']+[f'rng_state_{r}.pth' for r in range(4)]:
            path=(folder/name).resolve()
            assert path.is_relative_to(root) and path.parent==folder.resolve()
            if path.is_file():path.unlink();removed.append(str(path))
        write(folder/'retention.json',dict(weights_preserved=True,full_resume_state=False,
            reason='Protocol keeps latest two full states plus full-dev milestones'))
    return removed


def verify_actual_order(run,checkpoints,through):
    expected=read(run/'data_order.json')['epochs']
    totals={}
    for rank in range(4):
        path=checkpoints/f'train-rank-{rank}.jsonl'
        rows=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        assert [r['step'] for r in rows]==list(range(1,through+1)),(rank,through)
        for row in rows:
            epoch,offset=divmod(row['step']-1,134)
            assert len(row['microbatches'])==1
            m=row['microbatches'][0]
            assert m['indices']==[expected[epoch][rank][offset]],(rank,row['step'],m['indices'])
            totals[row['step']]=totals.get(row['step'],0)+m['local_supervised_tokens']
    for rank in range(4):
        rows=[json.loads(s) for s in (checkpoints/f'train-rank-{rank}.jsonl').read_text().splitlines() if s.strip()]
        assert all(r['microbatches'][0]['global_supervised_tokens']==totals[r['step']] for r in rows)
    write(run/'actual_data_order_verified.json',dict(through_step=through,all_ranks_match=True,
         supervised_tokens=sum(totals.values()),global_denominators_match=True,utc=now()))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--gate-run',type=Path,required=True)
    ap.add_argument('--checkpoints',type=Path,required=True)
    a=ap.parse_args()
    run=a.run;run.mkdir(parents=True,exist_ok=True);(run/'logs').mkdir(exist_ok=True)
    try:
        status(run,'validating_execution_bundle')
        gate=read(a.gate_run/'entry_gates_passed.json');assert gate['passed']
        manifest=read(HERE/'source_manifest.json')
        for name,value in manifest.items():assert sha(HERE/name)==value,name
        critical=['train.py','common.py','memory_sft.py','design_protocol.json','preflight.json',
                  'dev_inputs.json','sentinel_manifest.json']
        assert all(manifest[n]==gate['source_manifest'][n] for n in critical)
        for path,value in read(a.gate_run/'environment.json')['hashes'].items():
            assert sha(path)==value,path
        assert shutil.disk_usage(a.checkpoints.parent).free>=110*2**30
        write(run/'entry_gates.json',dict(gate_run=str(a.gate_run),**gate))
        write(run/'execution_contract.json',dict(checkpoints=str(a.checkpoints),source_manifest=manifest,
            gate_run=str(a.gate_run),schedule_horizon=402,first_stage=134,max_steps=402,
            evaluation_inputs_sha256=sha(OLD_EVAL/'inputs.json'),references_sha256=sha(OLD_EVAL/'references.json'),
            stop_50_policy='Pause on truncation, empty parse or major parse loss for review; no output editing',
            final_test='Only after checkpoint selection is frozen; original 56 test meetings'))
        execute(run,[METRIC_PY,HERE/'test_execution.py','metrics'],'selection-tests')
        execute(run,[sys.executable,HERE/'test_execution.py','statistics'],'statistics-tests')
        execute(run,[sys.executable,HERE/'test_host_runtime.py'],'host-memory-maintenance-test')
        if not (run/'data_order.json').exists():
            execute(run,[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                         HERE/'sampler_gate.py','--run',run],'shuffled-sampler-gate')
        fast(run,BASE,0)
        a.checkpoints.mkdir(parents=True,exist_ok=True)
        write(a.checkpoints/'step-0-base.json',model_manifest(BASE))
        phases=sorted(set(range(25,403,25))|set(MILESTONES))
        previous=None
        for step in phases:
            checkpoint=a.checkpoints/f'checkpoint-{step}'
            if not (a.checkpoints/f'phase-{step}-complete.json').is_file():
                status(run,'training',step_target=step,resume=str(previous) if previous else None,
                       checkpoints=str(a.checkpoints),schedule_horizon=402)
                cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                     HERE/'host_runtime.py','--output',a.checkpoints,'--stop',str(step)]
                if previous:cmd+=['--resume',previous]
                execute(run,cmd,f'train-to-{step}')
            assert read(checkpoint/'checkpoint_complete.json')['complete']
            verify_actual_order(run,a.checkpoints,step)
            previous=checkpoint
            model_manifest(checkpoint)
            if step%25==0:fast(run,checkpoint,step)
            if step==50:
                folder=full(run,checkpoint,step,'sentinel')
                if read(folder/'quality_status.json')['catastrophic_count_with_overlap']:
                    status(run,'paused_for_sentinel_failure',step=step,evaluation=str(folder))
                    write(run/'outcome.json',dict(status='paused_for_sentinel_failure',step=step,
                        reason='Full sentinel generation requires review before further updates.'))
                    return
            if step in MILESTONES:
                full(run,checkpoint,step,'dev')
                decision=read(run/'selection.json')
                if not decision['continue_training']:break
            removed=prune_states(a.checkpoints)
            if removed:write(run/f'retention-after-{step}.json',dict(removed_full_state_files=removed))
        decision=read(run/'selection.json')
        write(run/'selection_frozen.json',dict(**decision,frozen_utc=now(),protocol_sha256=sha(HERE/'design_protocol.json')))
        selected=decision['selected_step']
        if selected is not None:
            full(run,a.checkpoints/f'checkpoint-{selected}',selected,'test')
        outcome=dict(status='completed' if selected is not None else 'completed_no_eligible_sft',
                     selected_step=selected,selection=decision,test_performed=selected is not None,utc=now())
        write(run/'outcome.json',outcome)
        status(run,**{'stage':outcome['status'],'selected_step':selected})
    except Exception as exc:
        write(run/'failure.json',dict(utc=now(),type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc()))
        status(run,'failed',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__':main()
