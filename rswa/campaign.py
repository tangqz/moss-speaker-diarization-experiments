"""Bounded, resumable stages of the approved experiment; no implicit gate bypass."""
import argparse,fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from common import ROOT,BASE,read,write,sha,MILESTONES
from run_eval import evaluate
from assess import assess,choose,extending
from retention import prune
from order_audit import verify as verify_order,prepare as prepare_order_log
HERE=Path(__file__).resolve().parent
TASK=ROOT/'dkucc/rswa_20260920'
ACTIVE_TRAIN_WINDOWS=['256']
SENTINEL_INTERVAL=20

def group(window,seed):return f'{window}-seed{seed}'

def evaluate_at(step):return step in MILESTONES or step%SENTINEL_INTERVAL==0

def cadence_targets(start,target):
    """Evaluate every 20 updates, resetting the cadence at full-dev milestones."""
    assert start<target<=402
    result=[];cursor=start
    boundaries=[m for m in MILESTONES if start<m<target]+[target]
    for boundary in boundaries:
        while cursor+SENTINEL_INTERVAL<boundary:
            cursor+=SENTINEL_INTERVAL;result.append(cursor)
        if cursor<boundary:
            result.append(boundary);cursor=boundary
    return result

def checkpoint_ready(path,step,window):
    assert read(path/'trainer_state.json')['global_step']==step
    assert read(path/'config.json')['moss_rswa']['window']==(None if window=='full' else int(window))
    for name in ['optimizer.bin','scheduler.pt','pytorch_model_fsdp.bin','model.safetensors']:
        assert (path/name).is_file() and (path/name).stat().st_size>0,(path,name)
    assert len(list(path.glob('rng_state_*.pth')))==4

def train_stage(root,state,window,seed,target):
    assert window in ACTIVE_TRAIN_WINDOWS,'Training disabled by the 2026-09-21 protocol amendment'
    key=group(window,seed);gs=state['groups'].setdefault(key,dict(window=window,seed=seed,step=0,history=[]))
    assert gs['step']<=target<=402
    assert sum(g['step'] for g in state['groups'].values())+target-gs['step']<=2010
    train=root/'training'/key;train.mkdir(parents=True,exist_ok=True)
    endpoints=sorted(set(range((gs['step']//10+1)*10,target+1,10))|({target} if target>gs['step'] else set()))
    for end in endpoints:
        ckpt=train/f'checkpoint-{end}'
        ready=train/f'checkpoint-{end}-ready.json'
        if not ready.exists():
            prepare_order_log(train,gs['step'])
            arguments=['bash',str(HERE/'native_sft.sh'),str(TASK/'data/train.jsonl'),str(train),'4','4',
                       '--fsdp',str(HERE/'fsdp1_offload.json'),'--gradient_checkpointing','false',
                       '--vit_gradient_checkpointing','false','--seed',str(seed),'--data_seed',str(seed)]
            if gs['step']:
                previous=train/f"checkpoint-{gs['step']}";checkpoint_ready(previous,gs['step'],window)
                arguments+=['--resume_from_checkpoint',str(previous)]
            env=dict(os.environ,NPROC_PER_NODE='4',MOSS_STOP_STEP=str(end),MOSS_RSWA_WINDOW=window)
            env.pop('MOSS_RESUME_AUDIT_CHECKPOINT',None);env.pop('MOSS_SAVE_STEP_ONE',None)
            write(root/'status.json',dict(stage='training',group=key,from_step=gs['step'],to_step=end))
            started=time.monotonic()
            subprocess.run(arguments,env=env,check=True)
            checkpoint_ready(ckpt,end,window)
            verify_order(train,end)
            elapsed=time.monotonic()-started
            write(ready,dict(step=end,window=window,config_sha256=sha(ckpt/'config.json'),
                from_step=gs['step'],actual_updates=end-gs['step'],stage_wall_seconds=elapsed,
                allocated_gpu_hours=4*elapsed/3600,includes_loading_and_checkpoint_save=True))
        full=end in MILESTONES
        if evaluate_at(end):
            run=root/'dev'/key/f'step-{end}'
            evaluate(run,ckpt,window,root/'dev/base',sentinel=not full)
            assessment=assess(run)
            # Preserve the per-stage observed data order for matched-group checks.
            if (train/'sample_order.jsonl').exists():
                shutil.copy2(train/'sample_order.jsonl',run/'sample_order_through_stage.jsonl')
                observed=(run/'sample_order_through_stage.jsonl').read_text().splitlines()
                for peer in state['groups'].values():
                    if peer['seed']!=seed or peer['window']==window:continue
                    other=root/'dev'/group(peer['window'],seed)/f'step-{end}'/'sample_order_through_stage.jsonl'
                    if other.exists():
                        expected=other.read_text().splitlines()
                        assert observed==expected,('Matched groups consumed different batches',key,str(other))
            # Commit only after matched-order checks; a failed check must not be
            # silently skipped when this phase is resumed.
            if full:
                assessment.update(step=end,checkpoint=str(ckpt));gs['history'].append(assessment)
        gs['step']=end;write(root/'state.json',state)
        prune(root,state)
    assert gs['step']==target,(gs,target)

def plan_extension(state,root,previous,current,next_target):
    active=[];benefit=False
    for window in ACTIVE_TRAIN_WINDOWS:
        g=state['groups'][group(window,0)];rows={r['step']:r for r in g['history']}
        if current not in rows or previous not in rows:continue
        old,new=rows[previous],rows[current]
        if old['functional'] or new['functional']:active.append(window)
        if window!='full' and extending(old,new):benefit=True
    result=dict(previous=previous,current=current,next_target=next_target,benefit=benefit,active=active)
    write(root/f'extension-{current}.json',result)
    if benefit:
        for target in cadence_targets(current,next_target):
            for window in active:state['queue'].append(dict(kind='train',window=window,seed=0,target=target))
        state['queue'].append(dict(kind='extend' if next_target==268 else 'select',previous=current,current=next_target,
                                   next_target=402))
    else:state['queue'].append(dict(kind='select'))

def select_models(state,root):
    full=state['groups'][group('full',0)]
    selected_full=choose(full['history']) or choose(full['history'],require_functional=False)
    reference=selected_full if selected_full['functional'] else assess(root/'dev/base')
    choices={}
    for window in ['128','256']:
        g=state['groups'][group(window,0)]
        # Zero-shot window ablations remain separate from trained candidates.
        choices[window]=choose([r for r in g['history'] if r['step']>0],reference=reference)
    valid=[dict(row,window=w) for w,row in choices.items() if row is not None]
    winner=choose(valid) if valid else None
    state['selection']=dict(full=selected_full,windows=choices,winner=winner)
    write(root/'selection_frozen.json',state['selection'])
    for window in ['128','256']:
        state['queue'].append(dict(kind='ablate',window=window,checkpoint=selected_full['checkpoint']))
    # Freeze the checkpoint of every model before opening the existing Test set.
    state['queue'].append(dict(kind='test_base'))
    for window in ['full','128','256']:
        g=state['groups'][group(window,0)]
        selected=selected_full if window=='full' else choices[window]
        if selected is None:selected=choose([r for r in g['history'] if r['step']>0],require_functional=False)
        state['selection'].setdefault('test_checkpoints',{})[window]=dict(checkpoint=selected['checkpoint'],
            step=selected['step'],functional=selected['functional'],dev_qualified=(window=='full' or choices[window] is not None))
        state['queue'].append(dict(kind='test',window=window,seed=0,checkpoint=selected['checkpoint']))
    write(root/'selection_frozen.json',state['selection'])
    if winner:
        window=winner['window'];cap=state['groups'][group(window,0)]['step']
        # The amended replication keeps only the selected R-SWA window.
        for target in cadence_targets(0,cap):
            state['queue'].append(dict(kind='train',window=window,seed=1,target=target))
        state['queue'].append(dict(kind='select_seed1',window=window))
    else:queue_analysis(state)

def queue_analysis(state):
    state['queue'] += [dict(kind='performance',window=w,sample=s)
                       for s in ['smoke','typical','longest'] for w in ['full','128','256']]
    state['queue'].append(dict(kind='analysis'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--chain',action='store_true')
    a=ap.parse_args();root=a.root;root.mkdir(parents=True,exist_ok=True)
    lock=(root/'campaign.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate=read(root/'qualification.json');assert gate['passed'],'Engineering qualification required'
    assert gate['source_manifest_sha256']==sha(HERE/'source_manifest.json')
    from freeze_inputs import main as verify_frozen_inputs
    verify_frozen_inputs()
    path=root/'state.json'
    if path.exists():state=read(path)
    else:
        queue=[dict(kind='zero',window=w) for w in ['full','128','256']]
        queue += [dict(kind='train',window=w,seed=0,target=30) for w in ['128','full','256']]
        queue += [dict(kind='train',window=w,seed=0,target=t) for t in range(40,151,10) for w in ['128','full','256']]
        queue += [dict(kind='extend',previous=100,current=150,next_target=268)]
        state=dict(queue=queue,groups={},completed=[],approved_max_updates=2010)
    if not state['queue']:return
    phase=state['queue'][0];write(root/'status.json',dict(stage='running_phase',phase=phase))
    try:
        kind=phase['kind']
        if kind=='zero':
            w=phase['window'];run=root/'dev'/('base' if w=='full' else f'zero-{w}')
            evaluate(run,BASE,w,None if w=='full' else root/'dev/base')
            assessment=assess(run);assessment.update(step=0,checkpoint=str(BASE))
            state['groups'][group(w,0)]=dict(window=w,seed=0,step=0,history=[assessment])
        elif kind=='train':train_stage(root,state,phase['window'],phase['seed'],phase['target'])
        elif kind=='extend':plan_extension(state,root,phase['previous'],phase['current'],phase['next_target'])
        elif kind=='select':select_models(state,root)
        elif kind=='ablate':
            evaluate(root/'dev'/f"full-to-{phase['window']}",Path(phase['checkpoint']),phase['window'],
                     root/'dev/base',ablation=True)
        elif kind=='test_base':evaluate(root/'test/base',BASE,'full',split='test')
        elif kind=='test':
            evaluate(root/'test'/group(phase['window'],phase['seed']),Path(phase['checkpoint']),phase['window'],
                     root/'test/base',split='test')
        elif kind=='select_seed1':
            reference=state['selection']['full'];w=phase['window']
            row=choose(state['groups'][group(w,1)]['history'],reference=reference)
            if row is None:row=choose(state['groups'][group(w,1)]['history'],require_functional=False)
            chosen={w:row};state['queue'].append(dict(kind='test',window=w,seed=1,checkpoint=row['checkpoint']))
            write(root/'selection_seed1_frozen.json',chosen);queue_analysis(state)
        elif kind=='performance':
            from run_eval import VENV
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
            dest=root/'performance'/f"{phase['sample']}-{phase['window']}"
            if not (dest/'complete.json').exists():
                subprocess.run([str(VENV),str(HERE/'performance.py'),'--run',str(dest),
                    '--window',phase['window'],'--sample',phase['sample']],env=env,check=True)
        elif kind=='analysis':
            from run_eval import METRIC
            results=[read(p) for p in (root/'performance').glob('*/complete.json')]
            assert len(results)==9 and all(r['complete'] and not r['engineering_only'] for r in results)
            write(root/'performance/complete.json',dict(complete=True,cases=results))
            subprocess.run([str(METRIC),str(HERE/'final_analysis.py'),'--root',str(root)],check=True)
        else:raise ValueError(kind)
        state['completed'].append(state['queue'].pop(0));write(path,state)
        write(root/'status.json',dict(stage='phase_complete',phase=phase,remaining_phases=len(state['queue'])))
        if not state['queue']:
            write(root/'status.json',dict(stage='experiment_complete',
                report=str(root/'analysis/结果报告.md'),updates=sum(g['step'] for g in state['groups'].values())))
        if a.chain and state['queue']:
            job=subprocess.check_output(['sbatch','--parsable','--export=ALL,MOSS_CAMPAIGN='+str(root),str(HERE/'campaign.slurm')],text=True).strip()
            write(root/'next_job.json',dict(job=job,phase=state['queue'][0]));print('CAMPAIGN_NEXT_JOB '+job,flush=True)
    except Exception:
        write(root/'status.json',dict(stage='failed',phase=phase,traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
