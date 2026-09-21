"""Independent one-epoch R256/R512 runs with fixed endpoints and bounded retention."""
import argparse, fcntl, json, os, subprocess, time, traceback
from pathlib import Path
from common import ROOT, BASE, read, write, sha, now
from run_eval import evaluate, HENV, VENV, METRIC
from assess import assess
from order_audit import prepare as prepare_order, verify as verify_order

HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE.parent
ENDPOINTS=[1,25,50,75,100,125,134]
DEV={50,100,125}

def verify_source():
    manifest=read(HERE/'source_manifest.json')
    for name,digest in manifest.items():assert sha(HERE/name)==digest,('changed source',name)
    for name,digest in read(CAMPAIGN/'data_manifest.json').items():
        assert sha(CAMPAIGN/'data'/name)==digest,('changed input',name)

def checkpoint_ready(path,step,window):
    assert read(path/'trainer_state.json')['global_step']==step
    assert read(path/'config.json')['moss_rswa']['window']==int(window)
    for name in ['optimizer.bin','scheduler.pt','pytorch_model_fsdp.bin','model.safetensors']:
        assert (path/name).is_file() and (path/name).stat().st_size>0,(path,name)
    assert len(list(path.glob('rng_state_*.pth')))==4

def retain(training,step):
    """Keep two resume states; milestone and final inference weights. Never remove evidence."""
    root=training.resolve();removed=[]
    endpoints=[s for s in ENDPOINTS if s<=step]
    resume=set(endpoints[-2:]);weights=DEV|{25,75,134}|resume
    for p in root.glob('checkpoint-*'):
        if not p.is_dir() or p.is_symlink():continue
        n=int(p.name.split('-')[-1]);assert p.resolve().is_relative_to(root)
        names=[] if n in resume else ['optimizer.bin','scheduler.pt','pytorch_model_fsdp.bin']
        if n not in weights:names+=['model.safetensors']
        for name in names:
            item=p/name
            if item.exists():
                assert not item.is_symlink() and item.resolve().is_relative_to(root)
                removed.append(dict(path=str(item),bytes=item.stat().st_size));item.unlink()
    if removed:
        path=training/'retention.json';old=read(path) if path.exists() else []
        write(path,old+removed)

def release_export(run):
    for p in (run/'inference_model').glob('*.safetensors'):
        assert p.parent.resolve().is_relative_to(CAMPAIGN.resolve())
        p.unlink()  # Only duplicate export or link, never native weights.

def gate512():
    gate=CAMPAIGN/'engineering/r512';gate.mkdir(parents=True,exist_ok=True)
    if (gate/'complete.json').exists():return
    one=dict(os.environ,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0])
    subprocess.run([str(VENV),str(HERE/'vllm_contract.py'),'--output',str(gate/'vllm_contract.json')],env=one,check=True)
    subprocess.run([str(HENV.parent/'torchrun'),'--standalone','--nproc_per_node=4',str(HERE/'sp_contract.py'),
        '--output',str(gate/'sp4_contract.json')],check=True)
    trace=gate/'trace'
    for backend,python in [('vllm',VENV),('hf',HENV)]:
        subprocess.run([str(python),str(HERE/'inference_gate.py'),'--run',str(trace),'--backend',backend,
            '--window','512','--sample','smoke','--tokens','1026','--attention-backend','TRITON_ATTN','--record-only'],env=one,check=True)
    replay=read(trace/'hf_alignment.json')
    assert replay['all_layer_reference_unchanged']
    assert not any(row['high_margin_disagreement'] for row in replay['token_checks'])
    assert all(row['occupied']==replay['prefix']+512 and row['capacity']==replay['prefix']+512 for row in replay['hf_cache'])
    blocks=[json.loads(line) for line in (trace/'kv_blocks.jsonl').read_text().splitlines()]
    assert blocks and all(row['window']==512 for row in blocks)
    assert all(row['physical_blocks']<=((row['prefix_tokens']+15)//16)+32+2 for row in blocks if row['prefix_tokens'])
    assert read(gate/'vllm_contract.json')['passed'] and read(gate/'sp4_contract.json')['passed']
    write(gate/'complete.json',dict(passed=True,utc=now(),window=512,source_manifest_sha256=sha(HERE/'source_manifest.json'),
        scope='SP4 forward/backward, inference kernel, eviction, penalty and 1026-token real-MOSS smoke replay',quality_claim=False))

def finalize():
    if not all((CAMPAIGN/f'state-{w}.json').exists() and read(CAMPAIGN/f'state-{w}.json').get('complete') for w in ['256','512']):return
    lock=(CAMPAIGN/'summary.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    rows=[]
    for w in ['256','512']:
        state=read(CAMPAIGN/f'state-{w}.json')
        test=read(CAMPAIGN/f'test/{w}-seed0/assessment.json')
        rows.append(dict(window=w,step=134,test=test,dev_history=state['history']))
    write(CAMPAIGN/'summary.json',dict(complete=True,utc=now(),results=rows,
        interpretation='Repeated evaluation on an established Test set; not a newly blind holdout'))
    lines=['# MOSS R-SWA 第二轮结果','', '两组从同一 Base 初始化，LR 1e-6，标准 CE，训练 1 epoch（134 updates/536 场），penalty 1.02。',
        '', '| 窗口 | Test S | 功能失败/56 | Ali CER | AMI CER |','|---|---:|---:|---:|---:|']
    for row in rows:
        t=row['test'];c=t['corpora']
        lines.append(f"| R{row['window']} | {t['S']:.4f} | {len(t['failures'])}/56 | {c['alimeeting']['CER']:.4f} | {c['ami']['CER']:.4f} |")
    lines+=['','S 为两语料 cpCER 与 DER(0.25s) 的等权百分制均值；越低越好。功能失败单独报告。',
        'Test 为既有官方测试集的再次测评。本轮固定使用 epoch 末 checkpoint，不按 Test 选模型。']
    (CAMPAIGN/'结果摘要.md').write_text('\n'.join(lines)+'\n')

def train(window):
    assert window in ('256','512')
    statefile=CAMPAIGN/f'state-{window}.json';statusfile=CAMPAIGN/f'status-{window}.json'
    lock=(CAMPAIGN/f'{window}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=read(statefile) if statefile.exists() else dict(window=window,seed=0,step=0,history=[],complete=False)
    training=CAMPAIGN/f'training/{window}-seed0';training.mkdir(parents=True,exist_ok=True)
    try:
        if window=='512':
            write(statusfile,dict(stage='engineering_r512',utc=now()));gate512()
        for target in ENDPOINTS:
            if state['step']>=target:continue
            ckpt=training/f'checkpoint-{target}';ready=training/f'checkpoint-{target}-ready.json'
            write(statusfile,dict(stage='training',window=window,from_step=state['step'],target=target,utc=now()))
            if not ready.exists():
                prepare_order(training,state['step'])
                args=['bash',str(HERE/'native_sft.sh'),str(CAMPAIGN/'data/train.jsonl'),str(training),'4','4',
                    '--fsdp',str(HERE/'fsdp1_offload.json'),'--gradient_checkpointing','false',
                    '--vit_gradient_checkpointing','false','--seed','0','--data_seed','0']
                if state['step']:
                    previous=training/f"checkpoint-{state['step']}";checkpoint_ready(previous,state['step'],window)
                    args+=['--resume_from_checkpoint',str(previous)]
                env=dict(os.environ,NPROC_PER_NODE='4',MOSS_STOP_STEP=str(target),MOSS_RSWA_WINDOW=window)
                env.pop('MOSS_RESUME_AUDIT_CHECKPOINT',None);env.pop('MOSS_SAVE_STEP_ONE',None)
                start=time.monotonic();subprocess.run(args,env=env,check=True)
                checkpoint_ready(ckpt,target,window);order=verify_order(training,target)
                runtime=read(training/'rswa_runtime.json')
                assert runtime['initial_lr']==1e-6 and runtime['window']==window
                write(ready,dict(step=target,window=window,stage_wall_seconds=time.monotonic()-start,
                    actual_updates=target-state['step'],order=order,config_sha256=sha(ckpt/'config.json')))
            if target in (25,50,75,100,125):
                full=target in DEV;run=CAMPAIGN/f'dev/{window}-seed0/step-{target}'
                write(statusfile,dict(stage='full_dev' if full else 'sentinel',window=window,step=target,utc=now()))
                evaluate(run,ckpt,window,CAMPAIGN/'dev/base',sentinel=not full)
                row=assess(run);row.update(step=target,full_dev=full)
                state['history'].append(row);release_export(run)
            state['step']=target;write(statefile,state);retain(training,target)
        # Freeze the epoch-end endpoint before observing Test outcomes.
        write(CAMPAIGN/f'test_checkpoint-{window}.json',dict(step=134,window=window,
            checkpoint=str(training/'checkpoint-134'),selection='fixed epoch-end endpoint; no Test selection',utc=now()))
        base=CAMPAIGN/'test/base'
        if window=='256' and not (base/'evaluation_complete.json').exists():
            write(statusfile,dict(stage='shared_base_test',window=window,utc=now()))
            write(CAMPAIGN/'status-base.json',dict(stage='test_base',utc=now()))
            evaluate(base,BASE,'full',split='test')
            write(CAMPAIGN/'status-base.json',dict(stage='complete',utc=now()))
        write(statusfile,dict(stage='waiting_base_test',window=window,utc=now()))
        started=time.monotonic()
        while not (base/'evaluation_complete.json').exists():
            b=CAMPAIGN/'status-base.json'
            if b.exists() and read(b).get('stage')=='failed':raise RuntimeError('Shared Base Test job failed')
            if time.monotonic()-started>6*3600:raise RuntimeError('Shared Base Test exceeded six-hour wait')
            time.sleep(30)
        write(statusfile,dict(stage='test',window=window,step=134,utc=now()))
        run=CAMPAIGN/f'test/{window}-seed0'
        evaluate(run,training/'checkpoint-134',window,base,split='test')
        assess(run);release_export(run)
        # Matched input order is verified as soon as the other arm has finished training.
        peer=CAMPAIGN/f"training/{'512' if window=='256' else '256'}-seed0/order_audit.json"
        if peer.exists() and read(peer)['optimizer_steps']==134:
            assert (training/'sample_order.jsonl').read_bytes()==(peer.parent/'sample_order.jsonl').read_bytes()
            write(CAMPAIGN/'matched_order.json',dict(passed=True,meetings=536,updates=134))
        state['complete']=True;state['completed_utc']=now();write(statefile,state)
        write(statusfile,dict(stage='complete',window=window,step=134,utc=now()));finalize()
    except Exception:
        write(statusfile,dict(stage='failed',window=window,utc=now(),traceback=traceback.format_exc()));raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--window',choices=['256','512','base','gate'],required=True);a=ap.parse_args()
    verify_source()
    if a.window=='gate':
        write(CAMPAIGN/'status-512.json',dict(stage='engineering_r512',utc=now()))
        gate512()
        write(CAMPAIGN/'status-512.json',dict(stage='waiting_training',utc=now()))
        return
    if a.window=='base':
        try:
            write(CAMPAIGN/'status-base.json',dict(stage='test_base',utc=now()))
            evaluate(CAMPAIGN/'test/base',BASE,'full',split='test')
            write(CAMPAIGN/'status-base.json',dict(stage='complete',utc=now()))
        except Exception:
            write(CAMPAIGN/'status-base.json',dict(stage='failed',utc=now(),traceback=traceback.format_exc()));raise
    else:train(a.window)

if __name__=='__main__':main()
