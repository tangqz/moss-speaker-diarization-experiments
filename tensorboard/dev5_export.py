"""Read-only campaign export; each LR has a separate TensorBoard run."""
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REMOTE = r'''
import json,sys,subprocess
from pathlib import Path
r=Path(sys.argv[1]);job=int(sys.argv[2])
assert r.is_relative_to(Path('/work/qt28/moss/results')) and r.name=='full-attention-v2-'+str(job)
def read(p):return json.loads(p.read_text())
def lines(p):
    rows=[]
    if p.exists():
        for s in p.read_text().splitlines():
            try:rows.append(json.loads(s))
            except json.JSONDecodeError:pass
    return rows
trials=[]
paths=[r/'baseline']+sorted((r/'trials').glob('*'))
for path in paths:
    if not path.is_dir():continue
    config=read(path/'trial.json') if (path/'trial.json').exists() else {}
    cp=Path(config['checkpoint_root']) if config else None
    steps=[]
    if cp:
        ranks=[{x['step']:x for x in lines(cp/f'train-rank-{i}.jsonl')} for i in range(4)]
        logs={x['step']:x for x in lines(cp/'trainer-log.jsonl')}
        for step in sorted(set.intersection(*(set(x) for x in ranks))):
            rs=[x[step] for x in ranks]
            for x in rs:x.pop('updates',None)
            steps.append(dict(step=step,ranks=rs,trainer=logs.get(step,{})))
    losses=[]
    for p in sorted((path/'evaluations').glob('devloss-*/dev_loss_summary.json')):
        d=read(p)
        if d['complete']:losses.append({k:d[k] for k in ['step','groups','complete']})
    full=[]
    for p in sorted((path/'evaluations').glob('dev-*/scoring_complete.json')):
        d=read(p);s=read(p.parent/'metrics_summary.json')
        if d['complete'] and s['complete'] and s['scored_predictions']==52 and not s['missing'] and not s['errors']:
            full.append(dict(step=int(p.parent.name.split('-')[-1]),summary=s,decision=d))
    trials.append(dict(id=path.name,steps=steps,losses=losses,full=full,
      status=read(path/'status.json') if (path/'status.json').exists() else {},
      selection=read(path/'selection.json') if (path/'selection.json').exists() else {}))
accounting=subprocess.run(['sacct','-j',str(job),'--format=JobIDRaw,State,ExitCode','--parsable2','--noheader'],capture_output=True,text=True,timeout=15,check=True).stdout
slurm=next((dict(state=s.split('|')[1],exit_code=s.split('|')[2]) for s in accounting.splitlines() if s.split('|')[0]==str(job)),{})
print(json.dumps(dict(trials=trials,slurm=slurm,status=read(r/'status.json') if (r/'status.json').exists() else {'stage':'queued'},active=read(r/'active_trial.json') if (r/'active_trial.json').exists() else None)))
'''


def fetch(exporter,current,ssh,here):
    job=current['job']
    cmd='/dkucc/home/qt28/envs/moss312/bin/python - '+shlex.quote(current['remote_run'])+' '+str(job)
    response=subprocess.run(ssh+[cmd],input=REMOTE.encode(),capture_output=True,timeout=45,check=True)
    data=json.loads(response.stdout)
    observed=[]
    for trial in data['trials']:
        label=f"dev5_{trial['id']}_{job}"
        for item in trial['steps']:
            rank=item['ranks'];ms=[x['microbatches'][0] for x in rank]
            n=sum(x['local_supervised_tokens'] for x in ms)
            assert len(ms)==4 and all(x['global_supervised_tokens']==n for x in ms)
            wall=max(datetime.fromisoformat(x['utc']).timestamp() for x in rank)
            for tag,value in {'train/loss':sum(x['loss_numerator'] for x in ms)/n,
                'train/learning_rate':rank[0]['lr'],'train/supervised_tokens':n,
                'train/grad_norm':item['trainer'].get('grad_norm'),
                'performance/step_seconds':max(x['seconds'] for x in rank),
                'performance/peak_gpu_gib':max(x['peak_allocated_gib'] for x in rank)}.items():
                exporter.scalar(label,tag,item['step'],value,wall)
        for evaluation in trial['losses']:
            for corpus,g in evaluation['groups'].items():
                tag='dev/loss' if corpus=='all' else 'dev/loss_'+corpus
                exporter.scalar(label,tag,evaluation['step'],g['loss'])
                exporter.scalar(label,'dev/supervised_tokens_'+corpus,evaluation['step'],g['tokens'])
        for evaluation in trial['full']:
            for key,g in evaluation['summary']['groups'].items():
                model,corpus,split=key.split('/')
                if model!='sft':continue
                for metric,value in {'CER_percent':100*g['CER'],'cpCER_percent':100*g['cpCER'],
                    'DER025_percent':100*g['DER_collar_0.25']['DER'],
                    'truncated':g['truncated'],'empty_predictions':g['empty_predictions']}.items():
                    exporter.scalar(label,f'dev/{corpus}/{metric}',evaluation['step'],value)
            for key in ['score','eligible','catastrophic']:
                exporter.scalar(label,'dev/'+key,evaluation['step'],evaluation['decision'][key])
        last=max((x['step'] for x in trial['steps']),default=0)
        exporter.text(label,'status/current',last,json.dumps(trial['status'],ensure_ascii=False,indent=2))
        exporter.text(label,'status/protocol',0,
            '普通CE；train/loss为当前四场训练batch；dev/loss为完整26场dev，每5步重算。'
            'dev生成指标需全部26场评分完成后更新。step0 loss新计算，base完整生成预测复用历史生产记录。')
        observed.append(dict(id=trial['id'],tensorboard_run=label,last_train_step=last))
    active=(data['active'] or {}).get('id')
    last=next((x['last_train_step'] for x in observed if x['id']==active),0)
    exporter.text(f'dev5_campaign_{job}','status/current',last,json.dumps(data['status'],ensure_ascii=False,indent=2))
    exporter.flush()
    snapshot=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status='connected',job=job,
        tensorboard_run=f'dev5_{active}_{job}' if active else f'dev5_baseline_{job}',
        pipeline=data['status'],slurm=data['slurm'],last_train_step=last,trials=observed,
        refresh_seconds=60,source='Read-only SSH; all completed trial metrics retained')
    (Path(here)/'bridge_status.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
    print(json.dumps(snapshot),flush=True)
