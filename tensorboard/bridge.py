"""Read existing MOSS logs through authenticated SSH and append TensorBoard events.

This process has no write access to remote training files and imports no torch.
Metrics refresh every 60 seconds; conversational monitoring remains 45 minutes.
"""
import argparse
import hashlib
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.compat.proto.types_pb2 import DT_STRING
from tensorboard.plugins.text.metadata import create_summary_metadata
from tensorboard.summary.writer.event_file_writer import EventFileWriter

HERE=Path(__file__).resolve().parent
EXECUTION=HERE.parent/'full_attention_v2'
SSH=[r'C:\Program Files\Git\usr\bin\ssh.exe','-T','-O','proxy','-S',
     'C:/Users/qizhi/.ssh/cm-moss-check-20260912','-o','BatchMode=yes','qt28@dkucc-login-01.rc.duke.edu']
REMOTE=r'''
import json,sys,subprocess
from pathlib import Path
run=Path(sys.argv[1]);job=int(sys.argv[2]);after=int(sys.argv[3]);formal=sys.argv[4]=='formal'
assert run.is_relative_to(Path('/work/qt28/moss/results')) and run.name.startswith('full-attention-v2-')
def lines(p):
    if not p.exists():return []
    rows=[]
    for s in p.read_text().splitlines():
        try:rows.append(json.loads(s))
        except json.JSONDecodeError:pass
    return rows
def read(p):return json.loads(p.read_text())
root=Path('/work/qt28/moss/checkpoints')/('full-attention-v2-'+str(job)) if formal else run/'gates/ddp'
by_rank=[{r['step']:r for r in lines(root/f'train-rank-{i}.jsonl')} for i in range(4)]
common=set.intersection(*(set(r) for r in by_rank))
logs={r['step']:r for r in lines(root/'trainer-log.jsonl') if 'loss' in r}
steps=[dict(step=s,ranks=[r[s] for r in by_rank],trainer=logs.get(s,{})) for s in sorted(common) if s>=after-2]
fast=[read(p) for p in sorted((run/'evaluations').glob('fast-*/fast_summary.json'))]
full=[dict(folder=p.parent.name,summary=read(p),quality=read(p.parent/'quality_status.json') if (p.parent/'quality_status.json').exists() else {})
      for p in sorted((run/'evaluations').glob('*/metrics_summary.json')) if read(p).get('complete')]
status=read(run/'status.json') if (run/'status.json').exists() else {'stage':'qualification'}
accounting=subprocess.run(['sacct','-j',str(job),'--format=JobIDRaw,State,ExitCode','--parsable2','--noheader'],capture_output=True,text=True,timeout=15,check=True).stdout
slurm=next((dict(state=s.split('|')[1],exit_code=s.split('|')[2]) for s in accounting.splitlines() if s.split('|')[0]==str(job)),{})
host=[]
for rank in range(4):
    host.extend(r for r in lines(root/f'host-memory-rank-{rank}.jsonl') if r['step']>=after-2)
print(json.dumps(dict(job=job,steps=steps,fast=fast,full=full,status=status,slurm=slurm,host=host)))
'''


class Exporter:
    def __init__(self):
        self.index_path=HERE/'event_index.json'
        self.index=json.loads(self.index_path.read_text()) if self.index_path.exists() else {}
        self.writers={}

    def scalar(self,run,tag,step,value,wall=None):
        if value is None or not math.isfinite(float(value)):return
        key=f'{run}|{tag}|{step}'
        if key in self.index:return
        writer=self.writers.setdefault(run,None)
        if writer is None:
            writer=EventFileWriter(str(HERE/'events'/run));self.writers[run]=writer
        writer.add_event(Event(wall_time=wall or time.time(),step=step,
            summary=Summary(value=[Summary.Value(tag=tag,simple_value=float(value))])))
        self.index[key]=float(value)

    def text(self,run,tag,step,value):
        key=f'{run}|text|{tag}|{hashlib.sha256(value.encode()).hexdigest()}'
        if key in self.index:return
        if self.writers.get(run) is None:
            self.writers[run]=EventFileWriter(str(HERE/'events'/run))
        tensor=TensorProto(dtype=DT_STRING,string_val=[value.encode('utf-8')],
            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]))
        self.writers[run].add_event(Event(wall_time=time.time(),step=step,
            summary=Summary(value=[Summary.Value(tag=tag,tensor=tensor,
                metadata=create_summary_metadata(display_name='训练状态',description='实际作业与日志状态'))])))
        self.index[key]=True

    def flush(self):
        for writer in self.writers.values():writer.flush()
        tmp=self.index_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.index),encoding='utf-8');tmp.replace(self.index_path)

    def production_evaluation(self, config):
        if not config:
            return None
        run=Path(config['local_run']).resolve()
        assert run.is_relative_to((EXECUTION/'artifacts').resolve())
        read=lambda p:json.loads(p.read_text(encoding='utf-8'))
        outcome=read(run/'outcome.json')
        assert outcome['status']=='completed' and outcome['training_updates']==0
        label=f"production_eval_{config['job']}"
        wall=datetime.fromisoformat(outcome['completed_utc']).timestamp()
        metrics=read(run/'evaluations/sentinel-50/metrics_summary.json')
        assert metrics['complete'] and metrics['scored_predictions']==12
        assert not metrics['errors'] and not metrics['missing']
        for key,group in metrics['groups'].items():
            model,dataset,split=key.split('/')
            step=0 if model=='base' else 50
            for name in ['CER','cpCER']:
                self.scalar(label,f'six_meeting/{dataset}/{name}_percent',step,100*group[name],wall)
            self.scalar(label,f'six_meeting/{dataset}/DER025_percent',step,100*group['DER_collar_0.25']['DER'],wall)
            self.scalar(label,f'six_meeting/{dataset}/truncated',step,group['truncated'],wall)
        for case in read(run/'case_manifest.json'):
            pred=read(run/'cases'/case['id']/'predictions/sft'/(case['item']['key']+'.json'))
            records=read(run/'evaluations'/case['group']/'per_record_metrics.json')
            score=next(r for r in records if r['key']==pred['key'] and r['model']=='sft')
            step=case['step'];prefix=f"case/{pred['dataset']}/{pred['session_id']}"
            values={'cpCER_percent':100*score['text']['cpCER'],
                'DER025_percent':100*score['diarization']['DER_collar_0.25']['diarization error rate'],
                'generated_tokens':pred['generated_tokens'],'ended_eos':int(pred['ended_eos']),
                'invalid_tags':pred['diagnostics']['invalid_tag_count'],
                'longest_identical_token_run':pred['diagnostics']['longest_identical_run']['count']}
            for tag,value in values.items():
                self.scalar(label,f'{prefix}/{tag}',step,value,wall)
        self.text(label,'status/current',50,
            '生产入口完整复评已完成（作业 63441）。第 50 步六场中两场达到 token 上限；正式训练继续暂停。\n\n'
            'six_meeting 是每个语料三场；step 0 复用历史生产 base，step 50 为修正预处理后的 SFT。'
            'case 是单场诊断，TS3004c 的 step 0/10/50 均为本次完整生成。'
            '这两种范围不可合并为更多独立会议。原预处理路径的指标保留在旧 run。\n\n'
            'base 整场 26,337 个 token 与历史逐个相同；完整 dev/test 尚未执行。\n\n'
            f"归档 SHA256：{config['archive_sha256']}")
        return dict(job=config['job'],tensorboard_run=label,status='completed',
                    last_train_step=50,step50_truncated=2,quality_pass=False,
                    completed_utc=outcome['completed_utc'])

    def fetch(self):
        formal=(EXECUTION/'current_run.json').exists()
        current=json.loads((EXECUTION/('current_run.json' if formal else 'gate_run.json')).read_text(encoding='utf-8'))
        if current.get('mode')=='dev5_lr_search':
            from dev5_export import fetch
            return fetch(self,current,SSH,HERE)
        job=current['job'];label=current.get('tensorboard_run') or (f'full_attention_v2_{job}' if formal else f'qualification_{job}')
        done=[int(k.rsplit('|',1)[1]) for k in self.index if k.startswith(label+'|train/loss|')]
        after=max(done,default=0)
        cmd='/dkucc/home/qt28/envs/moss312/bin/python - '+shlex.quote(current['remote_run'])+' '+str(job)+' '+str(after)+' '+('formal' if formal else 'gate')
        result=subprocess.run(SSH+[cmd],input=REMOTE.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45,check=True)
        data=json.loads(result.stdout)
        for item in data['steps']:
            step=item['step'];ranks=item['ranks']
            micros=[m for r in ranks for m in r['microbatches']]
            tokens=sum(m['local_supervised_tokens'] for m in micros)
            assert len(micros)==4 and all(m['global_supervised_tokens']==tokens for m in micros)
            wall=max(datetime.fromisoformat(r['utc']).timestamp() for r in ranks)
            values={'train/loss':sum(m['loss_numerator'] for m in micros)/tokens,
                'train/learning_rate':ranks[0]['lr'],'train/learning_rate_x1e6':ranks[0]['lr']*1e6,'train/supervised_tokens':tokens,
                'train/grad_norm':item['trainer'].get('grad_norm'),
                'performance/compute_step_seconds':max(r['seconds'] for r in ranks),
                'performance/peak_gpu_gib':max(r['peak_allocated_gib'] for r in ranks)}
            for tag,value in values.items():self.scalar(label,tag,step,value,wall)
            for row in ranks:
                self.scalar(label,f"memory/rank_{row['rank']}_gpu_gib",step,row['peak_allocated_gib'],wall)
                self.scalar(label,f"host_memory/rank_{row['rank']}_peak_rss_gib",step,row['peak_process_rss_gib'],wall)
        for row in data['host']:
            for name,value in row['after'].items():
                self.scalar(label,f"host_memory/rank_{row['rank']}_after_cleanup_{name}",row['step'],value)
            self.scalar(label,f"host_memory/rank_{row['rank']}_pinned_before_cleanup_gib",row['step'],row['before']['pinned_owned_gib'])
        for evaluation in data['fast']:
            step=evaluation['step']
            for dataset,group in evaluation['groups'].items():
                self.scalar(label,f'validation/loss_{dataset}',step,group['loss'])
                for kind,v in group['by_type'].items():
                    if v['tokens']:
                        self.scalar(label,f'validation_types/{dataset}/{kind}_loss',step,v['nll_sum']/v['tokens'])
                        self.scalar(label,f'entropy/{dataset}/{kind}_nats',step,v['entropy_sum']/v['tokens'])
        for evaluation in data['full']:
            scope,step_text=evaluation['folder'].rsplit('-',1);step=int(step_text)
            for key,group in evaluation['summary']['groups'].items():
                model,dataset,split=key.split('/')
                if model!='sft':continue
                for metric in ['CER','cpCER']:
                    self.scalar(label,f'{scope}/{dataset}/{metric}_percent',step,100*group[metric])
                self.scalar(label,f'{scope}/{dataset}/DER025_percent',step,100*group['DER_collar_0.25']['DER'])
                self.scalar(label,f'{scope}/{dataset}/truncated',step,group['truncated'])
                self.scalar(label,f'{scope}/{dataset}/empty_parse',step,group['empty_predictions'])
            if evaluation['quality']:
                self.scalar(label,f'{scope}/major_parse_loss',step,evaluation['quality']['major_parse_loss'])
        last_step=max([s['step'] for s in data['steps']],default=after)
        stage=data['status'].get('stage','unknown')
        slurm_state=data['slurm'].get('state','unknown')
        stopped=stage=='failed' or slurm_state in ['OUT_OF_MEMORY','FAILED','TIMEOUT','NODE_FAIL','CANCELLED']
        names={'training':'正在训练','fast_validation':'正在验证 loss','full_generation':'正在完整会议生成验证','paused_for_sentinel_failure':'训练已按质量规则暂停','completed':'阶段流程已完成','validating_execution_bundle':'正在启动检查'}
        names.update(paused_for_probe_failure='第 10 步完整生成检查失败，短对照已暂停',
                     completed_short_trial='50 步短对照已完成，等待结果审阅')
        headline='训练已停止' if stopped else names.get(stage,'当前阶段：'+stage)
        status_text=f"作业 **{job}** · {headline}\n\n最后完成 **{last_step}** 步。调度器状态：**{slurm_state}**。"
        if stage=='paused_for_sentinel_failure':
            status_text+='\n\n第 50 步完整会议验证触发预定暂停条件。训练尚未完成，检查点已保留；调度器 COMPLETED 表示进程按规则退出。请查看 sentinel 指标与本次阶段报告。'
        if stage=='paused_for_probe_failure':
            status_text+='\n\n完整 TS3004c 触发预定灾难性输出门槛，未继续更新到第 25/50 步。检查点和失败输出保留。'
        if stage=='completed_short_trial':
            status_text+='\n\n短对照达到预定停止点；尚未执行全部 dev/test，不能据此宣称已收敛或产生正式合格 SFT。'
        if slurm_state=='OUT_OF_MEMORY':
            status_text+='\n\n服务器作业内存超限，进程被终止。TensorBoard 同步连接正常；曲线停止是因为没有新的训练更新。'
        elif stopped:
            status_text+='\n\n'+data['status'].get('error','请检查任务日志。')
        if stage in ['fast_validation','full_generation']:
            status_text+='\n\n正在验证，训练曲线暂时保持不变；验证结束后同步相应指标。'
        if current.get('evaluation_note'):
            status_text+='\n\n'+current['evaluation_note']
            self.text(label,'status/evaluation_protocol',last_step,current['evaluation_note'])
        self.text(label,'status/current',last_step,status_text)
        production=self.production_evaluation(current.get('production_evaluation'))
        self.flush()
        snapshot=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status='connected',job=job,
            tensorboard_run=label,pipeline=data['status'],slurm=data['slurm'],last_train_step=last_step,
            refresh_seconds=60,source='Read-only SSH export of existing logs',
            evaluation_note=current.get('evaluation_note'),production_evaluation=production)
        (HERE/'bridge_status.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
        print(json.dumps(snapshot),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--once',action='store_true');a=ap.parse_args()
    exporter=Exporter()
    while True:
        try:exporter.fetch()
        except Exception as exc:
            (HERE/'bridge_status.json').write_text(json.dumps(dict(status='connection_error',
                error=str(exc),updated_utc=datetime.now(timezone.utc).isoformat())),encoding='utf-8')
            traceback.print_exc()
            if a.once:raise
        if a.once:break
        time.sleep(60)


if __name__=='__main__':main()
