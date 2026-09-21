"""Reviewed phase-50 evidence, CSVs and standalone scientific figures."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ART=HERE/'artifacts/63421'
RUN=ART/'raw/run'
EVAL=RUN/'evaluations/sentinel-50'
OUT=HERE/'stage50-report'
OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
verified=read(ART/'download-verification.json')
assert verified['files_verified']==247
assert hashlib.sha256((ART/'logs-63421.tar.gz').read_bytes()).hexdigest()==verified['sha256']
summary=read(EVAL/'metrics_summary.json')
assert summary['complete'] and summary['scored_predictions']==12 and not summary['errors'] and not summary['missing']
assert read(RUN/'outcome.json')['status']=='paused_for_sentinel_failure'

rank_rows=[[json.loads(s) for s in (ART/f'raw/checkpoints/train-rank-{rank}.jsonl').read_text(encoding='utf-8').splitlines()] for rank in range(4)]
training=[]
for i in range(50):
    batch=[r[i] for r in rank_rows]
    assert all(r['step']==i+1 for r in batch)
    micros=[r['microbatches'][0] for r in batch]
    denominator=sum(r['local_supervised_tokens'] for r in micros)
    assert all(r['global_supervised_tokens']==denominator for r in micros)
    training.append(dict(step=i+1,loss=sum(r['loss_numerator'] for r in micros)/denominator,lr=batch[0]['lr'],supervised_tokens=denominator))
teacher=[]
for step in [0,25,50]:
    s=read(RUN/f'evaluations/fast-{step}/fast_summary.json')
    teacher.append(dict(step=step,**{k:v['loss'] for k,v in s['groups'].items()}))
scores=[]
for dataset in ['alimeeting','ami']:
    row=dict(dataset=dataset)
    for model in ['base','sft']:
        v=summary['groups'][f'{model}/{dataset}/dev']
        row.update({f'{model}_CER':100*v['CER'],f'{model}_cpCER':100*v['cpCER'],f'{model}_DER025':100*v['DER_collar_0.25']['DER'],f'{model}_truncated':v['truncated']})
    scores.append(row)
per=[]
for v in read(EVAL/'per_record_metrics.json'):
    per.append(dict(meeting=v['session_id'],dataset=v['dataset'],model=v['model'],CER=100*v['text']['CER'],cpCER=100*v['text']['cpCER'],DER025=100*v['diarization']['DER_collar_0.25']['diarization error rate'],tokens=v['generated_tokens'],truncated=v['truncated']))
timeline=[];cases=[]
for model in ['base','sft']:
    v=read(EVAL/f'predictions/{model}/ami/dev/TS3004c.json')
    segments=v['segments'];groups=[]
    for txt,g in itertools.groupby(enumerate(segments),key=lambda x:x[1]['text'].strip().lower()):
        vals=list(g)
        groups.append(dict(text=txt,count=len(vals),segment_index=vals[0][0],start=vals[0][1]['start'],end=vals[-1][1]['end']))
    longest=max(groups,key=lambda g:g['count'])
    cases.append(dict(model=model,tokens=v['generated_tokens'],segments=len(segments),ended_eos=v['ended_eos'],last_end=segments[-1]['end'],repeat_count=longest['count'],repeat_text=longest['text'],repeat_start=longest['start'],repeat_end=longest['end'],repeat_segment_index=longest['segment_index']))
    for i,s in enumerate(segments):timeline.append(dict(model=model,segment=i+1,audio_seconds=s['end']))
assert cases[1]['repeat_count']==1695 and cases[1]['tokens']==65536
memory=read(HERE/'recovery/verification-63421.json')['steps']
for name,rows in [('training',training),('teacher',teacher),('scores',scores),('per_meeting',per),('case_summary',cases),('timeline',timeline),('memory',memory)]:
    with (OUT/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
sources={
 'training':('逐步监督损失',training,['raw/checkpoints/train-rank-*.jsonl'],'四 rank loss_numerator 合计 / 全局非忽略的监督 token 数；不同步的会议不同。'),
 'teacher':('固定六场 teacher 验证',teacher,['raw/run/evaluations/fast-{0,25,50}/fast_summary.json'],'同六场完整音频、正确历史前缀的逐 token CE；all 为监督 token 加权，不能代替自由生成。'),
 'scores':('第 50 步完整生成指标',scores,['raw/run/evaluations/sentinel-50/metrics_summary.json'],'CER/cpCER 为累计字符错误数除以累计参考字符数；DER025 为总错误时长除以 0.25 秒 collar 后参考时长；表中单位为百分比。'),
 'per_meeting':('逐场原始评分',per,['raw/run/evaluations/sentinel-50/per_record_metrics.json'],'原评分器逐场 CER/cpCER/DER025 百分比；truncated 表示达到 token 上限且没有 EOS。'),
 'case_summary':('TS3004c 结构化重复诊断',cases,['raw/run/evaluations/sentinel-50/predictions/{base,sft}/ami/dev/TS3004c.json'],'对原 parser 的 segments 按相邻 text.strip().lower() 分组，取连续相同文本最长组；此诊断不改变评分或暂停标准。'),
 'timeline':('TS3004c 输出覆盖时间',timeline,['raw/run/evaluations/sentinel-50/predictions/{base,sft}/ami/dev/TS3004c.json'],'每个已解析片段的结束秒数对其 1-based 输出序号；不是 wall-clock 或 token 位置。'),
 'memory':('修复后的 CPU 内存',memory,['recovery/verification-63421.json'],'各 rank 在步末的 RSS 采样值合计；清理前后可比较，不等同于同一瞬间的 Slurm 峰值。'),
}
queries={}
for name,(label,rows,files,definition) in sources.items():
    queries[name]=dict(rows=rows,source=dict(label=label,files=files,
        metricDefinitions=[definition],evidenceFlow=[dict(title='已校验日志归档',detail=f'训练 63421；{verified["files_verified"]} 个文件；SHA256 {verified["sha256"]}'),dict(title='重现',detail='运行 dkucc/full_attention_v2/build_stage_report.py；原 JSON、JSONL 与 CSV 均保留。')],
        caveats=['范围为预先选定的 6 场 dev 哨兵，不是全部 26 场 dev 或 56 场 test。','base 采用已完成的 63363 原始预测；质量协议一致，运行时间不是同轮配对测量。']),
        methods=[dict(language='text',code=definition)])
snapshot=dict(surface='report',title='Loss 已下降，长会议生成仍在第 50 步触发暂停',
    generatedAt=read(RUN/'status.json')['updated_utc'],status='observed',buildStatus='creating',filters=[],queries=queries,
    report=dict(asOf=read(RUN/'status.json')['updated_utc'],scope='MOSS full attention SFT v2 · 50 步阶段报告 · 6 场 dev'))
write(OUT/'reviewed.json',snapshot)
write(OUT/'analysis.json',dict(training_steps=50,teacher_loss=teacher,metrics=scores,cases=cases,
    paused=True,full_dev_performed=False,test_performed=False,loop_case='ami/dev/TS3004c',
    quality=read(EVAL/'quality_status.json'),archive=verified,diagnostic=read(HERE/'diagnostics/current_run.json')))

plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
blue,orange='#176b91','#ce6234'
def save(fig,name):
    fig.tight_layout();fig.savefig(OUT/f'{name}.png',dpi=175,bbox_inches='tight');fig.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].plot([r['step'] for r in training],[r['loss'] for r in training],color=blue)
ax[0].set(title='不同训练会议的逐步 loss',xlabel='优化器步数',ylabel='监督 token 平均 CE')
for k,label,color in [('all','六场合计',blue),('alimeeting','AliMeeting',orange),('ami','AMI','#42875b')]:
    ax[1].plot([r['step'] for r in teacher],[r[k] for r in teacher],marker='o',label=label,color=color)
ax[1].set(title='固定六场 teacher loss',xlabel='检查点步数');ax[1].legend(frameon=False)
save(fig,'loss')
fig,ax=plt.subplots(1,2,figsize=(11,4))
for axis,metric,title in zip(ax,['cpCER','DER025'],['cpCER (%)','DER，collar 0.25 s (%)']):
    for j,model in enumerate(['base','sft']):
        values=[r[f'{model}_{metric}'] for r in scores];xs=[i+(j-.5)*.32 for i in range(2)]
        bars=axis.bar(xs,values,.30,label='base' if model=='base' else 'SFT 第 50 步',color=blue if model=='base' else orange)
        axis.bar_label(bars,fmt='%.2f',padding=3,fontsize=10)
    axis.set(title=title,xticks=[0,1],xticklabels=['AliMeeting · 3 场','AMI · 3 场'],ylim=(0,max(r[f'{m}_{metric}'] for r in scores for m in ['base','sft'])*1.24))
    axis.legend(frameon=False)
save(fig,'generation')
fig,ax=plt.subplots(figsize=(10.5,4.5))
for model,color,label in [('base',blue,'base'),('sft',orange,'SFT 第 50 步')]:
    rows=[r for r in timeline if r['model']==model]
    ax.plot([r['segment'] for r in rows],[r['audio_seconds'] for r in rows],label=label,color=color)
ax.axhline(2970,color='#888',linestyle='--',lw=1,label='原音频 2,970 秒')
ax.axvspan(899,2593,color=orange,alpha=.09)
ax.annotate('连续 1,695 个 Mm. 片段\n覆盖时间仅由 2,502 推进到 2,640 秒',xy=(1500,2550),xytext=(1200,1550),arrowprops=dict(arrowstyle='->',color=orange),color=orange)
ax.set(xlabel='模型输出的已解析片段序号',ylabel='片段结束时间（秒）',title='TS3004c：结构化重复消耗生成预算');ax.legend(frameon=False,loc='lower right')
save(fig,'structured_loop')
fig,ax=plt.subplots(figsize=(10.5,3.5))
for field,label,color in [('summed_rank_rss_before_gib','步末清理前',orange),('summed_rank_rss_after_gib','步末清理后',blue)]:
    ax.plot([r['step'] for r in memory],[r[field] for r in memory],label=label,color=color)
ax.set(xlabel='优化器步数',ylabel='各 rank RSS 合计（GiB）',title='CPU 缓存修复后，每步内存均有回落');ax.legend(frameon=False)
save(fig,'host_memory')
print(json.dumps(dict(output=str(OUT),queries=len(queries),plots=4,verified_predictions=12)))
