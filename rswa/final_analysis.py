"""Meeting-paired intervals and global-mapping speaker diagnostics; never selection."""
import argparse,collections,json,subprocess
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from rapidfuzz.distance import Levenshtein
from common import read,write,ROOT,now
from score_v1 import normalize

def pooled(rows):
    chars=sum(r['text']['reference_characters'] for r in rows)
    speech=sum(r['diarization']['DER_collar_0.25']['total'] for r in rows)
    return dict(CER=sum(r['text']['character_errors'] for r in rows)/chars,
                cpCER=sum(r['text']['cp_character_errors'] for r in rows)/chars,
                DER025=sum(sum(r['diarization']['DER_collar_0.25'][k] for k in
                    ['confusion','missed detection','false alarm']) for r in rows)/speech,
                RTF=sum(r['e2e_seconds'] for r in rows)/sum(r['duration'] for r in rows))

def paired(candidate,reference,seed=20260920):
    rng=np.random.default_rng(seed);corpora=sorted({r['dataset'] for r in candidate.values()})
    output={}
    for corpus in corpora:
        keys=sorted(k for k in candidate.keys()&reference.keys() if candidate[k]['dataset']==corpus)
        assert keys
        x=[candidate[k] for k in keys];y=[reference[k] for k in keys]
        px,py=pooled(x),pooled(y);draws=collections.defaultdict(list)
        for _ in range(1000):
            ix=rng.integers(0,len(keys),len(keys))
            a,b=pooled([x[i] for i in ix]),pooled([y[i] for i in ix])
            for metric in a:draws[metric].append(a[metric]-b[metric])
        output[corpus]=dict(paired_meetings=len(keys),candidate_total=sum(r['dataset']==corpus for r in candidate.values()),
            reference_total=sum(r['dataset']==corpus for r in reference.values()),
            differences={m:dict(point=px[m]-py[m],ci95=np.quantile(draws[m],[.025,.975]).tolist(),
                                 unit='ratio; multiply errors by 100 for percentage points') for m in px},
            sampling='1000 paired meeting resamples within corpus',seed=seed)
    return output

def merge(intervals):
    out=[]
    for lo,hi in sorted(intervals):
        if hi<=lo:continue
        if out and lo<=out[-1][1]:out[-1][1]=max(out[-1][1],hi)
        else:out.append([lo,hi])
    return out

def clip(intervals,uem):
    return merge((max(a,c),min(b,d)) for a,b in intervals for c,d in uem if min(b,d)>max(a,c))

def overlap(a,b):
    i=j=0;total=0.
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<=b[j][1]:i+=1
        else:j+=1
    return total

def speaker_diagnostics(ref,pred,duration):
    uem=merge((float(x.split()[2]),float(x.split()[3])) for x in ref['uem'].splitlines() if x.strip())
    reference=collections.defaultdict(list);hypothesis=collections.defaultdict(list)
    for line in ref['rttm'].splitlines():
        if not line.strip():continue
        x=line.split();lo=float(x[3]);reference[x[7]].append((lo,lo+float(x[4])))
    for s in pred['segments']:
        if s['end']>s['start']>=0:hypothesis[str(s['speaker'])].append((s['start'],s['end']))
    reference={k:clip(v,uem) for k,v in reference.items()};hypothesis={k:clip(v,uem) for k,v in hypothesis.items()}
    rkeys,hkeys=sorted(reference),sorted(hypothesis)
    costs=np.array([[overlap(reference[r],hypothesis[h]) for h in hkeys] for r in rkeys])
    mapping={}
    if rkeys and hkeys:
        ri,hi=linear_sum_assignment(-costs)
        mapping={hkeys[j]:rkeys[i] for i,j in zip(ri,hi) if costs[i,j]>0}
    all_hyp=merge(v for values in hypothesis.values() for v in values)
    events_for_overlap=sorted((t,delta) for values in reference.values() for lo,hi in values for t,delta in [(lo,1),(hi,-1)])
    active=0;previous=0.;overlap_seconds=0.
    for t,delta in events_for_overlap:
        if active>=2:overlap_seconds+=t-previous
        active+=delta;previous=t
    buckets={name:dict(events=0,reference_seconds=0.,covered_seconds=0.,confused_seconds=0.,missing_seconds=0.)
             for name in ['under30','30to120','over120']}
    events=[];fragmentation=[]
    for ri,r in enumerate(rkeys):
        intervals=reference[r];last=None
        mapped=merge(v for h,values in hypothesis.items() if mapping.get(h)==r for v in values)
        total=sum(b-a for a,b in intervals)
        substantial=[hkeys[j] for j in range(len(hkeys)) if costs[ri,j]>=max(1.,.1*total)] if hkeys else []
        fragmentation.append(dict(reference_speaker=r,substantial_hypothesis_speakers=substantial,
                                  count=len(substantial),threshold_seconds=max(1.,.1*total),reference_seconds=total))
        for lo,hi in intervals:
            if last is not None:
                gap=lo-last;bucket='under30' if gap<30 else '30to120' if gap<=120 else 'over120'
                covered=overlap([[lo,hi]],all_hyp);correct=overlap([[lo,hi]],mapped)
                row=dict(speaker=r,start=lo,end=hi,gap_seconds=gap,bucket=bucket,
                         reference_seconds=hi-lo,covered_seconds=covered,
                         confused_seconds=max(0,covered-correct),missing_seconds=hi-lo-covered)
                events.append(row);b=buckets[bucket];b['events']+=1
                for key in ['reference_seconds','covered_seconds','confused_seconds','missing_seconds']:b[key]+=row[key]
            last=hi
    # Full-stream character alignment assigns deletions to each reference
    # character's time bucket, so omitted speech remains in the denominator.
    rtext='';character_buckets=[]
    for s in sorted(ref['segments'],key=lambda s:(s['start'],s['end'])):
        value=normalize(s['text']);rtext+=value
        character_buckets.extend([min(2,int(3*((s['start']+s['end'])/2)/duration))]*len(value))
    htext=''.join(normalize(s['text']) for s in sorted(pred['segments'],key=lambda s:(s['start'],s['end'])))
    time_bins=[dict(reference_characters=character_buckets.count(i),deletions=0,substitutions=0) for i in range(3)]
    for edit in Levenshtein.editops(rtext,htext):
        if edit.tag in ('delete','replace'):
            key='deletions' if edit.tag=='delete' else 'substitutions';time_bins[character_buckets[edit.src_pos]][key]+=1
    return dict(global_hypothesis_to_reference=mapping,reentry_buckets=buckets,reentry_events=events,
        fragmentation=fragmentation,reference_time_thirds=time_bins,
        reference_speakers=len(rkeys),reference_overlap_fraction=overlap_seconds/max(sum(hi-lo for lo,hi in uem),1e-9),
        reference_characters=len(rtext),
        interpretation='auxiliary UEM-clipped temporal-overlap diagnostics; one global mapping per meeting; not standard DER')

def stratify(metrics,diagnostics):
    out={};reentry={}
    for corpus in sorted({r['dataset'] for r in metrics.values()}):
        keys=sorted(k for k,r in metrics.items() if r['dataset']==corpus)
        lengths=[diagnostics[k]['reference_characters'] for k in keys];low,high=np.quantile(lengths,[1/3,2/3])
        groups=collections.defaultdict(list)
        for key in keys:
            d=diagnostics[key];n=d['reference_speakers'];o=d['reference_overlap_fraction'];length=d['reference_characters']
            labels={'speakers':'1to3' if n<=3 else '4' if n==4 else '5plus',
                    'overlap':'under10pct' if o<.1 else '10to30pct' if o<=.3 else 'over30pct',
                    'text_length':'lower_third' if length<=low else 'middle_third' if length<=high else 'upper_third'}
            for dimension,label in labels.items():groups[dimension+'/'+label].append(metrics[key])
        out[corpus]=dict(reference_length_cutpoints=[float(low),float(high)],
            strata={name:dict(meetings=len(rows),**pooled(rows)) for name,rows in groups.items()})
        reentry[corpus]={}
        for bucket in ['under30','30to120','over120']:
            totals={field:sum(diagnostics[k]['reentry_buckets'][bucket][field] for k in keys)
                    for field in ['events','reference_seconds','covered_seconds','confused_seconds','missing_seconds']}
            denominator=totals['reference_seconds']
            totals.update(coverage=totals['covered_seconds']/denominator if denominator else None,
                          confusion_fraction=totals['confused_seconds']/denominator if denominator else None)
            reentry[corpus][bucket]=totals
    return dict(strata=out,reentry=reentry,exploratory=True,checkpoint_selection_uses_these=False)

def main(root):
    root=Path(root);report=root/'analysis';report.mkdir(exist_ok=True)
    runs={p.parent.name:p.parent for p in sorted((root/'test').glob('*/metrics_summary.json'))}
    assert 'base' in runs
    metrics={};summaries={}
    for name,path in runs.items():
        summary=read(path/'metrics_summary.json');assert summary['complete'],path
        summaries[name]=summary
        metrics[name]={r['key']:r for r in read(path/'per_record_metrics.json') if r['model']=='sft'}
        refs=read(path/'references.json');items=read(path/'inputs.json');assert len(items)==56
        diagnostics={item['key']:dict(dataset=item['dataset'],duration=item['duration'],
            **speaker_diagnostics(refs[item['key']],read(path/'predictions/sft'/f"{item['key']}.json"),item['duration'])) for item in items}
        write(report/f'{name}-speaker_diagnostics.json',diagnostics)
        write(report/f'{name}-stratified_summary.json',stratify(metrics[name],diagnostics))
    comparisons={}
    for name,rows in metrics.items():
        if name=='base':continue
        for ref in ['base','full-seed'+name.rsplit('seed',1)[-1]]:
            if ref in metrics and ref!=name:comparisons[f'{name} versus {ref}']=paired(rows,metrics[ref])
    write(report/'paired_intervals.json',comparisons)
    rows=['# MOSS 输出 KV 滑动窗口实验结果','',
          '各组统一 LR=1e-7，greedy，repetition penalty=1.02，仅惩罚已生成输出。',
          '以下为既有官方 Test 的复测；超参和检查点选择使用 dev。错误率包含坏输出，未删除失败会议。','',
          '| 模型 | 选中步数 | 语料 | CER | cpCER | DER025 | RTF | 截断 |',
          '|---|---:|---|---:|---:|---:|---:|---:|']
    for name,summary in summaries.items():
        checkpoint=Path(read(runs[name]/'evaluation_complete.json')['checkpoint'])
        step=read(checkpoint/'trainer_state.json')['global_step'] if (checkpoint/'trainer_state.json').exists() else 0
        for key,value in summary['groups'].items():
            if not key.startswith('sft/'):continue
            rows.append(f"| {name} | {step} | {key.split('/')[1]} | {100*value['CER']:.2f}% | {100*value['cpCER']:.2f}% | {100*value['DER_collar_0.25']['DER']:.2f}% | {value['RTF']:.3f} | {value['truncated']} |")
    selection=read(root/'selection_frozen.json')
    if selection['winner'] is None:
        rows+=['','本轮 dev 未获得满足预设功能资格与非劣门槛的 R-SWA 候选；表中保留各组冻结选点的结果，不宣称得到可用胜者。']
    else:
        winner=selection['winner'];rows+=['',f"dev 冻结的主窗口为 {winner['window']}，选中 step {winner['step']}。Test 结果不参与窗口选择。"]
    rows+=['','选中步数为 0 代表未微调 Base 权重；不能将该行视为窗口训练成功。各组同预算 dev 比较保留在原始 dev 目录。']
    state=read(root/'state.json')
    histories={key:g['history'] for key,g in state['groups'].items()}
    write(report/'dev_matched_budget.json',dict(histories=histories,
        score='equal corpus average of cpCER and DER025, percentage points',
        functional='all frozen functional checks pass; failures are retained'))
    rows+=['','| Dev 比较组 | 更新步数 | S（百分点） | 标准 CE | 功能资格 | 失败会议数 |',
           '|---|---:|---:|---:|---|---:|']
    for key,history in sorted(histories.items()):
        for record in sorted(history,key=lambda r:r['step']):
            ce=f"{record['ce']:.5f}" if record['ce'] is not None else 'N/A'
            rows.append(f"| {key} | {record['step']} | {record['S']:.3f} | {ce} | "
                        f"{'通过' if record['functional'] else '未通过'} | {len(record['failures'])} |")
    rows+=['','同一步数的行构成同预算比较；各组最佳检查点的 Test 表可能包含不同更新次数。',
           '逐语料 CER/cpCER/DER 与全部失败原因保存在 dev_matched_budget.json。']
    job_ids=sorted(p.stem.removeprefix('job-') for p in root.glob('job-*.log')
                   if p.stem.removeprefix('job-').isdigit())
    assert job_ids,'Missing campaign job provenance'
    accounting=subprocess.check_output(['sacct','-n','-P','-j',','.join(job_ids),
        '--format=JobIDRaw,State,ElapsedRaw,AllocTRES'],text=True)
    allocations=[]
    for line in accounting.splitlines():
        values=line.split('|')
        if values[0] not in job_ids:continue
        fields=dict(part.split('=',1) for part in values[3].split(',') if '=' in part)
        gpus=int(fields.get('gres/gpu',0));elapsed=int(values[2])
        allocations.append(dict(job_id=values[0],state=values[1],elapsed_seconds=elapsed,
            allocated_gpus=gpus,allocated_gpu_hours=gpus*elapsed/3600))
    assert {r['job_id'] for r in allocations}==set(job_ids)
    gpu_hours=sum(r['allocated_gpu_hours'] for r in allocations)
    write(report/'resource_accounting.json',dict(as_of_utc=now(),jobs=allocations,
        allocated_gpu_hours=gpu_hours,includes='all campaign training, evaluation and performance jobs, including failed attempts',
        excludes='earlier engineering qualification jobs; current analysis job has not ended'))
    rows+=['',f'截至报告生成时，正式作业累计分配 {gpu_hours:.2f} GPU-hours；包含训练、测评、性能回放和失败重试。',
           '工程验收作业另计，当前分析作业尚未结束；明细与采样时间见 resource_accounting.json。']
    rows+=['','配对会议 bootstrap 95% 区间保存在 paired_intervals.json；错误率差值乘 100 为百分点。',
           '说话人回归使用整场唯一映射，漏识别时间保留在分母；其诊断结果不替代标准 DER。',
           '性能回放和工程资格由各自原始报告支持。缺少性能结果时不得宣称达到加速目标。']
    performance=root/'performance/complete.json'
    if performance.exists():
        cases=read(performance)['cases'];lookup={(r['sample'],r['window']):r for r in cases}
        rows+=['','| 固定轨迹 | 输出长度 | 窗口 | decode 中位秒 | 相对 Full 变化 |',
               '|---|---:|---|---:|---:|']
        for (sample,window),case in lookup.items():
            full=lookup[(sample,'full')]
            for length,summary in case['summaries'].items():
                value=summary['decode_seconds']['median'];reference=full['summaries'][length]['decode_seconds']['median']
                rows.append(f'| {sample} | {length} | {window} | {value:.2f} | {100*(value/reference-1):+.1f}% |')
        rows+=['','固定轨迹使用同一 Base 权重及冻结训练目标，仅测计算效率；不能用于识别质量结论。',
               '总显存包含预分配 KV 池；实际在用的 KV 页见各性能目录 kv.jsonl。超出原始轨迹或逻辑上下文的长度标为不适用。',
               '输出 KV 有界不保证整个解码耗时恒定：1.02 penalty 仍使用全部已生成历史，采样与文本处理也计入实际延迟。',
               'measurements.json 同时保留主干及 penalty 的抽样 CUDA 时间；这些分项不包含全部主机开销，不能代替端到端计时。']
        # Keep the established scoring environment unchanged; plotting uses
        # the existing training environment's matplotlib installation.
        subprocess.run([str(ROOT/'envs/ms-swift-20260914/bin/python'),str(Path(__file__).with_name('plots.py')),
                        '--root',str(root),'--report',str(report)],check=True)
        rows+=['','![固定轨迹解码时间](fixed_trajectory_latency.png)','',
               '![实际 KV 占用](live_kv_pages.png)','',
               '![cpCER 配对区间](cpCER_paired_intervals.png)','',
               '![DER 配对区间](DER025_paired_intervals.png)']
    (report/'结果报告.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    write(report/'quality_analysis_complete.json',dict(complete=True,models=list(runs),meetings_per_model=56,
         performance_complete=(root/'performance/complete.json').exists()))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();main(a.root)
