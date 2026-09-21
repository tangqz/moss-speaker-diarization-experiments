"""Build an offline visual report from the verified, downloaded evaluation bundle."""
import argparse
import base64
import csv
from datetime import datetime
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--artifact',type=Path,required=True)
    a=p.parse_args()
    root=a.artifact.resolve()
    assert load(root/'download-verification.json')['verified_files']>0
    raw=root/'raw'
    runs=list((raw/'results').glob('eval-62994-*'))
    assert len(runs)==1,runs
    run=runs[0]
    summary=load(run/'metrics_summary.json')
    assert summary['complete'] and not summary['errors'] and not summary['missing']
    assert load(run/'job_exit.json')['exit_code']==0
    per=load(run/'per_record_metrics.json')
    training=load(raw/'checkpoints/full-attention-ddp4-62994/trainer_state.json')
    train_metrics=load(raw/'checkpoints/full-attention-ddp4-62994/train_results.json')
    audit=load(run/'data_audit.json')
    with (run/'gpu_start.csv').open(encoding='utf-8') as f:
        gpu_rows=list(csv.DictReader(f,skipinitialspace=True))
    occupancy='；'.join(f"GPU {r['index']} 启动前已有 {float(r['memory.used [MiB]'].split()[0]):.0f} MiB 显存占用" for r in gpu_rows)
    out=root/'report'
    out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'white','axes.facecolor':'white',
        'savefig.facecolor':'white','svg.fonttype':'none'})
    blue,orange='#176b91','#d56836'
    charts=[]
    def save(fig,name,title,note):
        fig.tight_layout()
        fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(out/(name+'.svg'),bbox_inches='tight')
        charts.append((name,title,note))
        plt.close(fig)
    logs=[r for r in training['log_history'] if 'loss' in r]
    steps=np.array([r['step'] for r in logs]); losses=np.array([r['loss'] for r in logs])
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    axes[0].plot(steps,losses,color=blue,alpha=.3,lw=.8,label='每步训练 loss')
    axes[0].plot(steps[24:],np.convolve(losses,np.ones(25)/25,mode='valid'),color=blue,lw=2,label='最近 25 步平均')
    axes[0].set(ylabel='训练 loss');axes[0].legend(frameon=False,ncol=2)
    axes[1].plot(steps,[r['learning_rate'] for r in logs],color=orange)
    axes[1].set(xlabel='优化器步数',ylabel='学习率')
    for ax in axes:
        ax.grid(axis='y',alpha=.15)
        for s in [134,268]:ax.axvline(s,color='#999999',ls=':',lw=.8)
    save(fig,'training','训练过程','细线为每步 loss，粗线为最近 25 步算术平均；竖线标记轮次边界。训练 loss 与验证集准确率分开判断。')
    group_keys=sorted(summary['paired_comparisons'],key=lambda k:(k.endswith('/test'),k))
    x=np.arange(len(group_keys));width=.36
    fig,axes=plt.subplots(1,3,figsize=(13,4.7))
    for ax,metric in zip(axes,['CER','cpCER','DeltaCP']):
        for j,label in enumerate(['base','sft']):
            values=[summary['paired_comparisons'][k][label][metric]*100 for k in group_keys]
            ax.bar(x+(j-.5)*width,values,width,label='原始模型' if label=='base' else '训练后',
                   color=blue if label=='base' else orange)
        ax.set_xticks(x,group_keys,rotation=40,ha='right')
        ax.set_title(metric);ax.set_ylabel('百分点' if metric=='DeltaCP' else '错误率 (%)');ax.grid(axis='y',alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.07),ncol=2,frameon=False)
    save(fig,'accuracy','训练前后文字与说话人归属错误','按每个数据集的参考字符总数汇总。dev 与 test 独立显示；CER、cpCER 越低越好。DeltaCP = cpCER − CER，可为负值，不能单独当作说话人错误率或优劣判据。')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for ax,variant in zip(axes,['DER_collar_0','DER_collar_0.25']):
        for j,label in enumerate(['base','sft']):
            values=[summary['paired_comparisons'][k][label][variant]['DER']*100 for k in group_keys]
            ax.bar(x+(j-.5)*width,values,width,label='原始模型' if label=='base' else '训练后',
                   color=blue if label=='base' else orange)
        ax.set_xticks(x,group_keys,rotation=40,ha='right')
        ax.set_title('DER / collar = '+('0 秒' if variant=='DER_collar_0' else '0.25 秒'))
        ax.set_ylabel('DER (%)');ax.grid(axis='y',alpha=.15)
    axes[0].legend(frameon=False)
    save(fig,'diarization','说话人时间区间错误','两种 collar 均保留重叠语音；使用既有 RTTM/UEM，以累计 reference speaker-time 汇总。')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for label,color,title in [('base',blue,'原始模型'),('sft',orange,'训练后')]:
        rows=[r for r in per if r['model']==label]
        axes[0].scatter([r['duration']/60 for r in rows],[r['rtf'] for r in rows],s=22,alpha=.7,color=color,label=title)
        axes[1].scatter([r['duration']/60 for r in rows],[r['peak_allocated_gib'] for r in rows],s=22,alpha=.7,color=color,label=title)
    axes[0].set(xlabel='会议时长 (分钟)',ylabel='RTF（处理时间 / 音频时长）')
    axes[1].set(xlabel='会议时长 (分钟)',ylabel='进程峰值显存 (GiB)')
    axes[0].legend(frameon=False)
    for ax in axes:ax.grid(alpha=.15)
    save(fig,'performance','完整会议推理资源与时延','RTF 包括预处理与生成，不含模型加载和独立预热；显存为当前进程统计。多卡并发的总墙钟时间不等于逐条处理时间之和。')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    bins=[(0,20,'<20 分钟'),(20,40,'20–40 分钟'),(40,float('inf'),'≥40 分钟')]
    for ax,metric in zip(axes,['rtf','peak_allocated_gib']):
        for j,(label,color) in enumerate([('base',blue),('sft',orange)]):
            values=[]
            for lo,hi,_ in bins:
                rows=[r for r in per if r['model']==label and lo<=r['duration']/60<hi]
                value=(sum(r['e2e_seconds'] for r in rows)/sum(r['duration'] for r in rows)
                       if metric=='rtf' and rows else np.mean([r[metric] for r in rows]) if rows else np.nan)
                values.append(value)
            ax.bar(np.arange(3)+(j-.5)*width,values,width,color=color,label='原始模型' if label=='base' else '训练后')
        ax.set_xticks(np.arange(3),[b[2] for b in bins])
        ax.set_ylabel('累计 RTF' if metric=='rtf' else '平均进程峰值显存 (GiB)')
        ax.grid(axis='y',alpha=.15)
    axes[0].legend(frameon=False)
    save(fig,'duration_buckets','按会议时长分桶','RTF 按音频秒数加权；显存图为逐会议峰值的平均。桶内同时包含验证集和测试集，仅用于资源分析。')
    table=[]
    for key in group_keys:
        for label in ['base','sft']:
            m=summary['paired_comparisons'][key][label]
            table.append([key,'原始模型' if label=='base' else '训练后',f"{m['completed']}/{m['expected']}",
                *[f"{m[k]*100:.2f}" for k in ['CER','cpCER','DeltaCP']],
                f"{m['DER_collar_0']['DER']*100:.2f}",f"{m['DER_collar_0.25']['DER']*100:.2f}",
                f"{m['RTF']:.3f}",str(m['truncated'])])
    headers=['数据 / 划分','模型','覆盖','CER %','cpCER %','Δcp 百分点','DER 0 %','DER .25 %','RTF','截断数']
    with (out/'aggregate_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(headers);w.writerows(table)
    def table_html(headers,rows):
        return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
    comparisons=[]
    by={(r['model'],r['key']):r for r in per}
    for r in per:
        if r['model']!='sft':continue
        b=by[('base',r['key'])]
        comparisons.append((100*(r['text']['cpCER']-b['text']['cpCER']),r['key'],
            f"{b['text']['cpCER']*100:.2f}",f"{r['text']['cpCER']*100:.2f}"))
    comparisons.sort(reverse=True)
    worst=[[key,b,s,f'{delta:+.2f}'] for delta,key,b,s in comparisons[:10]]
    test_changes=[]
    for key in group_keys:
        if not key.endswith('/test'):continue
        m=summary['paired_comparisons'][key]
        test_changes.append(f"{html.escape(key.split('/')[0])} 的 cpCER 变化为 <b>{100*(m['sft']['cpCER']-m['base']['cpCER']):+.2f}</b> 个百分点")
    total_trunc=sum(r['truncated'] for r in per)
    total_empty=sum(r['parse_empty'] for r in per)
    test_table=[]
    der_components=[]
    for key in group_keys:
        if not key.endswith('/test'):continue
        pair=summary['paired_comparisons'][key]
        b,s=pair['base'],pair['sft']
        test_table.append([key.split('/')[0], *[f'{b[m]*100:.2f} → {s[m]*100:.2f}' for m in ['CER','cpCER']],
            f"{b['DER_collar_0.25']['DER']*100:.2f} → {s['DER_collar_0.25']['DER']*100:.2f}"])
        bd,sd=b['DER_collar_0.25'],s['DER_collar_0.25']
        der_components.append([key.split('/')[0],*[f'{100*(sd[m]/sd["total"]-bd[m]/bd["total"]):+.2f}'
            for m in ['missed detection','false alarm','confusion']],f'{100*(sd["DER"]-bd["DER"]):+.2f}'])
    anomaly_rows=[[r['model'],r['key'],r['generated_tokens'],'是' if r['truncated'] else '否',
        '是' if r['parse_empty'] else '否'] for r in per if r['truncated'] or r['parse_empty']]
    anomaly_meetings=len(anomaly_rows)
    anomaly_overlap=sum(r['truncated'] and r['parse_empty'] for r in per)
    verification=load(root/'download-verification.json')
    finished=load(run/'job_exit.json')['finished_utc']
    epoch_means=[float(np.mean([r['loss'] for r in logs if (i-1)*134<r['step']<=i*134])) for i in [1,2,3]]
    figures=''.join('<section><h2>'+title+'</h2><div class="figure-wrap"><img alt="'+html.escape(title)+'" src="data:image/png;base64,'+
        base64.b64encode((out/(name+'.png')).read_bytes()).decode()+'"></div><p class="caption">'+note+'</p></section>'
        for name,title,note in charts)
    stamp=datetime.now().astimezone().isoformat(timespec='seconds')
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOSS 训练与评估报告 · 62994</title><link rel="icon" href="data:,"><style>
body{{font-family:"Microsoft YaHei",system-ui,sans-serif;background:#eff3f5;color:#20333e;margin:0;line-height:1.7}}
main{{max-width:1200px;margin:40px auto;padding:0 24px}}header,section{{background:white;border-radius:14px;padding:28px 32px;margin-bottom:22px}}
header{{border-top:6px solid {blue}}}.eyebrow{{letter-spacing:.12em;color:{blue};font-size:13px}}h1{{font-size:32px;margin:8px 0}}h2{{font-size:21px;margin-top:0}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:24px}}.card{{background:#f0f6f8;border-radius:10px;padding:16px}}.card b{{font-size:24px;display:block}}
.muted,.caption{{color:#536975;font-size:14px}}img{{width:100%;height:auto}}.table-wrap,.figure-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}}th,td{{padding:10px;border-bottom:1px solid #dfe7eb;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf4f7}}.note{{border-left:4px solid {orange};padding:12px 18px;background:#fff7f0}}
a{{color:{blue}}}@media(max-width:700px){{main{{padding:0 10px;margin-top:12px}}header,section{{padding:20px}}.cards{{grid-template-columns:repeat(2,minmax(0,1fr))}}.card b{{font-size:20px}}.figure-wrap img{{min-width:760px}}.figure-wrap::before{{content:'图表可横向滚动查看';display:block;font-size:13px;color:#536975}}h1{{font-size:25px}}}}@media print{{body{{background:white}}main{{margin:0;padding:0}}section,header{{break-inside:avoid;border:1px solid #ddd}}}}
</style><main><header><div class="eyebrow">MOSS · FULL ATTENTION · 4 × A40</div><h1>训练与评估报告</h1>
<p>原始模型与全注意力微调模型的完整会议比较 · 训练作业 62994</p>
<div class="cards"><div class="card">训练完成<b>{training['global_step']} 步 / {training['epoch']:g} 轮</b></div>
<div class="card">正式训练耗时<b>{train_metrics['train_runtime']/3600:.2f} 小时</b></div>
<div class="card">独立会议音频<b>{audit['records']} 场 / {audit['hours']:.2f} h</b></div>
<div class="card">完成预测<b>{summary['scored_predictions']} / {summary['expected_predictions']}</b></div></div></header>
<section><h2>AMI 测试集改善，两个中文测试集退步</h2>
<p>本次微调的收益不一致：AMI 测试集的文字、说话人归属与时间区间错误均下降，AliMeeting 和 AISHELL-4 测试集的这三项错误均上升。两个验证集也均退步，因此目前不支持把该最终权重作为原始模型的全面替代。</p>
{table_html(['测试集','CER %：原始 → 训练后','cpCER %：原始 → 训练后','DER %：原始 → 训练后'],test_table)}
<p class="caption">此处 DER 的边界容差 collar 为 0.25 秒，保留重叠语音。三项指标均越低越好；完整表和图同时提供 collar=0 的 DER。</p>
<p>{'；'.join(test_changes)}。变化值为训练后减原始模型，负值表示改善。</p>
<p>每轮平均训练 loss：{' → '.join(f'{v:.4f}' for v in epoch_means)}。末期学习率已接近零；loss 变平本身不足以证明收敛，也不替代独立数据上的识别与说话人评估。</p>
<p class="muted">{html.escape(occupancy)}。峰值显存图仅统计评估进程；启动前显存占用不能当作本模型的开销。</p>
<div class="note">训练后模型出现预测截断 {total_trunc} 条、空解析 {total_empty} 条；原始模型均为 0。两类情况均保留并评分，不从结果中排除。空解析表示官方解析器未提取到完整片段，不一定表示模型没有输出文字；格式错误可能导致全文无法解析。达到生成上限的重复输出也保留在原始预测中，CER/cpCER 按既定协议仅对官方解析出的片段计算。因此，错误率必须结合原始输出、截断和解析失败情况共同解读。指标采用本项目明确的归一化和 RTTM/UEM 协议，不能直接等同论文分数。</div></section>
<section><h2>准确率与资源总表</h2>{table_html(headers,table)}<p class="caption">误差以百分数显示；Δcp 为百分点。各数据集按参考字符数或说话人秒数累计。dev 为验证集，test 为测试集。</p></section>
{figures}<section><h2>cpCER 变化较差的会议</h2>{table_html(['会议','原始 cpCER %','训练后 cpCER %','变化百分点'],worst)}
<p class="caption">按训练后减原始模型排序。表中可能仍为改善，只表示相对改善较少；用于查阅原始预测，不能据此再调整测试集参数。</p></section>
<section><h2>中文测试集的 DER 增量主要来自漏检</h2>
{table_html(['测试集','漏检变化 / 百分点','误报变化 / 百分点','说话人混淆变化 / 百分点','总 DER 变化 / 百分点'],der_components)}
<p>以上是 collar=0.25 秒下三个错误分量的累计变化。中文测试集漏检增加，是 DER 退步的主要数值来源；AMI 测试集漏检减少，抵消了误报与说话人混淆的增加。这是评分分量的拆解，尚不能据此确定训练退步的根因。</p>
<p>两类异常合计涉及 {anomaly_meetings} 场不同会议，其中 {anomaly_overlap} 场同时存在截断和空解析。</p>
<details><summary>查看全部截断或空解析记录</summary>{table_html(['模型','会议','生成 token 数','截断','空解析'],anomaly_rows)}</details>
<p>后续应先在验证集检查重复生成、时间戳格式和漏识别，并对照已有训练配置与数据目标；在基线输出稳定后再推进既定的 RSWA 实验。本次未执行新训练、改写预测或更换解析器。</p></section>
<section><h2>如何理解这些指标</h2><p>CER 按时间顺序合并文字，计算字符编辑错误；cpCER 按说话人连接文字，经过最优说话人匹配后计算字符错误。两种排序方式在重叠语音中可能不同，所以 cpCER 可以小于 CER，差值 Δcp 也可以为负。该差值不等于纯粹的说话人错误率。</p>
<p>文字先做 Unicode NFKC 与大小写归一化，再去空白、标点、符号和控制字符；AMI 英文同样按字符计算，不能当作 WER。DER 依据现有 RTTM/UEM 标注，以参考说话人总时长累计，保留重叠语音。AMI 使用 Mix-Headset，其他数据使用既有 16 kHz 单声道输入。</p>
<p>两模型使用相同完整会议、原始处理器与官方解析器，采用贪心解码和全注意力，最大生成 65,536 token，总上下文上限 131,072。训练集与本次验证、测试集的音频路径及会议 ID 已核对无重叠。不同数据集独立汇总，不用跨语料总平均掩盖分化。</p></section>
<section><h2>证据与复现</h2><p>完整日志包已校验归档 SHA-256 和每个文件的大小及 SHA-256。raw 目录保留项目全部训练日志、本次逐会议预测、token IDs、评分明细、配置、数据与权重指纹；不包含模型权重和原始音频。</p>
<p>已核验 {verification['verified_files']} 个文件，压缩包 {verification['bytes']/1e6:.2f} MB。评估成功结束于 {finished}（UTC），主作业耗时 9 小时 00 分 38 秒；报告生成于 {stamp}。四个推理进程均正常结束，164/164 条完成评分，无推理错误或缺失。</p>
<p>图表另存 PNG 与 SVG，汇总表另存 UTF-8 CSV。最终指标范围只覆盖这次固定模型的比较，未运行基于验证集的早停或多检查点选择。</p>
<p><a href="aggregate_metrics.csv">下载指标 CSV</a> · <a href="../{run.name}-all-logs.tar.gz">完整日志压缩包</a> · <a href="../download-verification.json">下载校验记录</a> · <a href="../raw/results/{run.name}/metrics_summary.json">完整指标 JSON</a></p>
<p class="muted">评分定义参考 <a href="https://github.com/fgnt/meeteval">MeetEval</a> 与 <a href="https://pyannote.github.io/pyannote-metrics/_modules/pyannote/metrics/diarization.html">pyannote.metrics</a>。具体运行协议见原始归档内 PROTOCOL.md。</p></section></main></html>'''
    (out/'MOSS_训练与评估报告.html').write_text(body,encoding='utf-8')
    (out/'report_manifest.json').write_text(json.dumps(dict(generated=stamp,
        training_job=62994,evaluation=run.name,complete=True,plots=[c[0] for c in charts]),indent=2))
    print(out/'MOSS_训练与评估报告.html')


if __name__=='__main__':
    main()
