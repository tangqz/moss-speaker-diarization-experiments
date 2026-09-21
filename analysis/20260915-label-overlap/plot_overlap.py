import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
audit = json.loads((HERE/'overlap_audit.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
                     'axes.unicode_minus':False, 'font.size':10})
choices = [
    ('ms-swift-63521/vllm_exports/checkpoint-20','alimeeting/test/R8005_M8009','MS-Swift Step 20 · R8005_M8009'),
    ('ms-swift-63521/vllm_exports/checkpoint-20','alimeeting/test/R8005_M8008','MS-Swift Step 20 · R8005_M8008'),
    ('ms-swift-63521/vllm_exports/checkpoint-40','alimeeting/dev/R8003_M8001','MS-Swift Step 40 · R8003_M8001'),
    ('full-attention-v2-63421/checkpoint-50','ami/dev/TS3004c','早期 SFT Step 50 · TS3004c'),
    ('full-attention-ddp4-62994','aishell4/test/L_R003S04C02','早期 SFT Step 402 · L_R003S04C02'),
    ('lr2e-6/checkpoint-20','alimeeting/dev/R8003_M8001','早期 LR 2e-6 Step 20 · R8003_M8001'),
]
fig, axes = plt.subplots(3,2,figsize=(15,10),layout='constrained')
for ax,(model,key,title) in zip(axes.flat,choices):
    row = next(r for r in audit['cases'] if model in r['model_path'] and r['key']==key)
    refs = json.loads(Path(row['reference_source']).read_text(encoding='utf-8'))
    ref = refs[key]
    t = row['anchor_seconds']
    start,end = max(0,t-10),t+15
    speakers = sorted({s['speaker'] for s in ref['segments']})
    for i,spk in enumerate(speakers):
        for s in ref['segments']:
            a,b=max(start,s['start']),min(end,s['end'])
            if s['speaker']==spk and b>a:
                ax.broken_barh([(a,b-a)],(i-.28,.56),facecolors=plt.cm.tab10(i),alpha=.85)
    ax.axvline(t,color='#ba3030',ls='--',lw=1.3)
    ax.set_xlim(start,end)
    ax.set_ylim(-.65,len(speakers)-.35)
    ax.set_yticks(range(len(speakers)),speakers)
    ax.set_xlabel('参考标注时间（秒）；红线为生成时间锚点，非词级对齐')
    ax.set_title(title,loc='left',fontweight='bold')
    ax.grid(axis='x',alpha=.2)
    if not any(s['end']>start and s['start']<end for s in ref['segments']):
        last=max(s['end'] for s in ref['segments'])
        ax.text(.5,.5,f'该区间无参考语音标注\n最后一条参考语音结束于 {last:.2f} 秒',
                ha='center',va='center',transform=ax.transAxes,color='#555')
fig.suptitle('复读案例与重叠语音：既有多人重叠，也有单人语音及标注结束后的持续输出',
             fontsize=16,fontweight='bold')
fig.savefig(HERE/'reference_overlap_examples.png',dpi=150)
fig.savefig(HERE/'reference_overlap_examples.svg')
print('Saved six reference timeline examples')
