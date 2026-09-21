"""Create static research figures and a compact evidence ledger from real probes."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent
OUT=R/'figures'
OUT.mkdir(exist_ok=True)
def read(p): return json.loads(p.read_text(encoding='utf-8'))
replay=read(R/'evidence-63399/replay.json')
contracts=read(R/'evidence-63400/contracts.json')
pilots={k:read(R/f'evidence-63401/pilot-{k}.json') for k in ['bf16','fp32']}
post=read(R/'evidence-63402/post-pilot.json')
assert all(x['status']=='completed' for x in [replay,contracts,post,*pilots.values()])
assert pilots['bf16']['data_indices']==pilots['fp32']['data_indices']
assert all(len(x['steps'])==4 for x in pilots.values())

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
    'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
colors={'base':'#34699A','sft402':'#C56C28','bf16':'#C56C28','fp32':'#34699A'}
def save(fig,name):
    for ext in ['png','svg']: fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

fig,axes=plt.subplots(2,3,figsize=(14.2,8.4),sharex=True)
keys=sorted({x['key'] for x in replay['records']})
for row,key in enumerate(keys):
    for record in [x for x in replay['records'] if x['key']==key]:
        ps=record['points'];label='Base' if record['model']=='base' else 'SFT step 402'
        for col,field in enumerate(['probability','margin','entropy_nats']):
            axes[row,col].plot([p['k'] for p in ps],[p[field] for p in ps],
                color=colors[record['model']],marker='o' if label=='Base' else 's',
                linestyle='-' if label=='Base' else '--',linewidth=1.8,label=label,markersize=5)
    for col in range(3):
        ax=axes[row,col]
        ax.set_xscale('symlog',base=2,linthresh=1)
        ax.set_xticks([0,1,2,4,8,16,32],labels=['0','1','2','4','8','16','32'])
        ax.set_xlabel('Forced repeat count k (symlog scale)')
        ax.grid(axis='y',color='#E5E7EB',linewidth=.7)
        ax.set_title(key.split('/')[-1]+' · '+['Repeat probability','Repeat logit margin','Distribution entropy'][col])
    axes[row,0].set_ylim(0,1.05);axes[row,0].set_ylabel('Probability')
    axes[row,1].set_ylim(-2.1,10.6);axes[row,1].set_ylabel('Logit difference')
    axes[row,1].axhline(0,color='#444444',linewidth=.9)
    axes[row,2].set_ylim(0,3.5);axes[row,2].set_ylabel('Nats')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=2,bbox_to_anchor=(.5,.94),frameon=False)
fig.suptitle('Controlled repetition prefixes: identical history and full audio',fontsize=17,y=.99)
fig.text(.5,.02,'Two known AliMeeting dev failures. Forced prefixes do not measure natural recovery or failure prevalence.',ha='center',fontsize=10,color='#555555')
fig.tight_layout(rect=(0,.05,1,.9))
save(fig,'prefix_replay')

names={
 'model.language_model.embed_tokens.weight':'LM tied embedding / output',
 'model.language_model.layers.0.self_attn.q_proj.weight':'LM layer 0 · Q projection',
 'model.language_model.layers.14.mlp.down_proj.weight':'LM layer 14 · FFN down',
 'model.language_model.layers.27.mlp.down_proj.weight':'LM layer 27 · FFN down',
 'model.whisper_encoder.layers.0.fc1.weight':'Audio layer 0 · FFN up',
 'model.whisper_encoder.layers.23.fc2.weight':'Audio layer 23 · FFN down',
 'model.vq_adaptor.layers.0.weight':'Audio adapter · input projection'}
params=list(names)
fig,ax=plt.subplots(figsize=(11.8,6.2))
y=np.arange(len(params))
update_rows=[]
for precision,offset in [('bf16',-.16),('fp32',.16)]:
    lookup={x['parameter']:x for x in pilots[precision]['first_step_update_audit']}
    values=[lookup[k]['lost_fraction']*100 for k in params]
    ax.barh(y+offset,values,height=.28,color=colors[precision],label=precision.upper()+' updates')
    ax.scatter(values,y+offset,color=colors[precision],marker='o' if precision=='fp32' else 's',s=19,zorder=3)
    for yy,value,param in zip(y+offset,values,params):
        ax.text(value+1.1,yy,f'{value:.1f}%',va='center',fontsize=10)
        update_rows.append(dict(precision=precision,component=names[param],**lookup[param]))
ax.set_yticks(y,labels=[names[k] for k in params]);ax.invert_yaxis();ax.set_xlim(0,108)
ax.set_xticks([0,20,40,60,80,100],labels=['0%','20%','40%','60%','80%','100%'])
ax.set_xlabel('Zero stored update / nonzero independent FP32 reference update')
ax.grid(axis='x',color='#E5E7EB',linewidth=.7);ax.set_axisbelow(True)
ax.set_title('First optimizer step: updates lost at writeback',loc='left',fontsize=17,pad=40)
ax.legend(loc='lower left',bbox_to_anchor=(0,1.015),ncol=2,frameon=False)
fig.text(.03,.015,'65,536 positions sampled per component; each recipe uses its own actual clipped gradient for the FP32 reference.\nThis demonstrates a numerical update problem, not the cause of long-loop generation.',fontsize=10,color='#555555')
fig.tight_layout(rect=(0,.075,1,1))
save(fig,'update_writeback')

base={x['key']:x for x in pilots['bf16']['initial_validation']}
common={x['training_precision']:{v['key']:v for v in x['validation']} for x in post['models']}
loss_rows=[]
fig,axes=plt.subplots(1,2,figsize=(10.2,4.8))
for ax,key in zip(axes,base):
    vals=[base[key]['loss'],common['bf16'][key]['loss'],common['fp32'][key]['loss']]
    ax.bar(['Base','4-step BF16','4-step FP32'],vals,color=['#7B8490',colors['bf16'],colors['fp32']],width=.55)
    ax.set_ylim(0,.5);ax.set_ylabel('Mean next-token cross-entropy');ax.set_title(key.split('/')[-1])
    ax.grid(axis='y',color='#E5E7EB',linewidth=.7);ax.set_axisbelow(True)
    for i,v in enumerate(vals): ax.text(i,v+.009,f'{v:.4f}',ha='center')
    loss_rows.append(dict(key=key,supervised_tokens=base[key]['supervised_tokens'],base=vals[0],bf16_4step=vals[1],fp32_4step=vals[2]))
fig.suptitle('Full-meeting validation loss with the same BF16 inference path',fontsize=14,y=.99)
fig.text(.5,.02,'Two dev meetings; correct-history loss only. These are not CER/DER or full-generation results.',ha='center',fontsize=10,color='#555555')
fig.tight_layout(rect=(0,.06,1,.94));save(fig,'pilot_validation')

inputs=read(R/'inputs/dev_inputs.json')
sentinels=[]
for corpus in ['alimeeting','ami']:
    rows=sorted([x for x in inputs if x['split']=='dev' and x['dataset']==corpus],key=lambda x:(x['duration'],x['key']))
    for q in [0,.5,1]:
        x=rows[int(q*(len(rows)-1))]
        sentinels.append({k:x[k] for k in ['key','audio','duration','prompt_len','prompt']})
(R/'sentinel_manifest.json').write_text(json.dumps(dict(selection='Each dev corpus: floor(q*(n-1)) duration ranks, q=0,0.5,1; tie by key. No outcome filtering.',records=sentinels),ensure_ascii=False,indent=2),encoding='utf-8')
preflight=read(R/'inputs/preflight.json')
samples={s['index']:s for s in preflight['samples']}
selected=[samples[i] for i in pilots['bf16']['data_indices']]
training_seconds={k:sum(x['seconds'] for x in v['steps']) for k,v in pilots.items()}
summary=dict(jobs={'replay':'63399','contracts':'63400','paired_training':'63401','common_inference':'63402'},
    actual_train_records=16,optimizer_steps=4,supervised_tokens=sum(x['supervised_tokens'] for x in pilots['bf16']['steps']),
    pilot_max_context=max(x['tokens'] for x in selected),formal_max_context=preflight['max_tokens'],
    update_audit=update_rows,common_inference_validation=loss_rows,training_seconds=training_seconds,
    fp32_time_increase_percent=(training_seconds['fp32']/training_seconds['bf16']-1)*100,
    peak_allocated_gib={k:max(x['peak_allocated_gib'] for x in v['steps']) for k,v in pilots.items()},
    max_long_prefix_cache_logit_rms=max(x['rms'] for r in replay['records'] for x in r['cache_vs_recompute']),
    all_8_long_prefix_cache_top1_same=all(x['top1_same'] for r in replay['records'] for x in r['cache_vs_recompute']),
    full_generation_validated=False,formal_ddp4_fp32_qualified=False)
(R/'evidence_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
for name,rows in [('update_audit',update_rows),('validation_loss',loss_rows)]:
    with (R/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

items=[]
items.append(dict(id='update-precision',title='首步参数更新精度',queries=[dict(id='first-update',source=dict(
    label='A40 成对小实验 63401',files=['pilot-bf16.json','pilot-fp32.json'],
    metricDefinitions=['写回为零比例：独立 FP32 Adam 首步参照更新非零，但实际参数更新为零的抽样位置比例。'],
    caveats=['每个组件抽样 65,536 个位置；不是全部参数的总体比例。','两条轨迹的数值配置不同，各自以实际裁剪后梯度计算 FP32 参照。','4 步实验不证明 BF16 是长循环的根因。']),rows=update_rows)]))
items.append(dict(id='prefix-replay',title='相同前缀上的重复偏好',queries=[dict(id='repeat-prefixes',source=dict(
    label='完整音频前缀回放 63399',files=['replay.json'],links=['https://arxiv.org/html/2206.02369'],
    metricDefinitions=['Margin：重复 token 的 logit 减去其他候选中的最大 logit。','熵：完整词表概率分布的 Shannon 熵，单位 nat。'],
    caveats=['只包含两场已知验证集失败；人工重复前缀不能用于估计自然失败率。','整段前缀预填与逐 token 生成可能存在 BF16 数值差异；本实验不定位原始轨迹的精确首次分叉。']),
    rows=[dict(meeting=r['key'],model=r['model'],k=x['k'],probability=x['probability'],margin=x['margin'],entropy_nats=x['entropy_nats']) for r in replay['records'] for x in r['points']])]))
items.append(dict(id='pilot-validation',title='统一推理路径的验证 loss',queries=[dict(id='common-bf16',source=dict(
    label='保存重载复核 63402',files=['post-pilot.json','pilot-bf16.json'],
    metricDefinitions=['验证 loss：完整音频与正确历史文本上的平均 next-token 交叉熵。'],
    caveats=['仅两场验证会议，未完成这两个小实验权重的整场自由生成 CER/DER 评估。','16 场训练会议、4 次更新，两条轨迹均从同一 base 启动。']),rows=loss_rows)]))
approved=[
    [('precision','更新精度'),('component','组件'),('sampled','抽样位置数'),('reference_nonzero','参照非零位置数'),('reference_nonzero_but_actual_zero','实际更新为零的位置数'),('lost_fraction','写回为零比例 0–1')],
    [('meeting','验证会议'),('model','模型'),('k','强制重复次数'),('probability','重复概率 0–1'),('margin','重复 logit margin'),('entropy_nats','熵 nats')],
    [('key','验证会议'),('supervised_tokens','监督 token 数'),('base','Base loss'),('bf16_4step','BF16 更新 4 步 loss'),('fp32_4step','FP32 更新 4 步 loss')]
]
for item,fields in zip(items,approved):
    query=item['queries'][0]
    query['columns']=[dict(field=field,label=label) for field,label in fields]
    query['rows']=[{field:row[field] for field,label in fields} for row in query['rows']]
(R/'sources-receipt.json').write_text(json.dumps(dict(schemaVersion=1,items=items),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['update_audit']},ensure_ascii=True,indent=2))
