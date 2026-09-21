"""Render full-Dev measurements only; sentinel results are excluded."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE=Path(__file__).resolve().parent
data=json.loads((HERE/'state.json').read_text(encoding='utf-8'))
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10,
    'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
fig,axes=plt.subplots(1,3,figsize=(14.4,5.3))
series=[('full-seed0','Full','#426C95'),('128-seed0','R128','#D36149'),('256-seed0','R256','#278879')]
for group,label,color in series:
    rows=sorted(data['groups'][group]['history'],key=lambda r:r['step'])
    x=[r['step'] for r in rows]
    for ax,values in zip(axes,[[r['ce'] for r in rows],[r['S'] for r in rows],[len(r['failures']) for r in rows]]):
        ax.plot(x,values,marker='o',lw=2.2,ms=6,label=label,color=color)
    end=rows[-1]
    for ax,value,fmt in zip(axes,[end['ce'],end['S'],len(end['failures'])],['.3f','.2f','d']):
        ax.annotate(format(value,fmt),(end['step'],value),xytext=(6,6),textcoords='offset points',color=color,fontsize=10)
for ax,title,ylabel in zip(axes,['Teacher-forced CE','自由生成综合评分 S','功能失败会议数'],
                          ['按目标 token 加权的 CE ↓','S（越低越好，可超过 100） ↓','失败会议数 / 26 ↓']):
    ax.set_title(title,loc='left',fontsize=12,pad=12)
    ax.set_xlabel('优化器更新步数')
    ax.set_ylabel(ylabel)
    ax.set_xlim(-6,172);ax.set_xticks([0,30,100,150]);ax.grid(axis='y',alpha=.18)
axes[0].set_ylim(.48,.85)
axes[1].set_ylim(0,112)
axes[2].set_ylim(-1,27);axes[2].set_yticks([0,6,12,18,26])
fig.suptitle('完整 Dev：CE 下降未带来稳定的自由生成',x=.045,y=.97,ha='left',fontsize=17,fontweight='bold')
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper right',bbox_to_anchor=(.97,.99),ncol=3,frameon=False)
fig.text(.045,.085,'每次完整 Dev 为 26 场；step 0 为 Base 在对应注意力模式下的零样本评估；不含六场哨兵。',fontsize=10,color='#4D5661')
fig.text(.045,.04,'仅绘制已完成的测量。Full 的完整 Dev 仅有 0 / 30 步；连线用于阅读，不表示中间步骤已测评。',fontsize=10,color='#4D5661')
fig.subplots_adjust(left=.065,right=.985,top=.79,bottom=.23,wspace=.35)
for suffix in ['png','svg']:
    fig.savefig(HERE/f'完整Dev_CE_S_功能失败.{suffix}',dpi=190)
rows=[dict(group=group,step=r['step'],CE=r['ce'],S=r['S'],failures=len(r['failures']),meetings=26)
      for group,_,_ in series for r in data['groups'][group]['history']]
(HERE/'完整Dev_作图数据.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print('Wrote full-Dev figure and source data.')
