"""Standalone scientific figures from saved measurements only."""
import argparse,collections,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import read

COLORS={'full':'#52616b','128':'#138a80','256':'#d5653c'}

def save(fig,destination):
    fig.tight_layout();fig.savefig(str(destination)+'.png',dpi=240)
    fig.savefig(str(destination)+'.svg');plt.close(fig)

def render(root,report):
    root,report=Path(root),Path(report)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,sample in zip(axes,['smoke','typical','longest']):
        for window in ['full','128','256']:
            result=read(root/'performance'/f'{sample}-{window}'/'complete.json')
            rows=sorted((int(n),s['decode_seconds']) for n,s in result['summaries'].items())
            ax.plot([n for n,s in rows],[s['median'] for n,s in rows],marker='o',color=COLORS[window],label=window)
            ax.fill_between([n for n,s in rows],[s['minimum'] for n,s in rows],[s['maximum'] for n,s in rows],
                            color=COLORS[window],alpha=.12)
        ax.set(title=sample,xlabel='Generated tokens',ylabel='Decode seconds');ax.set_xscale('log',base=2)
        ax.grid(alpha=.2);ax.legend(title='Window')
    save(fig,report/'fixed_trajectory_latency')
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,sample in zip(axes,['smoke','typical','longest']):
        for window in ['full','128','256']:
            path=root/'performance'/f'{sample}-{window}'/'kv.jsonl'
            groups=collections.defaultdict(list)
            for line in path.read_text().splitlines():
                row=json.loads(line)
                if row['prefix_tokens'] is not None:groups[row['request_id']].append(row)
            # Requests run in increasing length; the last request is the last
            # measured repeat at the largest available original trajectory.
            rows=list(groups.values())[-1]
            x=[r['logical_tokens']-r['prefix_tokens'] for r in rows]
            y=[r['physical_blocks']*r['bytes_per_layer_block']*28/2**30 for r in rows]
            ax.plot(x,y,color=COLORS[window],label=window)
        ax.set(title=sample,xlabel='Output positions written',ylabel='Live KV GiB, all 28 layers')
        ax.grid(alpha=.2);ax.legend(title='Window')
    save(fig,report/'live_kv_pages')
    comparisons=read(report/'paired_intervals.json')
    for metric in ['cpCER','DER025']:
        fig,ax=plt.subplots(figsize=(9,5));labels=[];i=0
        for name,corpora in comparisons.items():
            if 'versus full-seed' not in name:continue
            window=name.split('-')[0]
            for corpus,value in corpora.items():
                record=value['differences'][metric];lo,hi=[100*x for x in record['ci95']]
                ax.hlines(i,lo,hi,color=COLORS.get(window,'black'))
                ax.scatter(100*record['point'],i,color=COLORS.get(window,'black'),s=22)
                labels.append(f'{name}: {corpus}');i+=1
        if i:
            ax.axvline(0,color='#888888',linewidth=.8);ax.set_yticks(range(i),labels)
            ax.set_xlabel(f'{metric} difference vs matched Full, percentage points (95% paired CI)')
            ax.grid(axis='x',alpha=.2);save(fig,report/f'{metric}_paired_intervals')
        else:plt.close(fig)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True);args=ap.parse_args();render(args.root,args.report)
