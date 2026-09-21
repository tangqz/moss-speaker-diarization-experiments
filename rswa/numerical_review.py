"""Retain raw errors and measure distributional differences under the real penalty."""
import argparse,json
from pathlib import Path
import torch
from common import read,write

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(4);reports={}
    for path in sorted(a.run.glob('*/hf*alignment.json')):
        data=read(path);trace=read(path.parent/'vllm_trace.json')
        directory=path.parent/('hf_native_logits' if data['native_full'] else 'hf_logits')
        records=[]
        for record in data['records']:
            step=record['step'];x=torch.load(directory/f'logits-{step}.pt',weights_only=True).double()
            y=torch.load(path.parent/'raw_logits'/f'logits-{step}.pt',weights_only=True)['logits'].double()
            delta=x-y;bias=delta.mean(-1,keepdim=True);centered=delta-bias
            history=sorted(set(trace['generated_ids'][:step]))
            if history:
                for value in [x,y]:
                    selected=value[:,history];value[:,history]=torch.where(selected<0,selected*1.02,selected/1.02)
            lx,ly=x.log_softmax(-1),y.log_softmax(-1);px,py=lx.exp(),ly.exp()
            records.append(dict(step=step,raw_signed_bias=float(bias.squeeze()),
                centered_mean_abs=float(centered.abs().mean()),centered_p99=float(torch.quantile(centered.abs(),.99)),
                centered_max_abs=float(centered.abs().max()),
                penalized_kl_hf_to_vllm=float((px*(lx-ly)).sum()),
                penalized_total_variation=float((px-py).abs().sum()/2)))
        reports[str(path.relative_to(a.run))]=dict(records=records,
            maxima={key:max(r[key] for r in records) for key in ['centered_mean_abs','centered_p99','centered_max_abs',
                'penalized_kl_hf_to_vllm','penalized_total_variation']})
    write(a.run/'distribution_review.json',dict(reports=reports,
        interpretation='Raw thresholds remain recorded. Centered errors diagnose common offsets; KL and TV use actual 1.02 output-only penalty.',
        qualification='diagnostics only; compare to matched Full and review before qualification'))
    print(json.dumps({key:value['maxima'] for key,value in reports.items()}),flush=True)

if __name__=='__main__':main()
