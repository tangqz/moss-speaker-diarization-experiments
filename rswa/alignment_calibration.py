"""Report native Full cross-backend noise before qualifying BF16 comparisons."""
import argparse,json
from pathlib import Path
import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    load=lambda p:json.loads(p.read_text())
    native=load(a.run/'full/hf_native_alignment.json')
    records={w:load(a.run/w/'hf_alignment.json') for w in ['full','128','256']}
    # Native Full establishes a baseline independent of the custom mask path.
    baseline={key:max(r[key] for r in native['records']) for key in ['max_abs','mean_abs','p99']}
    internal=[]
    for row in native['records']:
        step=row['step']
        x=torch.load(a.run/'full/hf_native_logits'/f'logits-{step}.pt',weights_only=True)
        y=torch.load(a.run/'full/hf_logits'/f'logits-{step}.pt',weights_only=True)
        error=(x-y).abs()
        internal.append(dict(step=step,max_abs=float(error.max()),mean_abs=float(error.mean()),p99=float(torch.quantile(error,.99))))
    summary=dict(native_full_baseline=baseline,custom_full_vs_native=internal,
                 reports={w:dict(original_tolerance_passed=r['passed'],
                   high_margin_disagreements=[x for x in r['token_checks'] if x['high_margin_disagreement']],
                   top1_agreement=sum(x['hf_top1']==x['vllm_top1'] for x in r['token_checks'])/len(r['token_checks']),
                   maxima={key:max(x[key] for x in r['records']) for key in baseline}) for w,r in records.items()},
                 qualification='diagnostic calibration; review before changing acceptance criteria')
    (a.run/'alignment_calibration.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
