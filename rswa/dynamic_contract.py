"""More than the default recompile limit of distinct meeting lengths/boundaries."""
import argparse,json
from pathlib import Path
import torch
from attention import attention,dense_reference,MaskSpec

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    torch.manual_seed(901);torch.set_num_threads(4)
    # A reached recompilation limit must fail this gate, not silently fall back.
    torch._dynamo.config.fail_on_recompile_limit_hit=True
    from torch._dynamo.utils import counters
    before=dict(counters['stats']);records=[]
    for window in [None,128,256]:
        for i in range(14):
            n=177+20*i;p=33+7*i;valid=n-(i%4);spec=MaskSpec(p,valid,window)
            q=torch.randn(1,4,n,32,device='cuda',requires_grad=True)
            k=torch.randn(1,2,n,32,device='cuda',requires_grad=True);v=torch.randn_like(k,requires_grad=True)
            out=attention(q,k,v,spec);ref=dense_reference(q,k,v,spec)
            direction=torch.randn_like(out)
            grads=torch.autograd.grad(out,(q,k,v),direction)
            refs=torch.autograd.grad(ref,(q,k,v),direction)
            torch.testing.assert_close(out,ref,atol=1e-5,rtol=1e-4)
            for x,y in zip(grads,refs):torch.testing.assert_close(x,y,atol=1e-5,rtol=1e-4)
            records.append(dict(window=window,length=n,prefix=p,valid=valid,max_abs=float((out-ref).abs().max())))
    report=dict(passed=True,cases=records,dynamo_stats_before=before,dynamo_stats_after=dict(counters['stats']))
    a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)

if __name__=='__main__':main()
