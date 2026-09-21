"""SP attention forward/backward against an independent unsharded reference."""
import argparse,json,os
from pathlib import Path
import torch
import torch.distributed as dist
from attention import MaskSpec,hf_attention
from contract_checks import independently_masked

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    dist.init_process_group('nccl')
    rank=dist.get_rank();world=dist.get_world_size()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));torch.set_num_threads(2)
    from swift.sequence_parallel import sequence_parallel as sp
    sp.world_size=world;sp.num_heads=4;sp._init_device_mesh()
    torch.manual_seed(87)
    n,p=512,173
    global_inputs=[torch.randn(1,h,n,32,device='cuda',requires_grad=True) for h in (8,4,4)]
    probe=torch.randn(1,8,n,32,device='cuda')
    out=[]
    for window in [128,256,None]:
        lo,hi=rank*(n//world),(rank+1)*(n//world)
        local=[x[...,lo:hi,:].detach().clone().requires_grad_() for x in global_inputs]
        actual,_=hf_attention(torch.nn.Identity(),*local,MaskSpec(p,n,window),scaling=32**-.5)
        ref,_=independently_masked(*global_inputs,p,window)
        torch.testing.assert_close(actual,ref[...,lo:hi,:].transpose(1,2),atol=2e-5,rtol=2e-4)
        grads=torch.autograd.grad((actual.transpose(1,2)*probe[...,lo:hi,:]).sum(),local)
        refgrads=torch.autograd.grad((ref*probe).sum(),global_inputs)
        errors=[]
        for ga,gr in zip(grads,refgrads):
            torch.testing.assert_close(ga,gr[...,lo:hi,:],atol=3e-5,rtol=3e-4)
            errors.append(float((ga-gr[...,lo:hi,:]).abs().max()))
        out.append(dict(window=window,rank=rank,gradient_errors=errors))
    gathered=[None]*world;dist.all_gather_object(gathered,out)
    if rank==0:
        result=dict(passed=True,sp=world,records=gathered)
        args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
