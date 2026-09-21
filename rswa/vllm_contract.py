"""Independent checks for the pinned inference kernel, eviction and penalty."""
import argparse,json,math
from pathlib import Path
import torch
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import RSWAManager
from vllm.v1.kv_cache_interface import RSWASpec
import moss_vllm_rswa
from vllm.v1.sample.ops.penalties import apply_all_penalties

def kernels():
    torch.manual_seed(712);records=[]
    for window in [None,128,256]:
        for prefix,n,qn in [(173,173,173),(173,174,1),(173,301,1),
                            (173,302,1),(173,700,1),(173,700,31)]:
            bs=16;blocks=math.ceil(n/bs);hq,hk,d=16,8,128
            q=torch.randn(qn,hq,d,device='cuda',dtype=torch.bfloat16)
            k=torch.randn(blocks,bs,hk,d,device='cuda',dtype=q.dtype)
            v=torch.randn_like(k);out=torch.empty_like(q)
            args=dict(q=q,k=k,v=v,out=out,
                cu_seqlens_q=torch.tensor([0,qn],device='cuda',dtype=torch.int32),
                max_seqlen_q=qn,seqused_k=torch.tensor([n],device='cuda',dtype=torch.int32),
                max_seqlen_k=n,softmax_scale=d**-.5,causal=True,window_size=(-1,-1),
                block_table=torch.arange(blocks,device='cuda',dtype=torch.int32)[None,:],
                softcap=0.,q_descale=None,k_descale=None,v_descale=None,
                rswa_prefix_lens=torch.tensor([prefix],device='cuda',dtype=torch.int32),rswa_window=window)
            unified_attention(**args)
            iq=torch.arange(n-qn,n,device='cuda')[:,None]
            ik=torch.arange(n,device='cuda')[None,:]
            mask=ik<=iq
            if window is not None:mask=mask&((ik<prefix)|(ik>=iq-window+1))
            kk=k.reshape(-1,hk,d)[:n].transpose(0,1).repeat_interleave(hq//hk,0).float()
            vv=v.reshape(-1,hk,d)[:n].transpose(0,1).repeat_interleave(hq//hk,0).float()
            logits=q.transpose(0,1).float()@kk.transpose(-1,-2)*d**-.5
            ref=(logits.masked_fill(~mask,-torch.inf).softmax(-1)@vv).transpose(0,1)
            torch.testing.assert_close(out.float(),ref,atol=.008,rtol=.03)
            records.append(dict(window=window,prefix=prefix,length=n,queries=qn,
                                max_abs_error=float((out.float()-ref).abs().max())))
    return records

def eviction():
    records=[]
    for window in [128,256]:
        bs,prefix=16,173
        pool=BlockPool(num_gpu_blocks=64,enable_caching=False,hash_block_size=bs)
        spec=RSWASpec(block_size=bs,num_kv_heads=8,head_size=128,dtype=torch.bfloat16,rswa_window=window)
        manager=RSWAManager(spec,block_pool=pool,enable_caching=False,kv_cache_group_id=0,scheduler_block_size=bs)
        manager.allocate_new_blocks('meeting',prefix,prefix)
        pinned=[b.block_id for b in manager.req_to_blocks['meeting']]
        max_live=0
        for n in range(prefix+1,prefix+window*12):
            manager.remove_skipped_blocks('meeting',n-1,prefix)
            manager.allocate_new_blocks('meeting',n,n)
            blocks=manager.req_to_blocks['meeting']
            assert [b.block_id for b in blocks[:len(pinned)]]==pinned
            alive=sum(not b.is_null for b in blocks)
            max_live=max(max_live,alive)
            assert alive<=math.ceil(prefix/bs)+math.ceil(window/bs)+2
        records.append(dict(window=window,prefix=prefix,max_live_blocks=max_live,
                            block_size=bs,output_tokens=n-prefix,prefix_pinned=True))
    return records

def penalty():
    scores=torch.tensor([[1.,2.,4.,-2.,.1,5.,.2,6.,.3]],device='cuda')
    actual=apply_all_penalties(scores.clone(),torch.tensor([[5,3,2]],device='cuda'),
          torch.zeros(1,device='cuda'),torch.zeros(1,device='cuda'),
          torch.tensor([1.02],device='cuda'),[[3,3,7]])
    expected=scores.clone();expected[0,3]*=1.02;expected[0,7]/=1.02
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    return dict(passed=True,scope='all_generated_tokens_only')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    torch.set_num_threads(4)
    report={}
    for name,fn in [('kernel',kernels),('eviction',eviction),('penalty',penalty)]:
        print('VLLM_CONTRACT_START '+name,flush=True)
        report[name]=fn();a.output.write_text(json.dumps(report,indent=2))
        print('VLLM_CONTRACT_PASSED '+name,flush=True)
    report['passed']=True;a.output.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
