"""Independent correctness checks before allowing any formal training."""
import argparse,json,time
from pathlib import Path
import torch
from attention import MaskSpec,attention,allowed,block_mask
from cache import ReferenceWindowCache
from model_adapter import install,OutputRepetitionPenalty


def independently_masked(q,k,v,prefix,window,valid=None,offset=0):
    valid=valid or k.shape[-2]
    mask=torch.zeros(q.shape[-2],k.shape[-2],dtype=torch.bool,device=q.device)
    for row in range(q.shape[-2]):
        logical=row+offset
        if logical>=valid:
            mask[row,0]=True
            continue
        for col in range(min(logical+1,valid)):
            if col<prefix or window is None or logical-col<window:
                mask[row,col]=True
    kk=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
    vv=v.repeat_interleave(q.shape[1]//v.shape[1],dim=1)
    weights=((q @ kk.transpose(-1,-2))*q.shape[-1]**-.5).float().masked_fill(~mask,-torch.inf).softmax(-1)
    return weights.to(q.dtype) @ vv,mask


def masks_and_gradients(device):
    rows=[]
    for p,w,n,pad in [(7,4,31,0),(129,128,385,3),(31,256,297,0),(15,None,137,2)]:
        torch.manual_seed(112)
        inputs=[torch.randn(1,h,n+pad,32,device=device,requires_grad=True) for h in (4,2,2)]
        ref,expected=independently_masked(*inputs,p,w,n)
        spec=MaskSpec(p,n,w)
        indices=torch.arange(n+pad,device=device)
        assert torch.equal(expected,allowed(spec,indices[:,None],indices[None,:]))
        before=torch.cuda.max_memory_allocated() if device=='cuda' else 0
        actual=attention(*inputs,spec)
        torch.testing.assert_close(actual,ref,atol=1e-5,rtol=1e-4)
        probe=torch.randn_like(ref)
        ga=torch.autograd.grad((actual*probe).sum(),inputs)
        gr=torch.autograd.grad((ref*probe).sum(),inputs)
        errors=[]
        for aa,rr in zip(ga,gr):
            torch.testing.assert_close(aa,rr,atol=2e-5,rtol=2e-4)
            errors.append(float((aa-rr).abs().max()))
        rows.append(dict(prefix=p,window=w,length=n,padding=pad,
                         max_output_error=float((actual-ref).abs().max()),gradient_errors=errors))
    return rows


@torch.no_grad()
def cache_contract(device):
    torch.manual_seed(92)
    p,w,n=19,8,101
    q=torch.randn(1,4,n,32,device=device)
    k=torch.randn(1,2,n,32,device=device)
    v=torch.randn(1,2,n,32,device=device)
    cache=ReferenceWindowCache(p,w,1)
    cache.update(k[...,:p,:],v[...,:p,:],0)
    pointer=cache.layers[0].keys.data_ptr()
    prefix=cache.layers[0].keys[...,:p,:].clone()
    max_error=0.
    for i in range(p,n):
        kk,vv=cache.update(k[...,i:i+1,:],v[...,i:i+1,:],0)
        actual=attention(q[...,i:i+1,:],kk,vv,MaskSpec(p,i+1,w,query_start=i))
        ref,_=independently_masked(q[...,i:i+1,:],k[...,:i+1,:],v[...,:i+1,:],p,w,offset=i)
        torch.testing.assert_close(actual,ref,atol=1e-5,rtol=1e-4)
        max_error=max(max_error,float((actual-ref).abs().max()))
        assert cache.get_seq_length()==i+1 and kk.shape[-2]<=p+w
        assert pointer==cache.layers[0].keys.data_ptr()
        assert torch.equal(prefix,cache.layers[0].keys[...,:p,:])
    return dict(max_output_error=max_error,ring_wraps=(n-p)//w,stats=cache.stats())


@torch.no_grad()
def tiny_model_contract(device):
    from transformers import Qwen3Config,Qwen3ForCausalLM,LogitsProcessorList
    from transformers.cache_utils import DynamicCache
    torch.manual_seed(104)
    cfg=Qwen3Config(vocab_size=97,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
                   num_attention_heads=4,num_key_value_heads=2,head_dim=16,
                   max_position_embeddings=131072,attention_dropout=0.,pad_token_id=0,
                   eos_token_id=96,tie_word_embeddings=True)
    original=Qwen3ForCausalLM(cfg).to(device).eval()
    original.config._attn_implementation='eager'
    ids=torch.randint(1,90,(1,58),device=device)
    original_logits=original(input_ids=ids,use_cache=False).logits
    adapted=Qwen3ForCausalLM(cfg).to(device).eval()
    adapted.load_state_dict(original.state_dict())
    install(adapted,window=128)
    full=adapted(input_ids=ids,use_cache=False,rswa_prefix_length=13).logits
    torch.testing.assert_close(full,original_logits,atol=1e-5,rtol=1e-4)
    adapted.config.moss_rswa['window']=8
    reference=adapted(input_ids=ids,use_cache=False,rswa_prefix_length=13).logits
    cache=ReferenceWindowCache(13,8,2)
    actual=adapted(input_ids=ids[:,:13],past_key_values=cache,use_cache=True).logits
    outputs=[actual]
    for i in range(13,ids.shape[1]):
        outputs.append(adapted(input_ids=ids[:,i:i+1],past_key_values=cache,use_cache=True).logits)
    cached=torch.cat(outputs,dim=1)
    torch.testing.assert_close(cached,reference,atol=2e-5,rtol=2e-4)
    changed=ids.clone();changed[:,-8:]=torch.randint(1,90,(1,8),device=device)
    future=adapted(input_ids=changed,use_cache=False,rswa_prefix_length=13).logits
    torch.testing.assert_close(future[:,:-8],reference[:,:-8],atol=0,rtol=0)
    # Long absolute positions are independent of physical cache slots.
    positions=torch.arange(100000,100058,device=device).unsqueeze(0)
    long_ref=adapted(input_ids=ids,position_ids=positions,use_cache=False,rswa_prefix_length=13).logits
    cache=ReferenceWindowCache(13,8,2)
    pieces=[adapted(input_ids=ids[:,:13],position_ids=positions[:,:13],past_key_values=cache,use_cache=True).logits]
    for i in range(13,58):
        pieces.append(adapted(input_ids=ids[:,i:i+1],position_ids=positions[:,i:i+1],past_key_values=cache,use_cache=True).logits)
    torch.testing.assert_close(torch.cat(pieces,dim=1),long_ref,atol=2e-5,rtol=2e-4)
    generated=adapted.generate(input_ids=ids[:,:13],attention_mask=torch.ones_like(ids[:,:13]),
                             past_key_values=ReferenceWindowCache(13,8,2),max_new_tokens=20,
                             do_sample=False,repetition_penalty=1.,
                             logits_processor=LogitsProcessorList([OutputRepetitionPenalty(13,1.02)]))
    return dict(full_equivalence_max_error=float((full-original_logits).abs().max()),
                cache_max_error=float((cached-reference).abs().max()),generation_tokens=generated.shape[1]-13,
                future_isolated=True,long_position=100000)


def penalty_contract():
    ids=torch.tensor([[5,3,2,3,3,7]])
    scores=torch.tensor([[1.,2.,4.,-2.,.1,5.,.2,6.,.3]])
    out=OutputRepetitionPenalty(3,1.02)(ids,scores.clone())
    assert out[0,5]==scores[0,5] and out[0,2]==scores[0,2]
    torch.testing.assert_close(out[0,3],scores[0,3]*1.02)
    torch.testing.assert_close(out[0,7],scores[0,7]/1.02)
    return dict(generated_only=True,positive_and_negative_logits=True,unique_penalty_once=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();torch.set_num_threads(4)
    assert torch.cuda.is_available(),'GPU allocation required'
    report=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),started=time.time())
    for name,fn in [('mask_and_gradient',lambda:masks_and_gradients('cuda')),
                    ('ring_cache',lambda:cache_contract('cuda')),
                    ('tiny_qwen3',lambda:tiny_model_contract('cuda')),
                    ('penalty',penalty_contract)]:
        print(json.dumps(dict(event='contract_start',name=name)),flush=True)
        report[name]=fn()
        args.output.write_text(json.dumps(report,indent=2))
        print(json.dumps(dict(event='contract_passed',name=name,result=report[name])),flush=True)
    report['passed']=True;report['seconds']=time.time()-report['started']
    args.output.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
