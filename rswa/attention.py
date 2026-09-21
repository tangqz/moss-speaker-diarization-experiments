"""Reference-preserving causal attention. All indices are global token positions."""
from dataclasses import dataclass
from functools import lru_cache
import os
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention


@dataclass(frozen=True)
class MaskSpec:
    prefix: int
    valid: int
    window: int | None
    backend: str = 'flex'
    query_start: int = 0

    def __post_init__(self):
        if not (0 < self.prefix <= self.valid):
            raise ValueError(f'Invalid prefix/length: {self}')
        if self.window is not None and self.window < 1:
            raise ValueError('Output window must be positive')


def allowed(spec, q, k):
    causal = k <= q
    window = True if spec.window is None else ((k < spec.prefix) | (k >= q-spec.window+1))
    # Pad queries receive one harmless key to avoid an all-masked softmax.
    # They carry no labels and are invisible to all valid queries.
    return ((q < spec.valid) & (k < spec.valid) & causal & window) | ((q >= spec.valid) & (k == 0))


def dense_reference(q, k, v, spec, scale=None):
    qi=torch.arange(q.shape[-2], device=q.device)+spec.query_start
    ki=torch.arange(k.shape[-2], device=q.device)
    mask=allowed(spec,qi[:,None],ki[None,:])
    k=k.repeat_interleave(q.shape[1]//k.shape[1],dim=1)
    v=v.repeat_interleave(q.shape[1]//v.shape[1],dim=1)
    scores=(q @ k.transpose(-1,-2))*(scale or q.shape[-1]**-.5)
    weights=scores.float().masked_fill(~mask,float('-inf')).softmax(-1).to(q.dtype)
    return weights @ v


@lru_cache(maxsize=8)
def block_mask(spec, qlen, kvlen, device, block=128):
    """Construct on the block grid, never on a token-by-token LxL grid."""
    nq=(qlen+block-1)//block
    nk=(kvlen+block-1)//block
    qlo=torch.arange(nq,device=device)[:,None]*block+spec.query_start
    qhi=torch.minimum(qlo+block-1,torch.tensor(spec.query_start+qlen-1,device=device))
    klo=torch.arange(nk,device=device)[None,:]*block
    khi=torch.minimum(klo+block-1,torch.tensor(kvlen-1,device=device))
    last_valid_q=torch.minimum(qhi,torch.tensor(spec.valid-1,device=device))
    causal_any=klo <= last_valid_q
    if spec.window is None:
        local_any=torch.ones_like(causal_any)
        local_all=local_any
    else:
        local_any=(klo < spec.prefix) | (khi >= qlo-spec.window+1)
        local_all=(khi < spec.prefix) | (klo >= qhi-spec.window+1)
    nonempty=((qlo < spec.valid)&(klo < spec.valid)&causal_any&local_any)
    nonempty=nonempty | ((qhi >= spec.valid)&(klo == 0))
    full=(qhi < spec.valid)&(khi < spec.valid)&(khi <= qlo)&local_all
    partial=nonempty & ~full
    ids=torch.arange(nk,device=device).expand(nq,nk)
    def pack(grid):
        # Fixed width at block granularity; compact valid columns are first.
        order=torch.where(grid,ids,ids+nk).argsort(dim=-1,stable=True).to(torch.int32)
        return grid.sum(-1).to(torch.int32)[None,None,:],order[None,None,:,:]
    pn,pi=pack(partial)
    fn,fi=pack(full)
    # Scalar tensor captures keep meeting-specific boundaries as runtime data.
    # Capturing the Python MaskSpec would specialize every new audio length.
    prefix_tensor=torch.tensor(spec.prefix,device=device,dtype=torch.int64)
    valid_tensor=torch.tensor(spec.valid,device=device,dtype=torch.int64)
    start_tensor=torch.tensor(spec.query_start,device=device,dtype=torch.int64)
    window=spec.window
    def mask_mod(b,h,q_idx,kv_idx):
        q=q_idx+start_tensor
        local=True if window is None else ((kv_idx<prefix_tensor)|(kv_idx>=q-window+1))
        return ((q<valid_tensor)&(kv_idx<valid_tensor)&(kv_idx<=q)&local)|((q>=valid_tensor)&(kv_idx==0))
    return BlockMask.from_kv_blocks(pn,pi,fn,fi,BLOCK_SIZE=block,mask_mod=mask_mod,
                                   seq_lengths=(qlen,kvlen))


_compiled_flex=None


def attention(q,k,v,spec,scale=None):
    global _compiled_flex
    if spec.backend=='dense':
        return dense_reference(q,k,v,spec,scale)
    if q.shape[0]!=1:
        raise NotImplementedError('Initial contract is one complete meeting per microbatch')
    if q.shape[-2]==1 and spec.query_start >= spec.prefix:
        # Cache contains only valid reference plus the current output window.
        return F.scaled_dot_product_attention(q,k,v,is_causal=False,scale=scale,enable_gqa=True)
    if _compiled_flex is None:
        if os.environ.get('MOSS_RSWA_STRICT_COMPILE')=='1':
            torch._dynamo.config.fail_on_recompile_limit_hit=True
        _compiled_flex=torch.compile(flex_attention,dynamic=True)
    mask=block_mask(spec,q.shape[-2],k.shape[-2],str(q.device))
    return _compiled_flex(q,k,v,block_mask=mask,scale=scale,enable_gqa=True)


def hf_attention(module,query,key,value,attention_mask,scaling=None,dropout=0.,**kwargs):
    if not isinstance(attention_mask,MaskSpec):
        raise TypeError('MOSS R-SWA requires explicit MaskSpec; refusing full-attention fallback')
    if dropout:
        raise NotImplementedError('Frozen MOSS config must have attention_dropout=0')
    module.rswa_forward_calls=getattr(module,'rswa_forward_calls',0)+1
    # Resolve Swift only if its sequence-parallel package has been loaded.
    import sys
    package=sys.modules.get('swift.sequence_parallel')
    sp=getattr(package,'sequence_parallel',None) if package else None
    if sp is not None and (getattr(sp,'world_size',1) or 1)>1:
        if sp.rp_world_size != 1:
            raise NotImplementedError('Only Ulysses SP is validated, not ring parallelism')
        from swift.sequence_parallel.ulysses import _SeqAllToAll
        def exchange(x):
            return _SeqAllToAll.apply(sp.sp_group,x.transpose(1,2).contiguous(),2,1).transpose(1,2)
        query,key,value=exchange(query),exchange(key),exchange(value)
        out=attention(query,key,value,attention_mask,scaling).transpose(1,2).contiguous()
        out=_SeqAllToAll.apply(sp.sp_group,out,1,2)
    else:
        out=attention(query,key,value,attention_mask,scaling).transpose(1,2).contiguous()
    return out,None
