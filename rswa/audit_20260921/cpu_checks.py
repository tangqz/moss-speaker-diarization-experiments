"""Independent equation/index audit; executed on CPU against deployed sources."""
import ast
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path
import torch

torch.set_num_threads(2)
ROOT=Path('/work/qt28/moss')
SOURCE=ROOT/'results/moss-rswa-r2-20260921/source'
sys.path.insert(0,str(SOURCE))
from attention import MaskSpec, allowed, block_mask, attention
from cache import ReferenceWindowCache

def oracle(prefix, window, q, kvlen, valid):
    # Paper (2) is one-based and t denotes the token to be predicted.
    if q>=valid:
        return {0}
    if q<prefix:
        return set(range(q+1))
    t=q-prefix+2
    first=prefix+1 if window is None else max(prefix+1,prefix+t-window)
    one_based_output=range(first,prefix+t)
    return set(range(prefix)) | {j-1 for j in one_based_output if j<=valid}

vroot=next((ROOT/'envs/vllm-moss-20260914/lib').glob('python*/site-packages'))/'vllm'
helper=vroot/'v1/attention/ops/triton_attention_helpers.py'
def plain_function(path,name,namespace):
    tree=ast.parse(path.read_text())
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[];node.returns=None
    for arg in node.args.args+node.args.kwonlyargs: arg.annotation=None
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path),'exec'),namespace)
    return namespace[name]
tl=types.SimpleNamespace(load=lambda x:x,minimum=min,maximum=max)
env={'tl':tl,'cdiv_fn':lambda a,b:(a+b-1)//b}
kernel_mask=plain_function(helper,'compute_kv_seq_mask',env)
bounds=plain_function(helper,'compute_tile_loop_bounds',env)
rows=0
for prefix in [1,7,127,128,129,173,511]:
    for window in [1,2,4,128,256,512]:
        valid=prefix+2*window+5
        for t in sorted(set([1,2,window,window+1,window+2,2*window+1,2*window+3])):
            q=prefix+t-2
            keys=torch.arange(valid)
            expected=torch.tensor([j in oracle(prefix,window,q,valid,valid) for j in range(valid)])
            actual=allowed(MaskSpec(prefix,valid,window),torch.tensor(q),keys)
            assert torch.equal(actual,expected),(prefix,window,t,'training')
            triton=kernel_mask(torch.tensor([[q]]),keys,0,valid,None,0,False,0,
                rswa_prefix_lens_ptr=prefix,R_SWA_WINDOW=window,USE_R_SWA=True)[0]
            assert torch.equal(triton,expected),(prefix,window,t,'triton')
            rows+=1

block_cases=0
for prefix,window,valid,pad,start in [
    (7,4,31,3,0),(129,128,385,3,0),(31,256,553,0,0),
    (173,512,1210,3,0),(15,None,137,2,0),(127,1,270,1,0),
    (128,128,385,0,127),(129,256,650,3,400),
    (173,512,1210,3,1100),(173,256,200,0,100)]:
    kvlen=valid+pad;qlen=kvlen-start
    spec=MaskSpec(prefix,valid,window,query_start=start)
    bm=block_mask(spec,qlen,kvlen,'cpu')
    actual=torch.zeros(qlen,kvlen,dtype=torch.bool)
    for qb in range((qlen+127)//128):
        qr=torch.arange(qb*128,min((qb+1)*128,qlen))
        for full,counts,indices in [(False,bm.kv_num_blocks,bm.kv_indices),
                                   (True,bm.full_kv_num_blocks,bm.full_kv_indices)]:
            for kb in indices[0,0,qb,:int(counts[0,0,qb])].tolist():
                kr=torch.arange(kb*128,min((kb+1)*128,kvlen))
                if not kr.numel():continue
                tile=True if full else bm.mask_mod(0,0,qr[:,None],kr[None,:])
                actual[qr[:,None],kr[None,:]]=tile
    expected=torch.zeros_like(actual)
    for localq in range(qlen):
        expected[localq,sorted(oracle(prefix,window,localq+start,kvlen,valid))]=True
    assert torch.equal(actual,expected),(prefix,window,valid,pad,start,'block grid')
    block_cases+=1

cache_checks=[]
with torch.inference_mode():
    torch.manual_seed(92021)
    for window in [1,4,128,256,512]:
        prefix=17;n=prefix+2*window+3
        q=torch.randn(1,4,n,8,dtype=torch.float64)
        k=torch.randn(1,2,n,8,dtype=torch.float64)
        v=torch.randn_like(k)
        cache=ReferenceWindowCache(prefix,window,1)
        cache.update(k[...,:prefix,:],v[...,:prefix,:],0)
        initial=cache.layers[0].keys[...,:prefix,:].clone()
        pointer=cache.layers[0].keys.data_ptr();error=0.
        for i in range(prefix,n):
            kk,vv=cache.update(k[...,i:i+1,:],v[...,i:i+1,:],0)
            got=attention(q[...,i:i+1,:],kk,vv,MaskSpec(prefix,i+1,window,query_start=i))
            ids=sorted(oracle(prefix,window,i,n,n))
            key=k[...,ids,:].repeat_interleave(2,dim=1)
            value=v[...,ids,:].repeat_interleave(2,dim=1)
            scores=q[...,i:i+1,:]@key.transpose(-1,-2)*8**-.5
            expected=scores.softmax(-1)@value
            error=max(error,float((got-expected).abs().max()))
            assert torch.allclose(got,expected,atol=1e-12,rtol=1e-12)
            assert cache.get_seq_length()==i+1
            assert torch.equal(initial,cache.layers[0].keys[...,:prefix,:])
            assert cache.layers[0].keys.data_ptr()==pointer
            assert kk.shape[-2]==prefix+min(window,i-prefix+1)
        cache_checks.append(dict(window=window,updates=n-prefix,max_error=error,capacity=prefix+window))

tile_counts=[]
for seq in [512,4096,65536]:
    lo,hi,_=bounds(seq-1,seq,1,0,0,0,32,16,8,2,0,True,False)
    tile_counts.append(dict(logical_length=seq,tile_size=32,loop_tiles=hi-lo))
important=['attention.py','cache.py','model_adapter.py','moss_rswa_plugin.py']
hashes={}
for name in important:
    old=ROOT/'results/moss-rswa-20260920/source'/name
    new=SOURCE/name
    hashes[name]=dict(round1=hashlib.sha256(old.read_bytes()).hexdigest(),round2=hashlib.sha256(new.read_bytes()).hexdigest())
    assert hashes[name]['round1']==hashes[name]['round2']
result=dict(passed=True,device='cpu',paper_vs_training_vs_triton_rows=rows,
    block_mask_cases=block_cases,ring_cache=cache_checks,triton_logical_tile_counts=tile_counts,
    same_round1_round2_core=hashes,swift_origin=importlib.util.find_spec('swift').origin,
    limitation='CPU execution of extracted Triton mask/bounds, not a fresh compiled GPU kernel test')
print(json.dumps(result))
