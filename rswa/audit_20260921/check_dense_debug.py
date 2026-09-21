"""Determine whether the optional dense debug backend supports a wrapped cache."""
import json,sys
from pathlib import Path
import torch
torch.set_num_threads(2)
sys.path.insert(0,'/work/qt28/moss/results/moss-rswa-r2-20260921/source')
from attention import MaskSpec,attention
from cache import ReferenceWindowCache
torch.manual_seed(221)
p,w,n=17,4,27
q=torch.randn(1,4,n,8,dtype=torch.float64)
k=torch.randn(1,2,n,8,dtype=torch.float64);v=torch.randn_like(k)
rows=[]
with torch.inference_mode():
    cache=ReferenceWindowCache(p,w,1)
    cache.update(k[...,:p,:],v[...,:p,:],0)
    for i in range(p,n):
        kk,vv=cache.update(k[...,i:i+1,:],v[...,i:i+1,:],0)
        correct=attention(q[...,i:i+1,:],kk,vv,MaskSpec(p,i+1,w,query_start=i))
        dense=attention(q[...,i:i+1,:],kk,vv,MaskSpec(p,i+1,w,backend='dense',query_start=i))
        rows.append(dict(i=i,output_kv_count=i-p+1,dense_vs_formal_max_error=float((dense-correct).abs().max())))
print(json.dumps({'scope':'unused dense backend plus ring cache','rows':rows}))
