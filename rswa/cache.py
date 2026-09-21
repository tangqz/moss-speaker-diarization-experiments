"""Bounded reference-plus-output cache; logical length never wraps."""
import torch
from transformers.cache_utils import Cache, CacheLayerMixin


class ReferenceWindowLayer(CacheLayerMixin):
    is_compileable=False
    is_sliding=False  # Not a standard sliding cache: the prefix is permanent.

    def __init__(self,prefix,window):
        super().__init__()
        self.prefix,self.window=int(prefix),int(window)
        self.cumulative_length=0
        self.ring_pos=0
        self.occupied=0

    def lazy_initialization(self,key_states,value_states):
        self.device,self.dtype=key_states.device,key_states.dtype
        shape=(*key_states.shape[:-2],self.prefix+self.window,key_states.shape[-1])
        self.keys=torch.empty(shape,device=self.device,dtype=self.dtype)
        self.values=torch.empty(shape,device=value_states.device,dtype=value_states.dtype)
        self.is_initialized=True

    def update(self,key_states,value_states,*args,**kwargs):
        if torch.is_grad_enabled():
            raise RuntimeError('Inference ring cache must not be used for training')
        if key_states.shape[0]!=1:
            raise NotImplementedError('Only batch=1 cache is validated')
        if not self.is_initialized:
            self.lazy_initialization(key_states,value_states)
        n=key_states.shape[-2]
        if self.cumulative_length < self.prefix:
            start=self.cumulative_length
            if start+n > self.prefix:
                raise ValueError('Prefill must end exactly at the reference boundary')
            self.keys[...,start:start+n,:].copy_(key_states)
            self.values[...,start:start+n,:].copy_(value_states)
            self.cumulative_length+=n
            self.occupied=self.cumulative_length
        else:
            if n!=1:
                raise NotImplementedError('Multi-token output decode requires separate causal validation')
            slot=self.prefix+self.ring_pos
            self.keys[...,slot:slot+1,:].copy_(key_states)
            self.values[...,slot:slot+1,:].copy_(value_states)
            self.ring_pos=(self.ring_pos+1)%self.window
            self.cumulative_length+=1
            self.occupied=min(self.prefix+self.window,self.cumulative_length)
        return self.keys[...,:self.occupied,:],self.values[...,:self.occupied,:]

    def get_seq_length(self):
        return self.cumulative_length

    def get_mask_sizes(self,query_length):
        return min(self.prefix+self.window,self.cumulative_length+query_length),0

    def get_max_cache_shape(self):
        return self.prefix+self.window

    def reset(self):
        self.cumulative_length=0
        self.ring_pos=0
        self.occupied=0

    def crop(self,*args,**kwargs):
        raise NotImplementedError('Speculative decode/crop is outside the initial contract')


class ReferenceWindowCache(Cache):
    def __init__(self,prefix,window,layers):
        super().__init__(layers=[ReferenceWindowLayer(prefix,window) for _ in range(layers)])
        self.prefix,self.window=prefix,window

    def stats(self):
        return [dict(logical_length=l.cumulative_length,occupied=l.occupied,
                     capacity=l.prefix+l.window,ring_pos=l.ring_pos,
                     storage_bytes=(l.keys.untyped_storage().nbytes()+l.values.untyped_storage().nbytes())
                     if l.is_initialized else 0) for l in self.layers]
