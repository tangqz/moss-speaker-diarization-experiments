"""Attach the R-SWA interface without changing weights, architecture, or CE."""
import types,json,os
from pathlib import Path
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from attention import MaskSpec,hf_attention
from cache import ReferenceWindowCache


def rswa_forward(self,input_ids=None,attention_mask=None,position_ids=None,
                 past_key_values=None,inputs_embeds=None,labels=None,use_cache=None,
                 rswa_prefix_length=None,rswa_valid_length=None,rswa_sample_key=None,**kwargs):
    cfg=self.config.moss_rswa
    window=cfg['window']
    if self.training:
        if past_key_values is not None:
            raise ValueError('Teacher-forced training must not receive a KV cache')
        use_cache=False
    x=input_ids if input_ids is not None else inputs_embeds
    if x is None or x.shape[0]!=1:
        raise ValueError('R-SWA initial contract requires one complete unpadded meeting')
    n=x.shape[1]
    seen=past_key_values.get_seq_length() if past_key_values is not None else 0
    if use_cache and window is not None and not isinstance(past_key_values,ReferenceWindowCache):
        raise TypeError('Windowed generation must provide ReferenceWindowCache explicitly')
    if rswa_prefix_length is None:
        if isinstance(past_key_values,ReferenceWindowCache):
            prefix=past_key_values.prefix
        elif labels is not None:
            raise ValueError('Training requires explicit prefix length before SP label shifting')
        else:
            prefix=seen+n if seen==0 else int(getattr(self,'_rswa_generation_prefix',0))
            if seen==0:
                self._rswa_generation_prefix=prefix
    else:
        prefix=int(rswa_prefix_length)
    valid=int(rswa_valid_length) if rswa_valid_length is not None else seen+n
    audit=os.environ.get('MOSS_RSWA_ORDER_LOG')
    if self.training and audit:
        assert rswa_sample_key is not None,'Actual training forward must identify its meeting'
        step=int(os.environ['MOSS_RSWA_OPTIMIZER_STEP'])
        previous=getattr(self,'_rswa_audit_step',None)
        microbatch=getattr(self,'_rswa_audit_microbatch',0)+1 if previous==step else 1
        self._rswa_audit_step=step;self._rswa_audit_microbatch=microbatch
        rank=int(os.environ.get('RANK','0'))
        path=Path(audit) if rank==0 else Path(audit).with_suffix(f'.rank{rank}.jsonl')
        with path.open('a') as stream:
            stream.write(json.dumps(dict(audio=rswa_sample_key,prefix=prefix,length=valid,
                optimizer_step=step,microbatch=microbatch))+'\n')
    spec=MaskSpec(prefix=prefix,valid=valid,window=window,backend=self._rswa_backend,query_start=seen)
    if position_ids is None:
        position_ids=torch.arange(seen,seen+n,device=x.device).unsqueeze(0)
    if attention_mask is not None and torch.is_tensor(attention_mask):
        if attention_mask.ndim!=2 or attention_mask.shape[0]!=1:
            raise ValueError('Only a 2D right-padding mask is accepted')
    return self._rswa_original_forward(
        input_ids=input_ids,attention_mask={'full_attention':spec},position_ids=position_ids,
        past_key_values=past_key_values,inputs_embeds=inputs_embeds,labels=labels,
        use_cache=use_cache,**kwargs)


def install(model,window,backend='flex'):
    if hasattr(model,'_rswa_original_forward'):
        raise ValueError('R-SWA has already been installed')
    if backend not in ('dense','flex'):
        raise ValueError(backend)
    if window is not None and window<=0:
        raise ValueError(window)
    ALL_ATTENTION_FUNCTIONS.register('moss_rswa',hf_attention)
    lm=model.model.language_model if hasattr(model.model,'language_model') else model.model
    lm.config._attn_implementation='moss_rswa'
    lm.config.use_cache=False
    model.config.moss_rswa=dict(version=1,window=window,reference='complete_prompt',
                               positions='absolute',window_includes_current_input=True)
    model._rswa_backend=backend
    model._rswa_original_forward=model.forward
    model.forward=types.MethodType(rswa_forward,model)
    return model


class OutputRepetitionPenalty:
    """HF/vLLM-aligned multiplicative penalty on generated tokens only."""
    def __init__(self,prompt_length,penalty=1.02):
        if penalty<=0:
            raise ValueError(penalty)
        self.prompt_length,self.penalty=int(prompt_length),float(penalty)

    def __call__(self,input_ids,scores):
        generated=input_ids[:,self.prompt_length:]
        if not generated.numel():
            return scores
        values=scores.gather(1,generated)
        values=torch.where(values<0,values*self.penalty,values/self.penalty)
        return scores.scatter(1,generated,values)
