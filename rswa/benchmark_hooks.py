"""Explicit performance-only fixed-token continuation, after ordinary sampling."""
import os
import torch

def install(worker):
    assert os.environ.get('MOSS_BENCHMARK_ONLY')=='1','Forced histories forbidden in quality evaluation'
    if hasattr(worker,'_rswa_benchmark'):return worker._rswa_benchmark
    state=dict(active=False,step=0,ids=None,events=[])
    runner=worker.model_runner;sampler=runner.sampler
    original_sample=sampler.forward;original_forward=runner._model_forward
    from vllm.v1.sample.ops import penalties
    original_penalty=penalties.apply_penalties
    def measured(kind,function,*args,**kwargs):
        step=state['step'];sample=state['active'] and (step==0 or step%64==0)
        if sample:
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
        result=function(*args,**kwargs)
        if sample:
            end.record();state['events'].append((kind,step,begin,end))
        return result
    def model_forward(*args,**kwargs):return measured('backbone_cuda',original_forward,*args,**kwargs)
    def penalty(*args,**kwargs):return measured('output_penalty_cuda',original_penalty,*args,**kwargs)
    def sample(*args,**kwargs):
        result=original_sample(*args,**kwargs)
        if state['active']:
            step=state['step'];assert step<len(state['ids']) and result.sampled_token_ids.numel()==1
            result.sampled_token_ids.copy_(state['ids'][step:step+1].reshape_as(result.sampled_token_ids))
            state['step']+=1
        return result
    sampler.forward=sample;runner._model_forward=model_forward;penalties.apply_penalties=penalty
    worker._rswa_benchmark=state;return state

def begin(worker,ids):
    state=install(worker);assert not state['active']
    state.update(active=True,step=0,ids=torch.tensor(ids,dtype=torch.int32,device='cuda'),events=[])
    return dict(active=True,tokens=len(ids),quality_metrics_forbidden=True)

def end(worker):
    state=worker._rswa_benchmark;torch.cuda.synchronize()
    assert state['step']==len(state['ids'])
    rows=[dict(kind=kind,output_index=step,milliseconds=start.elapsed_time(stop)) for kind,step,start,stop in state['events']]
    result=dict(tokens=state['step'],samples=rows,sampling_stride=64,
                timing_scope='backbone excludes LM head and sampler; output penalty measured separately',
                forced_token_copy_and_sampling_included_in_wall_latency=True)
    state.update(active=False,ids=None,events=[]);return result
