"""Scoped vLLM worker extension using its pinned upstream R-SWA kernels/cache.

Only Qwen3 decoder Attention construction is adapted. The Whisper encoder,
weights, rotary embeddings and multimodal processor retain the upstream path.
"""
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention.rswa_attention import RSWAAttention
from vllm.model_executor.models import qwen3
from kv_audit import install as _install_kv_audit
_install_kv_audit()

_original_attention=qwen3.Attention

def configured_attention(*args,**kwargs):
    config=get_current_vllm_config()
    window=config.model_config.rswa_window
    if window is None:return _original_attention(*args,**kwargs)
    assert window in (128,256),window
    return RSWAAttention(*args,rswa_window=window,**kwargs)

qwen3.Attention=configured_attention

# Upstream repetition penalty includes prompt IDs. Our frozen evaluation
# protocol uses every generated ID, independently of the evicted KV window.
from vllm.v1.sample.ops import penalties as _penalties
_original_penalties=_penalties.apply_penalties

def output_only_penalties(logits,prompt_tokens_tensor,output_tokens_tensor,
                          presence_penalties,frequency_penalties,repetition_penalties):
    empty_prompt=prompt_tokens_tensor[:,:0]
    return _original_penalties(logits,empty_prompt,output_tokens_tensor,
                               presence_penalties,frequency_penalties,repetition_penalties)

_penalties.apply_penalties=output_only_penalties

class WorkerExtension:
    def rswa_encoder_cache_size(self):
        return len(self.model_runner.encoder_cache)

    def rswa_benchmark_begin(self,ids):
        from benchmark_hooks import begin
        return begin(self,ids)

    def rswa_benchmark_end(self):
        from benchmark_hooks import end
        return end(self)

    def rswa_reset_memory_stats(self):
        import torch
        torch.cuda.reset_peak_memory_stats()
        return True

    def rswa_memory_stats(self):
        import torch
        return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),
                    interpretation='worker allocator including preallocated vLLM KV pool')

    def rswa_start_logit_audit(self,output_path,steps):
        import torch
        from pathlib import Path
        path=Path(output_path);path.mkdir(parents=True,exist_ok=True)
        model=self.model_runner.model
        original=model.compute_logits
        counter=[0];selected=set(steps)
        def audited(*args,**kwargs):
            logits=original(*args,**kwargs)
            step=counter[0];counter[0]+=1
            if step in selected:
                torch.save(dict(step=step,logits=logits.detach().float().cpu()),path/f'logits-{step}.pt')
            return logits
        model.compute_logits=audited
        return dict(audit_started=True,steps=steps)

    def rswa_runtime_receipt(self):
        from vllm.v1.kv_cache_interface import RSWASpec
        config=self.vllm_config
        runner=type(self.model_runner).__module__
        assert runner=='vllm.v1.worker.gpu_model_runner',runner
        layers=config.compilation_config.static_forward_context
        receipts=[]
        for name,module in layers.items():
            if hasattr(module,'get_kv_cache_spec'):
                spec=module.get_kv_cache_spec(config)
                if spec is None:continue
                receipts.append(dict(name=name,spec=type(spec).__name__,
                                     window=getattr(spec,'rswa_window',None),
                                     block_size=spec.block_size))
        window=config.model_config.rswa_window
        assert len(receipts)==28,receipts
        assert all(r['window']==window for r in receipts),receipts
        if window is not None:assert all(r['spec']=='RSWASpec' for r in receipts)
        assert _penalties.apply_penalties is output_only_penalties
        return dict(window=window,layers=receipts,penalty_scope='all_generated_tokens_only',model_runner=runner)

def overrides(window,allow_window_override=False):
    def apply(config):
        saved=getattr(config,'moss_rswa',None)
        if saved is not None and saved['window']!=window and not allow_window_override:
            raise ValueError(f'Checkpoint window {saved} disagrees with inference {window}')
        config.rswa_window=window
        # vLLM first probes the callback with a generic dummy config to infer
        # model_type, then applies it to the actual multimodal config.
        if hasattr(config,'text_config'):
            config.text_config.rswa_window=window
        return config
    return apply
