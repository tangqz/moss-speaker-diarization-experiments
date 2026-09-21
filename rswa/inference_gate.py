"""Actual MOSS vLLM trace and HF replay across both eviction boundaries."""
import argparse,copy,json,os,sys,time,hashlib
from pathlib import Path
BASE='/work/qt28/moss/models/MOSS-Transcribe-Diarize'
REPO='/work/qt28/moss/MOSS-Transcribe-Diarize'
TASK=Path('/work/qt28/moss/dkucc/rswa_20260920')
STEPS=[0,1,127,128,129,255,256,257,511]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--backend',choices=['vllm','hf'],required=True)
    ap.add_argument('--window',required=True)
    ap.add_argument('--sample',default='smoke',choices=['smoke','typical','longest'])
    ap.add_argument('--tokens',type=int,default=512)
    ap.add_argument('--native-full',action='store_true')
    ap.add_argument('--attention-backend',default='FLEX_ATTENTION',choices=['FLEX_ATTENTION','TRITON_ATTN'])
    ap.add_argument('--record-only',action='store_true');a=ap.parse_args()
    a.run.mkdir(parents=True,exist_ok=True)
    window=None if a.window=='full' else int(a.window)
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    os.environ['VLLM_USE_V2_MODEL_RUNNER']='0'
    os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    os.environ['MOSS_KV_AUDIT_PATH']=str(a.run/'kv_blocks.jsonl')
    import torch
    from transformers import AutoProcessor
    sys.path.insert(0,REPO)
    from moss_transcribe_diarize.inference_utils import build_transcription_messages,prepare_inputs,process_audio_info
    torch.set_num_threads(4);torch.manual_seed(0)
    processor=AutoProcessor.from_pretrained(BASE,trust_remote_code=True,local_files_only=True)
    row=json.loads((TASK/f'data/{a.sample}.jsonl').read_text().splitlines()[0])
    prompt=row['messages'][0]['content'].replace('<audio>','').strip()
    messages=build_transcription_messages(row['audios'][0],prompt)
    text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    audios=process_audio_info(messages,sampling_rate=processor.feature_extractor.sampling_rate)
    expected=processor(text=text,audio=audios,return_tensors='pt')
    prefix=expected['input_ids'].shape[1]
    steps=sorted(set([s for s in STEPS+[1023,1024,2047,2048] if s<a.tokens]+[a.tokens-1]))
    assert prefix+a.tokens<=131072
    if a.backend=='vllm':
        from vllm import LLM,SamplingParams
        from moss_vllm_rswa import overrides
        llm=LLM(model=BASE,tokenizer=BASE,trust_remote_code=True,model_impl='vllm',dtype='bfloat16',
            tensor_parallel_size=1,max_model_len=131072,max_num_seqs=1,
            max_num_batched_tokens=max(8192,((prefix+8191)//8192)*8192),
            gpu_memory_utilization=.55,enable_prefix_caching=False,mm_processor_cache_gb=0,
            disable_log_stats=False,generation_config='vllm',seed=0,
            limit_mm_per_prompt={'audio':1},hf_overrides=overrides(window),
            attention_config={'backend':a.attention_backend},
            worker_extension_cls='moss_vllm_rswa.WorkerExtension')
        receipt=llm.collective_rpc('rswa_runtime_receipt')
        (a.run/'runtime_receipt.json').write_text(json.dumps(receipt,indent=2))
        llm.collective_rpc('rswa_start_logit_audit',args=(str(a.run/'raw_logits'),steps))
        # This fixed-length engineering trace intentionally crosses k=256;
        # ignore_eos is NEVER used by quality evaluation.
        params=SamplingParams(temperature=0,repetition_penalty=1.02,max_tokens=a.tokens,
                              ignore_eos=True,skip_special_tokens=False,seed=0)
        out=llm.generate([dict(prompt=text,multi_modal_data={'audio':(audios[0],processor.feature_extractor.sampling_rate)})],params)[0]
        assert out.prompt_token_ids==expected['input_ids'][0].tolist()
        record=dict(window=window,prefix=prefix,prompt_ids=out.prompt_token_ids,
                    generated_ids=list(out.outputs[0].token_ids),penalty=1.02,
                    penalty_scope='all_generated_tokens_only',engineering_only=True,attention_backend=a.attention_backend)
        assert len(record['generated_ids'])==a.tokens
        (a.run/'vllm_trace.json').write_text(json.dumps(record))
    else:
        from transformers import AutoModelForCausalLM
        from transformers.cache_utils import DynamicCache
        from model_adapter import install,OutputRepetitionPenalty
        from cache import ReferenceWindowCache
        trace=json.loads((a.run/'vllm_trace.json').read_text())
        assert trace['prompt_ids']==expected['input_ids'][0].tolist()
        model=AutoModelForCausalLM.from_pretrained(BASE,trust_remote_code=True,local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
        if a.native_full:
            assert window is None
        else:install(model,window)
        inputs=expected.to('cuda')
        cache=DynamicCache(config=model.config.text_config) if window is None else ReferenceWindowCache(prefix,window,28)
        records=[];history=inputs['input_ids'];penalty=OutputRepetitionPenalty(prefix,1.02)
        token_checks=[]
        hf_dir=a.run/('hf_native_logits' if a.native_full else 'hf_logits');hf_dir.mkdir(exist_ok=True)
        def prefix_hashes():
            return [hashlib.sha256(x[...,:prefix,:].contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                    for layer in cache.layers for x in [layer.keys,layer.values]]
        reference_hashes=None
        with torch.inference_mode():
            for step in range(a.tokens):
                kwargs=dict(inputs) if step==0 else dict(input_ids=history[:,-1:])
                out=model(**kwargs,past_key_values=cache,use_cache=True,logits_to_keep=1)
                logits=out.logits[:,-1,:].float()
                if step==0:reference_hashes=prefix_hashes()
                penalized=penalty(history,logits)
                best=penalized.topk(2,dim=-1)
                margin=float(best.values[0,0]-best.values[0,1])
                chosen=int(best.indices[0,0]);vchosen=trace['generated_ids'][step]
                token_checks.append(dict(step=step,hf_top1=chosen,vllm_top1=vchosen,margin=margin,
                    high_margin_disagreement=chosen!=vchosen and margin>.5))
                if step in steps:
                    ref=torch.load(a.run/'raw_logits'/f'logits-{step}.pt',weights_only=True)['logits'].cuda()
                    diff=(logits-ref).abs()
                    refpenalty=penalty(history,ref)
                    vchosen=trace['generated_ids'][step]
                    assert int(refpenalty.argmax(-1))==vchosen,('penalty sampler mismatch',step)
                    row=dict(step=step,max_abs=float(diff.max()),mean_abs=float(diff.mean()),
                             p99=float(torch.quantile(diff,.99)),hf_top1=int(penalized.argmax(-1)),
                             vllm_top1=vchosen,penalty_agrees=True)
                    row['passed']=row['max_abs']<=.5 and row['mean_abs']<=.05 and row['p99']<=.15
                    records.append(row);print(json.dumps(row),flush=True)
                    torch.save(logits.cpu(),hf_dir/f'logits-{step}.pt')
                history=torch.cat([history,torch.tensor([[trace['generated_ids'][step]]],device='cuda')],dim=1)
        assert prefix_hashes()==reference_hashes,'Reference KV changed during decode'
        report=dict(passed=all(r['passed'] for r in records) and not any(r['high_margin_disagreement'] for r in token_checks),
                    records=records,prefix=prefix,window=window,token_checks=token_checks,native_full=a.native_full,
                    generated_tokens=a.tokens,reference_kv_sha256=reference_hashes,all_layer_reference_unchanged=True,
                    vllm_attention_backend=trace.get('attention_backend','FLEX_ATTENTION'),
                    hf_cache=cache.stats() if window is not None else {'length':cache.get_seq_length()})
        (a.run/('hf_native_alignment.json' if a.native_full else 'hf_alignment.json')).write_text(json.dumps(report,indent=2))
        if not a.record_only:
            assert report['passed'],'Actual MOSS HF/vLLM logits exceeded frozen BF16 tolerance'

if __name__=='__main__':main()
