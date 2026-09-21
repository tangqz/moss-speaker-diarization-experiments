"""Fixed original gold histories measure efficiency only, never recognition quality."""
import argparse,dataclasses,hashlib,json,os,sys,time,statistics
from pathlib import Path
from common import ROOT,BASE,REPO,read,write,sha
from protocol import ATTENTION_BACKEND
TASK=ROOT/'dkucc/rswa_20260920'

def summarize(values):
    ordered=sorted(values);position=.9*(len(ordered)-1);lo=int(position);hi=min(lo+1,len(ordered)-1)
    return dict(median=statistics.median(values),p90=ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo),
                minimum=min(values),maximum=max(values),repeats=len(values))

def inputs():
    from transformers import AutoProcessor
    sys.path.insert(0,str(REPO))
    from moss_transcribe_diarize.inference_utils import build_transcription_messages,process_audio_info
    processor=AutoProcessor.from_pretrained(BASE,trust_remote_code=True,local_files_only=True)
    result=[]
    for sample in ['smoke','typical','longest']:
        row=json.loads((TASK/f'data/{sample}.jsonl').read_text().splitlines()[0])
        prompt=row['messages'][0]['content'].replace('<audio>','').strip()
        messages=build_transcription_messages(row['audios'][0],prompt)
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        audios=process_audio_info(messages,sampling_rate=processor.feature_extractor.sampling_rate)
        official=processor(text=text,audio=audios,return_tensors='pt')
        ids=processor.tokenizer(row['messages'][1]['content'].strip()+processor.tokenizer.eos_token,add_special_tokens=False)['input_ids']
        p=official['input_ids'].shape[1];lengths=[1024,4096,8192,16384,32768,65536]
        result.append(dict(sample=sample,audio=row['audios'][0],text=text,prefix=p,
            prompt_ids=official['input_ids'][0].tolist(),gold_ids=ids,
            lengths=[n for n in lengths if n<=len(ids) and p+n<=131072],
            not_applicable=[dict(tokens=n,reason='original gold trajectory or logical context too short')
                            for n in lengths if n>len(ids) or p+n>131072]))
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--window',choices=['full','128','256'],required=True)
    ap.add_argument('--sample',choices=['smoke','typical','longest'],required=True)
    ap.add_argument('--engineering-tokens',type=int,default=0);a=ap.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    os.environ.update(VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_USE_V2_MODEL_RUNNER='0',
        VLLM_USE_FLASHINFER_SAMPLER='0',MOSS_BENCHMARK_ONLY='1',MOSS_KV_AUDIT_PATH=str(a.run/'kv.jsonl'))
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    import torch
    from vllm import LLM,SamplingParams
    from moss_vllm_rswa import overrides
    sys.path.insert(0,str(REPO))
    from moss_transcribe_diarize.inference_utils import build_transcription_messages,process_audio_info
    manifest=TASK/'data/performance_trajectories.json'
    if not manifest.exists():write(manifest,dict(selection='frozen training shortest, median duration, longest',records=inputs()))
    records=read(manifest)['records'];item=next(r for r in records if r['sample']==a.sample)
    torch.set_num_threads(4);window=None if a.window=='full' else int(a.window)
    start=time.perf_counter()
    llm=LLM(model=str(BASE),tokenizer=str(BASE),trust_remote_code=True,model_impl='vllm',dtype='bfloat16',
        tensor_parallel_size=1,max_model_len=131072,max_num_seqs=1,
        max_num_batched_tokens=max(8192,((max(r['prefix'] for r in records)+8191)//8192)*8192),
        gpu_memory_utilization=.55,enable_prefix_caching=False,mm_processor_cache_gb=0,disable_log_stats=False,
        generation_config='vllm',seed=0,limit_mm_per_prompt={'audio':1},hf_overrides=overrides(window),
        attention_config={'backend':ATTENTION_BACKEND},worker_extension_cls='moss_vllm_rswa.WorkerExtension')
    startup=time.perf_counter()-start;runtime=llm.collective_rpc('rswa_runtime_receipt')
    messages=build_transcription_messages(item['audio'],'unused');audios=process_audio_info(messages,sampling_rate=16000)
    request=dict(prompt=item['text'],multi_modal_data={'audio':(audios[0],16000)})
    lengths=[a.engineering_tokens] if a.engineering_tokens else item['lengths']
    results=[]
    for n in lengths:
        ids=item['gold_ids'][:n];assert len(ids)==n
        # Exact length warmup covers graph/kernel shapes before three measured repeats.
        repeats=1 if a.engineering_tokens else 3
        for repeat in range(-1,repeats):
            llm.llm_engine.reset_encoder_cache()
            assert llm.collective_rpc('rswa_encoder_cache_size')==[0]
            llm.collective_rpc('rswa_benchmark_begin',args=(ids,));llm.collective_rpc('rswa_reset_memory_stats')
            start=time.perf_counter()
            out=llm.generate([request],SamplingParams(temperature=0,repetition_penalty=1.02,
                             max_tokens=n,ignore_eos=True,seed=0),use_tqdm=False)[0]
            seconds=time.perf_counter()-start
            timing=llm.collective_rpc('rswa_benchmark_end')[0];memory=llm.collective_rpc('rswa_memory_stats')[0]
            assert list(out.outputs[0].token_ids)==ids and out.prompt_token_ids==item['prompt_ids']
            if repeat<0:continue
            metric=out.metrics;row=dict(tokens=n,repeat=repeat,seconds=seconds,
                prefill_seconds=metric.first_token_ts-metric.scheduled_ts,
                decode_seconds=metric.last_token_ts-metric.first_token_ts,
                timing=timing,memory=memory,encoder_cache_empty_before_request=True,
                trajectory_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                request_metrics=dataclasses.asdict(metric),request_id=out.request_id)
            results.append(row);write(a.run/'measurements.json',results)
            print(json.dumps({k:v for k,v in row.items() if k not in ['timing','memory','request_metrics']}),flush=True)
    write(a.run/'complete.json',dict(complete=True,window=a.window,sample=a.sample,weights=str(BASE),
        trajectory_manifest_sha256=sha(manifest),startup_seconds=startup,runtime=runtime,attention_backend=ATTENTION_BACKEND,
        quality_claim=False,engineering_only=bool(a.engineering_tokens),not_applicable=item['not_applicable'],
        summaries={str(n):{metric:summarize([r[metric] for r in results if r['tokens']==n])
                   for metric in ['seconds','prefill_seconds','decode_seconds']} for n in lengths}))

if __name__=='__main__':main()
