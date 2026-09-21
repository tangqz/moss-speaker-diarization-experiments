"""Same full-meeting BF16 greedy inference and parser as evaluation 63363."""
import argparse
import copy
import dataclasses
import gc
import time
import traceback
from pathlib import Path
import sys
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from common import BASE, REPO, read, write, emit, now
from diagnostics import output_diagnostics
sys.path.insert(0,str(REPO))
from moss_transcribe_diarize import parse_transcript
from moss_transcribe_diarize.inference_utils import build_transcription_messages,prepare_inputs,ProgressStreamer


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--rank',type=int,required=True)
    a=ap.parse_args()
    torch.cuda.set_device(a.rank);torch.set_num_threads(4);torch.manual_seed(0)
    device=torch.device('cuda',a.rank)
    items=[r for r in read(a.run/'inputs.json') if r['rank']==a.rank]
    processor=AutoProcessor.from_pretrained(BASE,trust_remote_code=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(a.checkpoint,trust_remote_code=True,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    status=dict(rank=a.rank,gpu=torch.cuda.get_device_name(device),checkpoint=str(a.checkpoint))
    def progress(**changes):
        status.update(changes);status['updated_utc']=now()
        write(a.run/f'worker-{a.rank}-status.json',status)
        emit('generation_progress',**status)
    warm=prepare_inputs(processor,build_transcription_messages(BASE.parents[1]/'data/samples/jfk.wav'),device=device).to(device)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        model.generate(**warm,max_new_tokens=4,do_sample=False,num_beams=1,logits_to_keep=1,use_cache=True)
    del warm
    failures=0
    for item in items:
        dest=a.run/'predictions/sft'/f"{item['key']}.json"
        if dest.is_file() and read(dest).get('status')=='ok': continue
        inputs=outputs=generated=None
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
        free,total=torch.cuda.mem_get_info(device)
        start=time.perf_counter()
        progress(stage='preprocessing',key=item['key'],generated_tokens=0)
        record=dict(item,model='sft',model_path=str(a.checkpoint),rank=a.rank,gpu=status['gpu'],
            model_manifest=read(a.run/'model_manifest.json'),free_gpu_bytes_before=free,total_gpu_bytes=total,
            decoding=dict(do_sample=False,num_beams=1,repetition_penalty=1,no_repeat_ngram_size=0,
                max_new_tokens_cap=65536,max_context=131072,attention='sdpa',dtype='bfloat16',
                logits_to_keep=1,use_cache=True,eos_token_id=151645,pad_token_id=151643),started_utc=now())
        try:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                inputs=prepare_inputs(processor,build_transcription_messages(item['audio'],item['prompt']),
                    max_length=131072,device=device).to(device)
            record['preprocessing']=dict(context='production_cuda_bfloat16_autocast',
                input_feature_dtype=str(inputs['input_features'].dtype))
            prompt_len=int(inputs['attention_mask'].sum())
            assert prompt_len==item['prompt_len']
            max_new=min(65536,131072-prompt_len)
            config=copy.deepcopy(model.generation_config)
            for k,v in dict(do_sample=False,num_beams=1,max_new_tokens=max_new,use_cache=True,
                eos_token_id=151645,pad_token_id=151643,repetition_penalty=1.,no_repeat_ngram_size=0).items():
                setattr(config,k,v)
            torch.cuda.synchronize(device)
            gen_start=time.perf_counter();first=[]
            def token_event(n):
                if not first:first.append(time.perf_counter())
                if n==1 or n%256==0:
                    progress(stage='generating',generated_tokens=n,elapsed_seconds=time.perf_counter()-start)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                outputs=model.generate(**inputs,generation_config=config,logits_to_keep=1,
                                       streamer=ProgressStreamer(token_event))
            torch.cuda.synchronize(device)
            seconds=time.perf_counter()-gen_start
            ids=outputs[0,prompt_len:].tolist()
            raw=processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
            segments=[dataclasses.asdict(s) for s in parse_transcript(raw)]
            eos=bool(ids and ids[-1]==151645)
            e2e=time.perf_counter()-start
            record.update(status='ok',raw_text=raw,segments=segments,generated_ids=ids,
                generated_tokens=len(ids),generated_absolute_position_start=prompt_len,
                ended_eos=eos,truncated=not eos,max_new_tokens=max_new,parse_empty=not segments,
                e2e_seconds=e2e,preprocessing_seconds=gen_start-start,generation_seconds=seconds,
                first_token_seconds=first[0]-gen_start if first else None,rtf=e2e/item['duration'],
                generated_tokens_per_second=len(ids)/seconds,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
            record['diagnostics']=output_diagnostics(record)
        except Exception as exc:
            failures+=1
            record.update(status='error',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
            traceback.print_exc()
        record['finished_utc']=now()
        write(dest,record)
        progress(stage='record_saved',key=item['key'],outcome=record['status'])
        del inputs,outputs,generated
        gc.collect();torch.cuda.empty_cache()
    progress(stage='done',failures=failures)
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
