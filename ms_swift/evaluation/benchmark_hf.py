"""Small HF control for the vLLM loading gate; never contributes dev scores."""
import argparse
from pathlib import Path
import sys
import time
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from common import BASE, REPO, read, write

sys.path.insert(0,str(REPO))
from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    start=time.perf_counter()
    processor=AutoProcessor.from_pretrained(BASE,trust_remote_code=True,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(BASE,trust_remote_code=True,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    startup=time.perf_counter()-start
    results=[]
    for item in read(args.run/'inputs.json'):
        start=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            inputs=prepare_inputs(processor,build_transcription_messages(item['audio'],item['prompt']),
                                  device=torch.device('cuda')).to('cuda')
            assert inputs['input_ids'].shape[-1]==item['prompt_len']
            torch.cuda.synchronize()
            gen_start=time.perf_counter()
            output=model.generate(**inputs,max_new_tokens=128,do_sample=False,num_beams=1,
                repetition_penalty=1,no_repeat_ngram_size=0,logits_to_keep=1,
                eos_token_id=151645,pad_token_id=151643,use_cache=True)
            torch.cuda.synchronize()
            seconds=time.perf_counter()-gen_start
            ids=output[0,item['prompt_len']:].tolist()
        results.append(dict(key=item['key'],generated_tokens=len(ids),generated_ids=ids,
            generation_seconds=seconds,e2e_seconds=time.perf_counter()-start,
            preprocessing_seconds=gen_start-start,startup_seconds=startup))
        del inputs,output
        torch.cuda.empty_cache()
    write(args.run/'hf_128_token_benchmark.json',dict(records=results,
        qualification='Same GPU allocation, cold first request; 128-token latency only, not full dev speedup'))


if __name__=='__main__':
    main()
