"""Full-audio teacher forcing and separate fixed dev failure-prefix probes."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT, BASE, HERE, REPO, OLD_EVAL, read, write, emit, now
from diagnostics import TYPES, target_types, longest_run
sys.path.insert(0,str(REPO))
from finetune import ConversationDataset, DataCollator
from transformers import AutoModelForCausalLM, AutoProcessor
from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor
from moss_transcribe_diarize.inference_utils import build_transcription_messages, prepare_inputs


def distribution(z, target):
    z=z.float()
    logz=torch.logsumexp(z,-1)
    nll=logz-z.gather(-1,target[:,None]).squeeze(-1)
    entropy=logz-(z.softmax(-1)*z).sum(-1)
    return nll,entropy


def teacher(model, processor, item, row, output, device):
    start=time.perf_counter()
    batch=DataCollator(processor,131072)([row])
    labels=batch.pop('labels')
    j=labels[0].ne(-100).nonzero().flatten()
    ids=labels[0,j]
    target=row['target']+processor.tokenizer.eos_token
    encoded=processor.tokenizer(target,add_special_tokens=False,return_offsets_mapping=True)
    assert encoded['input_ids']==ids.tolist(),item['key']
    types=np.asarray(target_types(target,encoded['offset_mapping'],encoded['input_ids'],151645),dtype=np.uint8)
    assert int(j[0])==item['prompt_len']
    assert ids[-1]==151645 and len(batch['input_ids'][0])<=131072
    inputs={k:v.to(device) for k,v in batch.items()}
    torch.cuda.reset_peak_memory_stats(device)
    nlls,entropies=[],[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        hidden=model.model(**inputs,use_cache=False,return_dict=True).last_hidden_state[0]
        positions=(j-1).to(device)
        for k in range(0,len(j),512):
            z=F.linear(hidden[positions[k:k+512]],model.lm_head.weight)
            nll,entropy=distribution(z,ids[k:k+512].to(device))
            assert bool(torch.isfinite(nll).all() & torch.isfinite(entropy).all())
            nlls.append(nll.cpu());entropies.append(entropy.cpu())
            del z,nll,entropy
    nll=np.concatenate([x.numpy() for x in nlls]);entropy=np.concatenate([x.numpy() for x in entropies])
    destination=output/'teacher'/item['key']
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(str(destination)+'.npz',target_ids=ids.numpy(),target_positions=j.numpy(),
        query_positions=(j-1).numpy(),token_types=types,nll=nll,entropy_nats=entropy,
        character_offsets=np.asarray(encoded['offset_mapping'],dtype=np.int32))
    by_type={name:dict(tokens=int((types==i).sum()),nll_sum=float(nll[types==i].sum(dtype=np.float64)),
        entropy_sum=float(entropy[types==i].sum(dtype=np.float64))) for i,name in enumerate(TYPES)}
    result=dict(key=item['key'],dataset=item['dataset'],supervised_tokens=len(ids),
        prompt_len=item['prompt_len'],sequence_tokens=len(batch['input_ids'][0]),
        loss=float(nll.mean(dtype=np.float64)),nll_sum=float(nll.sum(dtype=np.float64)),
        entropy_nats=float(entropy.mean(dtype=np.float64)),by_type=by_type,
        seconds=time.perf_counter()-start,peak_gib=torch.cuda.max_memory_allocated(device)/2**30,
        dtype='BF16_parameters_autocast_FP32_full_vocabulary_reduction',type_names=TYPES,
        offset_contract='Original target character offsets; target j is predicted by query j-1')
    write(str(destination)+'.json',result)
    emit('teacher_complete',**{k:v for k,v in result.items() if k not in ['by_type']})
    del hidden,inputs,batch
    gc.collect();torch.cuda.empty_cache()
    return result


def prefix(model, processor, key, output, device):
    pred=read(OLD_EVAL/'predictions/sft'/f'{key}.json')
    start,count,c=longest_run(pred['generated_ids'])
    prompt=prepare_inputs(processor,build_transcription_messages(pred['audio'],pred['prompt']),
                          max_length=131072,device=device).to(device)
    ids=torch.cat([prompt['input_ids'],torch.tensor([pred['generated_ids'][:start]],device=device)],-1)
    inputs=dict(prompt,input_ids=ids,attention_mask=torch.ones_like(ids))
    records=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        out=model(**inputs,use_cache=True,logits_to_keep=1)
        past=out.past_key_values
        for k in range(33):
            if k in [0,1,2,4,8,16,32]:
                z=out.logits[0,-1].float()
                p=z.softmax(-1)
                other=z.clone();other[c]=-torch.inf
                nll,entropy=distribution(z[None,:],torch.tensor([c],device=device))
                top=z.topk(5)
                records.append(dict(k=k,probability=float(p[c]),rank=int((z>z[c]).sum())+1,
                    margin=float(z[c]-other.max()),entropy_nats=float(entropy),
                    eos_probability=float(p[151645]),top_ids=top.indices.tolist()))
            if k<32:
                length=ids.shape[-1]+k+1
                out=model(input_ids=torch.tensor([[c]],device=device),
                    attention_mask=torch.ones((1,length),dtype=torch.long,device=device),
                    position_ids=torch.tensor([[length-1]],device=device),past_key_values=past,
                    use_cache=True,logits_to_keep=1)
                past=out.past_key_values
    result=dict(key=key,history_tokens=start,repeated_token=c,original_run=count,records=records,
                limitation='Known dev off-policy prefix; does not measure natural failure prevalence or audio grounding.')
    write(output/'prefix'/f'{key}.json',result)
    emit('prefix_complete',key=key)
    del model,out,past,prompt,inputs
    gc.collect();torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--rank',type=int,required=True)
    a=ap.parse_args()
    torch.cuda.set_device(a.rank);torch.set_num_threads(4);torch.manual_seed(0)
    device=torch.device('cuda',a.rank)
    a.run.mkdir(parents=True,exist_ok=True)
    model=AutoModelForCausalLM.from_pretrained(a.checkpoint,trust_remote_code=True,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to(device).eval()
    processor=MossTranscribeDiarizeProcessor.from_pretrained(BASE,trust_remote_code=True,local_files_only=True)
    rows=[]
    for filename in ['ami_dev.jsonl','alimeeting_dev.jsonl']:
        rows.extend(ConversationDataset(str(ROOT/'data/moss_jsonl'/filename)).samples)
    items=read(a.run/'inputs.json')
    for item in items:
        if item['rank']==a.rank:
            teacher(model,processor,item,next(r for r in rows if r['audio']==item['audio']),a.run,device)
    if a.rank<2:
        inference_processor=AutoProcessor.from_pretrained(BASE,trust_remote_code=True,local_files_only=True)
        key=['alimeeting/dev/R8001_M8004','alimeeting/dev/R8003_M8001'][a.rank]
        prefix(model,inference_processor,key,a.run,device)
    write(a.run/f'fast-rank-{a.rank}-complete.json',dict(complete=True,utc=now()))


if __name__=='__main__':main()
