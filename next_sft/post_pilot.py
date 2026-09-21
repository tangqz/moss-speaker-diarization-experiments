"""Evaluate both short pilots through the SAME BF16 inference storage path."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch
import probes as p

ap=argparse.ArgumentParser()
ap.add_argument('--pilot-run',type=Path,required=True)
ap.add_argument('--run',type=Path,required=True)
a=ap.parse_args()
torch.cuda.set_device(0)
p.DEVICE=torch.device('cuda',0)
torch.set_num_threads(4)
processor=p.MossTranscribeDiarizeProcessor.from_pretrained(p.BASE,trust_remote_code=True,local_files_only=True)
vals=p.get_val(processor)
result=dict(pilot_run=str(a.pilot_run),inference_parameter_dtype='bfloat16',models=[])
for label in ['bf16','fp32']:
    m=p.load(a.pilot_run/f'pilot-{label}-checkpoint',torch.bfloat16)
    p.install_memory_forward(m,512,offload_backbone=False)
    item=dict(training_precision=label,validation=p.eval_loss(m,vals))
    m.eval()
    # A bounded natural continuation after the IDENTICAL known dev history.
    # This is not a full-meeting performance score.
    pred=json.loads((p.EVAL/'predictions/sft/alimeeting/dev/R8001_M8004.json').read_text())
    start,count,c=p.longest_run(pred['generated_ids'])
    prompt=p.prepare_inputs(processor,p.build_transcription_messages(pred['audio'],pred['prompt']),
        max_length=131072,device=p.DEVICE).to(p.DEVICE)
    prefixes=[]
    with torch.inference_mode(),p.amp():
        for k in [0,8,32]:
            inputs=p.with_history(prompt,pred['generated_ids'][:start]+[c]*k)
            outputs=m.generate(**inputs,max_new_tokens=64,do_sample=False,num_beams=1,
                use_cache=True,logits_to_keep=1,eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id)
            ids=outputs[0,inputs['input_ids'].shape[-1]:].tolist()
            prefixes.append(dict(k=k,new_ids=ids,new_text=processor.tokenizer.decode(ids,skip_special_tokens=False),
                longest_identical_run=p.longest_run(ids)[1],continued_token_count=sum(t==c for t in ids),
                ended_eos=bool(ids and ids[-1]==processor.tokenizer.eos_token_id)))
            del inputs,outputs
    item['bounded_continuation']=dict(key=pred['key'],max_new_tokens=64,prefixes=prefixes,
        limitation='Off-policy known failure prefix; escaping this single-token loop is not proof of audio-grounded recovery.')
    result['models'].append(item)
    p.write(a.run/'post-pilot.json',result)
    p.emit('post_pilot_model',training_precision=label,validation=item['validation'])
    del m,prompt
    gc.collect();torch.cuda.empty_cache()
result['status']='completed'
p.write(a.run/'post-pilot.json',result)
