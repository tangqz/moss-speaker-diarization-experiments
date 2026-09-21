"""Teacher-forced CE with bounded logits workspace and unchanged targets."""
import argparse,json,os,sys,time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from attention import MaskSpec
from model_adapter import install
from common import BASE,REPO,read,write
sys.path.insert(0,str(REPO))
from finetune import DataCollator
from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor
from input_receipt import receipt as input_receipt

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--window',required=True)
    ap.add_argument('--rank',type=int,required=True);ap.add_argument('--verify',action='store_true');a=ap.parse_args()
    torch.cuda.set_device(a.rank);torch.set_num_threads(4)
    window=None if a.window=='full' else int(a.window)
    model=AutoModelForCausalLM.from_pretrained(a.checkpoint,trust_remote_code=True,local_files_only=True,
          dtype=torch.float32,attn_implementation='sdpa').to(f'cuda:{a.rank}').eval()
    install(model,window)
    processor=MossTranscribeDiarizeProcessor.from_pretrained(BASE,trust_remote_code=True,local_files_only=True)
    dataset=read(a.run/'inputs.json');refs=read(a.run/'references.json')
    manifest=[json.loads(x) for x in Path('/work/qt28/moss/dkucc/rswa_20260920/data/dev.jsonl').read_text().splitlines() if x]
    rows={str(Path(r['audios'][0]).resolve()):r for r in manifest}
    collator=DataCollator(processor,131072)
    for item in dataset:
        if item['rank']!=a.rank:continue
        row=rows[str(Path(item['audio']).resolve())]
        sample=dict(audio=row['audios'][0],prompt=row['messages'][0]['content'].replace('<audio>','').strip(),
                    target=row['messages'][1]['content'].strip())
        batch=collator([sample]);labels=batch.pop('labels').to(f'cuda:{a.rank}')
        n=labels.shape[1];prefix=int(torch.where(labels[0]!=-100)[0][0])
        assert prefix==item['prompt_len'] and int(labels[0,-1])==processor.tokenizer.eos_token_id
        observed=input_receipt(processor,batch['input_ids'][0,:prefix].tolist(),
                               batch['audio_feature_lengths'],batch['audio_chunk_mapping'])
        inference=read(a.run/'input_receipts'/f"{item['key']}.json")
        assert all(inference[k]==v for k,v in observed.items()),'Training/inference processor input mismatch'
        expected=processor.tokenizer(sample['target']+processor.tokenizer.eos_token,add_special_tokens=False)['input_ids']
        assert labels[0,prefix:].tolist()==expected,'Truncated CE target'
        batch={k:v.to(f'cuda:{a.rank}') if torch.is_tensor(v) else v for k,v in batch.items()}
        start=time.monotonic();total=0.;count=0
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            backbone=dict(batch,attention_mask={'full_attention':MaskSpec(prefix,n,window)},
                          position_ids=torch.arange(n,device=labels.device)[None,:],use_cache=False)
            hidden=model.model(**backbone).last_hidden_state
            for lo in range(prefix-1,n-1,512):
                hi=min(lo+512,n-1)
                logits=model.lm_head(hidden[:,lo:hi,:]).float()
                target=labels[:,lo+1:hi+1]
                total+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),reduction='sum'))
                count+=int(target.ne(-100).sum())
            if a.verify:
                # Keep all teacher-forced inputs and all targets, but avoid
                # materializing vocabulary logits for ignored prompt labels.
                direct=model(**batch,labels=labels[:,prefix-1:],logits_to_keep=n-prefix+1,
                             use_cache=False,rswa_prefix_length=prefix,rswa_valid_length=n).loss
                assert abs(float(direct)-total/count)<.005,(float(direct),total/count)
        result=dict(key=item['key'],window=window,target_tokens=count,loss_sum=total,
                    ce=total/count,seconds=time.monotonic()-start,penalty_applied=False,
                    verified_against_native_loss=a.verify,complete_target=True)
        result['training_inference_input_receipts_equal']=True
        write(a.run/'ce'/f"{item['key']}.json",result);print(json.dumps(result),flush=True)
        del hidden,batch,labels,logits
        torch.cuda.empty_cache()

if __name__=='__main__':main()
