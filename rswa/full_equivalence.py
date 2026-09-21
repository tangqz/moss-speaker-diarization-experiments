"""Unchanged real-MOSS weights: native Full repeats vs a nonbinding R-SWA window."""
import argparse,json,sys,gc,time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from common import BASE,REPO,ROOT,write
from attention import MaskSpec
from model_adapter import install
sys.path.insert(0,str(REPO))
from finetune import DataCollator
from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--sample',required=True);a=ap.parse_args();a.run.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(0)
    row=json.loads((ROOT/f'dkucc/rswa_20260920/data/{a.sample}.jsonl').read_text().splitlines()[0])
    processor=MossTranscribeDiarizeProcessor.from_pretrained(BASE,trust_remote_code=True,local_files_only=True)
    sample=dict(audio=row['audios'][0],prompt=row['messages'][0]['content'].replace('<audio>','').strip(),
                target=row['messages'][1]['content'].strip())
    batch=DataCollator(processor,131072)([sample]);labels=batch.pop('labels').cuda()
    n=labels.shape[1];p=int(torch.where(labels[0]!=-100)[0][0]);count=n-p
    batch={k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
    model=AutoModelForCausalLM.from_pretrained(BASE,trust_remote_code=True,local_files_only=True,
          dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
    model.requires_grad_(True)
    baseline={};custom_full={};records=[]
    for mode in ['native','native_repeat','custom_full','nonbinding_window']:
        if mode=='custom_full':install(model,None)
        if mode=='nonbinding_window':model.config.moss_rswa['window']=n
        model.zero_grad(set_to_none=True);torch.manual_seed(0);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        # CPU activation offload changes storage only. Ordinary CE covers every
        # original target and EOS; there is no optimizer update in this check.
        # Pinned save_on_cpu in this frozen PyTorch allocates contiguous CPU
        # tensors and loses FlexAttention's saved output strides. The unpinned
        # native path uses tensor.cpu(), preserving the strided layout.
        with torch.autograd.graph.save_on_cpu(pin_memory=False),torch.autocast('cuda',dtype=torch.bfloat16):
            kwargs=dict(batch,use_cache=False)
            if mode in ['custom_full','nonbinding_window']:
                kwargs.update(attention_mask={'full_attention':MaskSpec(p,n,None if mode=='custom_full' else n)},
                              position_ids=torch.arange(n,device='cuda')[None,:])
            hidden=model.model(**kwargs).last_hidden_state
            losses=[];selected=[]
            for lo in range(p-1,n-1,512):
                hi=min(lo+512,n-1);logits=model.lm_head(hidden[:,lo:hi,:]).float()
                losses.append(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels[:,lo+1:hi+1].reshape(-1),reduction='sum')/count)
                if lo==p-1:selected.append(logits[0,0].detach().cpu())
                if hi==n-1:selected.append(logits[0,-1].detach().cpu())
            loss=sum(losses)
        loss.backward()
        norms={};diff=0.;scale=0.;dot=0.;current=0.;maximum=0.;custom_diff=0.;custom_scale=0.
        for name,param in model.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(),name
            grad=param.grad.detach().cpu()
            if mode=='native':baseline[name]=grad
            else:
                ref=baseline[name];delta=grad-ref
                diff+=float(delta.double().square().sum());scale+=float(ref.double().square().sum())
                dot+=float((grad.double()*ref.double()).sum());current+=float(grad.double().square().sum())
                maximum=max(maximum,float(delta.abs().max()))
            if mode=='custom_full':custom_full[name]=grad
            if mode=='nonbinding_window':
                custom_diff+=float((grad-custom_full[name]).double().square().sum())
                custom_scale+=float(custom_full[name].double().square().sum())
            norms[name]=float(grad.norm());del grad
        record=dict(mode=mode,sample=a.sample,prefix=p,length=n,target_tokens=count,loss=float(loss.detach()),
                    gradient_norms=norms,gradient_relative_l2=(diff/max(scale,1e-30))**.5,
                    gradient_cosine=dot/max((scale*current)**.5,1e-30) if mode!='native' else 1.,
                    max_gradient_difference=maximum,seconds=time.monotonic()-start,
                    nonbinding_vs_custom_full_gradient_relative_l2=(custom_diff/max(custom_scale,1e-30))**.5,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        records.append(record);torch.save(torch.stack(selected),a.run/f'{mode}-logits.pt')
        write(a.run/'equivalence.json',dict(records=records,requires_numerical_review=True))
        print(json.dumps({k:v for k,v in record.items() if k!='gradient_norms'}),flush=True)
        del hidden,logits,loss,losses;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
