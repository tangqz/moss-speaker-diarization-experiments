"""End-to-end test of real free generation, checkpoint export, CE and scoring."""
import argparse,subprocess,os
from pathlib import Path
from common import ROOT,BASE,read,write
from prepare_eval import prepare
from export_vllm import export
from run_eval import workers,HENV,VENV,METRIC,HERE
from protocol import ATTENTION_BACKEND

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
    subprocess.run([str(HENV),str(HERE/'dynamic_contract.py'),'--output',str(a.run/'dynamic_contract.json')],check=True)
    subprocess.run([str(HENV),'-m','torch.distributed.run','--standalone','--nproc_per_node=4',
                    str(HERE/'sp_contract.py'),'--output',str(a.run/'sp4_dynamic_contract.json')],check=True)
    case=a.run/'typical-full';gpus=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    common=['--window','full','--sample','typical','--tokens','2050','--run',str(case)]
    subprocess.run([str(VENV),str(HERE/'inference_gate.py'),'--backend','vllm',*common,
        '--attention-backend',ATTENTION_BACKEND],env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[0]),check=True)
    processes=[]
    for rank,native in enumerate([True,False]):
        arguments=[str(HENV),str(HERE/'inference_gate.py'),'--backend','hf',*common,'--record-only']
        if native:arguments.append('--native-full')
        processes.append(subprocess.Popen(arguments,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[rank])))
    codes=[p.wait() for p in processes];assert codes==[0,0],codes
    subprocess.run([str(HENV),str(HERE/'prepare_gate_samples.py')],check=True)
    env=dict(os.environ,NPROC_PER_NODE='4',MOSS_RSWA_WINDOW='128',MOSS_STOP_STEP='3')
    env.pop('MOSS_RESUME_AUDIT_CHECKPOINT',None);env.pop('MOSS_SAVE_STEP_ONE',None)
    subprocess.run(['bash',str(HERE/'native_sft.sh'),str(ROOT/'dkucc/rswa_20260920/data/varied.jsonl'),
        str(a.run/'varied_training'),'4','4','--fsdp',str(HERE/'fsdp1_offload.json'),
        '--gradient_checkpointing','false','--vit_gradient_checkpointing','false'],env=env,check=True)
    checkpoint=a.run/'varied_training/checkpoint-3'
    target=a.run/'inference_model';export(checkpoint,target)
    prepare(a.run,checkpoint,'128')
    all_items=read(a.run/'inputs.json')
    items=[]
    for rank,corpus in enumerate(['alimeeting','ami']):
        item=min((r for r in all_items if r['dataset']==corpus),key=lambda r:r['duration'])
        item['rank']=rank;items.append(item)
    write(a.run/'inputs.json',items)
    refs=read(a.run/'references.json');write(a.run/'references.json',{r['key']:refs[r['key']] for r in items})
    for label,path,window in [('base',BASE,'full'),('sft',target,'128')]:
        workers('evaluate_generate.py',VENV,['--run',a.run,'--checkpoint',path,'--window',window,'--model-label',label],a.run,label)
    workers('evaluate_ce.py',HENV,['--run',a.run,'--checkpoint',target,'--window','128','--verify'],a.run,'ce')
    subprocess.run([str(METRIC),str(HERE/'score_v1.py'),'--run',str(a.run)],check=True)
    write(a.run/'pilot.json',dict(passed=True,meetings=len(items),quality_claim=False,meaning='pipeline operation, not scientific result'))

if __name__=='__main__':main()
