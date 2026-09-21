"""One complete evaluation stage; checkpoint reuse requires exact manifests."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from common import ROOT,BASE,read,write,sha
from prepare_eval import prepare
from export_vllm import export
from protocol import ATTENTION_BACKEND
HERE=Path(__file__).resolve().parent
VENV=ROOT/'envs/vllm-moss-20260914/bin/python'
HENV=ROOT/'envs/ms-swift-20260914/bin/python'
METRIC=ROOT/'dkucc/evaluation/metrics-env/bin/python'

def workers(script,python,arguments,run,tag):
    ranks=sorted({r['rank'] for r in read(run/'inputs.json')})
    procs=[];streams=[]
    try:
        for rank in ranks:
            stream=(run/f'{tag}-rank{rank}.log').open('a');streams.append(stream)
            proc=subprocess.Popen([str(python),str(HERE/script),*map(str,arguments),'--rank',str(rank)],
                                   stdout=stream,stderr=subprocess.STDOUT)
            procs.append(proc)
        while any(p.poll() is None for p in procs):
            failed=[p.returncode for p in procs if p.poll() not in (None,0)]
            if failed:
                for p in procs:
                    if p.poll() is None:p.terminate()
                raise RuntimeError(f'{tag} worker failed: {failed}; see {run}')
            print(json.dumps(dict(event='evaluation_running',tag=tag,run=str(run),
                  finished=sum(p.poll()==0 for p in procs),total=len(procs))),flush=True)
            time.sleep(30)
        assert all(p.returncode==0 for p in procs),[(p.pid,p.returncode) for p in procs]
    finally:
        for stream in streams:stream.close()

def evaluate(run,checkpoint,window,base_run=None,sentinel=False,split='dev',ablation=False):
    run=Path(run);checkpoint=Path(checkpoint)
    complete=run/'evaluation_complete.json'
    if complete.exists():
        old=read(complete)
        assert old['checkpoint']==str(checkpoint) and old['window']==window
        assert old['checkpoint_config_sha256']==sha(checkpoint/'config.json')
        assert old['penalty']==1.02 and old['penalty_scope']=='all_generated_tokens_only'
        assert old['attention_backend']==ATTENTION_BACKEND
        assert old['source_manifest_sha256']==sha(HERE/'source_manifest.json'),'Evaluation source changed; make a new run'
        return old
    prepare(run,checkpoint,window,split=split,sentinel=sentinel)
    native=checkpoint
    if checkpoint!=BASE:
        target=run/'inference_model'
        if not target.exists():export(checkpoint,target)
        receipt=read(target/'export_receipt.json');assert receipt['checkpoint']==str(checkpoint.resolve())
        checkpoint=target
    label='base' if base_run is None else 'sft'
    args=['--run',run,'--checkpoint',checkpoint,'--window',window,'--model-label',label]
    if ablation:args.append('--allow-inference-ablation')
    items=read(run/'inputs.json')
    same_base=base_run is not None and native==BASE and window=='full'
    if same_base:
        for item in items:
            pred=read(Path(base_run)/'predictions/base'/f"{item['key']}.json")
            assert pred['decoding']['repetition_penalty']==1.02
            write(run/'predictions/sft'/f"{item['key']}.json",dict(pred,model='sft'))
            write(run/'input_receipts'/f"{item['key']}.json",
                  read(Path(base_run)/'input_receipts'/f"{item['key']}.json"))
    else:workers('evaluate_generate.py',VENV,args,run,'generate')
    for item in items:
        source=(Path(base_run) if base_run else run)/'predictions/base'/f"{item['key']}.json"
        pred=read(source)
        assert pred['decoding']['repetition_penalty']==1.02
        assert pred['decoding']['penalty_scope']=='all_generated_tokens_only'
        if base_run:dest=run/'predictions/base'/f"{item['key']}.json"
        else:
            dest=run/'predictions/sft'/f"{item['key']}.json";pred=dict(pred,model='sft')
        write(dest,pred)
    if split=='dev' and not sentinel:
        workers('evaluate_ce.py',HENV,['--run',run,'--checkpoint',checkpoint,'--window',window],run,'teacher_ce')
    subprocess.run([str(METRIC),str(HERE/'score_v1.py'),'--run',str(run)],check=True)
    result=dict(checkpoint=str(native),window=window,checkpoint_config_sha256=sha(native/'config.json'),
                penalty=1.02,penalty_scope='all_generated_tokens_only',split=split,sentinel=sentinel,attention_backend=ATTENTION_BACKEND,
                source_manifest_sha256=sha(HERE/'source_manifest.json'),complete=True)
    write(complete,result);return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--window',required=True)
    ap.add_argument('--base-run',type=Path);ap.add_argument('--sentinel',action='store_true')
    ap.add_argument('--split',default='dev',choices=['dev','test']);ap.add_argument('--ablation',action='store_true')
    a=ap.parse_args();evaluate(a.run,a.checkpoint,a.window,a.base_run,a.sentinel,a.split,a.ablation)
