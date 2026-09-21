"""Build a separate frozen second-round campaign without modifying round-one source."""
import collections, hashlib, json, py_compile, shutil
from pathlib import Path
ROOT=Path('/work/qt28/moss')
OLD=ROOT/'results/moss-rswa-20260920'
NEW=ROOT/'results/moss-rswa-r2-20260921'
TASK=ROOT/'dkucc/rswa_20260920'
STAGING=ROOT/'dkucc/rswa_round2_staging'
def read(p):return json.loads(p.read_text())
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def edit(src,name,old,new):
    p=src/name;t=p.read_text();assert old in t,(name,old);p.write_text(t.replace(old,new))
def main():
    assert not NEW.exists(),'Never overwrite a started campaign'
    NEW.mkdir();source=NEW/'source';source.mkdir();data=NEW/'data';data.mkdir()
    for p in (OLD/'source').iterdir():
        if p.is_file() and p.name!='source_manifest.json':shutil.copy2(p,source/p.name)
    for name in ['round2.py','round2.slurm']:shutil.copy2(STAGING/name,source/name)
    for name in ['train.jsonl','dev.jsonl','dev_inputs.json','references.json']:
        shutil.copy2(TASK/'data'/name,data/name)
    oldsent=read(TASK/'data/sentinel.json');excluded={r['key'] for r in oldsent['inputs']}
    inputs=read(data/'dev_inputs.json');chosen=[]
    for corpus in ['alimeeting','ami']:
        rows=sorted((r for r in inputs if r['dataset']==corpus and r['key'] not in excluded),key=lambda r:(r['duration'],r['key']))
        for quantile in [1/3,2/3]:chosen.append(rows[round((len(rows)-1)*quantile)])
    assert len(chosen)==len({r['key'] for r in chosen})==4 and not ({r['key'] for r in chosen}&excluded)
    write(data/'sentinel.json',dict(inputs=chosen,selection='exclude all old sentinel; per-corpus duration 1/3 and 2/3 quantiles',
        outputs_used=False,excluded_previous_sentinel=sorted(excluded),reference_features={r['key']:oldsent['reference_features'][r['key']] for r in chosen}))
    train=[json.loads(line) for line in (data/'train.jsonl').read_text().splitlines() if line]
    assert len(train)==536 and len({r['audios'][0] for r in train})==536
    test=[r for r in read(ROOT/'results/eval-62994-63363/inputs.json') if r['split']=='test'];assert len(test)==56
    write(data/'test_inputs.json',test)
    edit(source,'prepare_eval.py','import argparse,math,json','import argparse,math,json,os')
    edit(source,'prepare_eval.py',"TASK=ROOT/'dkucc/rswa_20260920'","TASK=Path(__file__).resolve().parent.parent")
    edit(source,'prepare_eval.py',"inputs=[r for r in read(OLD_EVAL/'inputs.json') if r['split']=='test']","inputs=read(TASK/'data/test_inputs.json')")
    edit(source,'prepare_eval.py','loads=[0.]*4',"gpu_count=len(os.environ.get('CUDA_VISIBLE_DEVICES','0,1,2,3').split(','))\n    loads=[0.]*gpu_count")
    edit(source,'prepare_eval.py','min(range(4),','min(range(gpu_count),')
    edit(source,'native_sft.sh','--learning_rate 1e-7','--learning_rate 1e-6')
    edit(source,'native_sft.sh','--max_steps 402','--max_steps 134')
    edit(source,'native_sft.sh','--save_steps 10','--save_steps 25')
    edit(source,'native_sft.sh','MOSS_RSWA_INPUT_RECEIPTS=/work/qt28/moss/dkucc/rswa_20260920/evidence/training_inputs',
        'MOSS_RSWA_INPUT_RECEIPTS=/work/qt28/moss/results/moss-rswa-r2-20260921/evidence/training_inputs')
    edit(source,'runtime_callback.py',"if state.global_step==5 or (state.global_step==1 and os.environ.get('MOSS_SAVE_STEP_ONE')=='1'):",
        "if state.global_step==1 and os.environ.get('MOSS_SAVE_STEP_ONE')=='1':")
    edit(source,'moss_vllm_rswa.py','assert window in (128,256),window','assert window in (128,256,512),window')
    edit(source,'sp_contract.py','n,p=512,173','n,p=768,173')
    edit(source,'sp_contract.py','for window in [128,256,None]:','for window in [512]:')
    edit(source,'vllm_contract.py','for window in [None,128,256]:','for window in [512]:')
    edit(source,'vllm_contract.py','for window in [128,256]:','for window in [512]:')
    edit(source,'vllm_contract.py','(173,302,1),(173,700,1),(173,700,31)',
        '(173,685,1),(173,686,1),(173,1400,1),(173,1400,31)')
    edit(source,'inference_gate.py','511]','511,512,513]')
    edit(source,'assess.py',"items=read(run/'inputs.json');refs=read(run/'references.json')",
        "items=read(run/'inputs.json');refs=read(run/'references.json')\n    splits={r['split'] for r in items};assert len(splits)==1\n    split=next(iter(splits))")
    edit(source,'assess.py',"{c}/dev'","{c}/{split}'")
    # Reuse the completed matched Base Dev outputs, retaining their original receipts.
    shutil.copytree(OLD/'dev/base',NEW/'dev/base')
    write(NEW/'dev/base_reuse.json',dict(source=str(OLD/'dev/base'),original_complete=read(OLD/'dev/base/evaluation_complete.json'),
        reason='Same Base, Full backend, input manifest, penalty and scoring; only allowed sliding-window sizes expanded',
        predictions_sha256={str(p.relative_to(OLD/'dev/base')):sha(p) for p in (OLD/'dev/base/predictions/base').rglob('*.json')}))
    for p in source.glob('*.py'):py_compile.compile(str(p),doraise=True)
    manifest={p.name:sha(p) for p in sorted(source.iterdir()) if p.is_file()}
    write(source/'source_manifest.json',manifest)
    write(NEW/'data_manifest.json',{p.name:sha(p) for p in sorted(data.iterdir())})
    write(NEW/'protocol.json',dict(round=2,parent_round=str(OLD),windows=[256,512],initialization='same original Base; optimizer and scheduler reset',
        base=str(ROOT/'models/MOSS-Transcribe-Diarize'),learning_rate=1e-6,lr_scheduler='linear over all 134 steps; no warmup',
        epochs=1,meetings=536,effective_batch=4,updates=134,seed=0,full_parameter=True,loss='teacher-forced CE',
        checkpoint_steps=[25,50,75,100,125,134],engineering_step1=True,sentinel_steps=[25,75],
        sentinel_size=4,sentinel_keys=[r['key'] for r in chosen],full_dev_steps=[50,100,125],full_dev_size=26,
        full_dev_includes_sentinel=True,test_size=56,test_checkpoint=134,penalty=1.02,
        safety_stop='engineering errors, nonfinite gradients or resource failure; no quality-based extension/early-stop rule',
        test_interpretation='established Test set reused for final engineering comparison, not a new blind Test',
        reuse_original_qualification=str(OLD/'qualification.json'),
        r512_delta_qualification='must pass before training: SP4 forward/backward, kernel, eviction, penalty, 1026-token real smoke alignment',
        source_manifest_sha256=sha(source/'source_manifest.json')))
    # Estimate wall time using completed first-epoch timing, not the short sentinel.
    updates=[json.loads(line) for line in (OLD/'training/256-seed0/update_metrics.rank0.jsonl').read_text().splitlines()]
    updates={r['step']:r for r in updates if r['step']<=134}
    write(NEW/'timing_reference.json',dict(steps=len(updates),train_hours=sum(r['seconds'] for r in updates.values())/3600,
        mean_seconds_per_update=sum(r['seconds'] for r in updates.values())/len(updates),
        source=str(OLD/'training/256-seed0/update_metrics.rank0.jsonl')))
    print(json.dumps(dict(root=str(NEW),sentinel=[(r['key'],r['duration']) for r in chosen],
        test_corpora=dict(collections.Counter(r['dataset'] for r in test)),train_hours=read(NEW/'timing_reference.json')['train_hours'])))

if __name__=='__main__':main()
