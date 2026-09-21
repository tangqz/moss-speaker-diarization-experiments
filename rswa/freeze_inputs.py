"""Record exact initialization, external registration, and frozen data assets."""
from pathlib import Path
from common import ROOT,BASE,REPO,sha,read,write
TASK=ROOT/'dkucc/rswa_20260920'

def main():
    paths=[p for p in BASE.iterdir() if p.is_file() and p.suffix in
           ['.json','.py','.safetensors','.jinja','.txt']]
    paths += [ROOT/'dkucc/ms_swift_20260914/moss_plugin.py']
    paths += [REPO/'finetune.py']+list((REPO/'moss_transcribe_diarize').rglob('*.py'))
    paths += [TASK/f'data/{name}' for name in ['train.jsonl','dev.jsonl','dev_inputs.json','sentinel.json','references.json']]
    values={str(p.resolve()):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)}
    dest=TASK/'evidence/run_contract.json'
    if dest.exists():assert read(dest)['files']==values,'Frozen experiment inputs changed'
    else:write(dest,dict(files=values,scope='base weight bytes, processor, tokenizer, external plugin and frozen splits'))
    # Preserve the already established receipt and additionally freeze assets
    # prepared later during the benchmark smoke test, before formal results.
    extra=[ROOT/'results/eval-62994-63363/inputs.json',TASK/'data/performance_trajectories.json']
    if all(p.exists() for p in extra):
        values={str(p.resolve()):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in extra}
        additional=TASK/'evidence/additional_inputs_contract.json'
        if additional.exists():assert read(additional)['files']==values,'Frozen Test or performance manifest changed'
        else:write(additional,dict(files=values,scope='existing Test manifest and fixed original gold performance histories'))
    print('Base weights and experiment assets hashed and frozen',flush=True)

if __name__=='__main__':main()
