"""Hash-verified upload and once-only submission of the authorized low-LR trial."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent.parent))
import ssh_moss

REMOTE=r'''
import ast,fcntl,hashlib,io,json,os,subprocess,sys,tarfile
from pathlib import Path
root=Path('/work/qt28/moss/dkucc/full_attention_v2_lr2e6_20260913')
root.mkdir(parents=True,exist_ok=True)
payload=sys.stdin.buffer.read()
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as archive:
    files={}
    for entry in archive:
        assert entry.isfile() and Path(entry.name).name==entry.name and entry.name not in ['.','..']
        files[entry.name]=archive.extractfile(entry).read()
manifest=json.loads(files['source_manifest.json'])
manifest_sha=hashlib.sha256(files['source_manifest.json']).hexdigest()
assert set(files)==set(manifest)|{'source_manifest.json'}
for name,digest in manifest.items():assert hashlib.sha256(files[name]).hexdigest()==digest,name
with (root/'submit.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    receipt=root/'submission.json'
    if receipt.exists():
        record=json.loads(receipt.read_text())
        assert record['source_manifest_sha256']==manifest_sha,'Submitted bundle differs; refusing overwrite or duplicate.'
    else:
        bundle=root/'bundle';bundle.mkdir(exist_ok=True)
        for name,data in files.items():
            destination=bundle/name
            if destination.exists():assert destination.read_bytes()==data,name
            else:destination.write_bytes(data)
            if name.endswith('.py'):ast.parse(data.decode('utf-8'),filename=name)
        subprocess.run(['bash','-n',str(bundle/'trial.slurm')],check=True)
        checked=subprocess.run([sys.executable,'-B',str(bundle/'trial_checks.py')],capture_output=True,text=True,check=True)
        (root/'pre_submit_checks.log').write_text(checked.stdout+checked.stderr)
        if sys.argv[1]=='upload':
            print(json.dumps(dict(uploaded=len(files),source_manifest_sha256=manifest_sha,checks=checked.stdout.strip())))
            sys.exit(0)
        # Slurm allocates all actual GPU computation; login performs only lightweight validation.
        value=subprocess.run(['sbatch','--parsable','--exclude=dkucc-core-gpu-dkurc-d-it09-3',
            str(bundle/'trial.slurm')],capture_output=True,text=True,check=True).stdout.strip()
        assert value.isdigit(),value
        job=int(value)
        record=dict(job=job,remote_run=f'/work/qt28/moss/results/full-attention-v2-{job}',
            checkpoints=f'/work/qt28/moss/checkpoints/full-attention-v2-{job}',
            mode='lr2e6_short_trial',learning_rate=2e-6,scheduler_horizon=402,max_executed_steps=50,
            phase_boundaries=[10,25,50],source_manifest_sha256=manifest_sha,
            source_training_job=63421,comparison_evaluation_job=63441,
            remote_bundle=str(bundle),control_path='C:/Users/qizhi/.ssh/cm-moss-check-20260912')
        tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2));tmp.replace(receipt)
    print(json.dumps(record))
'''


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['upload','submit']);a=ap.parse_args()
    bundle=HERE/'bundle';manifest=json.loads((bundle/'source_manifest.json').read_text(encoding='utf-8'))
    local=HERE/'current_run.json'
    if local.exists():
        record=json.loads(local.read_text(encoding='utf-8'))
        assert record['source_manifest_sha256']==hashlib.sha256((bundle/'source_manifest.json').read_bytes()).hexdigest()
        print(json.dumps(record));return
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name in sorted(set(manifest)|{'source_manifest.json'}):
            data=(bundle/name).read_bytes()
            if name in manifest:assert hashlib.sha256(data).hexdigest()==manifest[name],name
            info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
    ssh_moss.SSH[ssh_moss.SSH.index('-S')+1]='C:/Users/qizhi/.ssh/cm-moss-check-20260912'
    cmd='/dkucc/home/qt28/envs/moss312/bin/python -B -c '+shlex.quote(REMOTE)+' '+a.mode
    result=subprocess.run(ssh_moss.SSH+[cmd],input=buffer.getvalue(),capture_output=True,check=True,timeout=120)
    record=json.loads(result.stdout)
    if 'job' in record:local.write_text(json.dumps(record,indent=2),encoding='utf-8')
    else:(HERE/'upload_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record))


if __name__=='__main__':
    main()
