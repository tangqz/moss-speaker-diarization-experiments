"""Compact live status or a verified local snapshot of all v2 logs/diagnostics.

Weights, optimizer/RNG binaries, original audio, and Python bytecode stay remote.
Training/evaluation text, token statistics, inputs, sources and receipts are kept.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import tarfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import ssh_moss


REMOTE = r'''
import hashlib,io,json,sys,tarfile,subprocess
from pathlib import Path
mode,run_text,job_text=sys.argv[1:]
run=Path(run_text).resolve();job=int(job_text)
root=Path('/work/qt28/moss').resolve()
assert run.is_relative_to(root/'results') and run.name.startswith('full-attention-v2-')
checkpoints=root/'checkpoints'/('full-attention-v2-'+str(job))
accounting=subprocess.run(['sacct','-j',str(job),'--format=JobID,State,ExitCode,Elapsed,NodeList','-n'],capture_output=True,text=True,check=True).stdout
if mode=='status':
    result={'job':job,'run':str(run),'slurm':accounting}
    for name in ['status.json','outcome.json','failure.json','entry_gates_passed.json','job_exit.json','selection.json','actual_data_order_verified.json']:
        if (run/name).exists():result[name]=json.loads((run/name).read_text())
    if (run/'active_trial.json').exists():
        active=json.loads((run/'active_trial.json').read_text())
        result['active_trial.json']=active
        candidate=Path(active['checkpoints']).resolve()
        assert candidate.is_relative_to(checkpoints.resolve())
        checkpoints=candidate
    if (run/'baseline_verified.json').exists():
        result['baseline_verified.json']=json.loads((run/'baseline_verified.json').read_text())
    if checkpoints.exists():
        result['ranks']=[json.loads(p.read_text()) for p in sorted(checkpoints.glob('rank-*-status.json'))]
        for row in result['ranks']:row.pop('updates',None)
    print(json.dumps(result))
else:
    allowed={'.json','.jsonl','.log','.out','.err','.txt','.csv','.md','.py','.slurm','.npz','.sha256','.diff'}
    files={}
    for label,path in [('run',run),('checkpoints',checkpoints)]:
        if path.exists():
            for p in path.rglob('*'):
                if p.is_file() and not p.is_symlink() and (p.suffix in allowed or
                    (p.suffix=='.pt' and p.name.startswith('gradients-step-'))):
                    files[label+'/'+p.relative_to(path).as_posix()]=p
    for prefix in ['moss-fa-v2','moss-v2-gates','moss-v2-repeat','moss-v2-audit','moss-v2-diag','moss-v2-early','moss-v2-preproc','moss-v2-prod','moss-v2-lr2e6','moss-dev5-lr']:
        for suffix in ['out','err']:
            p=root/'logs'/f'{prefix}-{job}.{suffix}'
            if p.exists():files['batch/'+p.name]=p
    manifest={}
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
        def add(name,data):
            info=tarfile.TarInfo(name);info.size=len(data)
            archive.addfile(info,io.BytesIO(data))
        for name,p in sorted(files.items()):
            data=p.read_bytes()
            manifest[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            add(name,data)
        data=accounting.encode();add('slurm-accounting.txt',data)
        manifest['slurm-accounting.txt']={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
        add('snapshot-manifest.json',json.dumps(manifest,indent=2).encode())
'''


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['status','download'])
    ap.add_argument('--run')
    ap.add_argument('--job',type=int)
    a=ap.parse_args()
    if not a.run:
        current=json.loads((HERE/'current_run.json').read_text(encoding='utf-8'))
        a.run=current['remote_run'];a.job=current['job']
    assert a.job
    ssh_moss.SSH[ssh_moss.SSH.index('-S')+1]='C:/Users/qizhi/.ssh/cm-moss-check-20260912'
    command='/dkucc/home/qt28/envs/moss312/bin/python - '+shlex.quote(a.mode)+' '+shlex.quote(a.run)+' '+str(a.job)
    output=ssh_moss.run(command,REMOTE.encode(),capture=True).stdout
    if a.mode=='status':
        result=json.loads(output)
        formal_job=json.loads((HERE/'current_run.json').read_text(encoding='utf-8'))['job']
        status_path=(HERE/'latest_status.json' if a.job==formal_job else
                     HERE/'diagnostics'/f'latest_status-{a.job}.json')
        status_path.parent.mkdir(parents=True,exist_ok=True)
        status_path.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result))
        return
    dest=HERE/'artifacts'/str(a.job)
    dest.mkdir(parents=True,exist_ok=True)
    archive=dest/f'logs-{a.job}.tar.gz'
    archive.write_bytes(output)
    extraction=dest/'raw';extraction.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(output),mode='r:gz') as tar:
        manifest=json.load(tar.extractfile('snapshot-manifest.json'))
        for item in tar:
            path=(extraction/item.name).resolve()
            assert path.is_relative_to(extraction.resolve()) and item.isfile(),item.name
            data=tar.extractfile(item).read()
            if item.name!='snapshot-manifest.json':
                assert hashlib.sha256(data).hexdigest()==manifest[item.name]['sha256']
                assert len(data)==manifest[item.name]['bytes']
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    result=dict(job=a.job,archive=str(archive),files_verified=len(manifest),bytes=len(output),
                sha256=hashlib.sha256(output).hexdigest(),complete_snapshot=True,
                note='A verified snapshot; training completion is determined by outcome and Slurm state separately.')
    (dest/'download-verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':main()
