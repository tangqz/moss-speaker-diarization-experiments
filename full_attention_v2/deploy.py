"""Upload a versioned execution bundle and explicitly submit the requested job."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import ssh_moss


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['gates','qualify','audit','upload','train'])
    ap.add_argument('--gate-run')
    ap.add_argument('--exclude')
    ap.add_argument('--dependency',type=int)
    a=ap.parse_args()
    ssh_moss.SSH[ssh_moss.SSH.index('-S')+1]='C:/Users/qizhi/.ssh/cm-moss-check-20260912'
    copies={
        'memory_sft.py':HERE.parent/'memory_sft/memory_sft.py',
        'design_protocol.json':HERE.parent/'next_sft/design_protocol.json',
        'preflight.json':HERE.parent/'next_sft/inputs/preflight.json',
        'dev_inputs.json':HERE.parent/'next_sft/inputs/dev_inputs.json',
        'sentinel_manifest.json':HERE.parent/'next_sft/sentinel_manifest.json',
        'score_v1.py':HERE.parent/'evaluation/score.py',
        'infer_worker_v1.py':HERE.parent/'evaluation/infer_worker.py',
    }
    for name,source in copies.items():
        (HERE/name).write_bytes(source.read_bytes().replace(b'\r\n',b'\n'))
    files=sorted(p for p in HERE.iterdir() if p.suffix in ['.py','.json','.slurm'] and
                 p.name not in ['source_manifest.json','current_run.json','gate_run.json','latest_status.json'])
    hashes={p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in files}
    (HERE/'source_manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    remote='/work/qt28/moss/dkucc/full_attention_v2'
    ssh_moss.run('mkdir -p '+shlex.quote(remote))
    for file in files+[HERE/'source_manifest.json']:
        data=file.read_bytes().replace(b'\r\n',b'\n')
        dest=remote+'/'+file.name
        ssh_moss.run('cat > '+shlex.quote(dest),data)
        found=ssh_moss.run('sha256sum '+shlex.quote(dest),capture=True).stdout.decode().split()[0]
        assert found==hashlib.sha256(data).hexdigest(),file.name
    if a.mode=='upload':
        print(json.dumps(dict(uploaded=len(files)+1,remote=remote)))
        return
    script={'gates':'gates.slurm','qualify':'qualify.slurm','audit':'audit.slurm','train':'train.slurm'}[a.mode]
    command='sbatch --parsable '
    if a.exclude:command+='--exclude='+shlex.quote(a.exclude)+' '
    if a.dependency:command+='--dependency=afterok:'+str(a.dependency)+' '
    command+=shlex.quote(remote+'/'+script)
    if a.mode in ['train','qualify','audit']:
        assert a.gate_run
        command+=' '+shlex.quote(a.gate_run)
    job=ssh_moss.run(command,capture=True).stdout.decode().strip()
    assert job.isdigit(),job
    run='/work/qt28/moss/results/full-attention-v2-'+({'gates':'gates-','qualify':'qualification-','audit':'audit-','train':''}[a.mode])+job
    record=dict(job=int(job),remote_run=run,control_path=ssh_moss.SSH[ssh_moss.SSH.index('-S')+1],
                mode=a.mode,gate_run=a.gate_run,excluded_nodes=a.exclude,dependency=a.dependency,source_manifest=hashes)
    (HERE/('gate_run.json' if a.mode in ['gates','qualify','audit'] else 'current_run.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record))


if __name__=='__main__':main()
