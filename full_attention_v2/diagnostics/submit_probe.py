"""Submit this one input-precision probe once; a receipt prevents duplication."""
import hashlib
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import ssh_moss


def main():
    receipt = HERE / 'preprocessing_run.json'
    if receipt.exists():
        print(receipt.read_text(encoding='utf-8'))
        return
    ssh_moss.SSH[ssh_moss.SSH.index('-S') + 1] = 'C:/Users/qizhi/.ssh/cm-moss-check-20260912'
    remote = '/work/qt28/moss/dkucc/full_attention_v2_preprocessing_probe'
    ssh_moss.run('mkdir -p ' + shlex.quote(remote))
    for name in ['preprocessing_probe.py', 'preprocessing_probe.slurm']:
        data = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
        ssh_moss.run('cat > ' + shlex.quote(remote + '/' + name), data)
        digest = ssh_moss.run('sha256sum ' + shlex.quote(remote + '/' + name), capture=True).stdout.decode().split()[0]
        assert digest == hashlib.sha256(data).hexdigest()
    script = '''import fcntl,json,subprocess
from pathlib import Path
folder=Path('/work/qt28/moss/dkucc/full_attention_v2_preprocessing_probe')
with (folder/'submit.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    receipt=folder/'submission.json'
    if receipt.exists(): job=json.loads(receipt.read_text())['job']
    else:
        value=subprocess.run(['sbatch','--parsable','--exclude=dkucc-core-gpu-dkurc-d-it09-3',str(folder/'preprocessing_probe.slurm')],capture_output=True,text=True,check=True).stdout.strip()
        assert value.isdigit(),value
        job=int(value);receipt.write_text(json.dumps({'job':job}))
    print(job)
'''
    job = int(ssh_moss.run('/dkucc/home/qt28/envs/moss312/bin/python -', script.encode(), capture=True).stdout)
    record = dict(job=job, remote_run=f'/work/qt28/moss/results/full-attention-v2-diagnostic-{job}',
        source_training_job=63421, checkpoints=[0, 10], key='ami/dev/TS3004c', training_updates=0,
        purpose='Paired CUDA audio preprocessing contexts and 512-token prefixes; not full generation.')
    receipt.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
