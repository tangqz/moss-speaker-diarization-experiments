"""One explicit submission; existing local/remote receipt prevents duplicate jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
import ssh_moss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if not args.submit:
        parser.error('Use --submit for the authorized inference-only diagnostic.')
    receipt = HERE / 'early_run.json'
    if receipt.exists():
        print(receipt.read_text(encoding='utf-8'))
        return
    ssh_moss.SSH[ssh_moss.SSH.index('-S') + 1] = 'C:/Users/qizhi/.ssh/cm-moss-check-20260912'
    remote = '/work/qt28/moss/dkucc/full_attention_v2_early_diagnostic'
    ssh_moss.run('mkdir -p ' + shlex.quote(remote))
    for name in ['early_recheck.py', 'early_recheck.slurm']:
        data = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
        ssh_moss.run('cat > ' + shlex.quote(remote + '/' + name), data)
        found = ssh_moss.run('sha256sum ' + shlex.quote(remote + '/' + name), capture=True).stdout.decode().split()[0]
        assert found == hashlib.sha256(data).hexdigest()
    script = '''import fcntl,json,subprocess
from pathlib import Path
folder=Path('/work/qt28/moss/dkucc/full_attention_v2_early_diagnostic')
with (folder/'submit.lock').open('w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    receipt=folder/'submission.json'
    if receipt.exists():
        job=json.loads(receipt.read_text())['job']
    else:
        result=subprocess.run(['sbatch','--parsable','--exclude=dkucc-core-gpu-dkurc-d-it09-3',str(folder/'early_recheck.slurm')],capture_output=True,text=True,check=True).stdout.strip()
        assert result.isdigit(),result
        job=int(result)
        receipt.write_text(json.dumps({'job':job}))
    print(job)
'''
    job = int(ssh_moss.run('/dkucc/home/qt28/envs/moss312/bin/python -', script.encode(), capture=True).stdout)
    record = dict(job=job, remote_run=f'/work/qt28/moss/results/full-attention-v2-diagnostic-{job}',
                  source_training_job=63421, checkpoints=[0, 1, 10, 20], key='ami/dev/TS3004c',
                  training_updates=0, previous_diagnostic_job=63432,
                  purpose='Locate onset before step 25 and recheck original base through the same frozen inference worker.')
    receipt.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
