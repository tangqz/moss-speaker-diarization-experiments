import hashlib,json,shlex,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent.parent))
import ssh_moss
ssh_moss.SSH[ssh_moss.SSH.index('-S')+1]='C:/Users/qizhi/.ssh/cm-moss-check-20260912'
remote='/work/qt28/moss/dkucc/full_attention_v2_diagnostic'
ssh_moss.run('mkdir -p '+shlex.quote(remote))
for name in ['recheck.py','recheck.slurm']:
    data=(HERE/name).read_bytes().replace(b'\r\n',b'\n')
    ssh_moss.run('cat > '+shlex.quote(remote+'/'+name),data)
    found=ssh_moss.run('sha256sum '+shlex.quote(remote+'/'+name),capture=True).stdout.decode().split()[0]
    assert found==hashlib.sha256(data).hexdigest()
job=ssh_moss.run('sbatch --parsable --exclude=dkucc-core-gpu-dkurc-d-it09-3 '+shlex.quote(remote+'/recheck.slurm'),capture=True).stdout.decode().strip()
assert job.isdigit()
record=dict(job=int(job),remote_run='/work/qt28/moss/results/full-attention-v2-diagnostic-'+job,
    source_training_job=63421,checkpoints=[25,50],key='ami/dev/TS3004c',training_updates=0,
    purpose='Earlier checkpoint comparison and same-checkpoint repeat; original inference and scoring')
(HERE/'current_run.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record))
