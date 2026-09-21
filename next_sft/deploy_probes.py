import hashlib
import json
from pathlib import Path
import shlex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ssh_moss

ssh_moss.SSH[ssh_moss.SSH.index('-S')+1]='C:/Users/qizhi/.ssh/cm-moss-check-20260912'
remote='/work/qt28/moss/dkucc/next_sft'
local=Path(__file__).resolve().parent
ssh_moss.run('mkdir -p '+shlex.quote(remote))
for name in ['probes.py','probes.slurm']:
    data=(local/name).read_bytes().replace(b'\r\n',b'\n')
    ssh_moss.run('cat > '+shlex.quote(remote+'/'+name),data)
    found=ssh_moss.run('sha256sum '+shlex.quote(remote+'/'+name),capture=True).stdout.decode().split()[0]
    assert found==hashlib.sha256(data).hexdigest()
job=ssh_moss.run('sbatch --parsable '+shlex.quote(remote+'/probes.slurm'),capture=True).stdout.decode().strip()
assert job.isdigit(),job
record=dict(job=job,remote_run='/work/qt28/moss/results/sft-design-'+job,purpose='bounded diagnostics, not formal next training')
(local/'probe_run.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record),flush=True)
