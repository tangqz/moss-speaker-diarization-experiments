import hashlib
import json
from pathlib import Path
import shlex
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ssh_moss
ssh_moss.SSH[ssh_moss.SSH.index('-S')+1]='C:/Users/qizhi/.ssh/cm-moss-check-20260912'
local=Path(__file__).resolve().parent
remote='/work/qt28/moss/dkucc/next_sft'
for name in ['post_pilot.py','post_pilot.slurm']:
    data=(local/name).read_bytes().replace(b'\r\n',b'\n')
    ssh_moss.run('cat > '+shlex.quote(remote+'/'+name),data)
    found=ssh_moss.run('sha256sum '+shlex.quote(remote+'/'+name),capture=True).stdout.decode().split()[0]
    assert found==hashlib.sha256(data).hexdigest()
job=ssh_moss.run('sbatch --parsable --dependency=afterok:63401 '+shlex.quote(remote+'/post_pilot.slurm'),capture=True).stdout.decode().strip()
assert job.isdigit(),job
record=dict(job=job,depends_on='63401',remote_run='/work/qt28/moss/results/sft-design-'+job)
(local/'post_run.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record),flush=True)
