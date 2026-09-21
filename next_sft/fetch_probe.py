import io
import json
from pathlib import Path
import sys
import tarfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ssh_moss

ssh_moss.SSH[ssh_moss.SSH.index('-S')+1]='C:/Users/qizhi/.ssh/cm-moss-check-20260912'
local=Path(__file__).resolve().parent
run=json.loads((local/'probe_run.json').read_text(encoding='utf-8'))
if len(sys.argv)>1:
    assert sys.argv[1].isdigit()
    run=dict(job=sys.argv[1],remote_run='/work/qt28/moss/results/sft-design-'+sys.argv[1])
remote=run['remote_run']
code="""from pathlib import Path
import sys, tarfile
r=Path(REMOTE)
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for p in sorted(r.rglob('*')):
        if p.is_file() and not any('checkpoint' in a for a in p.relative_to(r).parts):
            archive.add(p, arcname=str(p.relative_to(r)), recursive=False)
""".replace('REMOTE',repr(remote))
data=ssh_moss.run('/dkucc/home/qt28/envs/moss312/bin/python -',code.encode(),capture=True).stdout
dest=local/('evidence-'+run['job'])
dest.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
    for member in archive.getmembers():
        target=(dest/member.name).resolve()
        if not target.is_relative_to(dest.resolve()) or not member.isfile():
            raise ValueError(member.name)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(archive.extractfile(member).read())
print('LOCAL',dest)
for name in ['contracts','replay','pilot-bf16','pilot-fp32','post-pilot']:
    p=dest/(name+'.json')
    if p.exists():
        d=json.loads(p.read_text(encoding='utf-8'))
        print(name,json.dumps({k:d[k] for k in ['status','steps','initial_validation','final_validation'] if k in d}))
    out=dest/(name+'.out')
    if out.exists(): print(name+'_OUT',out.read_text(encoding='utf-8')[-700:])
    err=dest/(name+'.err')
    if err.exists():
        s=err.read_text(encoding='utf-8')
        if 'Traceback' in s: print(name+'_ERROR',s[s.rfind('Traceback'):][-2500:])
gpu=dest/'gpu-before.csv'
if gpu.exists(): print(gpu.read_text(encoding='utf-8'))
