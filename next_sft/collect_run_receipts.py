"""Collect final scheduler receipts and batch logs for this bounded experiment."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ssh_moss
ssh_moss.SSH[ssh_moss.SSH.index('-S')+1]='C:/Users/qizhi/.ssh/cm-moss-check-20260912'
r=Path(__file__).resolve().parent
accounting=ssh_moss.run('sacct -j 63399,63400,63401,63402 --format=JobID,State,ExitCode,Elapsed,AllocTRES -P',capture=True).stdout
(r/'slurm-accounting.txt').write_bytes(accounting)
remote_code="""import json
from pathlib import Path
r=Path('/work/qt28/moss')
out={}
for job in ['63399','63400','63401','63402']:
    name='moss-sft-post' if job=='63402' else 'moss-sft-design'
    for ext in ['out','err']:
        p=r/'logs'/f'{name}-{job}.{ext}'
        out[f'{job}/slurm.{ext}']=p.read_text() if p.exists() else None
print(json.dumps(out))
"""
logs=json.loads(ssh_moss.run('/dkucc/home/qt28/envs/moss312/bin/python -',remote_code.encode(),capture=True).stdout)
for name,content in logs.items():
    if content is None: raise FileNotFoundError(name)
    job,fn=name.split('/')
    (r/f'evidence-{job}'/fn).write_text(content,encoding='utf-8')
print(accounting.decode())
