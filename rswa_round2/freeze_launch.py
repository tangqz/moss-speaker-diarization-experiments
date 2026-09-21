import hashlib,json,py_compile,shutil,subprocess
from pathlib import Path
ROOT=Path('/work/qt28/moss/results/moss-rswa-r2-20260921');SRC=ROOT/'source'
def read(p):return json.loads(p.read_text())
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not subprocess.check_output(['squeue','-u','qt28','-h'],text=True).strip()
archive=ROOT/'launch_revision';archive.mkdir(exist_ok=True)
for name in ['source_manifest.json','round2.py','round2.slurm']:
    if not (archive/name).exists():shutil.copy2(SRC/name,archive/name)
for name in ['round2.py','round2.slurm']:
    shutil.copy2(Path('/work/qt28/moss/dkucc/rswa_round2_staging')/name,SRC/name)
for p in SRC.glob('*.py'):py_compile.compile(str(p),doraise=True)
subprocess.run(['bash','-n',str(SRC/'round2.slurm'),str(SRC/'native_sft.sh')],check=True)
for split in ['dev','test']:
    refs=read(ROOT/'data/references.json');inputs=read(ROOT/f'data/{split}_inputs.json')
    assert all(row['key'] in refs for row in inputs)
manifest={p.name:sha(p) for p in sorted(SRC.iterdir()) if p.is_file() and p.name!='source_manifest.json'}
write(SRC/'source_manifest.json',manifest)
protocol=read(ROOT/'protocol.json');protocol.update(scheduling='serial on four A40 GPUs; account MaxJobs=1, MaxTRES GPU=4',
    order=['R512 delta qualification','R256 train/dev','shared Base Test','R256 epoch-end Test','R512 train/dev/Test'],
    interrupted_launch=dict(jobs=[64635,64636,64637],reason='Association simultaneous-job and GPU limits discovered',
        committed_optimizer_updates=0,previous_source_manifest_sha256=sha(archive/'source_manifest.json')),
    source_manifest_sha256=sha(SRC/'source_manifest.json'))
write(ROOT/'protocol.json',protocol)
job=subprocess.check_output(['sbatch','--parsable','--job-name=moss-rswa2-serial',
    '--nodelist=dkucc-core-gpu-dkurc-d-it09-8',str(SRC/'round2.slurm')],text=True).strip()
write(ROOT/'launch.json',dict(job=job,root=str(ROOT),mode='serial',source_manifest_sha256=sha(SRC/'source_manifest.json')))
print(json.dumps(dict(job=job,source_manifest_sha256=sha(SRC/'source_manifest.json'))))
