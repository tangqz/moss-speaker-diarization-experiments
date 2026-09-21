"""Archive measurements before removing explicitly inventoried obsolete weights."""
import argparse, datetime, hashlib, json, subprocess, tarfile
from pathlib import Path

ROOT=Path('/work/qt28/moss/results/moss-rswa-20260920')
KEEP={ROOT/'training'/g/f'checkpoint-{s}'/'model.safetensors'
      for g,s in [('full-seed0',30),('128-seed0',30),('256-seed0',150)]}
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');a=ap.parse_args()
    assert ROOT.resolve()==ROOT and ROOT.is_dir()
    jobs=subprocess.check_output(['squeue','-u','qt28','-h','-o','%i %j'],text=True)
    assert 'moss-rswa-formal' not in jobs,jobs
    close=ROOT/'closure';close.mkdir(exist_ok=True)
    for name in ['state.json','status.json','next_job.json']:
        dest=close/f'pre_stop_{name}'
        if not dest.exists():dest.write_bytes((ROOT/name).read_bytes())
    state=json.loads((ROOT/'state.json').read_text())
    if state.get('queue'):
        write(close/'cancelled_queue.json',state['queue']);state['queue']=[]
    state['closed_by_user']=True;write(ROOT/'state.json',state)
    write(ROOT/'status.json',dict(stage='closed_by_user',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reason='Stop old round, archive available evidence, start independent R256/R512 LR1e-6 round',
        incomplete_evaluations_excluded=True,test_completed=False))
    targets=[]
    for p in sorted(ROOT.rglob('*')):
        rel=p.relative_to(ROOT)
        if p.is_dir() or rel.parts[0] not in ('training','dev','test'):continue
        if p in KEEP:continue
        if p.suffix in ('.safetensors','.bin','.pt','.pth'):
            assert p.parent.resolve().is_relative_to(ROOT)
            targets.append(dict(path=str(p),bytes=p.lstat().st_size,symlink=p.is_symlink()))
    plan=dict(keep_weights=[str(p) for p in sorted(KEEP)],remove=targets,
              planned_bytes=sum(r['bytes'] for r in targets),preserve='all logs, raw predictions, metrics, configurations and source')
    for p in KEEP:assert p.is_file() and p.stat().st_size>1_000_000_000
    write(close/'cleanup_plan.json',plan)
    print(json.dumps(dict(candidate_files=len(targets),planned_gib=plan['planned_bytes']/2**30,keep=plan['keep_weights'])),flush=True)
    if not a.apply:return
    archive=close/'evidence_before_cleanup.tar.gz'
    if not archive.exists():
        excluded={r['path'] for r in targets}|{str(p) for p in KEEP}
        with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
            for p in sorted(ROOT.rglob('*')):
                if p.is_file() and not p.is_symlink() and 'closure' not in p.relative_to(ROOT).parts and str(p) not in excluded:
                    tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
        with tarfile.open(archive,'r:gz') as tar:assert any(x.name=='state.json' for x in tar.getmembers())
    receipt=dict(archive=str(archive),archive_bytes=archive.stat().st_size,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),removed=[],freed_bytes=0)
    for row in targets:
        p=Path(row['path']);assert p.parent.resolve().is_relative_to(ROOT) and p not in KEEP
        assert p.lstat().st_size==row['bytes']
        p.unlink();receipt['removed'].append(row);receipt['freed_bytes']+=row['bytes']
    for p in KEEP:assert p.is_file()
    receipt['complete']=True;write(close/'cleanup_receipt.json',receipt)
    print(json.dumps(dict(freed_gib=receipt['freed_bytes']/2**30,archive_bytes=receipt['archive_bytes'],complete=True)),flush=True)

if __name__=='__main__':main()
