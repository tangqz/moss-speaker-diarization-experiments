"""Package all project training logs and this evaluation, excluding weights/audio."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path('/work/qt28/moss')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--allow-incomplete', action='store_true')
    a = p.parse_args()
    run = a.run.resolve()
    assert run.parent == ROOT/'results' and run.name.startswith('eval-62994-')
    if not a.allow_incomplete:
        assert json.loads((run/'metrics_summary.json').read_text())['complete']
        assert json.loads((run/'job_exit.json').read_text())['exit_code'] == 0
    job = run.name.rsplit('-',1)[-1]
    (run/'slurm-accounting.txt').write_bytes(subprocess.check_output([
        'sacct','-j',f'62994,{job}','--format=JobID,State,ExitCode,Elapsed,Start,End','-P']))
    paths = set(p for p in (ROOT/'logs').rglob('*') if p.is_file() and not p.is_symlink())
    paths.update(p for p in run.rglob('*') if p.is_file() and not p.is_symlink())
    for p in (ROOT/'checkpoints').rglob('*'):
        if p.is_file() and not p.is_symlink() and (p.suffix == '.json' or p.name == 'training_args.bin'):
            paths.add(p)
    for folder in [ROOT/'dkucc/evaluation',ROOT/'dkucc/memory_sft']:
        paths.update(p for p in folder.iterdir() if p.is_file() and p.suffix in ['.py','.slurm','.md','.sha256'])
    index=[]
    for path in sorted(paths):
        assert path.resolve().is_relative_to(ROOT)
        index.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=digest(path)))
    bundle=ROOT/'results'/(run.name+'-all-logs.tar.gz')
    temp=bundle.with_suffix('.tmp')
    with tarfile.open(temp,'w:gz',compresslevel=5) as tar:
        for entry in index:
            tar.add(ROOT/entry['path'],arcname=entry['path'],recursive=False)
        payload=json.dumps(dict(files=index,excludes=['model weights','optimizer tensors','raw audio']),indent=2).encode()
        info=tarfile.TarInfo('DOWNLOAD_MANIFEST.json')
        info.size=len(payload)
        tar.addfile(info,io.BytesIO(payload))
    temp.replace(bundle)
    result=dict(path=str(bundle),bytes=bundle.stat().st_size,sha256=digest(bundle),files=len(index))
    print(json.dumps(result))


if __name__=='__main__':
    main()
