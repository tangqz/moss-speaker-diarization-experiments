"""Collect compact engineering receipts, excluding all model/optimizer tensors."""
import json,tarfile
from pathlib import Path
ROOT=Path('/work/qt28/moss');TASK=ROOT/'dkucc/rswa_20260920'
dest=TASK/'evidence_bundle.tar.gz'
with tarfile.open(dest,'w:gz') as archive:
    for p in (TASK/'evidence').glob('*.json'):
        archive.add(p,arcname='task/'+p.name)
    for base in sorted((ROOT/'results').glob('rswa-*-64*')):
        if not base.is_dir():continue
        job=base.name
        for p in base.rglob('*.json'):
            relative=p.relative_to(base)
            if 'source' in relative.parts or any(x.startswith('checkpoint-') for x in relative.parts):continue
            if p.stat().st_size>10*1024*1024:continue
            archive.add(p,arcname=job+'/'+str(relative))
print(str(dest))
