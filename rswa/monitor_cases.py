"""Mirror changing parallel-case progress into the visible SSH console."""
import argparse,time
from pathlib import Path
from common import ROOT
ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);a=ap.parse_args()
last={};destination=ROOT/'logs/rswa-20260920-console.log'
while not (a.run/'job_exit.json').exists():
    for path in sorted(a.run.glob('case-*.log')):
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-5000));lines=stream.read().decode(errors='replace').replace('\r','\n').splitlines()
        value=next((x for x in reversed(lines) if x.strip()),'')
        if value and last.get(path)!=value:
            line=f'[{a.run.name}/{path.stem}] {value}\n'
            with destination.open('a') as out:out.write(line)
            print(line,flush=True);last[path]=value
    time.sleep(20)
