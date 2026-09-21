"""Transfer only this experiment's source files through the user-authenticated SSH master."""
import io,tarfile,sys,shlex,hashlib,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ssh_moss import run
HERE=Path(__file__).resolve().parent
TASK='/work/qt28/moss/dkucc/rswa_20260920'
payload=io.BytesIO()
manifest={}
with tarfile.open(fileobj=payload,mode='w:gz') as tf:
    for p in sorted(HERE.iterdir()):
        if p.suffix in ('.py','.json','.slurm','.sh','.md'):
            data=p.read_bytes().replace(b'\r\n',b'\n')
            manifest[p.name]=hashlib.sha256(data).hexdigest()
            entry=tarfile.TarInfo(p.name);entry.size=len(data);entry.mode=0o644
            tf.addfile(entry,io.BytesIO(data))
    data=json.dumps(manifest,indent=2).encode()
    entry=tarfile.TarInfo('source_manifest.json');entry.size=len(data);entry.mode=0o644
    tf.addfile(entry,io.BytesIO(data))
run('tar -xzf - -C '+shlex.quote(TASK),payload.getvalue())
print('RSWA source transfer complete')
