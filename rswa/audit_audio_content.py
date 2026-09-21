"""Stream checksums on a compute node; never decode or rewrite the corpus."""
import hashlib,json,os,time
from pathlib import Path
TASK=Path('/work/qt28/moss/dkucc/rswa_20260920')
rows=json.loads((TASK/'evidence/audio_files.json').read_text())
out=TASK/'evidence/audio_sha256.jsonl'
completed={}
if out.exists():
    for line in out.read_text().splitlines():
        row=json.loads(line);completed[row['path']]=row
start=time.monotonic()
with out.open('a') as dst:
    for i,row in enumerate(rows):
        p=Path(row['path']);st=p.stat()
        assert st.st_size==row['bytes'] and st.st_ino==row['inode']
        old=completed.get(row['path'])
        if old and old.get('mtime_ns')==st.st_mtime_ns:continue
        h=hashlib.sha256()
        with p.open('rb') as src:
            while block:=src.read(8*1024*1024):h.update(block)
        assert p.stat().st_mtime_ns==st.st_mtime_ns
        record=dict(row,sha256=h.hexdigest(),mtime_ns=st.st_mtime_ns)
        dst.write(json.dumps(record)+'\n');dst.flush();completed[row['path']]=record
        if i%25==0:print(f'AUDIO_HASH {i+1}/{len(rows)} elapsed={time.monotonic()-start:.1f}s',flush=True)
by_hash={}
for row in rows:
    record=completed[row['path']];by_hash.setdefault(record['sha256'],[]).append(record)
duplicates=[v for v in by_hash.values() if len(v)>1]
cross=[v for v in duplicates if len({r['split'] for r in v})>1]
report=dict(passed=not cross,records=len(rows),duplicates=duplicates,cross_split_duplicates=cross,
            seconds=time.monotonic()-start,host=os.uname().nodename)
(TASK/'evidence/audio_content_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
assert not cross,'Cross-split byte-identical audio: investigate before formal training'
