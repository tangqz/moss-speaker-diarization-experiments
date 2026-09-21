"""Freeze the source split metadata before touching any training data."""
import hashlib,json,os,shutil
from pathlib import Path
ROOT=Path('/work/qt28/moss')
TASK=ROOT/'dkucc/rswa_20260920'
OLD=ROOT/'dkucc/ms_swift_20260914'
def load(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def lines(path):return [json.loads(s) for s in path.read_text().splitlines() if s]
receipts=load(OLD/'data_receipts.json')
groups={};identities={};summary={};all_audio=[]
for name in ['train','dev']:
    src=Path(receipts[name]['path'])
    assert digest(src)==receipts[name]['sha256'],(name,'manifest changed')
    rows=lines(src)
    assert len(rows)==receipts[name]['records']
    groups[name]=rows
    identities[name]=set()
    for r in rows:
        p=Path(r['audios'][0]);st=p.stat()
        key=(st.st_dev,st.st_ino)
        assert key not in identities[name],(name,'duplicate audio identity',str(p))
        identities[name].add(key)
        all_audio.append(dict(split=name,path=str(p.resolve()),bytes=st.st_size,inode=st.st_ino,device=st.st_dev))
    summary[name]=dict(records=len(rows),sha256=digest(src))
    (TASK/'data').mkdir(exist_ok=True)
    shutil.copy2(src,TASK/'data'/f'{name}.jsonl')
test=[]
for corpus in ['alimeeting','aishell4','ami']:
    src=ROOT/f'data/manifests/{corpus}_test.jsonl'
    rows=lines(src)
    assert len(rows)==dict(alimeeting=20,aishell4=20,ami=16)[corpus]
    test.extend(rows)
identities['test']=set()
for row in test:
    p=Path(row['audio_path']);st=p.stat();key=(st.st_dev,st.st_ino)
    assert key not in identities['test']
    identities['test'].add(key)
    all_audio.append(dict(split='test',path=str(p.resolve()),bytes=st.st_size,inode=st.st_ino,device=st.st_dev))
for a,b in [('train','dev'),('train','test'),('dev','test')]:
    assert not identities[a]&identities[b],('split leakage',a,b)
for name in ['smoke','longest']:
    shutil.copy2(receipts[name]['path'],TASK/'data'/f'{name}.jsonl')
summary['test']={'records':len(test)}
summary.update(metadata_identity_disjoint=True,audio_content_hash_audit_pending=True,
               original_receipts=receipts,
               processor_sha256=digest(ROOT/'models/MOSS-Transcribe-Diarize/processor_config.json'),
               model_config_sha256=digest(ROOT/'models/MOSS-Transcribe-Diarize/config.json'))
(TASK/'evidence/data_metadata.json').write_text(json.dumps(summary,indent=2))
(TASK/'evidence/audio_files.json').write_text(json.dumps(all_audio,indent=2))
print(json.dumps(summary,indent=2),flush=True)
