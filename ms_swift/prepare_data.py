"""Convert existing official splits to Swift messages; never alter targets/audio."""
import json,hashlib,os,sys
from pathlib import Path
TASK=Path(__file__).resolve().parent
ROOT=Path('/work/qt28/moss')
sys.path.insert(0,str(ROOT/'MOSS-Transcribe-Diarize'))
from finetune import ConversationDataset

def write(name, rows):
    path=TASK/'data'/name
    path.parent.mkdir(exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    return dict(path=str(path),records=len(rows),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def convert(row):
    return dict(messages=[dict(role='user',content='<audio>\n'+row['prompt']),
                          dict(role='assistant',content=row['target'])],audios=[row['audio']])

train_path=ROOT/'data/moss_jsonl/train_mix.jsonl'
train=ConversationDataset(str(train_path)).samples
assert len(train)==536
devpaths=list((ROOT/'data/moss_jsonl').glob('*dev*.jsonl'))
print('DEV_MANIFESTS', [str(p) for p in devpaths],flush=True)
dev=[]
for p in devpaths:
    if p.name in ['dev_mix.jsonl','dev.jsonl']:continue
    dev+=ConversationDataset(str(p)).samples
if len(dev)!=26:
    raise ValueError(f'Expected exactly 26 dev meetings; found {len(dev)} using {devpaths}')
assert not {r['audio'] for r in train}&{r['audio'] for r in dev}
receipts=dict(train=write('train.jsonl',[convert(r) for r in train]),
              dev=write('dev.jsonl',[convert(r) for r in dev]),
              original_train_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest())
preflight=json.loads((ROOT/'dkucc/full_attention_dev5_20260913/bundle/preflight.json').read_text())
smallest=min(preflight['samples'],key=lambda r:r['tokens'])['index']
largest=max(preflight['samples'],key=lambda r:r['tokens'])['index']
receipts['smoke']=write('smoke.jsonl',[convert(train[smallest])]*4)
receipts['longest']=write('longest.jsonl',[convert(train[largest])]*4)
receipts.update(smoke_index=smallest,longest_index=largest,targets_unchanged=True)
(TASK/'data_receipts.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
print(json.dumps(receipts),flush=True)
