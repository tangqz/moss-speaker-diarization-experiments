"""Precheck complete supervised Dev lengths before a full model evaluation."""
import json
from pathlib import Path
from transformers import AutoTokenizer
from common import ROOT,BASE,read,write
TASK=ROOT/'dkucc/rswa_20260920'
tokenizer=AutoTokenizer.from_pretrained(BASE,trust_remote_code=True,local_files_only=True)
inputs={str(Path(r['audio']).resolve()):r for r in read(TASK/'data/dev_inputs.json')}
records=[]
for line in (TASK/'data/dev.jsonl').read_text().splitlines():
    row=json.loads(line);item=inputs[str(Path(row['audios'][0]).resolve())]
    target=row['messages'][1]['content'].strip()+tokenizer.eos_token
    tokens=len(tokenizer(target,add_special_tokens=False)['input_ids'])
    records.append(dict(key=item['key'],prompt=item['prompt_len'],target=tokens,total=item['prompt_len']+tokens))
assert len(records)==26 and max(r['total'] for r in records)<=131072
write(TASK/'evidence/dev_length_check.json',dict(passed=True,records=records,
    maximum_total=max(r['total'] for r in records),
    scope='tokenizer target counts plus frozen prompt lengths; real full CE rechecks actual tensors and EOS'))
print(json.dumps(dict(passed=True,meetings=26,maximum_total=max(r['total'] for r in records))),flush=True)
