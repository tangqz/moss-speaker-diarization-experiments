import json
from pathlib import Path
import difflib
from short_recovery.config import load_config
from short_recovery.prepare import load_official, _encode_official
from finetune import DataCollator
from moss_transcribe_diarize.processing_moss_transcribe_diarize import MossTranscribeDiarizeProcessor

c = load_config(Path('short_recovery/config.train150.json'))
a, ca = load_official(c)
b = MossTranscribeDiarizeProcessor.from_pretrained(c.model_path, trust_remote_code=True)
cb = DataCollator(b, c.max_length)
for filename in ('tokenizer.json',):
    x=json.loads((Path(c.processor_path)/filename).read_text())
    y=json.loads((Path(c.model_path)/filename).read_text())
    for k in x.keys() | y.keys():
        if x.get(k) != y.get(k):
            print('JSON_DIFF', k, repr(x.get(k))[:700], repr(y.get(k))[:700], flush=True)
rows=[json.loads(x) for x in Path('short_recovery/artifacts/plan.train150.sub/train.sub.jsonl').read_text().splitlines()]
for occurrence in (0,66):
    r=next(r for r in rows if r['chat_template_kwargs']['recovery_meta']['occurrence']==occurrence)
    args=(r['messages'][0]['content'],r['audios'][0],r['messages'][1]['content'])
    x=_encode_official(a,ca,*args); y=_encode_official(b,cb,*args)
    print('COMPARE',occurrence,'lengths',len(x[0]),len(y[0]),'target_starts',x[4],y[4],flush=True)
    for opcode, i,j,k,l in difflib.SequenceMatcher(None,x[0][:x[4]],y[0][:y[4]]).get_opcodes():
        if opcode!='equal':
            print(opcode,i,j,k,l,repr(a.tokenizer.decode(x[0][i:j])),repr(b.tokenizer.decode(y[0][k:l])),flush=True)
