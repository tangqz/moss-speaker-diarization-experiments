#!/usr/bin/env bash
set -euo pipefail
/dkucc/home/qt28/envs/moss312/bin/python - <<'PY'
from pathlib import Path
import json, hashlib, inspect, os
import transformers
from transformers.loss.loss_utils import ForCausalLMLoss
r=Path('/work/qt28/moss')
print('INSTALLED_CAUSAL_LOSS', inspect.getsource(ForCausalLMLoss))
for sub in ['models/MOSS-Transcribe-Diarize', 'checkpoints/full-attention-ddp4-62994']:
    p=r/sub
    for fn in ['generation_config.json','config.json']:
        d=json.loads((p/fn).read_text())
        if fn=='config.json':
            d={k:d.get(k) for k in ['architectures','model_type','dtype','torch_dtype','text_config','audio_config']}
        print('CONFIG', sub, fn, json.dumps(d))
p=r/'MOSS-Transcribe-Diarize/moss_transcribe_diarize/modeling_moss_transcribe_diarize.py'
print('MODEL_SOURCE', p.read_text())
for d in ['data/moss_jsonl','data','logs']:
    paths=[str(p.relative_to(r)) for p in (r/d).glob('*') if any(s in p.name for s in ['preflight','audit','token','length','dev','smoke'])]
    print('FILES', d, paths)
rows=[json.loads(x) for x in (r/'data/moss_jsonl/train_mix.jsonl').read_text().splitlines() if x.strip()]
print('TRAIN_EXAMPLE',json.dumps({k:v for k,v in rows[0].items() if k!='conversation'}))
print('TRAIN_AUDIO_FIRST', [x['conversation'][1]['content'] for x in rows[:3]])
PY
