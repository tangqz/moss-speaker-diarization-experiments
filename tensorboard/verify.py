"""Verify real TensorBoard HTTP data against independently checked training logs."""
import json
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode

HERE=Path(__file__).resolve().parent
launch=json.loads((HERE.parent/'full_attention_v2/launch-verification.json').read_text(encoding='utf-8'))
label=f"full_attention_v2_{launch['job']}"
base='http://127.0.0.1:6006'
def get(path):return json.load(urlopen(base+path,timeout=10))
tags=get('/data/plugin/scalars/tags')
required={'train/loss','train/grad_norm','train/learning_rate','performance/peak_gpu_gib','validation/loss_all'}
assert required<=tags[label].keys(),tags.get(label)
points=get('/data/plugin/scalars/scalars?'+urlencode({'run':label,'tag':'train/loss'}))
expected={r['step']:r['loss'] for r in launch['steps']}
compared=0
for wall,step,value in points:
    if step in expected:
        assert abs(value-expected[step])<1e-6,(step,value,expected[step])
        compared+=1
assert compared>=3,compared
baseline=get('/data/plugin/scalars/scalars?'+urlencode({'run':label,'tag':'validation/loss_all'}))
assert baseline and baseline[0][1]==0
assert abs(baseline[0][2]-launch['baseline_loss']['all'])<1e-6
result=dict(url=base,run=label,http_data_verified=True,training_points_verified=compared,
    latest_displayed_step=max(p[1] for p in points),scalar_tags=len(tags[label]),baseline_loss_matches=True)
(HERE/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
