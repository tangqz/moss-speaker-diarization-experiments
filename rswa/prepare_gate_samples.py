"""Select a median-duration training meeting without consulting predictions."""
import json,wave,hashlib
from pathlib import Path
TASK=Path('/work/qt28/moss/dkucc/rswa_20260920')
rows=[json.loads(x) for x in (TASK/'data/train.jsonl').read_text().splitlines() if x]
durations=[]
for i,row in enumerate(rows):
    with wave.open(row['audios'][0],'rb') as audio:
        durations.append((audio.getnframes()/audio.getframerate(),i))
duration,index=sorted(durations)[len(rows)//2]
ordered=sorted(durations)
varied_indices=[ordered[round(i*(len(ordered)-1)/11)][1] for i in range(12)]
(TASK/'data/varied.jsonl').write_text(''.join(json.dumps(rows[i],ensure_ascii=False)+'\n' for i in varied_indices))
line=json.dumps(rows[index],ensure_ascii=False)+'\n'
(TASK/'data/typical.jsonl').write_text(line*4)
samples={}
for name in ['smoke','typical','longest']:
    path=TASK/f'data/{name}.jsonl'
    records=[json.loads(x) for x in path.read_text().splitlines() if x]
    assert len(records)==4 and all(r==records[0] for r in records)
    samples[name]=dict(records=4,distinct_meetings=1,effective_batch=4,
        audio=records[0]['audios'][0],sha256=hashlib.sha256(path.read_bytes()).hexdigest())
report=dict(typical_index=index,typical_duration=duration,selection='upper median training audio duration',
            samples=samples,varied_indices=varied_indices,varied_selection='12 duration quantiles of training; originals unchanged',
            test_predictions_used=False)
(TASK/'evidence/gate_samples.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
