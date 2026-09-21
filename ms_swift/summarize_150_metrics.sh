#!/usr/bin/env bash
python3 - <<'PY'
import json
from pathlib import Path

r = Path('/work/qt28/moss/results/ms-swift-63643')
selection = json.loads((r / 'selection.json').read_text())
print('step,score,eligible,catastrophic,ali_CER,ali_cpCER,ali_DER025,ami_CER,ami_cpCER,ami_DER025,truncated,empty')
for row in selection['history']:
    step = row['step']
    groups = json.loads((r / f'evaluations/dev-{step}/metrics_summary.json').read_text())['groups']
    ali = groups['sft/alimeeting/dev']
    ami = groups['sft/ami/dev']
    values = [
        step, f"{row['score']:.6f}", row['eligible'], row['catastrophic'],
        f"{ali['CER']:.6f}", f"{ali['cpCER']:.6f}", f"{ali['DER_collar_0.25']['DER']:.6f}",
        f"{ami['CER']:.6f}", f"{ami['cpCER']:.6f}", f"{ami['DER_collar_0.25']['DER']:.6f}",
        ali['truncated'] + ami['truncated'], ali['empty_predictions'] + ami['empty_predictions'],
    ]
    print(','.join(map(str, values)))

groups = json.loads((r / 'evaluations/dev-150/metrics_summary.json').read_text())['groups']
ali = groups['base/alimeeting/dev']
ami = groups['base/ami/dev']
values = [
    'base', '11.762316', True, False,
    f"{ali['CER']:.6f}", f"{ali['cpCER']:.6f}", f"{ali['DER_collar_0.25']['DER']:.6f}",
    f"{ami['CER']:.6f}", f"{ami['cpCER']:.6f}", f"{ami['DER_collar_0.25']['DER']:.6f}",
    ali['truncated'] + ami['truncated'], ali['empty_predictions'] + ami['empty_predictions'],
]
print(','.join(map(str, values)))
PY
