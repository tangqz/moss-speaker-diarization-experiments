#!/usr/bin/env bash
set -euo pipefail
OLD=/work/qt28/moss/dkucc/ms_swift_20260914
NEW=/work/qt28/moss/dkucc/ms_swift_lr1e7_20260915
JOB=63643
RUN=/work/qt28/moss/results/ms-swift-63643

python3 - "$OLD" "$NEW" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

old, new = map(Path, sys.argv[1:])
same = [
    'moss_plugin.py', 'runtime_callback.py', 'export_vllm.py',
    'fsdp1_offload.json', 'data_receipts.json', 'resume_10step.py'
]
same += [str(p.relative_to(old)) for p in (old / 'evaluation').rglob('*.py')]
for rel in same:
    a = hashlib.sha256((old / rel).read_bytes()).hexdigest()
    b = hashlib.sha256((new / rel).read_bytes()).hexdigest()
    if a != b:
        raise RuntimeError(f'unexpected execution-code change: {rel}')

native_old = (old / 'native_sft.sh').read_text()
native_new = (new / 'native_sft.sh').read_text()
assert native_old.replace('--learning_rate 1e-6', '--learning_rate 1e-7') == native_new

pipeline_old = (old / 'pipeline.py').read_text()
pipeline_new = (new / 'pipeline.py').read_text()
assert pipeline_old.replace('initial_lr=1e-6', 'initial_lr=1e-7') == pipeline_new

for name in ['train.slurm', 'resume_10step.slurm', 'switch_after_step20.sh']:
    assert (old / name).read_text().replace(f'TASK={old}', f'TASK={new}') == (new / name).read_text()

print(json.dumps({
    'verified': True,
    'learning_rate': 1e-7,
    'unchanged_execution_files_checked': len(same),
    'data_receipts': json.loads((new / 'data_receipts.json').read_text()),
}, indent=2))
PY

echo JOB
squeue -j "$JOB" -o '%i|%j|%T|%M|%l|%R'
echo STATUS
if test -f "$RUN/status.json"; then cat "$RUN/status.json"; else echo status_not_created_yet; fi
echo LAUNCH_RECEIPT
cat "$RUN/launch_receipt.json"
echo WATCHER
watcher=$(cat "$RUN/switch-after-step20.pid")
if kill -0 "$watcher" 2>/dev/null; then echo "alive|$watcher"; else echo "dead|$watcher"; exit 3; fi
