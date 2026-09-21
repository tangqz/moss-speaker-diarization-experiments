#!/usr/bin/env bash
set -euo pipefail

TASK=/work/qt28/moss/dkucc/ms_swift_20260914
RUN=/work/qt28/moss/results/ms-swift-63643
ENV=/work/qt28/moss/envs/ms-swift-20260914

"$ENV/bin/python" -c 'from torch.utils.tensorboard import SummaryWriter; print("tensorboard_writer_ready")'
"$ENV/bin/python" -m py_compile "$TASK/final_test.py.new"
test "$(sha256sum "$TASK/final_test.py.new" | cut -d' ' -f1)" = '545337f367e1906ae77aa11336cb532b3cc0d7e0232c3e4737d34ea994d4accb'

python3 - <<'PY'
import json
from pathlib import Path

run = Path('/work/qt28/moss/results/ms-swift-63643')
outcome = json.loads((run / 'outcome.json').read_text())
selection = json.loads((run / 'selection.json').read_text())
assert outcome['status'] == 'completed'
assert outcome['selected_step'] == 150
assert selection['decision']['selected_step'] == 150
assert outcome['test_evaluated'] is False
assert (run / 'vllm_exports/checkpoint-150').is_dir()
assert not (run / 'evaluations/test-150/test_complete.json').exists()
print('frozen_step_150_test_preflight_passed')
PY

if [[ -f "$TASK/final_test.py" && ! -f "$TASK/final_test.py.pre-step150" ]]; then
  cp "$TASK/final_test.py" "$TASK/final_test.py.pre-step150"
fi
mv "$TASK/final_test.py.new" "$TASK/final_test.py"
sha256sum "$TASK/final_test.py"
