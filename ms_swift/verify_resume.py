"""Verify native checkpoint resume and dev CE before permitting full training."""
import json
import math
from pathlib import Path
import sys
import torch

run = Path(sys.argv[1])
checkpoint = run/'resume-sp4/checkpoint-2'
state = json.loads((checkpoint/'trainer_state.json').read_text())
assert state['global_step'] == 2
evaluation = [r for r in state['log_history'] if 'eval_loss' in r]
assert {r['step'] for r in evaluation} >= {1, 2}
assert all(math.isfinite(r['eval_loss']) and r['eval_loss'] > 0 for r in evaluation)
logs = [r for r in state['log_history'] if 'loss' in r and r['step'] == 2]
assert len(logs) == 1 and math.isfinite(logs[0]['loss'])
assert abs(logs[0]['learning_rate']-1e-6*(1-2/402)) < 1e-12
runtime=json.loads((run/'resume-sp4/native_runtime.json').read_text())
assert runtime['starting_global_step']==1 and runtime['effective_eval_steps']==1
assert all(abs(lr-1e-6*(1-1/402))<1e-12 for lr in runtime['optimizer_lr_at_start'])
optimizer = torch.load(checkpoint/'optimizer.bin', map_location='cpu', weights_only=True)
steps = [float(v['step']) for v in optimizer['state'].values() if 'step' in v]
assert steps and set(steps) == {2.}, (len(steps), sorted(set(steps)))
receipt = dict(native_resume_and_eval_passed=True, global_step=2, optimizer_steps=sorted(set(steps)),
    optimizer_parameter_states=len(steps), resumed_learning_rate=runtime['optimizer_lr_at_start'],
    scheduler_after_step2=logs[0]['learning_rate'],
    evaluation=evaluation, qualification='Four repeated short records, not full 26-meeting dev')
(run/'resume_check.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
