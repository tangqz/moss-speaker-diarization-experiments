"""Exercise resumed TrainerState with the installed native scheduling callback."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from transformers.trainer_callback import TrainerState, TrainerControl, DefaultFlowCallback
from transformers.trainer_utils import IntervalStrategy, SaveStrategy

task = Path('/work/qt28/moss/dkucc/ms_swift_lr1e7_20260915_cadence30')
sys.path.insert(0, str(task))
from runtime_callback import PhaseCallback
os.environ.update(MOSS_SAVE_STEPS='30', MOSS_EVAL_STEPS='5', MOSS_STOP_STEP='30')
args = SimpleNamespace(process_index=1, save_steps=30, eval_steps=5,
                       logging_first_step=False, logging_strategy=IntervalStrategy.STEPS,
                       eval_strategy=IntervalStrategy.STEPS, eval_delay=0,
                       save_strategy=SaveStrategy.STEPS)
state = TrainerState(global_step=5, save_steps=5, eval_steps=5,
                     logging_steps=1, max_steps=402)
phase = PhaseCallback(args, SimpleNamespace())
phase.on_train_begin(args, state, TrainerControl(), model=torch.nn.Linear(1, 1))
assert state.save_steps == 30 and state.eval_steps == 5
flow = DefaultFlowCallback()
saved, evaluated = [], []
for step in range(6, 31):
    state.global_step = step
    control = flow.on_step_end(args, state, TrainerControl())
    control = phase.on_step_end(args, state, control)
    if control.should_save:
        saved.append(step)
    if control.should_evaluate:
        evaluated.append(step)
    assert control.should_training_stop == (step == 30)
assert saved == [30]
assert evaluated == [10, 15, 20, 25, 30]
result = dict(passed=True, restored_from_step=5, native_flow_save_steps=saved,
              native_flow_dev_loss_steps=evaluated, final_stop_step=30,
              inherited_checkpoint_cadence_overridden=True)
(task / 'callback_verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
