"""Deploy only cadence/control changes; retain native model and optimizer code."""
from pathlib import Path
import shlex
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import ssh_moss

TASK = '/work/qt28/moss/dkucc/ms_swift_lr1e7_20260915_cadence30'
PREPARE = r'''
import json, shutil
from pathlib import Path
old = Path('/work/qt28/moss/dkucc/ms_swift_lr1e7_20260915')
new = Path('/work/qt28/moss/dkucc/ms_swift_lr1e7_20260915_cadence30')
assert not new.exists(), 'Refusing to overwrite an existing cadence task'
new.mkdir()
for path in old.iterdir():
    if path.is_file() and path.suffix in ('.py', '.json', '.sh', '.slurm'):
        shutil.copy2(path, new / path.name)
shutil.copytree(old / 'evaluation', new / 'evaluation')
callback = new / 'runtime_callback.py'
text = callback.read_text()
before = '        params=list(model.parameters())'
after = """
        # Restore requested cadence after loading TrainerState on resume.
        # Optimizer, scheduler, RNG and model use native resume semantics.
        if os.environ.get('MOSS_SAVE_STEPS'):
            args.save_steps = state.save_steps = int(os.environ['MOSS_SAVE_STEPS'])
        if os.environ.get('MOSS_EVAL_STEPS'):
            args.eval_steps = state.eval_steps = int(os.environ['MOSS_EVAL_STEPS'])
        params=list(model.parameters())"""
assert text.count(before) == 1
text = text.replace(before, after)
text = text.replace('effective_eval_steps=state.eval_steps,',
                    'effective_eval_steps=state.eval_steps,\n                effective_save_steps=state.save_steps,')
callback.write_text(text)
native = new / 'native_sft.sh'
text = native.read_text()
assert '--learning_rate 1e-7' in text
assert text.count('--save_steps 5') == 1
native.write_text(text.replace('--save_steps 5', '--save_steps 30'))
for name in ['moss_plugin.py', 'pipeline.py', 'fsdp1_offload.json', 'export_vllm.py', 'data_receipts.json']:
    assert (new/name).read_bytes() == (old/name).read_bytes()
print(json.dumps(dict(task=str(new), training_math_unchanged=True)))
'''

ssh_moss.run('python3 -', PREPARE.encode())
for name in ['selection_policy.py', 'resume_30step.py', 'resume_30step.slurm', 'dispatch_30step.py']:
    ssh_moss.run('cat > ' + shlex.quote(TASK + '/' + name), (HERE / name).read_bytes())
ssh_moss.run('python3 -m py_compile ' + TASK + '/runtime_callback.py ' + TASK + '/resume_30step.py ' + TASK + '/selection_policy.py ' + TASK + '/dispatch_30step.py')
ssh_moss.run('bash -n ' + TASK + '/native_sft.sh ' + TASK + '/resume_30step.slurm')
ssh_moss.run('nohup python3 -u ' + TASK + '/dispatch_30step.py > /work/qt28/moss/results/ms-swift-63643/logs/cadence30-dispatch.log 2>&1 < /dev/null &')
