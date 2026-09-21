"""Create a separately frozen low-LR trial from the archived qualified lineage."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
PARENT = V2 / 'artifacts/63421/raw/run/source'
BUNDLE = HERE / 'bundle'
FILES = ['common.py', 'train.py', 'host_runtime.py', 'memory_sft.py', 'run.py',
         'sampler_gate.py', 'fast_eval.py', 'generate.py', 'infer_worker_v1.py',
         'score_v1.py', 'diagnostics.py', 'selection.py', 'preflight.json',
         'dev_inputs.json', 'sentinel_manifest.json', 'design_protocol.json']


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    assert not (HERE / 'current_run.json').exists(), 'Submitted bundles are immutable.'
    BUNDLE.mkdir(exist_ok=True)
    parent_manifest = json.loads((PARENT / 'source_manifest.json').read_text(encoding='utf-8'))
    manifest, changes, diff = {}, {}, []
    for name in FILES:
        data = (PARENT / name).read_bytes()
        assert digest(data) == parent_manifest[name], name
        old = data.decode('utf-8')
        new = old
        if name == 'train.py':
            assert old.count('1e-5') == 3
            new = old.replace('1e-5', '2e-6')
        elif name == 'host_runtime.py':
            assert old.count('state.global_step in [1, 10, 20]') == 1
            new = old.replace('state.global_step in [1, 10, 20]', 'state.global_step in [1, 5]')
        elif name == 'sampler_gate.py':
            assert old.count('|{134,268}') == 1
            new = old.replace('|{134,268}', '|{10,134,268}')
        elif name == 'design_protocol.json':
            protocol = json.loads(old)
            protocol['experiment'] = 'moss-full-attention-sft-v2-lr2e6-short'
            protocol['status'] = 'authorized_short_trial'
            protocol['training']['learning_rate'] = 2e-6
            protocol['evaluation']['decoding']['audio_preprocessing'] = 'production_cuda_bfloat16_autocast'
            protocol['short_trial'] = dict(parent_training_job=63421, corrected_comparison_job=63441,
                phases=[10, 25, 50], max_executed_updates=50, scheduler_horizon=402,
                gate_at_10='Complete TS3004c; stop on truncation, empty parse or existing major-parse-loss gate.',
                gate_at_50='Original six full sentinels; stop either way for review.',
                full_resume_steps=[10, 25, 50], additional_weight_steps=[1, 5],
                teacher_steps=[0, 25, 50], inherited_teacher_base_job=63421,
                formal_selection_performed=False, test_performed=False,
                operational_difference='Additional phase boundary at 10; exact scheduler, data order and rank RNG resume semantics retained; not bitwise training determinism.')
            new = json.dumps(protocol, ensure_ascii=False, indent=2) + '\n'
        if new != old:
            changes[name] = dict(parent_sha256=digest(data), trial_sha256=digest(new.encode()),
                                 kind='learning_rate_only' if name == 'train.py' else 'protocol_or_checkpoint_audit')
            diff.extend(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                           fromfile='63421/' + name, tofile='lr2e6/' + name))
        (BUNDLE / name).write_text(new, encoding='utf-8', newline='\n')
        manifest[name] = digest((BUNDLE / name).read_bytes())
    prod = (V2 / 'diagnostics/production_generate.py').read_bytes()
    assert digest(prod) == 'b01db8701e04ecba615813839a2e419d0b687b1cb8d922f3509b19032501a7e3'
    (BUNDLE / 'production_generate.py').write_bytes(prod)
    manifest['production_generate.py'] = digest(prod)
    lineage = dict(parent_training_job=63421, qualification_job=63414,
        original_hashes={name: parent_manifest[name] for name in FILES}, changes=changes,
        model_and_training_data_unchanged=True, inference_source_job=63441,
        production_generate_sha256=digest(prod))
    (BUNDLE / 'lineage.json').write_text(json.dumps(lineage, indent=2), encoding='utf-8')
    (BUNDLE / 'trial_changes.diff').write_text(''.join(diff), encoding='utf-8')
    for name in ['trial_run.py', 'trial.slurm', 'check_device.py', 'trial_checks.py']:
        data = (HERE / name).read_bytes().replace(b'\r\n', b'\n')
        (BUNDLE / name).write_bytes(data)
    for path in BUNDLE.iterdir():
        if path.is_file() and path.name != 'source_manifest.json':
            manifest[path.name] = digest(path.read_bytes())
            if path.suffix == '.py':
                ast.parse(path.read_text(encoding='utf-8'), filename=path.name)
    (BUNDLE / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (HERE / 'bundle_review.json').write_text(json.dumps(dict(files=len(manifest), lineage=lineage,
        source_manifest_sha256=digest((BUNDLE/'source_manifest.json').read_bytes())), indent=2), encoding='utf-8')
    print(json.dumps(dict(files=len(manifest), changed_parent_files=list(changes), training_numeric_replacements=3)))


if __name__ == '__main__':
    main()
