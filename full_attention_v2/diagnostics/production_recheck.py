"""Repeat six step-50 sentinels, base and step 10 with production preprocessing."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
import sys
import threading

ROOT = Path('/work/qt28/moss')
ORIGINAL = ROOT / 'results/full-attention-v2-63421'
FROZEN = ORIGINAL / 'source'
sys.path.insert(0, str(FROZEN))
from common import BASE, now, read, sha, write


def main():
    run = ROOT / 'results' / f"full-attention-v2-diagnostic-{os.environ['SLURM_JOB_ID']}"
    run.mkdir(parents=True, exist_ok=True)
    source = run / 'source'
    source.mkdir(exist_ok=True)
    write(run / 'status.json', dict(stage='production_preprocessing_recheck', updated_utc=now(), training_updates=0))
    try:
        bundle = Path(__file__).resolve().parent
        fix = read(bundle / 'preprocessing_fix.json')
        assert sha(FROZEN / 'generate.py') == fix['original_sha256']
        assert sha(bundle / 'production_generate.py') == fix['corrected_sha256']
        probe = read(ROOT / 'results/full-attention-v2-diagnostic-63440/outcome.json')
        assert probe['status'] == 'completed' and probe['training_updates'] == 0
        base_probe = next(x for x in probe['results'] if x['step'] == 0 and x['mode'] == 'production_bf16_autocast')
        assert base_probe['historical_base_prefix_equal'] and base_probe['generated_tokens'] == 512
        frozen_manifest = read(FROZEN / 'source_manifest.json')
        for path in FROZEN.iterdir():
            if path.is_file() and path.suffix in {'.py', '.json', '.slurm'}:
                if path.name in frozen_manifest:
                    assert sha(path) == frozen_manifest[path.name], path.name
                shutil.copy2(path, source / path.name)
        for name in ['production_generate.py', 'production_recheck.py', 'preprocessing_fix.json', 'preprocessing_fix.diff']:
            shutil.copy2(bundle / name, source / name)
        write(run / 'probe_evidence.json', probe)
        write(run / 'derivative_source_manifest.json', {p.name: sha(p) for p in source.iterdir() if p.is_file()})
        original_eval = ORIGINAL / 'evaluations/sentinel-50'
        items = read(original_eval / 'inputs.json')
        refs = read(original_eval / 'references.json')
        key = 'ami/dev/TS3004c'
        target = next(x for x in items if x['key'] == key)
        model_paths = {0: BASE, 10: ROOT / 'checkpoints/full-attention-v2-63421/checkpoint-10',
                       50: ROOT / 'checkpoints/full-attention-v2-63421/checkpoint-50'}
        manifests = {0: read(ROOT / 'results/full-attention-v2-diagnostic-63439/checkpoint-0/model_manifest.json'),
                     10: read(ROOT / 'results/full-attention-v2-diagnostic-63439/checkpoint-10/model_manifest.json'),
                     50: read(original_eval / 'model_manifest.json')}
        for step, manifest in manifests.items():
            for name, digest in manifest['files'].items():
                assert sha(model_paths[step] / name) == digest, (step, name)
            assert sha(model_paths[step] / 'config.json') == manifest['config_sha256']
        write(run / 'protocol.json', dict(training_updates=0, source_training_job=63421,
            corrected_preprocessing='CUDA BF16 autocast around complete prepare_inputs, identical to historical and production entrypoint.',
            unchanged='Full audio, BF16 model/forward, greedy, EOS/pad, cap 65536, original parser/scorer.',
            formal_training='Remains paused. This evaluator never updates or resumes training.',
            base_comparator='Historical production-path predictions reused for six-meeting quality; separate fresh full TS3004c base recheck.',
            candidate_label='Frozen worker uses predictions/sft for all candidates, including the untrained base diagnostic.',
            fix=fix))
        cases = [dict(id='sentinel-target', step=50, item=target, group='sentinel-50'),
                 dict(id='base-check', step=0, item=target, group='base-check'),
                 dict(id='step10-check', step=10, item=target, group='step10-check')]
        for i, item in enumerate(sorted((x for x in items if x['key'] != key), key=lambda x: -x['duration'])):
            cases.append(dict(id=f'sentinel-{i}', step=50, item=item, group='sentinel-50'))
        assert len(cases) == 8
        write(run / 'case_manifest.json', cases)
        queue = Queue()
        for case in cases:
            queue.put(case)
        completed, lock = [], threading.Lock()

        def worker(rank):
            while True:
                try:
                    case = queue.get_nowait()
                except Empty:
                    return
                folder = run / 'cases' / case['id']
                folder.mkdir(parents=True, exist_ok=True)
                write(folder / 'inputs.json', [dict(case['item'], rank=rank)])
                write(folder / 'model_manifest.json', manifests[case['step']])
                write(folder / 'protocol.json', dict(group=case['group'], checkpoint_step=case['step'],
                    candidate_kind='untrained_base_recheck' if case['step'] == 0 else 'sft',
                    preprocessing='production_cuda_bfloat16_autocast', training_updates=0))
                with (folder / 'generate.log').open('w') as handle:
                    result = subprocess.run([sys.executable, str(source / 'production_generate.py'),
                        '--run', str(folder), '--checkpoint', str(model_paths[case['step']]), '--rank', str(rank)],
                        stdout=handle, stderr=subprocess.STDOUT)
                with lock:
                    completed.append(dict(case=case['id'], rank=rank, exit_code=result.returncode))
                    write(run / 'status.json', dict(stage='production_preprocessing_recheck', updated_utc=now(),
                        completed_cases=len(completed), expected_cases=len(cases), cases=completed, training_updates=0))

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(worker, rank) for rank in range(4)]
            for future in futures:
                future.result()
        assert all(x['exit_code'] == 0 for x in completed), completed
        scored = []
        for group in ['sentinel-50', 'base-check', 'step10-check']:
            group_cases = [c for c in cases if c['group'] == group]
            evaluation = run / 'evaluations' / group
            evaluation.mkdir(parents=True, exist_ok=True)
            write(evaluation / 'inputs.json', [c['item'] for c in group_cases])
            write(evaluation / 'references.json', {c['item']['key']: refs[c['item']['key']] for c in group_cases})
            write(evaluation / 'model_manifest.json', manifests[group_cases[0]['step']])
            for case in group_cases:
                case_key = case['item']['key']
                for model, src in [('base', original_eval / 'predictions/base' / f'{case_key}.json'),
                        ('sft', run / 'cases' / case['id'] / 'predictions/sft' / f'{case_key}.json')]:
                    dst = evaluation / 'predictions' / model / f'{case_key}.json'
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
            with (evaluation / 'scoring.log').open('w') as handle:
                subprocess.run([str(ROOT / 'dkucc/evaluation/metrics-env/bin/python'), str(source / 'selection.py'),
                    '--run', str(evaluation)], stdout=handle, stderr=subprocess.STDOUT, check=True)
            scored.append(dict(group=group, quality=read(evaluation / 'quality_status.json'),
                               metrics=read(evaluation / 'metrics_summary.json')))
        fresh = read(run / 'cases/base-check/predictions/sft' / f'{key}.json')['generated_ids']
        historical = read(original_eval / 'predictions/base' / f'{key}.json')['generated_ids']
        write(run / 'outcome.json', dict(status='completed', completed_utc=now(), training_updates=0,
            training_status='paused_pending_corrected_evaluation_review', results=scored,
            base_recheck=dict(all_generated_tokens_equal=fresh == historical,
                fresh_tokens=len(fresh), historical_tokens=len(historical))))
        write(run / 'status.json', dict(stage='completed', updated_utc=now(), training_updates=0))
    except Exception as exc:
        write(run / 'status.json', dict(stage='failed', updated_utc=now(), error=str(exc), training_updates=0))
        raise


if __name__ == '__main__':
    main()
