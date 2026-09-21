"""Frozen full-audio inference at base and existing steps 1/10/20; no updates."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/work/qt28/moss')
ORIGINAL = ROOT / 'results/full-attention-v2-63421'
SOURCE = ORIGINAL / 'source'
sys.path.insert(0, str(SOURCE))
from common import BASE, now, read, sha, write


def main():
    run = ROOT / 'results' / f"full-attention-v2-diagnostic-{os.environ['SLURM_JOB_ID']}"
    run.mkdir(parents=True, exist_ok=True)
    (run / 'logs').mkdir(exist_ok=True)
    shutil.copy2(__file__, run / 'early_recheck.py')
    write(run / 'status.json', dict(stage='early_checkpoint_inference', updated_utc=now(),
                                    training_job=63421, training_updates=0))
    children, handles, cases = [], [], []
    try:
        frozen = read(SOURCE / 'source_manifest.json')
        for name in ['generate.py', 'common.py', 'diagnostics.py', 'selection.py', 'score_v1.py']:
            assert sha(SOURCE / name) == frozen[name], name
        source_eval = ORIGINAL / 'evaluations/sentinel-50'
        key = 'ami/dev/TS3004c'
        item = next(i for i in read(source_eval / 'inputs.json') if i['key'] == key)
        for rank, step in enumerate([0, 1, 10, 20]):
            kind = 'untrained_base_recheck' if step == 0 else 'existing_sft_checkpoint'
            model = BASE if step == 0 else ROOT / f'checkpoints/full-attention-v2-63421/checkpoint-{step}'
            if step:
                complete = read(model / 'checkpoint_complete.json')
                assert complete['complete'] and complete['step'] == step
            weights = sorted(model.glob('*.safetensors'))
            assert weights, str(model)
            folder = run / f'checkpoint-{step}'
            folder.mkdir(exist_ok=True)
            model_manifest = dict(path=str(model), candidate_kind=kind, checkpoint_step=step,
                                  files={p.name: sha(p) for p in weights + [
                                      model / 'configuration_moss_transcribe_diarize.py',
                                      model / 'modeling_moss_transcribe_diarize.py']},
                                  config_sha256=sha(model / 'config.json'))
            write(folder / 'model_manifest.json', model_manifest)
            write(folder / 'inputs.json', [dict(item, rank=rank)])
            write(folder / 'references.json', {key: read(source_eval / 'references.json')[key]})
            dest = folder / 'predictions/base' / f'{key}.json'
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_eval / 'predictions/base' / f'{key}.json', dest)
            write(folder / 'protocol.json', dict(
                purpose='diagnostic_only_not_checkpoint_selection', source_training_job=63421,
                checkpoint_step=step, candidate_kind=kind, training_updates=0,
                source_manifest=frozen, inference='unchanged frozen generate.py; BF16 greedy; full audio; 65536 cap',
                candidate_output_label='Frozen worker writes predictions/sft for every candidate, including untrained base at step 0.',
                comparator='Historical base prediction 63363; timing is not contemporaneous.',
                optimizer_state_required=False))
            handle = (run / 'logs' / f'generate-{step}.log').open('w')
            handles.append(handle)
            children.append(subprocess.Popen([
                sys.executable, str(SOURCE / 'generate.py'), '--run', str(folder),
                '--checkpoint', str(model), '--rank', str(rank)], stdout=handle, stderr=subprocess.STDOUT))
            cases.append((step, kind, folder))
        codes = [p.wait() for p in children]
        assert not any(codes), codes
        for step, kind, folder in cases:
            with (run / 'logs' / f'score-{step}.log').open('w') as handle:
                subprocess.run([
                    str(ROOT / 'dkucc/evaluation/metrics-env/bin/python'), str(SOURCE / 'selection.py'),
                    '--run', str(folder)], stdout=handle, stderr=subprocess.STDOUT, check=True)
        write(run / 'outcome.json', dict(status='completed', training_updates=0,
            results=[dict(step=step, candidate_kind=kind, quality=read(folder / 'quality_status.json'),
                          metrics=read(folder / 'metrics_summary.json')) for step, kind, folder in cases],
            completed_utc=now()))
        write(run / 'status.json', dict(stage='completed', updated_utc=now(), training_updates=0))
    except Exception as exc:
        write(run / 'status.json', dict(stage='failed', updated_utc=now(), error=str(exc), training_updates=0))
        raise
    finally:
        # Never orphan another candidate if setup or one child fails.
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
