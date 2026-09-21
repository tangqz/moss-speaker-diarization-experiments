"""One-time Slurm handoff after the in-progress evaluation finishes."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone

RUN = Path('/work/qt28/moss/results/ms-swift-63643')
OLD = Path('/work/qt28/moss/dkucc/ms_swift_lr1e7_20260915')
TASK = Path('/work/qt28/moss/dkucc/ms_swift_lr1e7_20260915_cadence30')
JOB = 63643


def read(path):
    return json.loads(path.read_text())


def write(path, payload):
    temp = path.with_name(path.name + '.cadence30.tmp')
    temp.write_text(json.dumps(payload, indent=2))
    temp.replace(path)


def record(stage, **kwargs):
    data = dict(stage=stage, utc=datetime.now(timezone.utc).isoformat(), **kwargs)
    write(RUN / 'cadence30_dispatch.json', data)
    print(json.dumps(data), flush=True)


def main():
    pid_file = RUN / 'switch-after-step20.pid'
    if pid_file.exists():
        pid = int(pid_file.read_text())
        cmdline = Path(f'/proc/{pid}/cmdline')
        if cmdline.exists():
            command = cmdline.read_bytes().replace(b'\x00', b' ').decode()
            assert str(OLD / 'switch_after_step20.sh') in command
            assert str(RUN) in command and str(JOB) in command
            os.kill(pid, signal.SIGTERM)
            write(RUN / 'replaced_step20_watcher.json', dict(pid=pid, reason='cadence30'))

    record('waiting_for_current_full_dev_evaluation', old_job=JOB)
    deadline = time.monotonic() + 3 * 3600
    while True:
        status = read(RUN / 'status.json')
        receipts = sorted((RUN / 'evaluations').glob('dev-*/scoring_complete.json'),
                          key=lambda p: int(p.parent.name.split('-')[-1]))
        if receipts:
            receipt = receipts[-1]
            step = int(receipt.parent.name.split('-')[-1])
            evaluated = read(receipt)
            if evaluated.get('catastrophic'):
                record('stopped_on_existing_catastrophic_output', step=step)
                return
            if evaluated.get('complete') and step < 30:
                assert read(RUN / f'training/checkpoint-{step}/trainer_state.json')['global_step'] == step
                assert (RUN / f'training/checkpoint-{step}/optimizer.bin').is_file()
                break
        if status['stage'] == 'failed':
            raise RuntimeError(status)
        if time.monotonic() > deadline:
            raise TimeoutError('Current full-dev evaluation did not finish within three hours')
        time.sleep(2)

    job = int(subprocess.check_output([
        'sbatch', '--parsable', f'--dependency=afterany:{JOB}',
        f'--export=ALL,MOSS_RESUME_RUN={RUN},MOSS_RESUME_STEP={step}',
        str(TASK / 'resume_30step.slurm')], text=True).strip().split(';')[0])
    state = read(OLD / 'current_run.json')
    state.update(training_job=job, resumed_from_job=JOB, resume_start_step=step,
                 remote_task=str(TASK), stage='cadence30_resume_submitted',
                 full_dataset_training_started=True, watcher_pid=None,
                 checkpoint_interval=30, generation_interval=30, dev_loss_interval=5,
                 dev_schedule=[0, step, 30, 60, 90, 120, 150], maximum_updates=150,
                 minimum_plateau_stop_step=134, follow_remote_state=True)
    state.pop('sole_model_optimization_change', None)
    for path in [OLD / 'current_run.json', TASK / 'current_run.json',
                 Path('/work/qt28/moss/dkucc/ms_swift_20260914/current_run.json')]:
        write(path, state)
    write(RUN / 'cadence30_resume_receipt.json', dict(old_job=JOB, training_job=job,
          resumed_from_step=step, checkpoint_interval=30, generation_interval=30,
          dev_loss_interval=5, maximum_updates=150, minimum_plateau_stop_step=134,
          initial_learning_rate=1e-7, scheduler_horizon=402, task=str(TASK)))
    subprocess.run(['scancel', str(JOB)], check=True)
    record('resume_submitted', old_job=JOB, training_job=job, start_step=step,
           checkpoint_steps=[30, 60, 90, 120, 150])


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        record('failed', error=repr(exc))
        raise
