"""Continue native Swift from a complete checkpoint, with 30-step generation."""
import argparse
import math
import os
from pathlib import Path
import traceback

from pipeline import HERE, execute, generate_dev, status
from common import BASE, read, write, sha, now
from selection_policy import decide


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--start-step', type=int, required=True)
    args = parser.parse_args()
    run, start = args.run, args.start_step
    output = run / 'training'
    try:
        assert 0 < start < 30
        receipts = read(HERE / 'data_receipts.json')
        for name in ['train', 'dev']:
            assert sha(Path(receipts[name]['path'])) == receipts[name]['sha256']
        checkpoint = output / f'checkpoint-{start}'
        initial_state = read(checkpoint / 'trainer_state.json')
        assert initial_state['global_step'] == start
        assert any((checkpoint / name).is_file() for name in ['optimizer.pt', 'optimizer.bin'])
        assert (checkpoint / 'scheduler.pt').is_file()
        assert all((checkpoint / f'rng_state_{rank}.pth').is_file() for rank in range(4))
        initial_args = read(checkpoint / 'args.json')
        assert initial_args['learning_rate'] == 1e-7
        assert initial_args['max_steps'] == 402

        history = []
        for folder in sorted((run / 'evaluations').glob('dev-*'),
                             key=lambda p: int(p.name.split('-')[-1])):
            step = int(folder.name.split('-')[-1])
            receipt = folder / 'scoring_complete.json'
            if 0 < step <= start and receipt.is_file():
                row = read(receipt)
                assert row['complete']
                history.append(dict(step=step, **row))
        assert history and history[-1]['step'] == start
        decision = decide(history)
        assert not decision['stop'], decision

        protocol = read(run / 'protocol.json')
        protocol.update(
            maximum_updates_this_pilot=150, minimum_plateau_stop_step=134,
            checkpoint_interval=30, generation_interval=30, dev_loss_interval=5,
            evaluation_schedule=[0] + [r['step'] for r in history] + [30, 60, 90, 120, 150],
            planned_generation_steps=[30, 60, 90, 120, 150],
            dev_loss='native eval_loss over complete 26 meeting dev every 5 steps',
            generation='official vLLM BF16 greedy complete meetings every 30 steps after cadence change',
            schedule_change=dict(from_step=start, checkpoint_interval=30,
                                 generation_interval=30, dev_loss_interval=5,
                                 reason='user_requested', utc=now()))
        write(run / 'protocol.json', protocol)
        previous = start
        for step in [30, 60, 90, 120, 150]:
            status(run, 'native_swift_training', target_step=step,
                   resumed_from_step=previous, checkpoint_interval=30,
                   generation_interval=30, dev_loss_interval=5)
            env = dict(os.environ, NPROC_PER_NODE='4', MOSS_STOP_STEP=str(step),
                       MOSS_TASK_ROOT=str(HERE), MOSS_IMPLICIT_CAUSAL='1',
                       CELOSS_PARALLEL_SIZE='512', MOSS_SAVE_STEPS='30', MOSS_EVAL_STEPS='5')
            command = ['bash', HERE / 'native_sft.sh', receipts['train']['path'], output, '4', '4',
                       '--val_dataset', receipts['dev']['path'], '--eval_strategy', 'steps',
                       '--eval_steps', '5', '--save_steps', '30',
                       '--fsdp', HERE / 'fsdp1_offload.json',
                       '--gradient_checkpointing', 'false', '--vit_gradient_checkpointing', 'false',
                       '--resume_from_checkpoint', output / f'checkpoint-{previous}',
                       '--eval_on_start', 'false']
            execute(run, f'train-to-{step}', command, env)
            checkpoint = output / f'checkpoint-{step}'
            state = read(checkpoint / 'trainer_state.json')
            assert state['global_step'] == step and state['max_steps'] == 402
            assert state['save_steps'] == 30 and state['eval_steps'] == 5
            for dev_step in range((previous // 5 + 1) * 5, step + 1, 5):
                assert any(row.get('step') == dev_step and 'eval_loss' in row
                           for row in state['log_history'])
            assert all(math.isfinite(row[key]) for row in state['log_history']
                       for key in ['loss', 'eval_loss', 'grad_norm'] if key in row)
            assert any((checkpoint / name).is_file() for name in ['optimizer.pt', 'optimizer.bin'])
            assert (checkpoint / 'scheduler.pt').is_file()
            runtime = read(output / 'native_runtime.json')
            assert runtime['trainer_class'].startswith('swift.')
            assert not runtime['custom_forward'] and not runtime['custom_loss']
            assert runtime['effective_save_steps'] == 30 and runtime['effective_eval_steps'] == 5
            expected_lr = 1e-7 * (1 - previous / 402)
            assert all(math.isclose(lr, expected_lr, rel_tol=1e-6)
                       for lr in runtime['optimizer_lr_at_start'])
            status(run, 'vllm_dev_generation', step=step, meetings=26, generation_interval=30)
            history.append(dict(step=step, **generate_dev(run, checkpoint, step)))
            decision = decide(history)
            write(run / 'selection.json', dict(history=history, decision=decision, utc=now()))
            previous = step
            if decision['stop']:
                break
        selected = decision['selected_step']
        write(run / 'outcome.json', dict(status='completed', through_step=previous,
              reason=decision['reason'] if decision['stop'] else '150_update_budget',
              selected_step=selected,
              selected_model=str(output / f'checkpoint-{selected}') if selected is not None else str(BASE),
              test_evaluated=False, generation_interval=30, checkpoint_interval=30, utc=now()))
        status(run, 'completed', through_step=previous, selected_step=selected)
    except Exception as exc:
        status(run, 'failed', error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
