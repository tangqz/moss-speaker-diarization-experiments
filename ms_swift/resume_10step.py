"""Resume the formal pilot after step 20 with full dev evaluation every 10 steps."""
import argparse
import math
import os
from pathlib import Path
import traceback

from pipeline import HERE, execute, generate_dev, status
from common import read, write, sha, now
from search_selection import decide


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--start-step', type=int, default=20)
    parser.add_argument('--stop-step', type=int, default=50)
    parser.add_argument('--interval', type=int, default=10)
    args = parser.parse_args()
    run = args.run
    output = run / 'training'
    (run / 'logs').mkdir(parents=True, exist_ok=True)
    try:
        assert args.start_step == 20
        assert args.interval == 10 and args.stop_step == 50
        receipts = read(HERE / 'data_receipts.json')
        for name in ['train', 'dev']:
            assert sha(Path(receipts[name]['path'])) == receipts[name]['sha256']

        checkpoint = output / f'checkpoint-{args.start_step}'
        state = read(checkpoint / 'trainer_state.json')
        assert state['global_step'] == args.start_step
        assert any((checkpoint / name).exists() for name in ['optimizer.pt', 'optimizer.bin'])
        assert (checkpoint / 'scheduler.pt').exists()
        completed = read(run / f'evaluations/dev-{args.start_step}/scoring_complete.json')
        assert completed['complete']
        selection = read(run / 'selection.json')
        history = selection['history']
        assert [row['step'] for row in history] == [5, 10, 15, 20]

        protocol = read(run / 'protocol.json')
        protocol['dev_loss'] = 'native eval_loss over complete 26 meeting dev at steps 0,5,10,15,20,30,40,50'
        protocol['generation'] = 'official vLLM BF16 greedy complete meetings at steps 0,5,10,15,20,30,40,50; fresh same-engine Base'
        protocol['evaluation_schedule'] = [0, 5, 10, 15, 20, 30, 40, 50]
        protocol['schedule_change'] = dict(after_step=20, old_interval=5, new_interval=10,
                                           reason='user_requested', utc=now())
        write(run / 'protocol.json', protocol)
        write(run / 'resume_10step.json', dict(start_step=args.start_step,
              stop_step=args.stop_step, interval=args.interval,
              source_checkpoint=str(checkpoint), utc=now()))

        decision = selection['decision']
        previous_step = args.start_step
        for step in range(args.start_step + args.interval, args.stop_step + 1, args.interval):
            status(run, 'native_swift_training', target_step=step,
                   resumed_from_step=previous_step, dev_interval=args.interval)
            env = dict(os.environ, NPROC_PER_NODE='4', MOSS_STOP_STEP=str(step),
                       MOSS_TASK_ROOT=str(HERE), MOSS_IMPLICIT_CAUSAL='1',
                       CELOSS_PARALLEL_SIZE='512')
            command = ['bash', HERE / 'native_sft.sh', receipts['train']['path'], output, '4', '4',
                       '--val_dataset', receipts['dev']['path'], '--eval_strategy', 'steps',
                       '--eval_steps', str(args.interval), '--fsdp', HERE / 'fsdp1_offload.json',
                       '--gradient_checkpointing', 'false', '--vit_gradient_checkpointing', 'false',
                       '--resume_from_checkpoint', output / f'checkpoint-{previous_step}',
                       '--eval_on_start', 'false']
            execute(run, f'train-to-{step}', command, env)
            checkpoint = output / f'checkpoint-{step}'
            state = read(checkpoint / 'trainer_state.json')
            assert state['global_step'] == step
            assert any(row.get('step') == step and 'eval_loss' in row
                       for row in state['log_history'])
            assert all(math.isfinite(row[key]) for row in state['log_history']
                       for key in ['loss', 'eval_loss', 'grad_norm'] if key in row)
            assert any((checkpoint / name).exists() for name in ['optimizer.pt', 'optimizer.bin'])
            assert (checkpoint / 'scheduler.pt').exists()
            runtime = read(output / 'native_runtime.json')
            assert runtime['trainer_class'].startswith('swift.')
            assert not runtime['custom_forward'] and not runtime['custom_loss']

            status(run, 'vllm_dev_generation', step=step, meetings=26,
                   dev_interval=args.interval)
            history.append(dict(step=step, **generate_dev(run, checkpoint, step)))
            decision = decide(history)
            write(run / 'selection.json', dict(history=history, decision=decision, utc=now()))
            previous_step = step
            if decision['stop']:
                break

        selected = decision['selected_step']
        write(run / 'outcome.json', dict(status='completed', through_step=previous_step,
              reason=decision['reason'] if decision['stop'] else 'pilot_50_update_budget',
              selected_step=selected,
              selected_model=str(output / f'checkpoint-{selected}') if selected is not None else None,
              test_evaluated=False, resumed_after_step=20, dev_interval=10, utc=now()))
        status(run, 'completed', through_step=previous_step, selected_step=selected)
    except Exception as exc:
        status(run, 'failed', error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
