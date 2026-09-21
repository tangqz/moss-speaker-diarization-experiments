"""Native Swift SFT phases with official vLLM generation and unchanged scoring.

This orchestrator never imports the former training implementation. Training,
checkpoint resume, CE evaluation, optimizers and TensorBoard are owned by Swift.
Generation runs in its own pinned vLLM environment after training releases GPUs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import traceback
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'evaluation'))
from common import BASE, OLD_EVAL, read, write, sha, now
from search_selection import assess, decide
from export_vllm import export as export_vllm

VLLM_PY = '/work/qt28/moss/envs/vllm-moss-20260914/bin/python'
METRIC_PY = '/work/qt28/moss/dkucc/evaluation/metrics-env/bin/python'


def status(run, stage, **details):
    result = dict(stage=stage, utc=now(), **details)
    write(run/'status.json', result)
    print(json.dumps(result), flush=True)


def execute(run, name, command, env=None):
    with (run/'logs'/f'{name}.log').open('a') as log:
        subprocess.run([str(x) for x in command], env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def prepare_dev(run, checkpoint, step, copy_base=True):
    folder = run/'evaluations'/f'dev-{step}'
    items = read(HERE/'evaluation/dev_inputs.json')
    assert len(items) == 26 and all(r['split'] == 'dev' for r in items)
    references = read(OLD_EVAL/'references.json')
    write(folder/'inputs.json', items)
    write(folder/'references.json', {r['key']: references[r['key']] for r in items})
    write(folder/'model_manifest.json', dict(path=str(checkpoint), files={
        p.name: sha(p) for p in sorted(checkpoint.iterdir())
        if p.suffix in ('.safetensors', '.py') or p.name == 'config.json'}))
    hashes = {}
    for row in items if copy_base else []:
        src = run/'evaluations/dev-0/predictions/base'/f"{row['key']}.json"
        original = read(src)
        assert original['status'] == 'ok' and original['prompt_len'] == row['prompt_len']
        assert original['backend'] == 'vllm' and not original['engineering_smoke']
        dest = folder/'predictions/base'/f"{row['key']}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        hashes[row['key']] = sha(src)
    write(folder/'protocol.json', dict(scope='all_26_dev', step=step,
        base_prediction_source=str(run/'evaluations/dev-0'), base_prediction_hashes=hashes,
        base_generation='fresh vLLM base, cached only within this run',
        sft_generation='same pinned vLLM engine settings and original parser',
        scoring='unchanged cpCER and DER025, failure and parse-loss guards',
        no_test_selection=True))
    return folder


def generate_predictions(run, folder, checkpoint, label, step):
    processes, handles = [], []
    # Four independent single-GPU engines; inference parallelism is separate from SP4.
    env = dict(os.environ)
    for key in ['NPROC_PER_NODE', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'PYTORCH_ALLOC_CONF']:
        env.pop(key, None)
    try:
        for rank in range(4):
            log = (run/'logs'/f'dev-{step}-rank-{rank}.log').open('a')
            handles.append(log)
            processes.append(subprocess.Popen([
                VLLM_PY, str(HERE/'evaluation/vllm_generate.py'),
                '--run', str(folder), '--checkpoint', str(checkpoint), '--rank', str(rank),
                '--model-label', label], start_new_session=True,
                stdout=log, stderr=subprocess.STDOUT, env=env))
        deadline=time.monotonic()+3*3600
        while True:
            codes=[p.poll() for p in processes]
            assert all(code in (None,0) for code in codes), ('vllm_generation_failed',codes)
            if all(code==0 for code in codes):
                break
            if time.monotonic()>deadline:
                raise TimeoutError('vLLM dev generation exceeded three-hour limit')
            time.sleep(2)
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
        for p in processes:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL)
                p.wait()
        for log in handles:
            log.close()
    records=[read(folder/'predictions'/label/f"{r['key']}.json") for r in read(folder/'inputs.json')]
    assert len(records)==26 and all(r['status']=='ok' and not r['engineering_smoke'] for r in records)
    if label=='base':
        assert all(not r['truncated'] and not r['parse_empty'] and
                   not r['diagnostics']['major_parse_loss'] for r in records), 'Unhealthy fresh vLLM Base outputs; investigate before training'


def generate_dev(run, checkpoint, step):
    inference_checkpoint=export_vllm(checkpoint,run/'vllm_exports'/f'checkpoint-{step}')
    folder = prepare_dev(run, inference_checkpoint, step)
    started=time.perf_counter()
    generate_predictions(run, folder, inference_checkpoint, 'sft', step)
    write(folder/'timing.json', dict(generation_wall_seconds_including_startup=time.perf_counter()-started))
    execute(run, f'score-dev-{step}',
            [METRIC_PY, HERE/'evaluation/selection.py', '--run', folder])
    summary=read(folder/'metrics_summary.json')
    result = assess(summary, read(folder/'quality_status.json'))
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(str(run/'evaluation_events')) as writer:
        writer.add_scalar('dev/selection_score',result['score'],step)
        writer.add_scalar('dev/eligible',int(result['eligible']),step)
        writer.add_scalar('dev/catastrophic',int(result['catastrophic']),step)
        writer.add_scalar('dev/generation_wall_seconds',read(folder/'timing.json')['generation_wall_seconds_including_startup'],step)
        if step==5:
            writer.add_scalar('base/selection_score',result['base_score'],0)
        for key,group in summary['groups'].items():
            model,corpus,split=key.split('/')
            if model=='base' and step!=5:
                continue
            x=0 if model=='base' else step
            prefix=f'{model}/{corpus}/dev'
            for metric in ['CER','cpCER','truncated','empty_predictions']:
                if group.get(metric) is not None:
                    writer.add_scalar(f'{prefix}/{metric}',group[metric],x)
            writer.add_scalar(f'{prefix}/DER025',group['DER_collar_0.25']['DER'],x)
        writer.flush()
    write(folder/'scoring_complete.json', dict(complete=True, utc=now(), **result))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gate-run', type=Path, required=True)
    parser.add_argument('--resume-gate', type=Path, required=True)
    parser.add_argument('--vllm-gate', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    (run/'logs').mkdir(parents=True, exist_ok=True)
    try:
        gate = read(args.gate_run/'outcome.json')
        assert gate['status'] == 'completed_native_swift_smoke_and_longest'
        assert read(args.gate_run/'job_exit.json')['exit_code'] == 0
        assert read(args.resume_gate/'resume_check.json')['native_resume_and_eval_passed']
        assert read(args.vllm_gate/'vllm_check.json')['loading_and_long_audio_passed']
        receipts = read(HERE/'data_receipts.json')
        for name in ['train', 'dev']:
            assert sha(Path(receipts[name]['path'])) == receipts[name]['sha256']
        write(run/'protocol.json', dict(framework='MS-Swift', full_attention=True,
            full_parameter=True, base=str(BASE), initial_lr=1e-6, scheduler='linear',
            scheduler_horizon=402, warmup=0, sequence_parallel_size=4,
            fsdp_version=1, activation_cpu_offload=True,
            global_batch_meetings=4, train_loss='native CE every optimizer step',
            dev_loss='native eval_loss over complete 26 meeting dev every 5 steps',
            dev_loss_aggregation='MS-Swift/Transformers native evaluation reduction',
            generation='official vLLM BF16 greedy complete meetings every 5 steps; fresh same-engine Base',
            maximum_updates_this_pilot=50, dev_selection=dict(min_delta=.1, patience=3,
            tolerance=.1, catastrophic_stop=True), test_used=False,
            dataset=receipts, gate_run=str(args.gate_run), gate=gate,
            sampler='native Swift SP sampler currently hardcodes seed 42; CLI seed/data_seed remain 0',
            implicit_full_causal=True, native_ce_chunk_size=512))
        status(run, 'vllm_base_dev_generation', meetings=26)
        baseline=prepare_dev(run, BASE, 0, copy_base=False)
        started=time.perf_counter()
        generate_predictions(run, baseline, BASE, 'base', 0)
        write(baseline/'timing.json',dict(generation_wall_seconds_including_startup=time.perf_counter()-started))
        output = run/'training'
        history = []
        decision = None
        for step in range(5, 51, 5):
            status(run, 'native_swift_training', target_step=step)
            env = dict(os.environ, NPROC_PER_NODE='4', MOSS_STOP_STEP=str(step), MOSS_TASK_ROOT=str(HERE),
                       MOSS_IMPLICIT_CAUSAL='1', CELOSS_PARALLEL_SIZE='512')
            command = ['bash', HERE/'native_sft.sh', receipts['train']['path'], output, '4', '4',
                       '--val_dataset', receipts['dev']['path'], '--eval_strategy', 'steps',
                       '--eval_steps', '5', '--fsdp', HERE/'fsdp1_offload.json',
                       '--gradient_checkpointing', 'false', '--vit_gradient_checkpointing', 'false']
            if step == 5:
                command += ['--eval_on_start', 'true']
            else:
                previous = output/f'checkpoint-{step-5}'
                assert read(previous/'trainer_state.json')['global_step'] == step-5
                command += ['--resume_from_checkpoint', previous, '--eval_on_start', 'false']
            execute(run, f'train-to-{step}', command, env)
            checkpoint = output/f'checkpoint-{step}'
            state = read(checkpoint/'trainer_state.json')
            assert state['global_step'] == step
            assert any(r.get('step') == step and 'eval_loss' in r for r in state['log_history'])
            assert all(math.isfinite(r[key]) for r in state['log_history']
                       for key in ['loss', 'eval_loss', 'grad_norm'] if key in r)
            assert any((checkpoint/name).exists() for name in ['optimizer.pt', 'optimizer.bin'])
            assert (checkpoint/'scheduler.pt').exists()
            runtime = read(output/'native_runtime.json')
            assert runtime['trainer_class'].startswith('swift.')
            assert not runtime['custom_forward'] and not runtime['custom_loss']
            status(run, 'vllm_dev_generation', step=step, meetings=26)
            history.append(dict(step=step, **generate_dev(run, checkpoint, step)))
            decision = decide(history)
            write(run/'selection.json', dict(history=history, decision=decision, utc=now()))
            if decision['stop']:
                break
        selected = decision['selected_step']
        write(run/'outcome.json', dict(status='completed', through_step=step,
            reason=decision['reason'] if decision['stop'] else 'pilot_50_update_budget',
            selected_step=selected, selected_model=str(output/f'checkpoint-{selected}')
            if selected is not None else str(BASE), test_evaluated=False, utc=now()))
        status(run, 'completed', through_step=step, selected_step=selected)
    except Exception as exc:
        status(run, 'failed', error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
