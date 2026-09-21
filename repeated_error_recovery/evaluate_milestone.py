"""Evaluate R8005_M8009 at a training milestone and emit an early-stop decision."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError as exc:
    if exc.name != "tensorboard":
        raise

    class SummaryWriter:  # type: ignore[no-redef]
        """No-op fallback; evaluation JSON remains the source of truth."""
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_text(self, *args, **kwargs):
            pass

        def close(self):
            pass

TASK = Path('/work/qt28/moss/dkucc/ms_swift_20260914')
PREVIOUS = Path('/work/qt28/moss/results/ms-swift-63643/evaluations/test-150')
KEY = 'alimeeting/test/R8005_M8009'
VLLM_PY = '/work/qt28/moss/envs/vllm-moss-20260914/bin/python'
METRIC_PY = '/work/qt28/moss/dkucc/evaluation/metrics-env/bin/python'
sys.path[:0] = [str(TASK), str(TASK/'evaluation'), '/work/qt28/moss/dkucc']

from common import BASE, read, write, sha, now
from export_vllm import export
from final_test import manifest
from short_recovery.core import periodic_runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--training-root', required=True, type=Path)
    ap.add_argument('--step', required=True, type=int, choices=[30, 60, 90, 120, 150])
    args = ap.parse_args()
    checkpoint = args.training_root/f'checkpoint-{args.step}'
    export_dir = args.training_root/'vllm_exports'/f'checkpoint-{args.step}'
    run = args.training_root/'evaluations'/f'problem-R8005_M8009-step-{args.step}'
    folder = run/'results'
    assert not run.exists(), f'refusing to overwrite {run}'
    state = read(checkpoint/'recovery_state.json')
    assert state['global_step'] == args.step
    run.mkdir(parents=True)
    (run/'logs').mkdir()
    write(run/'status.json', {'stage':'exporting','step':args.step,'utc':now()})
    export(checkpoint, export_dir)
    receipt = read(export_dir/'export_receipt.json')
    assert receipt['checkpoint'] == str(checkpoint)
    for name, digest in receipt['files'].items():
        assert sha(export_dir/name) == digest

    original = next(x for x in read(PREVIOUS/'inputs.json') if x['key'] == KEY)
    item = dict(original, rank=0)
    references = {KEY: read(PREVIOUS/'references.json')[KEY]}
    prior_protocol = read(PREVIOUS/'protocol.json')
    decoding = prior_protocol['decoding']
    assert decoding['temperature'] == 0 and decoding['repetition_penalty'] == 1
    assert decoding['presence_penalty'] == 0 and decoding['frequency_penalty'] == 0
    write(folder/'inputs.json', [item])
    write(folder/'references.json', references)
    write(folder/'model_manifest.json', manifest(export_dir))
    write(folder/'protocol.json', {
        'scope':'30_step_training_gate_on_known_problem_meeting', 'step':args.step,
        'checkpoint':str(checkpoint), 'input':item, 'decoding':decoding,
        'generation':prior_protocol['generation'], 'inference_processor_source':str(BASE),
        'historical_comparison':str(PREVIOUS), 'utc':now()})
    base_rel = Path('predictions/base')/f'{KEY}.json'
    (folder/base_rel).parent.mkdir(parents=True)
    shutil.copy2(PREVIOUS/base_rel, folder/base_rel)
    write(folder/'base_reuse.json', {'source':str(PREVIOUS/base_rel), 'sha256':sha(folder/base_rel)})

    write(run/'status.json', {'stage':'generating','step':args.step,'utc':now()})
    generation_log = run/'logs/generation.log'
    env = dict(os.environ)
    for key in ('NPROC_PER_NODE','RANK','LOCAL_RANK','WORLD_SIZE','PYTORCH_ALLOC_CONF'):
        env.pop(key, None)
    started = time.monotonic()
    with generation_log.open('w') as log:
        subprocess.run([VLLM_PY, str(TASK/'evaluation/vllm_generate.py'),
            '--run', str(folder), '--checkpoint', str(export_dir), '--rank', '0',
            '--model-label', 'sft'], stdout=log, stderr=subprocess.STDOUT,
            env=env, check=True)
    write(run/'status.json', {'stage':'scoring','step':args.step,'utc':now()})
    with (run/'logs/scoring.log').open('w') as log:
        subprocess.run([METRIC_PY, str(TASK/'evaluation/selection.py'), '--run', str(folder)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)

    pred = read(folder/'predictions/sft'/f'{KEY}.json')
    prior = read(PREVIOUS/'predictions/sft'/f'{KEY}.json')
    assert pred['decoding']['temperature'] == 0 and pred['decoding']['repetition_penalty'] == 1
    assert pred['prompt_ids_sha256'] == prior['prompt_ids_sha256']
    loops = list(periodic_runs(pred['generated_ids']))
    segments = pred['segments']
    max_end = max((x['end'] for x in segments), default=0)
    reference_end = max(x['end'] for x in references[KEY]['segments'])
    early_end = max_end < reference_end - max(30, reference_end * .05)
    major_parse_loss = bool(pred['diagnostics']['major_parse_loss'])
    reasons = []
    if pred['truncated']: reasons.append('output_cap_truncation')
    if not pred['ended_eos']: reasons.append('missing_eos')
    if pred['parse_empty']: reasons.append('empty_transcript')
    if loops: reasons.append('raw_token_repetition')
    if early_end: reasons.append('premature_transcript_end')
    if major_parse_loss: reasons.append('major_parse_loss')
    passed = not reasons
    metrics = next(x for x in read(folder/'per_record_metrics.json') if x['model']=='sft')
    decision = {
        'schema_version':'error-repeat-milestone-gate-v1', 'step':args.step,
        'passed':passed, 'continue_training':passed and args.step < 150,
        'stop_training':not passed, 'reasons':reasons,
        'temperature':0, 'repetition_penalty':1.0,
        'generated_tokens':pred['generated_tokens'], 'ended_eos':pred['ended_eos'],
        'truncated':pred['truncated'], 'periodic_runs':loops,
        'maximum_segment_end':max_end, 'reference_maximum_end':reference_end,
        'last_segment':segments[-1] if segments else None,
        'CER':metrics['text']['CER'], 'cpCER':metrics['text']['cpCER'],
        'deletions':metrics['text']['deletions'],
        'generation_seconds':pred['generation_seconds'],
        'wall_seconds':time.monotonic()-started, 'utc':now()}
    write(folder/'gate_decision.json', decision)
    (folder/'transcript.txt').write_text(pred['raw_text'], encoding='utf-8')
    writer = SummaryWriter(str(args.training_root/'gate_events'), flush_secs=2)
    writer.add_scalar('gate/passed', int(passed), args.step)
    writer.add_scalar('gate/truncated', int(pred['truncated']), args.step)
    writer.add_scalar('gate/raw_long_loops', len(loops), args.step)
    writer.add_scalar('gate/maximum_segment_end', max_end, args.step)
    writer.add_scalar('gate/CER', metrics['text']['CER'], args.step)
    writer.add_scalar('gate/cpCER', metrics['text']['cpCER'], args.step)
    writer.add_text('gate/decision', json.dumps(decision, ensure_ascii=False, indent=2), args.step)
    writer.close()
    write(run/'status.json', {'stage':'passed' if passed else 'failed_stop_training', **decision})
    print(json.dumps(decision, ensure_ascii=False), flush=True)
    return 0 if passed else 42


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        print(json.dumps({'evaluation_error':type(exc).__name__, 'message':str(exc)}), flush=True)
        raise
