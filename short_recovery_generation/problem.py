"""Full-length, unpenalized generation of the known failing meeting."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run import (TASK, TRAIN, CHECKPOINT, PREVIOUS, EXPORT, BASE, read, write,
                 sha, now, manifest, METRIC_PY, periodic_runs, SummaryWriter)
from final_test import VLLM_PY, publish_text_logs

KEY = 'alimeeting/test/R8005_M8009'
RUN = TRAIN / 'evaluations/problem-R8005_M8009-20260916-r2'
FOLDER = RUN / 'results'


def main():
    assert not RUN.exists(), f'Refusing to overwrite {RUN}'
    assert read(CHECKPOINT/'recovery_state.json')['global_step'] == 150
    receipt = read(EXPORT/'export_receipt.json')
    assert receipt['checkpoint'] == str(CHECKPOINT)
    for name, digest in receipt['files'].items():
        assert sha(EXPORT/name) == digest, name
    original = next(x for x in read(PREVIOUS/'inputs.json') if x['key'] == KEY)
    item = dict(original, rank=0)
    refs = {KEY: read(PREVIOUS/'references.json')[KEY]}
    protocol = read(PREVIOUS/'protocol.json')
    assert protocol['decoding']['temperature'] == 0
    assert protocol['decoding']['repetition_penalty'] == 1
    RUN.mkdir(parents=True)
    (RUN/'logs').mkdir()
    write(FOLDER/'inputs.json', [item])
    write(FOLDER/'references.json', refs)
    write(FOLDER/'model_manifest.json', manifest(EXPORT))
    write(FOLDER/'protocol.json', dict(
        scope='full_length_problem_meeting_before_full_evaluation',
        checkpoint=str(CHECKPOINT), original_input=original, input=item,
        decoding=protocol['decoding'], generation=protocol['generation'],
        inference_processor_source=str(BASE), historical_comparison=str(PREVIOUS),
        source_hashes={str(p):sha(p) for p in [Path(__file__), TASK/'evaluation/vllm_generate.py']},
        full_evaluation_requires_review=True, utc=now()))
    base_rel = Path('predictions/base')/f'{KEY}.json'
    (FOLDER/base_rel).parent.mkdir(parents=True)
    shutil.copy2(PREVIOUS/base_rel, FOLDER/base_rel)
    write(FOLDER/'base_reuse.json', dict(source=str(PREVIOUS/base_rel), sha256=sha(FOLDER/base_rel)))
    writer = SummaryWriter(str(RUN/'test_events'), flush_secs=5)
    writer.add_scalar('problem/complete', 0, 0)
    writer.add_scalar('problem/repetition_penalty', 1.0, 0)
    writer.flush()
    write(RUN/'status.json', dict(stage='loading_and_generating', key=KEY, utc=now()))
    log_path = RUN/'logs/generation.log'
    env = dict(os.environ)
    for name in ('NPROC_PER_NODE','RANK','LOCAL_RANK','WORLD_SIZE','PYTORCH_ALLOC_CONF'):
        env.pop(name, None)
    started = time.monotonic()
    print('Full problem meeting: temperature=0 repetition_penalty=1.0', flush=True)
    offsets, steps = {}, {}
    with log_path.open('w') as log:
        proc = subprocess.Popen([VLLM_PY, str(TASK/'evaluation/vllm_generate.py'),
            '--run', str(FOLDER), '--checkpoint', str(EXPORT), '--rank', '0',
            '--model-label', 'sft'], stdout=log, stderr=subprocess.STDOUT, env=env)
        while proc.poll() is None:
            publish_text_logs(writer, 'problem', [log_path], offsets, steps)
            writer.add_scalar('problem/elapsed_minutes', (time.monotonic()-started)/60, int(time.monotonic()-started))
            writer.flush()
            time.sleep(10)
        assert proc.returncode == 0, f'Generation failed: {proc.returncode}; see {log_path}'
    publish_text_logs(writer, 'problem', [log_path], offsets, steps)
    write(RUN/'status.json', dict(stage='scoring', key=KEY, utc=now()))
    with (RUN/'logs/scoring.log').open('w') as log:
        subprocess.run([METRIC_PY, str(TASK/'evaluation/selection.py'), '--run', str(FOLDER)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    new = read(FOLDER/'predictions/sft'/f'{KEY}.json')
    old = read(PREVIOUS/'predictions/sft'/f'{KEY}.json')
    assert new['decoding']['temperature'] == 0 and new['decoding']['repetition_penalty'] == 1
    assert new['prompt_ids_sha256'] == old['prompt_ids_sha256']
    def details(rec):
        segs = rec['segments']
        return dict(ended_eos=rec['ended_eos'], truncated=rec['truncated'],
            empty=rec['parse_empty'], generated_tokens=rec['generated_tokens'],
            periodic_runs=list(periodic_runs(rec['generated_ids'])),
            first_segment=segs[0] if segs else None, last_segment=segs[-1] if segs else None,
            maximum_segment_end=max((s['end'] for s in segs), default=0),
            diagnostics=rec['diagnostics'])
    old_metrics = next(x for x in read(PREVIOUS/'per_record_metrics.json') if x['key']==KEY and x['model']=='sft')
    new_metrics = next(x for x in read(FOLDER/'per_record_metrics.json') if x['model']=='sft')
    report = dict(key=KEY, decoding=new['decoding'], previous_sft=details(old), recovery=details(new),
        previous_metrics=old_metrics, recovery_metrics=new_metrics,
        reference_last_segment=refs[KEY]['segments'][-1],
        reference_maximum_end=max(s['end'] for s in refs[KEY]['segments']),
        manual_review_required=True, full_evaluation_started=False, utc=now())
    write(FOLDER/'problem_review.json', report)
    (FOLDER/'transcript.txt').write_text(new['raw_text'], encoding='utf-8')
    for name in ('CER','cpCER','deletions','insertions'):
        writer.add_scalar('problem/'+name, new_metrics['text'][name], 150)
    writer.add_scalar('problem/long_loops', len(report['recovery']['periodic_runs']), 150)
    writer.add_scalar('problem/truncated', int(new['truncated']), 150)
    writer.add_scalar('problem/complete', 1, 150)
    writer.close()
    write(RUN/'status.json', dict(stage='complete_awaiting_review', key=KEY, utc=now()))
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if RUN.exists():
            write(RUN/'failure.json', dict(error_type=type(exc).__name__, error=str(exc), utc=now()))
            write(RUN/'status.json', dict(stage='failed', error=str(exc), utc=now()))
        raise
