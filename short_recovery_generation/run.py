"""Evaluate the fixed recovery checkpoint with the preceding 56-meeting protocol."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

TASK = Path('/work/qt28/moss/dkucc/ms_swift_20260914')
TRAIN = Path('/work/qt28/moss/results/short-recovery-sr-v2-sub-150-20260916-r3/sub/seed-0')
CHECKPOINT = TRAIN / 'checkpoint-150'
PREVIOUS = Path('/work/qt28/moss/results/ms-swift-63643/evaluations/test-150')
RUN = TRAIN / 'evaluations/test-150-20260916-r2'
FOLDER = RUN / 'results'
EXPORT = TRAIN / 'vllm_exports/checkpoint-150'
PROBLEM = TRAIN / 'evaluations/problem-R8005_M8009-20260916-r2/results'
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(TASK / 'evaluation'))
sys.path.insert(0, '/work/qt28/moss/dkucc')

from common import BASE, read, write, sha, now
from export_vllm import export
from final_test import generate, manifest, progress, METRIC_PY
from short_recovery.core import periodic_runs
from torch.utils.tensorboard import SummaryWriter


def prediction_scan(folder, label):
    records = []
    for item in read(FOLDER / 'inputs.json'):
        record = read(folder / 'predictions' / label / f"{item['key']}.json")
        assert record['status'] == 'ok' and not record['engineering_smoke']
        loops = list(periodic_runs(record['generated_ids']))
        records.append(dict(key=item['key'], dataset=item['dataset'],
                            truncated=record['truncated'], empty=record['parse_empty'],
                            long_loop=bool(loops), periodic_runs=loops,
                            generated_tokens=record['generated_tokens']))
    return dict(meetings=len(records), truncated=sum(r['truncated'] for r in records),
                empty=sum(r['empty'] for r in records),
                long_loop=sum(r['long_loop'] for r in records), records=records)


def main():
    assert not RUN.exists(), f'Refusing to overwrite evaluation: {RUN}'
    review = read(PROBLEM / 'review_decision.json')
    assert review['passed'], 'Problem meeting must pass review before full evaluation'
    assert review['problem_report_sha256'] == sha(PROBLEM / 'problem_review.json')
    state = read(CHECKPOINT / 'recovery_state.json')
    assert state['global_step'] == 150 and state['next_row_index'] == 1200
    assert read(PREVIOUS / 'test_complete.json')['complete']
    items = read(PREVIOUS / 'inputs.json')
    references = read(PREVIOUS / 'references.json')
    previous_protocol = read(PREVIOUS / 'protocol.json')
    counts = {name: sum(r['dataset'] == name for r in items)
              for name in ('alimeeting', 'aishell4', 'ami')}
    assert len(items) == 56 and counts == dict(alimeeting=20, aishell4=20, ami=16)
    assert all(r['split'] == 'test' for r in items)
    assert sorted({r['rank'] for r in items}) == [0, 1, 2, 3]
    assert previous_protocol['decoding']['temperature'] == 0
    assert previous_protocol['decoding']['repetition_penalty'] == 1.0
    assert previous_protocol['decoding']['presence_penalty'] == 0
    assert previous_protocol['decoding']['frequency_penalty'] == 0

    RUN.mkdir(parents=True)
    (RUN / 'logs').mkdir()
    writer = SummaryWriter(log_dir=str(RUN / 'test_events'), flush_secs=5)
    started = time.monotonic()
    os.environ['MOSS_REUSE_BASE'] = str(PREVIOUS)
    progress(writer, RUN, 'starting', 0, started)
    write(RUN / 'status.json', dict(stage='verifying_export', utc=now()))
    receipt = read(EXPORT / 'export_receipt.json')
    assert receipt['checkpoint'] == str(CHECKPOINT)
    for name, digest in receipt['files'].items():
        assert sha(EXPORT / name) == digest, name
    new_manifest = manifest(EXPORT)
    base_manifest = read(PREVIOUS / 'model_manifests.json')['base']
    assert manifest(BASE) == base_manifest
    # Keep the previous official inference processor/tokenizer and decoding route
    # exactly. The export does not convert weights; tied aliases are byte-checked.
    assert sha(EXPORT / 'tokenizer.json') == sha(BASE / 'tokenizer.json')
    write(FOLDER / 'inputs.json', items)
    write(FOLDER / 'references.json', references)
    write(FOLDER / 'model_manifest.json', new_manifest)
    write(FOLDER / 'model_manifests.json', {'base': base_manifest, 'sft': new_manifest})
    protocol = dict(scope='fixed_150_step_recovery_test_comparison',
                    checkpoint=str(CHECKPOINT), checkpoint_step=150,
                    checkpoint_rule='User-prespecified 150 optimizer updates; no claim of Dev selection.',
                    historical_comparison=str(PREVIOUS), datasets=counts,
                    problem_meeting_review=review,
                    models={'base': str(BASE), 'sft': str(EXPORT)},
                    generation=previous_protocol['generation'],
                    decoding=previous_protocol['decoding'], metrics=previous_protocol['metrics'],
                    inference_processor_source=str(BASE),
                    inference_tokenizer_sha256=sha(BASE / 'tokenizer.json'),
                    training_tokenizer_sha256=sha(CHECKPOINT / 'tokenizer.json'),
                    processor_policy='Same official inference preprocessing as the prior evaluation.',
                    base_runtime_is_historical=True,
                    split_note='Reuses the already evaluated test set for the fixed-checkpoint comparison.',
                    source_hashes={str(p):sha(p) for p in [
                        TASK/'evaluation/vllm_generate.py', TASK/'evaluation/selection.py',
                        TASK/'evaluation/score_v1.py', TASK/'evaluation/diagnostics.py',
                        TASK/'export_vllm.py', TASK/'final_test.py', Path(__file__)]}, utc=now())
    write(FOLDER / 'protocol.json', protocol)
    hashes = {}
    for item in items:
        rel = Path('predictions/base') / f"{item['key']}.json"
        record = read(PREVIOUS / rel)
        assert record['status'] == 'ok' and not record['engineering_smoke']
        assert record['backend_version'] == '0.23.1rc1.dev949+g68b4a1d58'
        assert record['model_manifest'] == base_manifest
        for key, value in item.items():
            assert record[key] == value, (item['key'], key)
        dest = FOLDER / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PREVIOUS / rel, dest)
        assert sha(dest) == sha(PREVIOUS / rel)
        hashes[str(rel)] = sha(dest)
    write(FOLDER / 'base_reuse.json', dict(source=str(PREVIOUS), records=56, sha256=hashes))
    progress(writer, RUN, 'base', 56, started)
    writer.add_text('test/protocol', json.dumps(protocol, ensure_ascii=False, indent=2), 0)
    writer.flush()
    write(RUN / 'status.json', dict(stage='generating', utc=now(), expected_new_predictions=56))
    print('STAGE=generating 56 complete meetings', flush=True)
    generation_start = time.monotonic()
    generate(RUN, FOLDER, EXPORT, 'sft', writer, 56, started)
    write(FOLDER / 'timing.json', dict(
        sft_wall_seconds_including_startup=time.monotonic()-generation_start,
        base_wall_seconds_including_startup=0,
        historical_base_wall_seconds_including_startup=read(PREVIOUS/'timing.json').get(
            'historical_base_wall_seconds_including_startup',
            read(PREVIOUS/'timing.json').get('base_wall_seconds_including_startup'))))
    progress(writer, RUN, 'scoring', 112, started)
    write(RUN / 'status.json', dict(stage='scoring', utc=now()))
    with (RUN/'logs/scoring.log').open('w') as log:
        subprocess.run([METRIC_PY, str(TASK/'evaluation/selection.py'), '--run', str(FOLDER)],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    metrics = read(FOLDER / 'metrics_summary.json')
    assert metrics['complete'] and metrics['scored_predictions'] == 112
    previous_metrics = read(PREVIOUS / 'metrics_summary.json')
    comparisons = {}
    for dataset in counts:
        key = f'sft/{dataset}/test'
        old, new = previous_metrics['groups'][key], metrics['groups'][key]
        comparisons[dataset] = {metric:dict(previous=old[metric], recovery=new[metric],
            difference=new[metric]-old[metric]) for metric in ('CER','cpCER','DeltaCP','truncated','empty_predictions')}
        comparisons[dataset]['DER025'] = dict(previous=old['DER_collar_0.25']['DER'],
            recovery=new['DER_collar_0.25']['DER'],
            difference=new['DER_collar_0.25']['DER']-old['DER_collar_0.25']['DER'])
        for metric in ('CER', 'cpCER', 'DeltaCP'):
            writer.add_scalar(f'recovery_test/{dataset}/{metric}', new[metric], 150)
        writer.add_scalar(f'recovery_test/{dataset}/DER025',new['DER_collar_0.25']['DER'],150)
    write(FOLDER / 'comparison_previous_sft.json', comparisons)
    scans = dict(base=prediction_scan(FOLDER, 'base'),
                 previous_sft=prediction_scan(PREVIOUS, 'sft'),
                 recovery=prediction_scan(FOLDER, 'sft'))
    write(FOLDER / 'raw_loop_comparison.json', scans)
    for label, scan in scans.items():
        for metric in ('truncated','empty','long_loop'):
            writer.add_scalar(f'recovery_test/{label}/{metric}',scan[metric],150)
    complete = dict(complete=True, checkpoint=str(CHECKPOINT), checkpoint_step=150,
                    new_predictions=56, reused_base_predictions=56,
                    metrics_sha256=sha(FOLDER/'metrics_summary.json'), utc=now())
    write(FOLDER / 'test_complete.json', complete)
    write(RUN / 'status.json', dict(stage='complete', **complete))
    progress(writer, RUN, 'complete', 112, started)
    writer.close()
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if RUN.exists():
            write(RUN/'failure.json', dict(error_type=type(exc).__name__, error=str(exc), utc=now()))
        raise
