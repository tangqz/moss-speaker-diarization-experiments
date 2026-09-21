"""Compare verified 25/50-step full generations without changing official scores."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V2 = Path(__file__).resolve().parent.parent
ART = V2 / 'artifacts/63432'
RUN = ART / 'raw/run'
ORIGINAL = V2 / 'artifacts/63421/raw/run/evaluations/sentinel-50'
OUT = V2 / 'stage50-report'
KEY = 'ami/dev/TS3004c'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    verified = read(ART / 'download-verification.json')
    assert sha(ART / 'logs-63432.tar.gz') == verified['sha256']
    assert read(RUN / 'job_exit.json')['exit_code'] == 0
    outcome = read(RUN / 'outcome.json')
    assert outcome['status'] == 'completed' and outcome['training_updates'] == 0
    for result in outcome['results']:
        metrics = result['metrics']
        assert metrics['complete'] and metrics['scored_predictions'] == 2
        assert not metrics['errors'] and not metrics['missing']
    rows, records = [], {}
    variants = [
        ('base', 0, '历史 base', ORIGINAL, 'base'),
        ('step25', 25, '第 25 步', RUN / 'checkpoint-25', 'sft'),
        ('step50_original', 50, '第 50 步·首次', ORIGINAL, 'sft'),
        ('step50_repeat', 50, '第 50 步·复核', RUN / 'checkpoint-50', 'sft'),
    ]
    for variant, step, label, folder, model in variants:
        path = folder / f'predictions/{model}/{KEY}.json'
        record = read(path)
        records[variant] = record
        metric = next(m for m in read(folder / 'per_record_metrics.json')
                      if m['session_id'] == 'TS3004c' and m['model'] == model)
        runs = []
        for text, group in itertools.groupby(enumerate(record['segments']),
                key=lambda item: item[1]['text'].strip().lower()):
            items = list(group)
            runs.append(dict(text=text, count=len(items), first_segment=items[0][0] + 1,
                             start=items[0][1]['start'], end=items[-1][1]['end']))
        longest = max(runs, key=lambda item: item['count'])
        rows.append(dict(variant=variant, label=label, checkpoint_step=step,
            generated_tokens=record['generated_tokens'], ended_eos=record['ended_eos'],
            truncated=record['truncated'], parsed_segments=len(record['segments']),
            repeat_text=longest['text'], repeat_count=longest['count'],
            repeat_start_seconds=longest['start'], repeat_end_seconds=longest['end'],
            CER=100 * metric['text']['CER'], cpCER=100 * metric['text']['cpCER'],
            DER025=100 * metric['diarization']['DER_collar_0.25']['diarization error rate'],
            source_file=str(path), source_sha256=sha(path)))
    old, repeated = records['step50_original'], records['step50_repeat']
    same = old['generated_ids'] == repeated['generated_ids']
    assert same and len(old['generated_ids']) == 65536
    identity = [dict(checkpoint_step=50, compared_tokens=65536, all_tokens_equal=same,
                     model_manifest_equal=old['model_manifest'] == repeated['model_manifest'],
                     raw_text_equal=old['raw_text'] == repeated['raw_text'],
                     scope='This one checkpoint and meeting; not a general determinism guarantee.')]
    reference = read(RUN / 'checkpoint-25/references.json')[KEY]['segments']
    last_end = max(s['end'] for s in reference)
    duration = old['duration']
    coverage = [dict(meeting=KEY, audio_seconds=duration, reference_last_end_seconds=last_end,
                     audio_after_last_reference_seconds=duration - last_end,
                     step25_loop_start_seconds=rows[1]['repeat_start_seconds'],
                     step50_loop_start_seconds=rows[2]['repeat_start_seconds'])]
    data = dict(job=63432, completed_utc=outcome['completed_utc'], rows=rows,
                identity=identity, reference_coverage=coverage, archive=verified,
                next_diagnostic=read(V2 / 'diagnostics/early_run.json'),
                scope='One known dev failure, diagnostic only; no formal selection or test use.')
    write(OUT / 'recheck_analysis.json', data)
    for name, values in [('recheck_comparison', rows), ('recheck_identity', identity), ('reference_coverage', coverage)]:
        with (OUT / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    queries = {}
    for name, values, definition, files in [
        ('recheck_comparison', rows,
         'TS3004c only; official CER/cpCER/DER025 percentages; adjacent normalized segment text identifies the longest structured repetition.',
         [str(p / 'per_record_metrics.json') for p in [ORIGINAL, RUN / 'checkpoint-25', RUN / 'checkpoint-50']] + [r['source_file'] for r in rows]),
        ('recheck_identity', identity,
         'Elementwise equality of every generated token ID and equality of model manifests and raw text; no tolerance or prefix-only comparison.',
         [rows[2]['source_file'], rows[3]['source_file']]),
        ('reference_coverage', coverage,
         'Maximum annotated segment end versus full audio duration and generated repetition start. An unannotated interval is not evidence of silence.',
         [str(RUN / 'checkpoint-25/references.json'), rows[1]['source_file'], rows[2]['source_file']]),
    ]:
        queries[name] = dict(rows=values, source=dict(label=name, files=files,
            metricDefinitions=[definition], evidenceFlow=[
                dict(title='Verified inference-only archive', detail=f"63432: {verified['files_verified']} files, SHA256 {verified['sha256']}"),
                dict(title='Reproduce', detail='dkucc/full_attention_v2/diagnostics/analyze_recheck.py')],
            caveats=['单场已知 dev 失败案例，不代表总体、不改变正式 checkpoint 选择。',
                     '历史 base 输出复用，仅比较质量；第 50 步两次输出在本案例逐 token 相同。',
                     '参考标注结束不代表真实语音结束；当前未听审尾段，不能将无标注区间直接称作静音。']),
            methods=[dict(language='text', code=definition)])
    for filename in ['reviewed.json', 'app/src/data.json']:
        path = OUT / filename
        snapshot = read(path)
        snapshot['queries'].update(queries)
        snapshot['generatedAt'] = outcome['completed_utc']
        snapshot['report']['asOf'] = '2026-09-13'
        snapshot['report']['scope'] = '50 步六场 dev sentinel；追加 TS3004c 的 25/50 步复核'
        snapshot['buildStatus'] = 'updating'
        write(path, snapshot)
    analysis = read(OUT / 'analysis.json')
    analysis['diagnostic_result'] = data
    analysis['active_diagnostic'] = data['next_diagnostic']
    write(OUT / 'analysis.json', analysis)
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'svg.fonttype': 'none',
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = [r['label'] for r in rows]
    colors = ['#176b91', '#cb963c', '#ce6234', '#a84930']
    bars = axes[0].bar(labels, [r['cpCER'] for r in rows], color=colors)
    axes[0].bar_label(bars, fmt='%.2f', padding=3)
    axes[0].set(title='同一场 TS3004c：cpCER（%）', ylim=(0, 28))
    bars = axes[1].bar(labels, [r['repeat_count'] for r in rows], color=colors)
    axes[1].bar_label(bars, fmt='%d', padding=3)
    axes[1].set(title='最长连续相同文本片段数', ylim=(0, 2000))
    for axis in axes:
        axis.tick_params(axis='x', labelrotation=15)
    fig.suptitle('第 25 步已出现循环；第 50 步两次输出逐 token 一致', fontsize=13)
    fig.tight_layout()
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'recheck_comparison.{ext}', dpi=175, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'rows': len(rows), 'repeat_identity': identity, 'coverage': coverage,
                      'archive': verified, 'next_job': data['next_diagnostic']['job']}, ensure_ascii=True))


if __name__ == '__main__':
    main()
