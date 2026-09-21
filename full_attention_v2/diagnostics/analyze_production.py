"""Review the archived production-preprocessing reevaluation without rescoring it."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

V = Path(__file__).resolve().parent.parent
OUT = V / 'stage50-report'
RUN = V / 'artifacts/63441/raw/run'
ORIGINAL = V / 'artifacts/63421/raw/run/evaluations/sentinel-50'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    archive = read(V / 'artifacts/63441/download-verification.json')
    assert sha(V / 'artifacts/63441/logs-63441.tar.gz') == archive['sha256']
    assert read(RUN / 'job_exit.json')['exit_code'] == 0
    outcome = read(RUN / 'outcome.json')
    assert outcome['status'] == 'completed' and outcome['training_updates'] == 0
    manifest = read(RUN / 'case_manifest.json')
    assert len(manifest) == 8
    summaries, records = {}, {}
    for name, count in [('sentinel-50', 12), ('base-check', 2), ('step10-check', 2)]:
        folder = RUN / 'evaluations' / name
        summary = read(folder / 'metrics_summary.json')
        assert summary['complete'] and summary['scored_predictions'] == count
        assert not summary['missing'] and not summary['errors']
        summaries[name] = summary
        records[name] = read(folder / 'per_record_metrics.json')
        assert len(records[name]) == count
    scores = []
    for dataset, label in [('alimeeting', 'AliMeeting'), ('ami', 'AMI')]:
        row = dict(dataset=label, dataset_key=dataset, meetings=3)
        for model in ['base', 'sft']:
            group = summaries['sentinel-50']['groups'][f'{model}/{dataset}/dev']
            selected = [r for r in records['sentinel-50'] if r['dataset'] == dataset and r['model'] == model]
            assert len(selected) == 3
            chars = sum(r['text']['reference_characters'] for r in selected)
            assert chars == group['reference_characters']
            for metric, numerator in [('CER', 'character_errors'), ('cpCER', 'cp_character_errors')]:
                errors = sum(r['text'][numerator] for r in selected)
                assert errors == group[numerator] and abs(errors / chars - group[metric]) < 1e-12
                row[f'{model}_{metric}'] = 100 * group[metric]
            diar = [r['diarization']['DER_collar_0.25'] for r in selected]
            total = sum(r['total'] for r in diar)
            errors = sum(r['confusion'] + r['false alarm'] + r['missed detection'] for r in diar)
            assert abs(errors / total - group['DER_collar_0.25']['DER']) < 1e-12
            row[f'{model}_DER025'] = 100 * group['DER_collar_0.25']['DER']
            row[f'{model}_truncated'] = group['truncated']
            row[f'{model}_empty'] = group['empty_predictions']
        for metric in ['CER', 'cpCER', 'DER025']:
            row[f'delta_{metric}_pp'] = row[f'sft_{metric}'] - row[f'base_{metric}']
        scores.append(row)

    cases, preds = [], {}
    for case in manifest:
        key = case['item']['key']
        path = RUN / 'cases' / case['id'] / 'predictions/sft' / (key + '.json')
        pred = read(path)
        assert pred['status'] == 'ok'
        assert pred['preprocessing']['context'] == 'production_cuda_bfloat16_autocast'
        assert len(pred['generated_ids']) == pred['generated_tokens']
        metric = next(r for r in records[case['group']] if r['key'] == key and r['model'] == 'sft')
        runs = []
        for text, g in itertools.groupby(enumerate(pred['segments']), key=lambda x: x[1]['text'].strip().lower()):
            items = list(g)
            runs.append(dict(text=text, count=len(items), first=items[0][0] + 1,
                             start=items[0][1]['start'], end=items[-1][1]['end']))
        longest = max(runs, key=lambda r: r['count'])
        diag = pred['diagnostics']
        row = dict(case_id=case['id'], group=case['group'], step=case['step'],
            label='base·生产入口' if case['step'] == 0 else f"第 {case['step']} 步",
            key=key, dataset=pred['dataset'], meeting=pred['session_id'],
            generated_tokens=pred['generated_tokens'], ended_eos=pred['ended_eos'], truncated=pred['truncated'],
            parsed_segments=len(pred['segments']), last_parsed_end=pred['segments'][-1]['end'],
            max_parsed_end=max(s['end'] for s in pred['segments']),
            official_characters=diag['official_characters'], legal_scan_characters=diag['legal_scan_characters'],
            major_parse_loss=diag['major_parse_loss'], invalid_tag_count=diag['invalid_tag_count'],
            longest_identical_token_run=diag['longest_identical_run']['count'],
            longest_text_run=longest['count'], repeat_text=longest['text'],
            repeat_start=longest['start'], repeat_end=longest['end'],
            repeated_tag_unit_occurrences=pred['raw_text'].count('[66.65][66.66][Música]') if case['step'] == 10 else 0,
            CER=100 * metric['text']['CER'], cpCER=100 * metric['text']['cpCER'],
            DER025=100 * metric['diarization']['DER_collar_0.25']['diarization error rate'],
            source_file=str(path), source_sha256=sha(path))
        cases.append(row)
        preds[case['id']] = pred

    base = preds['base-check']
    historical_path = RUN / 'evaluations/base-check/predictions/base/ami/dev/TS3004c.json'
    historical = read(historical_path)
    identity = [dict(meeting='TS3004c', compared_tokens=len(base['generated_ids']),
        all_generated_tokens_equal=base['generated_ids'] == historical['generated_ids'],
        raw_text_equal=base['raw_text'] == historical['raw_text'],
        ended_eos=base['ended_eos'], scope='This full meeting only; not a general determinism guarantee.')]
    assert identity[0]['all_generated_tokens_equal'] and identity[0]['raw_text_equal']
    assert outcome['base_recheck']['all_generated_tokens_equal']

    old_records = read(ORIGINAL / 'per_record_metrics.json')
    pairs = []
    for row in cases:
        if row['step'] != 50:
            continue
        old_path = ORIGINAL / 'predictions/sft' / (row['key'] + '.json')
        old = read(old_path)
        new = preds[row['case_id']]
        assert old['model_manifest']['files'] == new['model_manifest']['files']
        assert old['decoding'] == new['decoding']
        metric = next(r for r in old_records if r['key'] == row['key'] and r['model'] == 'sft')
        base_metric = next(r for r in records['sentinel-50'] if r['key'] == row['key'] and r['model'] == 'base')
        pairs.append(dict(meeting=row['meeting'], dataset=row['dataset'],
            historical_base_cpCER=100 * base_metric['text']['cpCER'],
            original_sft_cpCER=100 * metric['text']['cpCER'], corrected_sft_cpCER=row['cpCER'],
            corrected_minus_original_cpCER_pp=row['cpCER'] - 100 * metric['text']['cpCER'],
            original_generated_tokens=old['generated_tokens'], corrected_generated_tokens=row['generated_tokens'],
            original_ended_eos=old['ended_eos'], corrected_ended_eos=row['ended_eos'],
            weight_files_equal=True, decoding_settings_equal=True, original_source=str(old_path),
            corrected_source=row['source_file']))

    failures = []
    for row in cases:
        if not row['truncated']:
            continue
        detail = ('连续 1,657 个 Mm. 片段，时间微增' if row['step'] == 50 and row['dataset'] == 'ami'
                  else '连续 51,141 个相同 token（下）' if row['dataset'] == 'alimeeting'
                  else '4,085 个非法 Música 标签，时间停在约 66.7 秒')
        failures.append({**row, 'failure_detail': detail})
    ts = sorted([r for r in cases if r['meeting'] == 'TS3004c'], key=lambda r: r['step'])
    quality = read(RUN / 'evaluations/sentinel-50/quality_status.json')
    review = [dict(job=63441, completed_utc=outcome['completed_utc'], generated_cases=8,
        unique_sentinel_meetings=6, scored_records=16, execution_errors=0,
        step50_truncated=sum(r['truncated'] for r in cases if r['step'] == 50),
        step50_empty=sum(r['official_characters'] == 0 for r in cases if r['step'] == 50),
        step50_major_parse_loss=quality['major_parse_loss'], quality_pass=False,
        training_updates=0, last_train_step=50, formal_training='paused',
        full_dev_performed=False, test_performed=False)]
    assert review[0]['step50_truncated'] == quality['catastrophic_count_with_overlap'] == 2
    result = dict(job=63441, completed_utc=outcome['completed_utc'], archive=archive,
        scores=scores, cases=cases, identity=identity, pairs=pairs, failures=failures, review=review,
        conclusion='Production preprocessing is restored; full base output reproduces historical tokens, but step 50 fails 2/6 sentinels and step 10 also loops. Training remains paused.')
    write(OUT / 'production_analysis.json', result)
    analysis = read(OUT / 'analysis.json')
    analysis.update(production_review=result, production_comparison_requires_reevaluation=False,
                    active_diagnostic=None, latest_quality_eligible=False)
    write(OUT / 'analysis.json', analysis)

    definitions = {
        'production_scores': (scores, ['production-cpcer', 'production-score-table', 'production-summary'],
            'Six fixed full dev meetings, three per corpus. CER/cpCER are summed errors divided by summed reference characters; DER is summed error seconds divided by summed scored seconds, collar 0.25 s. Values are percent and deltas are percentage points. Historical production-path base predictions are reused; failures remain included.'),
        'production_cases': (cases, ['production-summary', 'production-failure-context', 'production-next', 'methods'],
            'Eight fresh full generations: six step-50 sentinels, one untrained base TS3004c and one step-10 TS3004c. Candidates in predictions/sft are labeled by actual checkpoint, including base step zero. Segment repetition groups adjacent stripped lower-case text; timestamp maxima and output-order final ends are kept separately.'),
        'production_identity': (identity, ['preprocessing-correction', 'production-summary', 'production-next'],
            'Elementwise equality of all 26337 fresh versus historical base token IDs and equality of raw text on TS3004c. No prefix-only substitution; scope is one meeting.'),
        'production_pairs': (pairs, ['production-preprocessing-impact'],
            'Same step-50 weight-file hashes and decoding settings, comparing original outside-autocast preparation with corrected production preparation on each meeting. Original and corrected scores are retained separately. GPU worker placement can differ; this is not a guarantee of global bitwise determinism.'),
        'production_failures': (failures, ['production-failure-context', 'production-failure-table'],
            'The three truncated fresh generations, including two of the six step-50 sentinels and the separate step-10 diagnostic. Token/segment/tag repetition counts describe different units and must not be added. Major-parse-loss false does not establish valid or complete raw output.'),
        'production_ts': (ts, ['production-ts-chart', 'production-failure-context'],
            'TS3004c base, step 10 and step 50, all using production CUDA BF16 audio preprocessing, full audio, greedy decoding and the unchanged scorer. Single meeting diagnostic, not formal checkpoint selection.'),
        'production_review': (review, ['production-summary', 'decision', 'production-next', 'methods'],
            '63441 completed 8 fresh outputs and 16 scoring records across three comparisons. Base copies and TS3004c repetitions overlap; only six independent sentinel meetings. Evaluation made zero weight updates; training is still paused at 50 and full dev/test are unperformed.'),
    }
    evidence_files = [str(RUN / 'outcome.json'), str(RUN / 'protocol.json'), str(RUN / 'case_manifest.json'),
                      str(RUN / 'evaluations/sentinel-50/per_record_metrics.json'),
                      str(RUN / 'evaluations/sentinel-50/metrics_summary.json')]
    for file in ['reviewed.json', 'app/src/data.json']:
        path = OUT / file
        snapshot = read(path)
        for name, (values, components, definition) in definitions.items():
            files = list(evidence_files)
            if name in ['production_cases', 'production_failures', 'production_ts']:
                files += [r['source_file'] for r in values]
            elif name == 'production_identity':
                files += [str(historical_path), next(r['source_file'] for r in cases if r['step'] == 0)]
            elif name == 'production_pairs':
                files += [r['original_source'] for r in pairs] + [r['corrected_source'] for r in pairs]
            snapshot['queries'][name] = dict(rows=values, source=dict(label=name, files=files,
                metricDefinitions=[dict(label=name, definition=definition, componentIds=components)],
                evidenceFlow=[dict(title='Verified complete archive', detail=f"63441: {archive['files_verified']} files, SHA256 {archive['sha256']}"),
                    dict(title='Reproduce', detail='dkucc/full_attention_v2/diagnostics/analyze_production.py')],
                caveats=['固定六场 sentinel；尚无全 26 场 dev、56 场 test 或合格 SFT。',
                         '历史 base 复用用于质量；不能用其计时与本轮推导同期速度变化。',
                         '参考尾段未听审，无标注不等于静音。']),
                methods=[dict(language='text', code=definition)])
        for query in snapshot['queries'].values():
            caveats = query.get('source', {}).get('caveats', [])
            query['source']['caveats'] = [c.replace('生产一致的完整复评正在运行。', '生产一致完整复评已完成，见 production_* 查询。') for c in caveats]
        snapshot['generatedAt'] = outcome['completed_utc']
        snapshot['report']['asOf'] = '2026-09-13'
        snapshot['report']['scope'] = '第 50 步暂停；预处理已对齐，完整复评六场中两场截断'
        snapshot['buildStatus'] = 'updating'
        write(path, snapshot)
    for name, (values, _, _) in definitions.items():
        with (OUT / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'],
        'axes.unicode_minus': False, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    x = np.arange(2)
    for ax, metric, label in zip(axes, ['cpCER', 'DER025'], ['cpCER (%)', 'DER .25 (%)']):
        for offset, model, color, title in [(-.19, 'base', '#176b91', 'base'), (.19, 'sft', '#b44235', 'SFT 50')]:
            bars = ax.bar(x + offset, [r[f'{model}_{metric}'] for r in scores], .36, label=title, color=color)
            ax.bar_label(bars, fmt='%.2f', padding=3)
        ax.set(xticks=x, xticklabels=[r['dataset'] for r in scores], ylabel=label, ylim=(0, 30))
        ax.legend(frameon=False)
    fig.suptitle('生产入口复评：第 50 步仍未通过六场质量检查', fontsize=15)
    fig.text(.5, .015, '每个语料三场完整 dev；全部失败计入。AliMeeting / AMI 各一场达到生成上限。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .94))
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'production_comparison.{ext}', dpi=175, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(dict(scores=scores, identity=identity, failures=[{k:r[k] for k in ['meeting', 'step', 'cpCER', 'failure_detail']} for r in failures], archive=archive), ensure_ascii=True))


if __name__ == '__main__':
    main()
