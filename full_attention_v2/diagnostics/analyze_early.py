"""Archive-backed early-checkpoint and preprocessing audit report data."""
import csv
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V = Path(__file__).resolve().parent.parent
OUT = V / 'stage50-report'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    archives = []
    for job in [63439, 63440]:
        folder = V / f'artifacts/{job}'
        verified = read(folder / 'download-verification.json')
        assert sha(folder / f'logs-{job}.tar.gz') == verified['sha256']
        assert read(folder / 'raw/run/job_exit.json')['exit_code'] == 0
        assert read(folder / 'raw/run/outcome.json')['status'] == 'completed'
        archives.append(verified)
    run = V / 'artifacts/63439/raw/run'
    probe = V / 'artifacts/63440/raw/run'
    rows = []
    for step in [0, 1, 10, 20]:
        folder = run / f'checkpoint-{step}'
        metrics = read(folder / 'metrics_summary.json')
        assert metrics['complete'] and metrics['scored_predictions'] == 2
        assert not metrics['errors'] and not metrics['missing']
        pred = read(folder / 'predictions/sft/ami/dev/TS3004c.json')
        m = metrics['groups']['sft/ami/dev']
        groups = []
        for text, g in itertools.groupby(enumerate(pred['segments']), key=lambda x: x[1]['text'].strip().lower()):
            items = list(g)
            groups.append(dict(text=text, count=len(items), start=items[0][1]['start'], end=items[-1][1]['end']))
        longest = max(groups, key=lambda x: x['count'])
        rows.append(dict(step=step, label='base·本轮预处理' if step == 0 else f'第 {step} 步',
            preprocessing='outside_cuda_autocast', generated_tokens=pred['generated_tokens'],
            ended_eos=pred['ended_eos'], truncated=pred['truncated'],
            CER=100 * m['CER'], cpCER=100 * m['cpCER'], DER025=100 * m['DER_collar_0.25']['DER'],
            parsed_segments=len(pred['segments']), last_parsed_end=pred['segments'][-1]['end'],
            longest_text_run=longest['count'], repeat_text=longest['text'], repeat_start=longest['start'],
            invalid_tag_count=pred['diagnostics']['invalid_tag_count'],
            step10_repeated_unit_occurrences=pred['raw_text'].count('[66.70][66.70][?]') if step == 10 else 0))
    outcome = read(probe / 'outcome.json')
    probe_rows = []
    for item in outcome['results']:
        probe_rows.append(dict(step=item['step'], mode=item['mode'], prefix_tokens=item['generated_tokens'],
            matches_recent_prefix=item['recent_prefix_equal'], matches_historical_base_prefix=item['historical_base_prefix_equal'],
            historical_base_common_prefix=item['historical_base_common_prefix'],
            invalid_tag_count=item['diagnostics']['invalid_tag_count'],
            invalid_tag_example=item['diagnostics']['invalid_tags'][0] if item['diagnostics']['invalid_tags'] else ''))
    features = outcome['input_differences']['input_features']
    feature_rows = [dict(feature_max_abs_difference=features['max_abs'], feature_rms_difference=features['rms'],
        changed_fraction=features['changed_fraction'], returned_dtype_both='torch.float32',
        input_ids_equal=True, attention_mask_equal=True, audio_lengths_equal=True, audio_chunk_mapping_equal=True)]
    for name, values in [('early_comparison', rows), ('preprocessing_probe', probe_rows), ('feature_precision', feature_rows)]:
        with (OUT / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    sources = {
        'early_comparison': (rows, [str(run / f'checkpoint-{s}/metrics_summary.json') for s in [0, 1, 10, 20]] +
            [str(run / f'checkpoint-{s}/predictions/sft/ami/dev/TS3004c.json') for s in [0, 1, 10, 20]],
            'All four candidates used the same outside-autocast audio preparation; base here is a fresh untrained base, not the historical production comparator. Official metrics remain unchanged.'),
        'preprocessing_probe': (probe_rows, [str(probe / 'outcome.json')],
            'Paired 512-token prefixes on identical weights and full audio; only audio-preprocessing CUDA BF16 autocast changes. Not a full-generation quality test.'),
        'feature_precision': (feature_rows, [str(probe / 'input_comparison.json'), str(V / 'diagnostics/inference_source_audit.jsonl')],
            'Elementwise full feature-tensor comparison; both output tensors are float32, but their intermediate precision differs. Token IDs and mappings match exactly.'),
    }
    for file in ['reviewed.json', 'app/src/data.json']:
        p = OUT / file
        data = read(p)
        for name, (values, files, definition) in sources.items():
            data['queries'][name] = dict(rows=values, source=dict(label=name, files=files,
                metricDefinitions=[definition], evidenceFlow=[dict(title='Verified archives', detail='63439: 60 files; 63440: 13 files; SHA256 receipts retained.'),
                dict(title='Reproduce', detail='dkucc/full_attention_v2/diagnostics/analyze_early.py')],
                caveats=['单场 TS3004c 的诊断，不代表全部开发集或测试集。',
                         '63421/63432/63439 的完整生成在音频预处理精度上与历史生产路径不同；旧输出保留为该路径的诊断。',
                         '512-token 探针不证明整场生成通过，生产一致的完整复评正在运行。']),
                methods=[dict(language='text', code=definition)])
        caveat = '更正：历史 base 音频预处理处于 CUDA BF16 autocast，本轮原生成脚本处于其外；该历史比较不能直接作为生产一致的模型验收，需看修正后的复评。'
        for name in ['scores', 'per_meeting', 'case_summary', 'timeline', 'recheck_comparison', 'recheck_identity', 'reference_coverage']:
            q = data['queries'][name]
            q['source']['caveats'] = [x for x in q['source'].get('caveats', []) if not x.startswith('更正：')]
            q['source']['caveats'].insert(0, caveat)
        data['generatedAt'] = outcome['completed_utc']
        data['report']['scope'] = '第 50 步暂停；早期权重与预处理精度审计；生产一致复评待完成'
        data['buildStatus'] = 'updating'
        write(p, data)
    result = dict(completed_early_job=63439, completed_probe_job=63440, rows=rows, probe=probe_rows,
        feature_precision=feature_rows, archives=archives, active_recheck=read(V / 'diagnostics/production_run.json'),
        conclusion='First checked full-generation failure is step 10 under the original v2 preprocessing path; production-matched full evaluation remains pending.')
    write(OUT / 'early_analysis.json', result)
    analysis = read(OUT / 'analysis.json')
    analysis.update(early_audit=result, production_comparison_requires_reevaluation=True, active_diagnostic=result['active_recheck'])
    write(OUT / 'analysis.json', analysis)
    plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei', 'SimHei', 'DejaVu Sans'], 'axes.unicode_minus': False,
                         'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar([r['label'] for r in rows], [r['cpCER'] for r in rows], color=['#176b91', '#438b73', '#b44235', '#ce7a37'])
    ax.bar_label(bars, fmt='%.2f', padding=3)
    ax.set(ylim=(0, 112), ylabel='cpCER (%)', title='TS3004c：原 v2 预处理路径下的早期检查点')
    fig.text(.5, .005, '四者使用同一预处理；生产路径完整复评仍待完成。第 10 步主要为时间戳/非法标签循环。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .035, 1, 1))
    for ext in ['png', 'svg']:
        fig.savefig(OUT / f'early_comparison.{ext}', dpi=175, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
