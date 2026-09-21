"""Reproduce the completed 63443 short-trial report from verified archives."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'lr2e6-report'
OUT.mkdir(exist_ok=True)

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def query(name, rows, files, definition, components):
    with (OUT / f'{name}.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return {'rows': rows, 'source': {
        'label': name, 'files': files,
        'metricDefinitions': [{'label': name, 'definition': definition, 'componentIds': components}],
        'evidenceFlow': [{'title': 'Verified archives', 'detail': '63443, 63421 and corrected production evaluation 63441; archived original predictions include all failed cases.'},
                         {'title': 'Reproduction', 'detail': 'Run dkucc/full_attention_v2/lr2e6/build_report_data.py from this workspace.'}],
        'caveats': ['Six fixed dev meetings, not full 26 dev or 56 test.', 'Base scores reused from historical production predictions; no concurrent timing comparison.']},
        'methods': [{'language': 'text', 'code': definition}]}

new = ROOT / 'artifacts/63443/raw'
old = ROOT / 'artifacts/63421/raw'
prod = ROOT / 'artifacts/63441/raw/run'
verification = read(ROOT / 'artifacts/63443/download-verification.json')
assert hashlib.sha256((ROOT / 'artifacts/63443/logs-63443.tar.gz').read_bytes()).hexdigest() == verification['sha256']
end = read(new / 'run/outcome.json')
assert end['status'] == 'paused_for_sentinel_failure' and end['step'] == 50
e = new / 'run/evaluations/sentinel-50'
summary = read(e / 'metrics_summary.json')
assert summary['complete'] and not summary['errors'] and not summary['missing'] and summary['scored_predictions'] == 12
# Find the corrected six-meeting result by its declared population, never a probe.
historical = []
for p in prod.rglob('metrics_summary.json'):
    d = read(p)
    if d.get('scored_predictions') == 12:
        historical.append((p, d))
assert len(historical) == 1, [str(x[0]) for x in historical]
old_score_path, previous = historical[0]
train = []
matched = {}
for label, path in [('old', old), ('new', new)]:
    ranks = [[json.loads(s) for s in (path / f'checkpoints/train-rank-{r}.jsonl').read_text().splitlines() if s.strip()] for r in range(4)]
    assert all(len(x) == 50 for x in ranks)
    matched[label] = ranks
for i in range(50):
    row = {'step': i+1}
    for label in ['old', 'new']:
        records = [r[i] for r in matched[label]]
        assert all(r['step'] == i+1 for r in records)
        ms = [r['microbatches'][0] for r in records]
        tokens = sum(m['local_supervised_tokens'] for m in ms)
        assert all(m['global_supervised_tokens'] == tokens for m in ms)
        row[label+'_loss'] = sum(m['loss_numerator'] for m in ms)/tokens
        row[label+'_lr'] = records[0]['lr']
        row[label+'_tokens'] = tokens
    for r in range(4):
        a,b = [matched[l][r][i]['microbatches'][0] for l in ['old','new']]
        assert a['indices'] == b['indices'] and a['local_supervised_tokens'] == b['local_supervised_tokens']
    assert abs(row['new_lr']/row['old_lr'] - .2) < 1e-12
    train.append(row)
assert sum(r['new_tokens'] for r in train) == 4076380
teacher = []
for step in [0,25,50]:
    row = {'step': step}
    for label,path in [('old',old),('new',new)]:
        d=read(path / f'run/evaluations/fast-{step}/fast_summary.json')
        assert d['complete'] and len(d['records'])==6
        assert d['groups']['all']['tokens']==109603
        row[label+'_loss']=d['groups']['all']['loss']
    teacher.append(row)
scores=[]
for corpus in ['alimeeting','ami']:
    row={'dataset':corpus}
    for label,d,model in [('base',summary,'base'),('old',previous,'sft'),('new',summary,'sft')]:
        g=d['groups'][f'{model}/{corpus}/dev']
        for metric in ['CER','cpCER']: row[label+'_'+metric]=100*g[metric]
        row[label+'_DER025']=100*g['DER_collar_0.25']['DER']
        row[label+'_truncated']=g['truncated']
    scores.append(row)
diag=read(e/'output_diagnostics.json')
records=read(e/'per_record_metrics.json')
old_records={m['key']:m for m in read(old_score_path.parent/'per_record_metrics.json') if m['model']=='sft'}
base_records={m['key']:m for m in records if m['model']=='base'}
cases=[]
for m in records:
    if m['model']!='sft': continue
    k=m['key'];d=diag['sft/'+k];b=base_records[k];o=old_records[k]
    row={'key':k,'dataset':m['dataset'],'meeting':m['session_id'],'tokens':m['generated_tokens'],'eos':m['ended_eos'],'truncated':m['truncated'],'major_parse_loss':d['major_parse_loss'],'invalid_tags':d['invalid_tag_count'],'official_characters':d['official_characters'],'legal_scan_characters':d['legal_scan_characters']}
    for label,rec in [('base',b),('old',o),('new',m)]:
        row[label+'_cpCER']=100*rec['text']['cpCER']
        row[label+'_DER025']=100*rec['diarization']['DER_collar_0.25']['diarization error rate']
    cases.append(row)
assert len(cases)==6 and sum(x['truncated'] for x in cases)==1 and sum(x['major_parse_loss'] for x in cases)==1
q={
 'training':query('training',train,[str(new/'checkpoints/train-rank-*.jsonl'),str(old/'checkpoints/train-rank-*.jsonl')],'Each step is summed NLL divided by global supervised tokens; matched indices and tokens checked at all 50 updates.',['train-chart','method']),
 'teacher':query('teacher',teacher,[str(new/'run/evaluations/fast-*/fast_summary.json'),str(old/'run/evaluations/fast-*/fast_summary.json')],'All 109603 supervised tokens of six fixed dev meetings; step zero reused, not newly measured.',['loss-chart','loss-context']),
 'scores':query('scores',scores,[str(e/'metrics_summary.json'),str(old_score_path)],'Error rates in percent; within-corpus reference character/time weighted aggregation, three meetings per corpus.',['cp-chart','der-chart','score-table','summary']),
 'cases':query('cases',cases,[str(e/'per_record_metrics.json'),str(e/'output_diagnostics.json')],'One row per meeting; EOS does not imply valid parsing. All failed outputs remain in scores.',['case-table','failure-context'])
}
snapshot={'surface':'report','title':'学习率降至五分之一：AMI 改善，中文长会议仍失败','generatedAt':end['completed_utc'],'buildStatus':'creating','filters':{},'queries':q,'report':{'asOf':end['completed_utc'],'scope':'作业 63443 · 50 步 full-attention SFT · 六场 dev 短对照'}}
(OUT/'reviewed.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'analysis_verification.json').write_text(json.dumps({'source_archive':verification,'matched_steps':50,'supervised_tokens':4076380,'complete_scored_records':12,'independent_meetings':6,'eos':5,'truncated':1,'major_parse_loss':1,'outcome':end,'report_pending':True},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'report_dir':str(OUT),'query_rows':{k:len(v['rows']) for k,v in q.items()},'failures':[x for x in cases if x['truncated'] or x['major_parse_loss']]},ensure_ascii=True))
