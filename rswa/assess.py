"""Frozen dev-only functional checks, ranking and extension conditions."""
from pathlib import Path
from common import read,write

def assess(run):
    run=Path(run);summary=read(run/'metrics_summary.json')
    assert summary['complete']
    items=read(run/'inputs.json');refs=read(run/'references.json')
    failures=[]
    for item in items:
        pred=read(run/'predictions/sft'/f"{item['key']}.json")
        reasons=[];d=pred['diagnostics'];segments=pred['segments']
        if pred['truncated']:reasons.append('budget_truncation')
        if pred['parse_empty']:reasons.append('empty_prediction')
        if d['major_parse_loss']:reasons.append('major_parse_loss')
        if d['longest_identical_run']['count']>=128:reasons.append('pathological_identical_token_loop')
        if d['timestamp_loop']:reasons.append('repeated_identical_timestamped_segment_loop')
        last_ref=max((s['end'] for s in refs[item['key']]['segments']),default=0)
        last_hyp=max((s['end'] for s in segments),default=0)
        ref_chars=sum(len(s['text']) for s in refs[item['key']]['segments'])
        hyp_chars=sum(len(s['text']) for s in segments)
        if last_ref>60 and last_hyp<.8*last_ref and hyp_chars<.7*ref_chars:
            reasons.append('severe_early_end_with_text_deficit')
        if reasons:failures.append(dict(key=item['key'],reasons=reasons))
    groups=summary['groups']
    corpora=sorted({r['dataset'] for r in items})
    score=sum(100*(groups[f'sft/{c}/dev']['cpCER']+groups[f'sft/{c}/dev']['DER_collar_0.25']['DER'])/2
              for c in corpora)/len(corpora)
    ce_files=list((run/'ce').rglob('*.json')) if (run/'ce').exists() else []
    ce=[read(p) for p in ce_files]
    if ce:assert len(ce)==len(items)
    result=dict(run=str(run),functional=not failures,failures=failures,S=score,
                ce=sum(r['loss_sum'] for r in ce)/sum(r['target_tokens'] for r in ce) if ce else None,
                RTF=sum(groups[f'sft/{c}/dev']['processing_hours'] for c in corpora)/
                    sum(groups[f'sft/{c}/dev']['audio_hours'] for c in corpora),
                corpora={c:groups[f'sft/{c}/dev'] for c in corpora})
    write(run/'assessment.json',result);return result

def extending(previous,current):
    if not previous['functional'] and not current['functional']:return False
    improved=previous['S']-current['S']>=.1
    ce_improved=(previous['ce'] is not None and current['ce'] is not None and
                 current['ce']<=.99*previous['ce'] and current['S']<=previous['S']+.1)
    return improved or ce_improved

def noninferior(candidate,reference):
    return all(candidate['corpora'][c]['CER']<=reference['corpora'][c]['CER']+.005 and
               candidate['corpora'][c]['cpCER']<=reference['corpora'][c]['cpCER']+.005 and
               candidate['corpora'][c]['DER_collar_0.25']['DER']<=reference['corpora'][c]['DER_collar_0.25']['DER']+.01
               for c in candidate['corpora'])

def choose(history,reference=None,require_functional=True):
    eligible=[r for r in history if (r['functional'] or not require_functional) and
              (reference is None or noninferior(r,reference))]
    if not eligible:return None
    minimum=min(r['S'] for r in eligible)
    return min((r for r in eligible if r['S']<=minimum+.1),key=lambda r:(r['RTF'],r['step']))
