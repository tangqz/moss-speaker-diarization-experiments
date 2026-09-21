"""Frozen dev-only selection rules; metric environment, no GPU dependency."""
import argparse
import sys
from common import read,write,emit,MILESTONES
from diagnostics import output_diagnostics
import score_v1


def selection_score(groups,model='sft'):
    return sum((100*groups[f'{model}/{dataset}/dev']['cpCER']+
        100*groups[f'{model}/{dataset}/dev']['DER_collar_0.25']['DER'])/2
        for dataset in ['alimeeting','ami'])/2


def assess(summary,major):
    assert summary['complete'] and summary['scored_predictions']==52
    groups=summary['groups']
    failures=[]
    for dataset in ['alimeeting','ami']:
        b,s=[groups[f'{label}/{dataset}/dev'] for label in ['base','sft']]
        if s['truncated'] or s['empty_predictions']:failures.append(dataset+':generation_failure')
        if s['cpCER']>b['cpCER']+.01:failures.append(dataset+':cpCER_regression_over_1pp')
        if s['DER_collar_0.25']['DER']>b['DER_collar_0.25']['DER']+.01:
            failures.append(dataset+':DER025_regression_over_1pp')
    if major:failures.append('major_parse_loss')
    score=selection_score(groups)
    base=selection_score(groups,'base')
    if score>=base:failures.append('selection_score_did_not_improve')
    return dict(score=score,base_score=base,eligible=not failures,ineligibility=failures)


def decide(history):
    assert history and len({r['step'] for r in history})==len(history)
    history=sorted(history,key=lambda r:r['step'])
    best=history[0]['base_score'];strikes=0
    for row in history:
        strikes=strikes+1 if row['score']>=best+.5 else 0
        best=min(best,row['score'])
    eligible=[r for r in history if r['eligible']]
    selected=None
    if eligible:
        minimum=min(r['score'] for r in eligible)
        selected=min((r for r in eligible if r['score']<=minimum+.1),key=lambda r:r['step'])['step']
    return dict(history=history,selected_step=selected,early_stop=strikes>=2,
        successive_worse_checkpoints=strikes,best_observed_score=best,
        continue_training=strikes<2 and history[-1]['step']<402)


def main():
    from pathlib import Path
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--training-run',type=Path)
    ap.add_argument('--step',type=int)
    a=ap.parse_args()
    # Keep the exact existing metric implementation, including failed outputs.
    sys.argv=[sys.argv[0],'--run',str(a.run)]
    score_v1.main()
    diagnostics={}
    for label in ['base','sft']:
        for item in read(a.run/'inputs.json'):
            pred=read(a.run/'predictions'/label/f"{item['key']}.json")
            diagnostics[f'{label}/{item["key"]}']=output_diagnostics(pred)
    write(a.run/'output_diagnostics.json',diagnostics)
    summary=read(a.run/'metrics_summary.json')
    major=sum(v['major_parse_loss'] for k,v in diagnostics.items() if k.startswith('sft/'))
    bad=sum(g['truncated']+g['empty_predictions'] for k,g in summary['groups'].items() if k.startswith('sft/'))+major
    write(a.run/'quality_status.json',dict(major_parse_loss=major,catastrophic_count_with_overlap=bad))
    if a.training_run and a.step in MILESTONES:
        current=assess(summary,major)
        path=a.training_run/'selection.json'
        history=read(path)['history'] if path.exists() else []
        assert a.step not in [r['step'] for r in history]
        history.append(dict(step=a.step,**current))
        result=decide(history)
        write(path,result)
        emit('checkpoint_selection',**result)


if __name__=='__main__':main()
