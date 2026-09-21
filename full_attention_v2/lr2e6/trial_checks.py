"""CPU-only checks of fail-closed quality gates and the isolated LR derivative."""
import ast
import copy
from pathlib import Path
from common import HERE, read
from trial_run import assess_complete


def sample(expected=1):
    return dict(complete=True,scored_predictions=2*expected,expected_predictions=2*expected,
        missing=[],errors=[],groups={'sft/ami/dev':dict(complete=True,completed=expected,
            truncated=0,empty_predictions=0)})


def rejected(fn):
    try:fn()
    except AssertionError:return
    raise AssertionError('Incomplete or inconsistent evaluation was accepted')


def main():
    quality={'major_parse_loss':0}
    assert assess_complete(sample(),quality,1)
    for metric in ['truncated','empty_predictions']:
        s=sample();s['groups']['sft/ami/dev'][metric]=1
        assert not assess_complete(s,quality,1)
    assert not assess_complete(sample(),{'major_parse_loss':1},1)
    for key,value in [('complete',False),('scored_predictions',1),('expected_predictions',12),
                      ('missing',['missing_case']),('errors',['failed_worker'])]:
        s=sample();s[key]=value
        rejected(lambda:assess_complete(s,quality,1))
    s=sample(6);s['groups']['sft/ami/dev']['completed']=5
    rejected(lambda:assess_complete(s,quality,6))
    assert assess_complete(sample(6),quality,6)

    source=(HERE/'train.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    args=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='TrainingArguments')
    kw={k.arg:k.value for k in args.keywords}
    assert ast.literal_eval(kw['learning_rate'])==2e-6
    assert ast.literal_eval(kw['warmup_steps'])==0
    assert isinstance(kw['max_steps'],ast.Name) and kw['max_steps'].id=='HORIZON'
    assert sum(isinstance(n,ast.Constant) and n.value==2e-6 for n in ast.walk(tree))==3
    assert 'state.global_step in [1, 5]' in (HERE/'host_runtime.py').read_text()
    protocol=read(HERE/'design_protocol.json')
    assert protocol['training']['learning_rate']==2e-6
    assert protocol['training']['scheduler_horizon_steps']==402
    assert protocol['short_trial']['full_resume_steps']==[10,25,50]
    assert not protocol['short_trial']['test_performed']
    print('TRIAL_SOURCE_AND_FAIL_CLOSED_QUALITY_CHECKS_PASS',flush=True)


if __name__=='__main__':
    main()
