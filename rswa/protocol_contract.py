"""Counterexamples for stopping, checkpoint selection and repetition diagnostics."""
import copy,tempfile
from pathlib import Path
from assess import choose,extending
from campaign import plan_extension
from diagnostics import output_diagnostics

def row(step,score,ce=.5,functional=True,rtf=1.):
    return dict(step=step,S=score,ce=ce,functional=functional,RTF=rtf,
        corpora={c:dict(CER=.1,cpCER=.12,DER_collar_0=.13,**{'DER_collar_0.25':dict(DER=.11)})
                 for c in ['ami','alimeeting']})

def main():
    a=row(100,10);b=row(150,9.95,rtf=.8)
    assert choose([a,b]) is b
    broken=row(30,1,functional=False);assert choose([broken,a]) is a
    inferior=copy.deepcopy(b);inferior['corpora']['ami']['cpCER']=.14
    assert choose([inferior],reference=a) is None
    assert not extending(a,b) and extending(a,row(150,9.8))
    assert extending(a,row(150,10.05,ce=.49))
    assert not extending(row(100,10,functional=False),row(150,9,functional=False))
    state=dict(groups={f'{w}-seed0':dict(history=[row(100,10),row(150,9.8)]) for w in ['128','full','256']},queue=[])
    with tempfile.TemporaryDirectory() as tmp:
        plan_extension(state,Path(tmp),100,150,268)
    targets=[p['target'] for p in state['queue'] if p['kind']=='train']
    assert max(targets)==268 and targets.count(268)==3 and 270 not in targets
    pred=dict(raw_text='[0][S01]word[1]'*16,segments=[dict(text='word')]*16,generated_ids=[1,2,3,4]*256)
    d=output_diagnostics(pred)
    assert d['timestamp_loop'] and d['longest_short_period']['period']==4 and d['longest_short_period']['length']==1024
    pred=dict(raw_text='[0][S01]word[1]',segments=[dict(text='word')],generated_ids=[1,2,3])
    assert not output_diagnostics(pred)['major_parse_loss']
    print('Protocol contracts passed',flush=True)

if __name__=='__main__':main()
