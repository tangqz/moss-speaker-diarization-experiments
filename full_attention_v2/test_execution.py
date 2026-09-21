"""Small independent checks for statistics and autonomous selection decisions."""
import argparse


def metrics_tests():
    from diagnostics import target_types,TYPES,output_diagnostics
    from selection import assess,decide,selection_score
    # Character spans include a deliberately mixed timestamp/speaker token.
    text='[0.0][S01]hi[1.0]'
    types=target_types(text,[(0,5),(5,10),(10,12),(4,6),(0,0)],[1,2,3,4,151645],151645)
    assert [TYPES[i] for i in types]==['timestamp','speaker','text','boundary','eos']
    malformed='[0.0][0.0][S01]'+'a'*101+'[1.0]'
    diag=output_diagnostics(dict(raw_text=malformed,segments=[],generated_ids=[7]*16+[8]))
    assert diag['major_parse_loss'] and diag['longest_identical_run']['count']==16
    groups={}
    for model in ['base','sft']:
        for dataset in ['ami','alimeeting']:
            error=.10 if model=='base' else .09
            groups[f'{model}/{dataset}/dev']=dict(cpCER=error,DER_collar_0=0,
                **{'DER_collar_0.25':dict(DER=error)},truncated=0,empty_predictions=0)
    summary=dict(complete=True,scored_predictions=52,groups=groups)
    assert selection_score(groups)==9
    assert assess(summary,0)['eligible']
    assert not assess(summary,1)['eligible']
    groups['sft/ami/dev']['cpCER']=.12
    assert not assess(summary,0)['eligible']
    def r(step,score,eligible=True):return dict(step=step,score=score,base_score=10,eligible=eligible)
    assert decide([r(134,10.6,False),r(268,10.7,False)])['early_stop']
    d=decide([r(134,9),r(268,8.95)])
    assert d['selected_step']==134 and d['continue_training']
    assert not decide([r(134,9),r(268,9.6),r(402,9.7)])['continue_training']
    assert decide([r(134,11,False)])['selected_step'] is None
    print('METADATA_AND_SELECTION_TESTS_PASS',flush=True)


def statistics_tests():
    import torch
    from fast_eval import distribution
    z=torch.tensor([[2.,-1.,.5],[0.,0.,0.]],device='cuda')
    target=torch.tensor([2,1],device='cuda')
    nll,entropy=distribution(z,target)
    manual=z.double().softmax(-1)
    expected=-manual[torch.arange(2),target].log()
    expected_entropy=-(manual*manual.log()).sum(-1)
    torch.testing.assert_close(nll.double(),expected,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(entropy.double(),expected_entropy,atol=1e-6,rtol=1e-6)
    print('FULL_VOCABULARY_STATISTICS_TESTS_PASS',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['metrics','statistics'])
    a=ap.parse_args()
    metrics_tests() if a.mode=='metrics' else statistics_tests()
