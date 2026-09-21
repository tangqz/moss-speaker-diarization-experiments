"""Synthetic counterexamples for global speaker mapping and paired resampling."""
import tempfile
from pathlib import Path
from final_analysis import speaker_diagnostics,paired,stratify

def segment(start,end,speaker,text='hello'):
    return dict(start=start,end=end,speaker=speaker,text=text)

def main():
    ref=dict(segments=[segment(0,10,'A','aaaaa'),segment(20,30,'B','bbbbb'),segment(200,210,'A','ccccc')],
        rttm='SPEAKER x 1 0 10 <NA> <NA> A <NA> <NA>\nSPEAKER x 1 20 10 <NA> <NA> B <NA> <NA>\nSPEAKER x 1 200 10 <NA> <NA> A <NA> <NA>',
        uem='x 1 0 220')
    pred=dict(segments=[segment(0,10,'X','aaaaa'),segment(20,30,'Y','bbbbb'),segment(200,210,'Y','ccccc')])
    result=speaker_diagnostics(ref,pred,220)
    assert result['global_hypothesis_to_reference']=={'X':'A','Y':'B'}
    assert result['reentry_buckets']['over120']['confused_seconds']==10
    assert result['fragmentation'][0]['count']==2
    missing=speaker_diagnostics(ref,dict(segments=pred['segments'][:2]),220)
    assert missing['reentry_buckets']['over120']['missing_seconds']==10
    assert missing['reference_time_thirds'][2]['deletions']==5
    # Identical predictions under any paired sample must have zero difference.
    row=dict(dataset='ami',duration=200,e2e_seconds=20,text=dict(reference_characters=100,character_errors=10,cp_character_errors=12),
        diarization={'DER_collar_0.25':dict(total=100,confusion=3,**{'missed detection':2,'false alarm':1})})
    compare=paired({'a':row,'b':row},{'a':row,'b':row})
    assert all(v['point']==0 and v['ci95']==[0,0] for v in compare['ami']['differences'].values())
    strata=stratify({'a':row,'b':row},{'a':result,'b':missing})
    assert strata['reentry']['ami']['over120']['coverage']==.5
    assert strata['reentry']['ami']['over120']['confusion_fraction']==.5
    print('Analysis contracts passed',flush=True)

if __name__=='__main__':main()
