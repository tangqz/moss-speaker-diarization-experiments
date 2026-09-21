"""Validate permutations, missing/extra speakers, overlap and micro denominators."""
import itertools
import random
import math
from meeteval.wer.wer.cp import cp_word_error_rate
from score import normalize, cp_distance, text_scores, diarization_scores, aggregate


def seg(a, b, speaker, text='x'):
    return dict(start=a, end=b, speaker=speaker, text=text)


assert normalize('Ａ b，中文！') == 'ab中文'
assert text_scores([seg(0,1,'A','你好')], [seg(0,1,'B','您好')])['CER'] == .5
ref = [seg(0,1,'A','abc'),seg(1,2,'B','def')]
hyp = [seg(0,1,'Z','abc'),seg(1,2,'Y','def')]
assert text_scores(ref,hyp)['cpCER'] == 0
assert text_scores(ref,[])['cpCER'] == 1
assert cp_distance({'A':'abc'}, {'X':'abc','Y':'def'}) == 3
rng = random.Random(47)
for _ in range(60):
    a={f'R{i}':''.join(rng.choice('abc') for _ in range(rng.randint(1,8))) for i in range(rng.randint(1,4))}
    b={f'H{i}':''.join(rng.choice('abc') for _ in range(rng.randint(1,8))) for i in range(rng.randint(1,4))}
    got=cp_distance(a,b)
    expected=cp_word_error_rate({k:' '.join(v) for k,v in a.items()}, {k:' '.join(v) for k,v in b.items()})
    assert got == expected.errors, (a,b,got,expected)
reference=dict(rttm='SPEAKER t 1 0 1 <NA> <NA> A <NA> <NA>\n'
                    'SPEAKER t 1 0 1 <NA> <NA> B <NA> <NA>\n', uem='t 1 0 1\n')
perfect=diarization_scores(reference,[seg(0,1,'X'),seg(0,1,'Y')],'t')
miss=diarization_scores(reference,[seg(0,1,'X')],'t')
empty=diarization_scores(reference,[],'t')
assert perfect['DER_collar_0']['diarization error rate'] == 0
assert miss['DER_collar_0']['diarization error rate'] == .5
assert empty['DER_collar_0']['diarization error rate'] == 1
assert miss['DER_collar_0']['total'] == 2
assert miss['DER_collar_0.25']['total'] < miss['DER_collar_0']['total']
print('METRICS_TEST_PASS normalization, CER, 60 MeetEval cpCER comparisons, extra/missing speakers, overlap DER, collars')
