"""Pure full-dev eligibility, early stopping and bounded LR allocation rules."""
import math

CORPORA = {'alimeeting': 8, 'ami': 18}


def assess(summary, quality):
    assert summary['complete'] and summary['expected_predictions'] == 52
    assert summary['scored_predictions'] == 52 and not summary['errors'] and not summary['missing']
    groups = summary['groups']
    assert set(groups) == {f'{m}/{c}/dev' for m in ['base', 'sft'] for c in CORPORA}
    failures = []
    scores = {}
    catastrophic = bool(quality['major_parse_loss'])
    for model in ['base', 'sft']:
        values = []
        for corpus, count in CORPORA.items():
            g = groups[f'{model}/{corpus}/dev']
            assert g['complete'] and g['completed'] == g['expected'] == count
            assert g['coverage'] == 1
            cer, der = g['cpCER'], g['DER_collar_0.25']['DER']
            assert all(math.isfinite(x) and x >= 0 for x in [cer, der])
            values.append(50*(cer+der))
            if model == 'sft':
                b = groups[f'base/{corpus}/dev']
                assert g['reference_characters'] == b['reference_characters']
                assert abs(g['DER_collar_0.25']['total']-b['DER_collar_0.25']['total']) < 1e-6
                if g['truncated'] or g['empty_predictions']:
                    catastrophic = True; failures.append(corpus+':generation_failure')
                if cer > b['cpCER']+.01: failures.append(corpus+':cpCER_regression')
                if der > b['DER_collar_0.25']['DER']+.01: failures.append(corpus+':DER_regression')
        scores[model] = sum(values)/2
    if quality['major_parse_loss']: failures.append('major_parse_loss')
    if scores['sft'] >= scores['base']: failures.append('no_improvement_over_base')
    return dict(score=scores['sft'], base_score=scores['base'], eligible=not failures,
                catastrophic=catastrophic, failures=failures)


def decide(history, min_delta=.1, patience=3, tolerance=.1):
    steps = [r['step'] for r in history]
    assert history and steps == sorted(set(steps)) and steps[0] == 5
    assert all(b-a == (5 if a < 20 else 10) for a, b in zip(steps, steps[1:]))
    best_anchor = history[0]['base_score']; stale = 0
    for row in history:
        assert abs(row['base_score']-history[0]['base_score']) < 1e-9
        if row['eligible'] and row['score'] <= best_anchor-min_delta:
            best_anchor = row['score']; stale = 0
        else: stale += 1
    eligible = [r for r in history if r['eligible']]
    minimum = min((r['score'] for r in eligible), default=None)
    selected = min((r['step'] for r in eligible if r['score'] <= minimum+tolerance), default=None)
    stop = history[-1]['catastrophic'] or stale >= patience
    return dict(selected_step=selected, best_eligible_score=minimum, stale_evaluations=stale,
                stop=stop, reason='catastrophic_output' if history[-1]['catastrophic'] else
                'dev_plateau' if stop else 'continue')


def survivors(trials, limit=2):
    candidates = [t for t in trials if not t['decision']['stop'] and
                  t['decision']['best_eligible_score'] is not None]
    return [t['id'] for t in sorted(candidates, key=lambda t:
            (t['decision']['best_eligible_score'], t['decision']['selected_step'], t['lr']))[:limit]]
