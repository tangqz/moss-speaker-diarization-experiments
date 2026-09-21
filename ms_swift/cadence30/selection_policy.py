"""Checkpoint selection with an explicit cadence and a minimum coverage budget."""


def decide(history, min_delta=.1, patience=3, tolerance=.1, minimum_stop_step=134):
    steps = [row['step'] for row in history]
    assert history and steps == sorted(set(steps))
    assert all(step in (5, 10, 15, 20) or step % 30 == 0 for step in steps)
    best_anchor = history[0]['base_score']
    stale = 0
    for row in history:
        assert abs(row['base_score'] - history[0]['base_score']) < 1e-9
        if row['eligible'] and row['score'] <= best_anchor - min_delta:
            best_anchor = row['score']
            stale = 0
        else:
            stale += 1
    eligible = [row for row in history if row['eligible']]
    minimum = min((row['score'] for row in eligible), default=None)
    selected = min((row['step'] for row in eligible
                    if row['score'] <= minimum + tolerance), default=None)
    catastrophic = history[-1]['catastrophic']
    plateau = stale >= patience
    stop = catastrophic or (steps[-1] >= minimum_stop_step and plateau)
    return dict(selected_step=selected, best_eligible_score=minimum,
                stale_evaluations=stale, stop=stop,
                minimum_stop_step=minimum_stop_step,
                plateau_deferred=bool(plateau and not stop),
                reason='catastrophic_output' if catastrophic else
                'dev_plateau' if stop else 'continue')
