python3 - <<'PY'
import datetime as dt
import json
import statistics
from pathlib import Path

folder = Path('/work/qt28/moss/results/ms-swift-63521/evaluations/test-20')
inputs = json.loads((folder / 'inputs.json').read_text())
now = dt.datetime.now(dt.timezone.utc)
print(json.dumps({'utc': now.isoformat(), 'input_count': len(inputs)}))

for label in ('base', 'sft'):
    rows = []
    pred_root = folder / 'predictions' / label
    if pred_root.exists():
        for path in pred_root.rglob('*.json'):
            try:
                rows.append(json.loads(path.read_text()))
            except Exception:
                pass
    completed = {row.get('key') for row in rows if row.get('status') == 'ok'}
    total_audio = sum(float(row.get('duration', 0)) for row in rows if row.get('status') == 'ok')
    total_e2e = sum(float(row.get('e2e_seconds', 0)) for row in rows if row.get('status') == 'ok')
    result = {
        'model': label,
        'completed': len(completed),
        'errors': sum(row.get('status') != 'ok' for row in rows),
        'audio_hours_completed': total_audio / 3600,
        'aggregate_rtf': total_e2e / total_audio if total_audio else None,
        'mean_record_seconds': statistics.mean([row['e2e_seconds'] for row in rows if row.get('status') == 'ok']) if completed else None,
        'ranks': [],
    }
    for rank in range(4):
        assigned = [item for item in inputs if item['rank'] == rank]
        done = [row for row in rows if row.get('rank') == rank and row.get('status') == 'ok']
        done_keys = {row['key'] for row in done}
        remaining = [item for item in assigned if item['key'] not in done_keys]
        rank_audio = sum(float(row.get('duration', 0)) for row in done)
        rank_e2e = sum(float(row.get('e2e_seconds', 0)) for row in done)
        status_path = folder / f'{label}-worker-{rank}-status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {'stage': 'not_started'}
        result['ranks'].append({
            'rank': rank,
            'completed': len(done),
            'assigned': len(assigned),
            'remaining_audio_hours': sum(float(item['duration']) for item in remaining) / 3600,
            'observed_rtf': rank_e2e / rank_audio if rank_audio else None,
            'status': status,
        })
    print(json.dumps(result, ensure_ascii=True))
PY
