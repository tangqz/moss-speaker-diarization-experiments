"""Inventory preserved predictions and package full-generation truncation evidence."""
import json
import os
from pathlib import Path
import tarfile
from datetime import datetime, timezone

root = Path('/work/qt28/moss/results')
out = Path('/work/qt28/moss/analysis/20260915-label-overlap')
out.mkdir(parents=True, exist_ok=True)
records, failures, errors = [], [], []
skip = {'training', 'source', 'vllm_exports', 'sft-export', 'checkpoint', '__pycache__'}
for parent, directories, files in os.walk(root):
    directories[:] = [d for d in directories if d not in skip
                      and not d.startswith(('source-', 'chrome-', '.'))]
    path = Path(parent)
    if 'predictions' not in path.parts:
        continue
    for name in files:
        if not name.endswith('.json'):
            continue
        source = path / name
        try:
            r = json.loads(source.read_text())
        except Exception as exc:
            errors.append(dict(path=str(source), error=str(exc)))
            continue
        if not isinstance(r, dict) or 'model' not in r or 'segments' not in r:
            continue
        row = {key: r.get(key) for key in [
            'key', 'model', 'model_path', 'backend', 'max_new_tokens', 'generated_tokens',
            'engineering_smoke', 'truncated', 'ended_eos', 'status', 'started_utc']}
        row['path'] = str(source)
        records.append(row)
        if r.get('truncated') and not r.get('engineering_smoke') and r.get('max_new_tokens', 0) >= 65536:
            failures.append(source)

inventory = dict(utc=datetime.now(timezone.utc).isoformat(), records=records,
                 full_generation_truncated_count=len(failures), errors=errors)
(out / 'remote_inventory.json').write_text(json.dumps(inventory, indent=2))
files = set(failures)
for source in failures:
    for parent in source.parents:
        if parent == root:
            break
        if (parent / 'references.json').is_file():
            files.add(parent / 'references.json')
            break
with tarfile.open(out / 'remote_failure_evidence.tar.gz', 'w:gz') as archive:
    for source in sorted(files):
        archive.add(source, arcname=str(source.relative_to(root)), recursive=False)
print(json.dumps(dict(scanned_records=len(records), full_generation_truncated=len(failures),
                      archive_bytes=(out/'remote_failure_evidence.tar.gz').stat().st_size,
                      errors=errors)))
