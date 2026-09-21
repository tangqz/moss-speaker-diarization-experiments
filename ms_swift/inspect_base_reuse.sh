#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

old = Path('/work/qt28/moss/results/ms-swift-63521/evaluations/test-20')
new = Path('/work/qt28/moss/results/ms-swift-63643/evaluations/test-150')
run = Path('/work/qt28/moss/results/ms-swift-63643')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert old.is_dir()
print('old_test_complete', (old / 'test_complete.json').read_text())
print('old_inputs_sha', sha(old / 'inputs.json'))
print('new_inputs_sha', sha(new / 'inputs.json') if (new / 'inputs.json').exists() else 'missing')
print('old_references_sha', sha(old / 'references.json'))
print('new_references_sha', sha(new / 'references.json') if (new / 'references.json').exists() else 'missing')
print('base_prediction_files', len(list((old / 'predictions/base').rglob('*.json'))))
print('old_manifests', json.dumps(json.loads((old / 'model_manifests.json').read_text())['base'], indent=2))
print('current_base_manifest', (run / 'evaluations/dev-0/model_manifest.json').read_text())
print('old_protocol', (old / 'protocol.json').read_text())
PY
