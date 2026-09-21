"""Download completed test evidence, without checkpoint weights."""
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ssh_moss import SSH

run = '/work/qt28/moss/results/ms-swift-63643'
dest = Path(__file__).resolve().parents[1] / 'artifacts/ms-swift-63643'
dest.mkdir(parents=True, exist_ok=True)
members = ['outcome.json', 'status.json', 'selection.json', 'test_progress.json',
           'final_test_job_exit.json', 'base_reuse_validation.json', 'test_reuse_submission.json',
           'evaluations/test-150', 'logs', 'source-final-test', 'training/logging.jsonl',
           'training/checkpoint-150/trainer_state.json', 'training/args.json']
remote = '''import sys,tarfile
from pathlib import Path
root=Path(sys.argv[1])
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name in sys.argv[2:]:
  p=root/name
  if p.exists(): archive.add(p,arcname=name)
'''
command = 'python3 -c ' + shlex.quote(remote) + ' ' + ' '.join(map(shlex.quote, [run]+members))
data = subprocess.run(SSH+[command], check=True, capture_output=True, timeout=180).stdout
archive_path = dest / 'test-150-evidence.tar.gz'
archive_path.write_bytes(data)
target = dest / 'raw'
target.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
    archive.extractall(target, filter='data')
outcome = json.loads((target/'outcome.json').read_text())
metric = target/'evaluations/test-150/metrics_summary.json'
digest = hashlib.sha256(metric.read_bytes()).hexdigest()
assert digest == outcome['test_metrics_sha256']
print(json.dumps({'archive':str(archive_path), 'bytes':len(data),
                  'metrics_sha256_verified':digest, 'outcome':outcome},indent=2))
