python3 - <<'PY'
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

run = Path('/work/qt28/moss/results/ms-swift-63521')
archive = run / 'ms-swift-63521-final-artifacts.tar.gz'
log_root = Path('/work/qt28/moss/logs')
slurm_names = [
    'moss-native-swift-63521.out', 'moss-native-swift-63521.err',
    'moss-swift-resume10-63532.out', 'moss-swift-resume10-63532.err',
    'moss-ms-swift-final-test-63536.out', 'moss-ms-swift-final-test-63536.err',
]

files = []
for path in sorted(run.rglob('*')):
    if not path.is_file() or path.is_symlink() or path == archive:
        continue
    rel = path.relative_to(run)
    if rel.parts[0] == 'vllm_exports':
        continue
    if rel.parts[0] == 'training' and len(rel.parts) > 1 and rel.parts[1].startswith('checkpoint-'):
        continue
    if path.suffixes[-2:] == ['.tar', '.gz']:
        continue
    files.append((path, Path('ms-swift-63521') / rel))
for name in slurm_names:
    path = log_root / name
    if path.is_file():
        files.append((path, Path('slurm_logs') / name))

manifest = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'run': str(run),
    'exclusions': ['training/checkpoint-* model weights', 'vllm_exports model weights'],
    'files': [],
}
for path, arcname in files:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['files'].append({'path': str(arcname), 'bytes': path.stat().st_size, 'sha256': digest})

manifest_path = run / 'final-artifact-manifest.json'
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
files.append((manifest_path, Path('ms-swift-63521/final-artifact-manifest.json')))

with tarfile.open(archive, 'w:gz', compresslevel=6) as bundle:
    for path, arcname in files:
        bundle.add(path, arcname=str(arcname), recursive=False)

print(json.dumps({
    'archive': str(archive),
    'bytes': archive.stat().st_size,
    'files': len(files),
    'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
}, ensure_ascii=True))
PY
