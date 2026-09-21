"""Frozen paths and auditable, atomic run metadata for full-attention SFT v2."""
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path('/work/qt28/moss')
BASE = ROOT/'models/MOSS-Transcribe-Diarize'
TRAIN = ROOT/'data/moss_jsonl/train_mix.jsonl'
OLD_EVAL = ROOT/'results/eval-62994-63363'
HERE = Path(__file__).resolve().parent
REPO = ROOT/'MOSS-Transcribe-Diarize'
HORIZON = 402
MILESTONES = [134, 268, 402]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def emit(event, **values):
    print(json.dumps(dict(event=event, utc=now(), **values), ensure_ascii=True), flush=True)


def append(path, obj):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=True)+'\n')


def seed_order():
    # Used by tests/reporting only; actual order is saved from Trainer's sampler.
    import torch
    return [torch.randperm(536, generator=torch.Generator().manual_seed(e)).tolist()
            for e in range(3)]
