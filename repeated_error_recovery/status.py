"""Publish controller stages before the training run itself has event files."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

from torch.utils.tensorboard import SummaryWriter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--control', required=True, type=Path)
    ap.add_argument('--stage', required=True, choices=[
        'starting', 'calibration', 'manifest', 'smoke', 'training', 'testing',
        'early_stop', 'complete'])
    ap.add_argument('--step', required=True, type=int)
    args = ap.parse_args()
    codes = {name:i for i, name in enumerate([
        'starting', 'calibration', 'manifest', 'smoke', 'training', 'testing',
        'early_stop', 'complete'])}
    args.control.mkdir(parents=True, exist_ok=True)
    payload = {'stage':args.stage, 'checkpoint_step':args.step,
               'slurm_job_id':os.environ.get('SLURM_JOB_ID'),
               'utc':datetime.now(timezone.utc).isoformat()}
    (args.control/'status.json').write_text(json.dumps(payload, indent=2)+'\n')
    writer = SummaryWriter(str(args.control/'events'), flush_secs=1)
    writer.add_scalar('pipeline/stage', codes[args.stage], args.step)
    writer.add_scalar('pipeline/checkpoint_step', args.step, args.step)
    writer.add_text('pipeline/status', json.dumps(payload, ensure_ascii=False, indent=2), args.step)
    writer.close()


if __name__ == '__main__':
    main()
