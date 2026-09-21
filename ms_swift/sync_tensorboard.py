"""Copy native Swift TensorBoard event files without reinterpreting metrics."""
import argparse
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
SSH = [r'C:\Program Files\Git\usr\bin\ssh.exe', '-T', '-O', 'proxy', '-S',
       'C:/Users/qizhi/.ssh/cm-moss-check-20260912', '-o', 'BatchMode=yes',
       'qt28@dkucc-login-01.rc.duke.edu']
REMOTE = r'''
import sys,tarfile
from pathlib import Path
root=Path(sys.argv[1])
assert root.parent == Path('/work/qt28/moss/results') and root.name.startswith('ms-swift-')
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
    for p in root.rglob('events.out.tfevents.*'):
        if p.is_file() and not p.is_symlink():
            archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
'''


def sync():
    connection_file = HERE.parent/'ssh_connection.json'
    if connection_file.exists():
        SSH[SSH.index('-S')+1] = json.loads(connection_file.read_text(encoding='utf-8-sig'))['control_path']
    config = json.loads((HERE/'current_run.json').read_text(encoding='utf-8-sig'))
    if config.get('follow_remote_state'):
        remote_task = config.get('remote_task', '/work/qt28/moss/dkucc/ms_swift_20260914')
        state_path = remote_task.rstrip('/') + '/current_run.json'
        refreshed=json.loads(subprocess.run(SSH+['cat '+shlex.quote(state_path)],
            capture_output=True,check=True,timeout=10).stdout)
        config.update(refreshed)
        config['follow_remote_state']=True
        (HERE/'current_run.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
        (HERE.parent/'evaluation/current_run.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    remote = config['remote_run']
    name = Path(remote).name
    dest = (HERE.parent/'tensorboard/events'/name).resolve()
    command = 'python3 -c '+shlex.quote(REMOTE)+' '+shlex.quote(remote)
    data = subprocess.run(SSH+[command], capture_output=True, check=True, timeout=50).stdout
    files = 0
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for member in archive:
            if not member.isfile():
                continue
            target = (dest/member.name).resolve()
            assert target.is_relative_to(dest)
            target.parent.mkdir(parents=True, exist_ok=True)
            content = archive.extractfile(member).read()
            if not target.exists() or target.stat().st_size != len(content):
                tmp = target.with_name(target.name+'.partial')
                tmp.write_bytes(content)
                tmp.replace(target)
            files += 1
    (HERE/'tensorboard_sync.json').write_text(json.dumps(dict(
        connected=True, utc=datetime.now(timezone.utc).isoformat(), native_event_files=files,
        remote_run=remote, local_events=str(dest), scalar_values_modified=False), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    while True:
        try:
            sync()
        except Exception as exc:
            (HERE/'tensorboard_sync.json').write_text(json.dumps(dict(
                connected=False, utc=datetime.now(timezone.utc).isoformat(), error=str(exc)), indent=2))
            if args.once:
                raise
        if args.once:
            break
        time.sleep(60)
