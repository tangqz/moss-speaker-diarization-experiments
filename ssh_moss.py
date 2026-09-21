"""Reuse Proma's authenticated OpenSSH master through no-fd-passing proxy mode."""
import argparse
import json
import pathlib
import shlex
import subprocess

SSH = [r"C:\Program Files\Git\usr\bin\ssh.exe", "-T", "-O", "proxy", "-S",
       "C:/Users/qizhi/.ssh/cm-moss", "-o", "BatchMode=yes",
       "qt28@dkucc-login-01.rc.duke.edu"]
connection_file = pathlib.Path(__file__).with_name('ssh_connection.json')
if connection_file.exists():
    SSH[SSH.index('-S') + 1] = json.loads(connection_file.read_text(encoding='utf-8-sig'))['control_path']


def run(command, data=None, capture=False):
    return subprocess.run(SSH + [command], input=data, check=True,
                          stdout=subprocess.PIPE if capture else None)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-path", default=SSH[SSH.index("-S") + 1],
                   help="Path of an existing authenticated SSH master socket")
    p.add_argument("--command")
    p.add_argument("--script", type=pathlib.Path)
    p.add_argument("--upload", type=pathlib.Path)
    p.add_argument("--download")
    p.add_argument("--destination")
    a = p.parse_args()
    SSH[SSH.index("-S") + 1] = a.control_path
    if a.upload:
        dest = shlex.quote(a.destination)
        run("cat > " + dest, a.upload.read_bytes())
    elif a.download:
        pathlib.Path(a.destination).write_bytes(run("cat " + shlex.quote(a.download), capture=True).stdout)
    elif a.script:
        run("bash -s", a.script.read_text(encoding="utf-8-sig").replace("\r\n", "\n").encode())
    else:
        run(a.command)
