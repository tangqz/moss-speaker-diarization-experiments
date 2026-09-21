"""Run after evaluation completes; verify archive and every downloaded file."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job',required=True,type=int)
    p.add_argument('--control-path',default='C:/Users/qizhi/.ssh/cm-moss-check-20260912')
    p.add_argument('--allow-incomplete',action='store_true')
    a=p.parse_args()
    name=f'eval-62994-{a.job}'
    ssh=[r'C:\Program Files\Git\usr\bin\ssh.exe','-T','-O','proxy','-S',a.control_path,
         '-o','BatchMode=yes','qt28@dkucc-login-01.rc.duke.edu']
    run='/work/qt28/moss/results/'+name
    command='/dkucc/home/qt28/envs/moss312/bin/python /work/qt28/moss/dkucc/evaluation/archive_results.py --run '+shlex.quote(run)
    if a.allow_incomplete:
        command+=' --allow-incomplete'
    result=subprocess.run(ssh+[command],check=True,stdout=subprocess.PIPE)
    metadata=json.loads(result.stdout)
    dest=Path(__file__).resolve().parents[1]/'artifacts'/name
    dest.mkdir(parents=True,exist_ok=True)
    archive=dest/(name+'-all-logs.tar.gz')
    with archive.with_suffix('.partial').open('wb') as f:
        subprocess.run(ssh+['cat '+shlex.quote(metadata['path'])],stdout=f,check=True)
    archive.with_suffix('.partial').replace(archive)
    assert archive.stat().st_size==metadata['bytes'] and sha(archive)==metadata['sha256']
    extracted=dest/'raw'
    extracted.mkdir(exist_ok=True)
    base=extracted.resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target=(base/member.name).resolve()
            assert target.is_relative_to(base) and member.isfile(), member.name
        tar.extractall(extracted,filter='data')
    index=json.loads((extracted/'DOWNLOAD_MANIFEST.json').read_text())
    for entry in index['files']:
        f=extracted/entry['path']
        assert f.stat().st_size==entry['bytes'] and sha(f)==entry['sha256'],entry['path']
    metadata['verified_files']=len(index['files'])
    (dest/'download-verification.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(dict(artifact_directory=str(dest),verified_files=len(index['files']),sha256=metadata['sha256'])))


if __name__=='__main__':
    main()
