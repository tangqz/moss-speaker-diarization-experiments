import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from ssh_moss import run
origin=json.loads((HERE/'cpu_checks_result.json').read_text())['swift_origin']
payload='''import json\nfrom pathlib import Path\nroot=Path(%r).parent\npaths=list((root/'sequence_parallel').rglob('*.py'))+list((root/'trainers').glob('*.py'))+list((root/'loss').glob('*.py'))\nprint(json.dumps({str(p.relative_to(root)):p.read_text() for p in paths}))\n''' % origin
files=json.loads(run('python -',payload.encode(),capture=True).stdout)
for name,content in files.items():
    path=HERE/'source/swift'/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(content,encoding='utf-8')
print(json.dumps({'files':list(files)}))
