import sys,json,hashlib,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import ssh_moss
ssh_moss.SSH[ssh_moss.SSH.index('-S')+1]='C:/Users/qizhi/.ssh/cm-moss-check-20260912'
remote="""import importlib.metadata as m,json,hashlib,pathlib,subprocess
out={}
for name in ['ms-swift','transformers','torch','accelerate']:
 try: out[name]=m.version(name)
 except m.PackageNotFoundError: out[name]=None
p=pathlib.Path('/work/qt28/moss/MOSS-Transcribe-Diarize/finetune.py')
out['finetune']=p.read_text()
out['finetune_sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
out['repo_head']=subprocess.check_output(['git','-C',str(p.parent),'rev-parse','HEAD'],text=True).strip()
out['repo_status']=subprocess.check_output(['git','-C',str(p.parent),'status','--short'],text=True)
print(json.dumps(out))
"""
try:
 d=json.loads(ssh_moss.run("/dkucc/home/qt28/envs/moss312/bin/python -",remote.encode(),capture=True).stdout.decode())
 (HERE/'remote_environment.json').write_text(json.dumps({k:v for k,v in d.items() if k!='finetune'},indent=2),encoding='utf-8')
 (HERE/'remote_finetune.py').write_text(d.pop('finetune'),encoding='utf-8')
 print('REMOTE',json.dumps(d))
except Exception as e:print('REMOTE_ERROR',str(e))
commit=json.loads((HERE/'swift_tree.json').read_text())['sha']
sources={
 'swift_model_qwen.py':f'modelscope/ms-swift/{commit}/swift/model/models/qwen.py',
 'swift_template_qwen.py':f'modelscope/ms-swift/{commit}/swift/template/templates/qwen.py',
 'swift_model_moss.py':f'modelscope/ms-swift/{commit}/swift/model/models/moss.py',
 'swift_audio.sh':f'modelscope/ms-swift/{commit}/examples/train/multimodal/audio.sh',
 'upstream_finetune_current.py':'OpenMOSS/MOSS-Transcribe-Diarize/main/finetune.py',
 'upstream_finetune_frozen.py':'OpenMOSS/MOSS-Transcribe-Diarize/61bc29cd4120be7b5d3b761b64cd5dff57263642/finetune.py',
}
def get(kv):
 name,ref=kv
 try:
  data=urllib.request.urlopen('https://raw.githubusercontent.com/'+ref,timeout=30).read()
  (HERE/name).write_bytes(data)
  return dict(file=name,url='https://raw.githubusercontent.com/'+ref,sha256=hashlib.sha256(data).hexdigest())
 except Exception as e:return dict(file=name,error=str(e))
with ThreadPoolExecutor(max_workers=4) as ex:results=list(ex.map(get,sources.items()))
(HERE/'upstream_sources.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
print('SOURCES',json.dumps(results))
