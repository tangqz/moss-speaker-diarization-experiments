"""Read-only snapshot of the code actually deployed for formula auditing."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from ssh_moss import run

REMOTE = r'''
import json, hashlib
from pathlib import Path
root=Path('/work/qt28/moss')
records=[]
def capture(label,path):
    if not path.is_file(): return
    data=path.read_bytes()
    records.append(dict(label=label,path=str(path),sha256=hashlib.sha256(data).hexdigest(),text=data.decode()))
names=['attention.py','cache.py','model_adapter.py','moss_rswa_plugin.py','moss_vllm_rswa.py',
       'evaluate_ce.py','evaluate_generate.py','native_sft.sh','protocol.py','source_manifest.json']
for label,dirname in [('round1','moss-rswa-20260920'),('round2','moss-rswa-r2-20260921')]:
    for name in names: capture(label+'/'+name,root/'results'/dirname/'source'/name)
vfiles=['v1/attention/ops/triton_attention_helpers.py','v1/attention/ops/triton_unified_attention.py',
        'v1/attention/backends/triton_attn.py','v1/core/single_type_kv_cache_manager.py',
        'v1/core/kv_cache_manager.py','v1/core/kv_cache_coordinator.py',
        'model_executor/layers/attention/rswa_attention.py','model_executor/models/qwen3.py',
        'model_executor/models/qwen2.py','model_executor/models/moss_transcribe_diarize.py',
        'v1/worker/gpu_model_runner.py','v1/sample/ops/penalties.py']
for site in (root/'envs/vllm-moss-20260914/lib').glob('python*/site-packages'):
    for name in vfiles: capture('vllm/'+name,site/'vllm'/name)
for site in (root/'envs/ms-swift-20260914/lib').glob('python*/site-packages'):
    for name in ['models/qwen3/modeling_qwen3.py','loss/loss_utils.py']:
        capture('transformers/'+name,site/'transformers'/name)
    for name in ['sequence_parallel/ulysses.py','sequence_parallel/base.py']:
        capture('swift/'+name,site/'swift'/name)
for name in ['config.json','modeling_moss_transcribe_diarize.py']:
    capture('base/'+name,root/'models/MOSS-Transcribe-Diarize'/name)
capture('official/finetune.py',root/'MOSS-Transcribe-Diarize/finetune.py')
capture('base_plugin.py',root/'dkucc/ms_swift_20260914/moss_plugin.py')
print(json.dumps(records))
'''

records = json.loads(run('python -', REMOTE.encode(), capture=True).stdout)
manifest=[]
for record in records:
    dest=HERE/'source'/record['label']
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(record['text'].encode())
    assert hashlib.sha256(dest.read_bytes()).hexdigest()==record['sha256']
    manifest.append({k:v for k,v in record.items() if k!='text'})
(HERE/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'files':len(manifest),'directory':str(HERE/'source')},ensure_ascii=True))
