"""Read-only inventory of the installed production inference implementation."""
import importlib.metadata as m,json,sysconfig,shutil
from pathlib import Path
site=Path(sysconfig.get_paths()['purelib']);root=site/'vllm'
out=Path('/work/qt28/moss/dkucc/rswa_20260920/vllm_snapshot');out.mkdir(exist_ok=True)
selected=[]
for name in ['model_executor/layers/attention/rswa_attention.py','model_executor/models/config.py',
             'config/model.py','transformers_utils/model_arch_config_convertor.py',
             'v1/core/single_type_kv_cache_manager.py','v1/worker/gpu_model_runner.py',
             'v1/attention/ops/triton_unified_attention.py','v1/attention/backends/flex_attention.py']:
    p=root/name
    if p.exists():
        dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);selected.append(name)
for base in ['model_executor/models','model_executor/layers/attention','attention','v1/attention','v1/core','v1']:
    folder=root/base
    if not folder.exists():continue
    for p in folder.glob('*.py'):
        if p.name in ['moss_transcribe_diarize.py','qwen3.py','unlimited_ocr.py','attention.py',
                      'layer.py','kv_cache_manager.py','kv_cache_utils.py','kv_cache_interface.py']:
            dest=out/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,dest);selected.append(str(p.relative_to(root)))
for p in (root/'v1/attention/backends').glob('*.py'):
    if p.name in ['flash_attn.py','flashinfer.py','triton_attn.py','utils.py']:
        dest=out/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        selected.append(str(p.relative_to(root)))
hits=[]
for sub in ['model_executor/models','v1/attention','v1/core']:
    for p in (root/sub).rglob('*.py'):
        for i,line in enumerate(p.read_text(errors='replace').splitlines(),1):
            if any(s in line.lower() for s in ['sliding_window_with','reference_window','unlimitedocr','unlimited_ocr','attention_sink','sink_tokens','prefix_lm']):
                hits.append(dict(path=str(p.relative_to(root)),line=i,text=line.strip()[:230]))
report=dict(version=m.version('vllm'),site_packages=str(site),selected=selected,hits=hits)
(out/'inventory.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
