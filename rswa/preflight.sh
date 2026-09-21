#!/usr/bin/env bash
set -euo pipefail
ROOT=/work/qt28/moss
TASK=$ROOT/dkucc/rswa_20260920
mkdir -p "$TASK/evidence" "$TASK/source_snapshot"
exec > >(tee -a "$ROOT/logs/rswa-20260920-console.log" "$TASK/evidence/preflight.log") 2>&1
date -Is
hostname
id
squeue -u qt28
sinfo -p common-gpu -o '%P %a %l %D %G'
sacctmgr -n -P show assoc where user=qt28 format=User,Account,Partition,MaxJobs,MaxTRES
df -h "$ROOT" /dkucc/home/qt28
du -sh "$ROOT/envs" "$ROOT/checkpoints" "$ROOT/results"
git -C "$ROOT/MOSS-Transcribe-Diarize" rev-parse HEAD
find "$ROOT/envs" -maxdepth 2 -name pyvenv.cfg -print
find "$ROOT/dkucc/ms_swift_20260914" -maxdepth 2 -type f \( -name 'data_receipts.json' -o -name '*inputs.json' -o -name 'native_sft.sh' -o -name 'moss_plugin.py' \) -print
cat "$ROOT/dkucc/ms_swift_20260914/data_receipts.json"
cat "$ROOT/models/MOSS-Transcribe-Diarize/processor_config.json"
"$ROOT/envs/ms-swift-20260914/bin/python" - <<'PY'
import importlib.metadata as m, json, sys, sysconfig
from pathlib import Path
root=Path('/work/qt28/moss')
info={'python':sys.version,'site_packages':sysconfig.get_paths()['purelib'],
      'packages':{p:m.version(p) for p in ['torch','transformers','ms-swift','accelerate','triton']}}
print(json.dumps(info,indent=2))
(root/'dkucc/rswa_20260920/evidence/environment.json').write_text(json.dumps(info,indent=2))
site=Path(sysconfig.get_paths()['purelib'])
targets={
 'modeling_qwen3.py':site/'transformers/models/qwen3/modeling_qwen3.py',
 'cache_utils.py':site/'transformers/cache_utils.py',
 'generation_utils.py':site/'transformers/generation/utils.py',
 'masking_utils.py':site/'transformers/masking_utils.py',
 'flex_attention.py':site/'torch/nn/attention/flex_attention.py',
 'modeling_moss_transcribe_diarize.py':root/'MOSS-Transcribe-Diarize/moss_transcribe_diarize/modeling_moss_transcribe_diarize.py',
 'finetune.py':root/'MOSS-Transcribe-Diarize/finetune.py',
}
import shutil
for name,path in targets.items():
 if not path.is_file(): raise FileNotFoundError(path)
 shutil.copy2(path,root/'dkucc/rswa_20260920/source_snapshot'/name)
print('RSWA_PREFLIGHT_INVENTORY_COMPLETE')
PY
