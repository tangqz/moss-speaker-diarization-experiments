#!/usr/bin/env bash
set -euo pipefail
TASK=/work/qt28/moss/dkucc/ms_swift_20260914
ENV=/work/qt28/moss/envs/vllm-moss-20260914
UV=/dkucc/home/qt28/.local/bin/uv
exec > >(tee -a "$TASK/bootstrap_vllm.log") 2>&1
"$UV" venv --python /dkucc/home/qt28/envs/moss312/bin/python "$ENV"
"$UV" pip install --python "$ENV/bin/python" \
  'vllm==0.23.1rc1.dev949+g68b4a1d58' \
  --extra-index-url https://wheels.vllm.ai/68b4a1d582818e67adc903bf1b8fc5a5447da2fa/cu130
"$UV" pip install --python "$ENV/bin/python" -e /work/qt28/moss/MOSS-Transcribe-Diarize
"$UV" pip freeze --python "$ENV/bin/python" > "$TASK/vllm_packages.txt"
"$ENV/bin/python" -c 'import vllm,torch,transformers; print("VLLM_ENV_READY",vllm.__version__,torch.__version__,transformers.__version__)'
