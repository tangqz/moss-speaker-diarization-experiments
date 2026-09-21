#!/usr/bin/env bash
#SBATCH --job-name=moss-vllm-check
#SBATCH --partition=common-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/work/qt28/moss/logs/%x-%j.out
#SBATCH --error=/work/qt28/moss/logs/%x-%j.err
set -euo pipefail
TASK=/work/qt28/moss/dkucc/ms_swift_20260914
ENV=/work/qt28/moss/envs/vllm-moss-20260914
RUN=/work/qt28/moss/results/vllm-check-${SLURM_JOB_ID}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
export PYTHONPATH=/work/qt28/moss/MOSS-Transcribe-Diarize:$TASK/evaluation
mkdir -p "$RUN"
export RUN TASK
"$ENV/bin/python" - <<'PY'
import os,sys
from pathlib import Path
from common import read,write,BASE
run=Path(os.environ['RUN']);task=Path(os.environ['TASK'])
items=read(task/'evaluation/dev_inputs.json')
short=min(items,key=lambda r:r['prompt_len']);long=max(items,key=lambda r:r['prompt_len'])
write(run/'inputs.json',[dict(r,rank=0) for r in [short,long]])
write(run/'model_manifest.json',dict(purpose='vLLM short and longest dev checkpoint loading gate'))
PY
for MODEL in base sft; do
    CHECKPOINT=/work/qt28/moss/models/MOSS-Transcribe-Diarize
    if [ "$MODEL" = sft ]; then
        CHECKPOINT="$RUN/sft-export"
        "$ENV/bin/python" "$TASK/export_vllm.py" \
            --checkpoint /work/qt28/moss/results/ms-swift-fsdp-check-63506/sp4/checkpoint-1 \
            --destination "$CHECKPOINT"
    fi
    "$ENV/bin/python" "$TASK/evaluation/vllm_generate.py" --run "$RUN" --checkpoint "$CHECKPOINT" \
        --rank 0 --model-label "$MODEL" --smoke-tokens 128 > "$RUN/$MODEL.log" 2>&1
done
/dkucc/home/qt28/envs/moss312/bin/python "$TASK/evaluation/benchmark_hf.py" --run "$RUN" > "$RUN/hf_benchmark.log" 2>&1
"$ENV/bin/python" - <<'PY'
import os
from pathlib import Path
from common import read,write,now
run=Path(os.environ['RUN']);items=read(run/'inputs.json')
records=[read(run/'predictions'/model/(r['key']+'.json')) for model in ['base','sft'] for r in items]
assert len(records)==4 and all(r['status']=='ok' and r['generated_tokens']>0 for r in records)
bench=read(run/'hf_128_token_benchmark.json')['records']
comparison=[]
for hf in bench:
    vl=read(run/'predictions/base'/(hf['key']+'.json'))
    comparison.append(dict(key=hf['key'],hf_seconds=hf['e2e_seconds'],vllm_seconds=vl['e2e_seconds'],
        hf_generated_tokens=hf['generated_tokens'],vllm_generated_tokens=vl['generated_tokens'],
        e2e_ratio_128_tokens=hf['e2e_seconds']/vl['e2e_seconds'],
        generated_ids_equal=hf['generated_ids']==vl['generated_ids']))
write(run/'benchmark_comparison.json',dict(comparison=comparison,
    qualification='128-token latency; not full meeting accuracy or full-dev speedup; startup excluded'))
write(run/'vllm_check.json',dict(loading_and_long_audio_passed=True,utc=now(),
    scope='128-token engineering gate, not complete dev scoring', keys=[r['key'] for r in items]))
PY
