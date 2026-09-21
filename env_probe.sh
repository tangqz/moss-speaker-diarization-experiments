#!/usr/bin/env bash

# Run this script inside a Slurm GPU allocation, not on the login node.
set -u

echo "===== DKUCC environment probe ====="
date -Is 2>/dev/null || date
echo "hostname=$(hostname)"
echo "user=$(id -un)"
echo "pwd=$(pwd)"
echo "slurm_job_id=${SLURM_JOB_ID:-<none>}"
echo "slurm_job_name=${SLURM_JOB_NAME:-<none>}"
echo "slurm_partition=${SLURM_JOB_PARTITION:-<none>}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"

echo
echo "===== Slurm allocation ====="
if command -v scontrol >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
  scontrol show job "$SLURM_JOB_ID" 2>&1 || true
else
  echo "No active Slurm job or scontrol unavailable"
fi

echo
echo "===== GPU ====="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi -L 2>&1 || true
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv 2>&1 || true
  echo "----- topology -----"
  nvidia-smi topo -m 2>&1 || true
else
  echo "nvidia-smi is unavailable; this shell may be on a login node or missing a GPU allocation"
fi

echo
echo "===== Modules ====="
if command -v module >/dev/null 2>&1; then
  module list 2>&1 || true
else
  echo "module command unavailable"
fi

echo
echo "===== Tool versions ====="
for tool in python python3 uv git; do
  if command -v "$tool" >/dev/null 2>&1; then
    echo "--- $tool ---"
    "$tool" --version 2>&1 || true
  else
    echo "--- $tool: unavailable ---"
  fi
done

echo
echo "===== Python packages ====="
if command -v python >/dev/null 2>&1; then
  python - <<'PY'
import importlib

for name in ("torch", "transformers", "accelerate", "soundfile", "soxr"):
    try:
        mod = importlib.import_module(name)
        print(f"{name}={getattr(mod, '__version__', 'version-unavailable')}")
    except Exception as exc:
        print(f"{name}=UNAVAILABLE ({type(exc).__name__}: {exc})")

try:
    import torch
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"torch.cuda.device_{index}={torch.cuda.get_device_name(index)}")
except Exception as exc:
    print(f"torch_cuda_probe=FAILED ({type(exc).__name__}: {exc})")
PY
else
  echo "python unavailable"
fi

echo
echo "===== Storage ====="
df -h /dkucc/home 2>&1 || true
df -h /work 2>&1 || true
echo "===== Probe complete ====="
