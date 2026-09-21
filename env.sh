#!/usr/bin/env bash
# =============================================================================
# DKUCC / MOSS-Transcribe-Diarize project environment
# Source this file in an interactive shell or a Slurm job script:
#     source /work/qt28/moss/dkucc/env.sh
#
# Created 2026-09-11 (Phase 0).
#
# Layout / conventions:
#   Python env     : /dkucc/home/qt28/envs/moss312   (conda, Python 3.12)
#   Project tree   : /work/qt28/moss                 (data/models/checkpoints/logs/results)
#   HF model cache : /dkucc/home/qt28/hf-cache       (survives /work 75-day cleanup)
#   Model weights  : /work/qt28/moss/models/MOSS-Transcribe-Diarize (plain dir, 1.7GB safetensors)
#
# Package installs: use uv (installed at ~/.local/bin/uv), NOT pip directly:
#   uv pip install --python /dkucc/home/qt28/envs/moss312/bin/python <package>
#
# Network notes (measured 2026-09-11 on compute node):
#   - pypi.org                  : direct OK
#   - github.com                : needs DKU proxy (restricted proxy is faster)
#   - huggingface.co            : needs DKU proxy; hf-mirror.com works direct (faster)
# =============================================================================

if command -v module >/dev/null 2>&1; then
  module load anaconda/2023.7 2>/dev/null || true
fi

CONDA_SH=/opt/apps/centos8/anaconda3/etc/profile.d/conda.sh
if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate /dkucc/home/qt28/envs/moss312 2>/dev/null || true
fi

# --- tooling (standalone binaries in home) ---
export PATH="/dkucc/home/qt28/.local/bin:$PATH"   # uv 0.12.13
# NAS-friendly uv settings: copy instead of hardlink (NFS may not support links)
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=120

# --- runtime niceties ---
export PYTHONUNBUFFERED=1

# --- project paths ---
export MOSS_HOME=/work/qt28/moss
export MOSS_ROOT="$MOSS_HOME/MOSS-Transcribe-Diarize"
export MOSS_DATA="$MOSS_HOME/data"
export MOSS_MODELS="$MOSS_HOME/models"
export MOSS_LOGS="$MOSS_HOME/logs"
export MOSS_CKPT="$MOSS_HOME/checkpoints"
export MOSS_RESULTS="$MOSS_HOME/results"

# --- caches ---
export HF_HOME=/dkucc/home/qt28/hf-cache
export HF_ENDPOINT=https://hf-mirror.com   # fast mirror; unset to use official HF via proxy
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

# --- DKU proxies ---
export DKU_PROXY_CHINA=http://proxy-china-prod-dku.oit.duke.edu:3128
export DKU_PROXY_RESTRICTED=http://proxy-dku.oit.duke.edu:3128

# Run a command through the DKU restricted proxy (github, huggingface official).
# Example: dku_proxy git clone https://github.com/OpenMOSS/MOSS-Transcribe-Diarize.git
dku_proxy() {
  https_proxy="$DKU_PROXY_RESTRICTED" http_proxy="$DKU_PROXY_RESTRICTED" "$@"
}

# module load cuda/12.6   # only needed when building CUDA extensions (e.g. flash-attn)
