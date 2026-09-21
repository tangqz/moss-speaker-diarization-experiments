#!/usr/bin/env bash
set -euo pipefail
TASK=/work/qt28/moss/dkucc/ms_swift_20260914
ENV=/work/qt28/moss/envs/ms-swift-20260914
cd "$TASK"
exec > >(tee -a bootstrap.log) 2>&1
tar -xzf upstream.tar.gz
/dkucc/home/qt28/.local/bin/uv venv --system-site-packages --python /dkucc/home/qt28/envs/moss312/bin/python "$ENV"
/dkucc/home/qt28/.local/bin/uv pip install --python "$ENV/bin/python" -e "$TASK/ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076" 'transformers==5.12.1'
"$ENV/bin/python" -c 'import swift,transformers,torch; print("SWIFT_ENV_READY",swift.__version__,transformers.__version__,torch.__version__)'
"$ENV/bin/python" -m pip freeze > "$TASK/packages.txt"
