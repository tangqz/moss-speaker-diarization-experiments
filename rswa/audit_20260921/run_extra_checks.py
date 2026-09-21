from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from ssh_moss import run
result=run('CUDA_VISIBLE_DEVICES= /work/qt28/moss/envs/ms-swift-20260914/bin/python -',
           (HERE/'check_dense_debug.py').read_bytes(),capture=True)
(HERE/'dense_debug_result.json').write_bytes(result.stdout)
print(result.stdout.decode())
