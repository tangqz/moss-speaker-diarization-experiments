from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]))
from ssh_moss import run
result=run('CUDA_VISIBLE_DEVICES= /work/qt28/moss/envs/ms-swift-20260914/bin/python -',
           (HERE/'cpu_checks.py').read_bytes(),capture=True)
(HERE/'cpu_checks_result.json').write_bytes(result.stdout)
print(result.stdout.decode())
