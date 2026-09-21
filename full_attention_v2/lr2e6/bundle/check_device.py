"""Short allocation-only GPU check; exits before training loads any weights."""
import argparse
import importlib.metadata
from pathlib import Path
import subprocess
import torch
from common import read, write, now


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    a=ap.parse_args()
    expected=read(Path('/work/qt28/moss/results/full-attention-v2-audit-63414/environment.json'))
    versions={name:importlib.metadata.version(name) for name in expected['versions']}
    assert versions==expected['versions'], (versions,expected['versions'])
    assert torch.cuda.is_available() and torch.cuda.device_count()==4
    devices=[]
    for rank in range(4):
        torch.cuda.set_device(rank)
        free,total=torch.cuda.mem_get_info(rank)
        name=torch.cuda.get_device_name(rank)
        assert name=='NVIDIA A40',name
        assert free>=32*2**30,(rank,free/2**30)
        devices.append(dict(rank=rank,name=name,free_gib=free/2**30,total_gib=total/2**30))
    topology=subprocess.run(['nvidia-smi','topo','-m'],capture_output=True,text=True,check=True).stdout
    write(a.run/'device_preflight.json',dict(utc=now(),devices=devices,versions=versions,topology=topology,
        world_size=4,sequence_parallel_size=1,passed=True))
    print('FOUR_A40_AND_ENVIRONMENT_PASS',flush=True)


if __name__=='__main__':
    main()
