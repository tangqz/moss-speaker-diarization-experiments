#!/usr/bin/env python
"""Minimal NCCL smoke test for multi-GPU DKUCC allocations.

Run inside a >=2 GPU Slurm allocation:
    torchrun --standalone --nproc_per_node=2 dkucc/dist_test.py
"""
import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    name = torch.cuda.get_device_name(device)

    dist.init_process_group("nccl")
    print(f"[nccl] rank {rank}/{world} using cuda:{local_rank} ({name})", flush=True)

    # all-reduce test: sum of (rank+1)
    t = torch.full((1,), float(rank + 1), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    expected = float(world * (world + 1) / 2)
    allreduce_ok = abs(t.item() - expected) < 1e-6

    # broadcast test
    t2 = torch.zeros(1, device=device)
    if rank == 0:
        t2.fill_(42.0)
    dist.broadcast(t2, src=0)
    broadcast_ok = abs(t2.item() - 42.0) < 1e-6

    dist.barrier()
    print(
        f"[nccl] rank {rank} allreduce={t.item()} (expected {expected}) "
        f"broadcast={t2.item()} ok={allreduce_ok and broadcast_ok}",
        flush=True,
    )
    if rank == 0:
        print("[nccl] ALL-DONE ok=" + str(allreduce_ok and broadcast_ok), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
