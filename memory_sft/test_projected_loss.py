"""Compare dense CE loss/gradients, masked labels, accumulation and DDP scaling."""
import json
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from memory_sft import projected_causal_loss

device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
torch.cuda.set_device(device)
world = int(os.environ.get("WORLD_SIZE", 1))
if world > 1:
    dist.init_process_group("nccl")
rank = int(os.environ.get("RANK", 0))
torch.manual_seed(918)
h0 = torch.randn(4, 37, 32, device=device) * 0.2
w0 = torch.randn(127, 32, device=device) * 0.2
y = torch.randint(0, 127, (4, 37), device=device)
for i in range(4):
    y[i, :3 + 4*i] = -100
y[0, :] = -100  # A completely unlabelled rank must still join gradient reduction.
y[:, -3:] = -100
shift = F.pad(y, (0, 1), value=-100)[:, 1:]
denom = shift.ne(-100).sum()
for dtype in [torch.float32, torch.bfloat16]:
    h, w = h0.to(dtype).detach().requires_grad_(), w0.to(dtype).detach().requires_grad_()
    dense = F.cross_entropy(F.linear(h, w).float().reshape(-1, 127), shift.reshape(-1), reduction="sum") / denom
    dense.backward()
    dh, dw = h.grad.clone(), w.grad.clone()
    hc, wc = h.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    chunked = projected_causal_loss(hc, wc, y, chunk_size=11)
    chunked.backward()
    tol = 1e-6 if dtype == torch.float32 else 2e-4
    torch.testing.assert_close(chunked, dense, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(hc.grad, dh, atol=tol, rtol=0.03 if dtype == torch.bfloat16 else 1e-5)
    torch.testing.assert_close(wc.grad, dw, atol=tol, rtol=0.03 if dtype == torch.bfloat16 else 1e-5)
    ha, wa = h.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    # Match Trainer's global denominator and world-size compensation before DDP mean.
    idxs = list(range(rank, 4, world))
    accum = sum(projected_causal_loss(ha[i:i+1], wa, y[i:i+1], denom, 11) * world for i in idxs)
    accum.backward()
    if world > 1:
        dist.all_reduce(wa.grad)
        wa.grad /= world
    torch.testing.assert_close(wa.grad, dw, atol=tol, rtol=0.03 if dtype == torch.bfloat16 else 1e-5)
    print(json.dumps(dict(event="projected_loss_test_pass",rank=rank,world=world,dtype=str(dtype),
        loss=float(chunked),loss_error=float((chunked-dense).abs()),head_grad_error=float((wc.grad-dw).abs().max()))),flush=True)
if world > 1:
    dist.destroy_process_group()
