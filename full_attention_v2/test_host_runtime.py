"""CUDA check: release cached host blocks, preserve live data, gradients and RNG."""
import json
import torch
from host_runtime import release_host_cache, host_snapshot


def main():
    torch.cuda.set_device(0)
    torch.manual_seed(71)
    live = torch.full((1024 * 1024,), 3, dtype=torch.uint8, pin_memory=True)
    temporary = torch.empty((64 * 1024 * 1024,), dtype=torch.uint8, pin_memory=True)
    temporary.fill_(5)
    del temporary
    torch.cuda.synchronize()
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    cache = release_host_cache()
    assert cache['before']['pinned_owned_gib'] - cache['after']['pinned_owned_gib'] >= 0.060
    assert cache['after']['pinned_active_gib'] > 0
    assert bool(live.eq(3).all()), 'Live pinned storage was changed'
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    x = torch.randn(4096, 512, device='cuda')
    w = torch.randn(512, 128, device='cuda', requires_grad=True)
    results = []
    for flush in [False, True]:
        w.grad = None
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            loss = (x @ w).square().mean()
        if flush:
            release_host_cache()
        loss.backward()
        results.append((loss.detach().clone(), w.grad.detach().clone()))
        del loss
        release_host_cache()
    assert torch.equal(results[0][0], results[1][0])
    assert torch.equal(results[0][1], results[1][1])
    print(json.dumps(dict(passed=True, cached_memory_reclaimed=cache,
          live_tensor_preserved=True, rng_preserved=True, loss_and_gradient_exact=True,
          final=host_snapshot(), torch_version=torch.__version__)), flush=True)


if __name__ == '__main__':
    main()
