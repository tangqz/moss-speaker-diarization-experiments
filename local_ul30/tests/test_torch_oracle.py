import torch

from local_ul30.core import loss_budget


def test_native_loss_scale_matches_direct_ce_autograd_fp64():
    budget = loss_budget(7, 4, update=3)
    torch.manual_seed(4)
    left = torch.randn(11, 13, dtype=torch.float64, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    a = torch.nn.functional.cross_entropy(left, labels, reduction="none")
    direct = a[:7].sum() / 7 + budget.lambda_value * a[7:].sum() / 4
    direct.backward()
    b = torch.nn.functional.cross_entropy(right, labels, reduction="none")
    scales = torch.tensor([budget.clean_scale] * 7 + [budget.aux_scale] * 4, dtype=torch.float64)
    native = (b * scales).sum() / budget.total
    native.backward()
    torch.testing.assert_close(direct, native, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(left.grad, right.grad, rtol=1e-12, atol=1e-14)
