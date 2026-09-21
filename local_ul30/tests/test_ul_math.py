import math

import torch

from local_ul30.core import apply_event, loss_budget


def stable_ul(logits, token_id):
    row = logits.float()
    other = torch.cat((row[:token_id], row[token_id + 1:]))
    return torch.nn.functional.softplus(row[token_id] - torch.logsumexp(other, 0))


def test_ul_matches_probability_formula_and_has_finite_gradient():
    logits = torch.tensor([2.0, 0.5, -1.0, 1.25], dtype=torch.float64, requires_grad=True)
    value = stable_ul(logits, 0)
    probability = torch.softmax(logits, 0)[0]
    expected = -torch.log1p(-probability)
    torch.testing.assert_close(value, expected.float(), rtol=2e-6, atol=2e-6)
    value.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0] > 0


def test_recovery_exit_and_tail_scales_are_event_normalized():
    budget = loss_budget(100, 64, update=0, maximum=0.1, warmup_updates=0)
    exit_scale = budget.lambda_value * budget.total / 4
    tail_scale = budget.lambda_value * budget.total / (4 * 15)
    assert math.isclose(exit_scale + 15 * tail_scale, budget.lambda_value * budget.total / 2)
    assert math.isclose(exit_scale / tail_scale, 15)


def test_fixed_repeat_event_keeps_inserted_copies_context_only():
    result = apply_event([1, 2, 7, 4, 5, 6], anchor=2, k=2,
                         kind="repeat_previous", repeat_count=2,
                         clean_scale=1.0, aux_scale=0.1)
    assert result.input_ids[2:5] == (7, 7, 7)
    assert result.labels[2:5] == (-100, -100, -100)
    assert result.labels[5:7] == (4, 5)
