import math

import numpy as np
import pytest

from local_ul30.core import (IGNORE_INDEX, apply_event, body_spans, candidate_set,
                                 clean_view, eligible_anchors, loss_budget,
                                 periodic_runs, sample_candidate, scaled_framework_loss,
                                 sparse_substitution, stable_seed, weighted_loss)


def test_frozen_golden_substitution_coordinates():
    z = [100, 101, 10, 11, 12, 13, 14, 99]
    result = apply_event(z, anchor=4, k=2, kind="substitute", replacement=77,
                         repeat_count=0, clean_scale=1.1, aux_scale=0.2)
    assert list(result.input_ids) == [100, 101, 10, 11, 77, 13, 14, 99]
    assert list(result.labels) == [-100, -100, -100, -100, -100, 13, 14, -100]
    assert result.recovery_start == 5
    assert sum(x != IGNORE_INDEX for x in result.labels) == 2


def test_frozen_golden_repeat3_coordinates():
    z = [100, 101, 10, 11, 12, 13, 14, 99]
    result = apply_event(z, anchor=4, k=2, kind="repeat_previous", repeat_count=2,
                         clean_scale=1.1, aux_scale=0.2)
    assert list(result.input_ids) == [100, 101, 10, 11, 12, 12, 12, 13, 14, 99]
    assert list(result.labels) == [-100, -100, -100, -100, -100, -100, -100, 13, 14, -100]
    assert (result.recovery_start, result.recovery_end) == (7, 9)


def test_substitute_then_repeat_is_explicitly_distinct():
    result = apply_event([1, 2, 3, 4, 5], anchor=2, k=2,
                         kind="substitute_then_repeat", replacement=9, repeat_count=1,
                         clean_scale=1, aux_scale=.1)
    assert result.input_ids == (1, 2, 9, 9, 4, 5)
    assert result.labels == (-100, -100, -100, -100, 4, 5)


@pytest.mark.parametrize("insert_count,total_repeats", [(2, 3), (3, 4)])
def test_error_is_repeated_three_or_four_times_before_recovery(insert_count, total_repeats):
    result = apply_event([1, 2, 3, 4, 5, 6], anchor=2, k=2,
                         kind="substitute_then_repeat", replacement=9,
                         repeat_count=insert_count, clean_scale=1, aux_scale=.1)
    assert result.input_ids[2:2 + total_repeats] == (9,) * total_repeats
    assert result.recovery_start == 2 + total_repeats
    assert result.labels[result.recovery_start:result.recovery_end] == (4, 5)


def test_loss_scale_exact_formula_and_fp64_gradient():
    budget = loss_budget(5, 3, update=2)
    clean = [1.0, 2.0, 3.0, 4.0, 7.0]
    aux = [5.0, 6.0, 9.0]
    assert math.isclose(weighted_loss(clean, aux, budget),
                        scaled_framework_loss(clean, aux, budget), rel_tol=1e-14)
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(8, 7)).astype(np.float64)
    labels = np.array([1, 2, 3, 4, 5, 1, 2, 3])
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    per_token_grad = probs.copy()
    per_token_grad[np.arange(8), labels] -= 1
    direct_grad = per_token_grad.copy()
    direct_grad[:5] /= 5
    direct_grad[5:] *= budget.lambda_value / 3
    scales = np.array([budget.clean_scale] * 5 + [budget.aux_scale] * 3)[:, None]
    native_grad = per_token_grad * scales / budget.total
    np.testing.assert_allclose(direct_grad, native_grad, rtol=1e-13, atol=1e-15)


def test_oracle_handles_unequal_lengths_and_empty_sp_shard():
    budget = loss_budget(7, 4, update=4)
    # Four conceptual SP shards; shard 2 has no valid labels.
    shards = [([1.0, 2.0], [8.0]), ([3.0, 4.0, 5.0], [9.0]),
              ([], []), ([6.0, 7.0], [10.0, 11.0])]
    clean = [x for c, _ in shards for x in c]
    aux = [x for _, a in shards for x in a]
    assert len(clean) == 7 and len(aux) == 4
    assert scaled_framework_loss(clean, aux, budget) == pytest.approx(weighted_loss(clean, aux, budget))


def test_candidate_filter_before_topk_tie_order_and_sampling():
    logits = [0.0] * 40
    candidates = candidate_set(logits, gold=5, legal_ids=range(40), min_prob=0,
                               relative_to_gold=0, top_k=32, include_gold=False)
    assert 5 not in candidates.ids
    assert candidates.ids == tuple([0, 1, 2, 3, 4] + list(range(6, 33)))
    seed = stable_seed(0, "source", 1, 4, "substitute")
    assert sample_candidate(candidates, seed, minimum_mass=0) == sample_candidate(candidates, seed, minimum_mass=0)


@pytest.mark.parametrize("values", [[], [0.0, float("nan")], [0.0, float("inf")]])
def test_bad_logits_fail_closed(values):
    with pytest.raises(ValueError):
        candidate_set(values, 0, [0])


def test_negative_infinity_is_allowed_when_some_logit_is_finite():
    result = candidate_set([0.0, float("-inf"), -1.0], 0, [0, 2], include_gold=True)
    assert result.ids == (0, 2)


def test_temperature_sampling_accepts_zero_probability_forced_gold():
    candidates = candidate_set([float("-inf"), 0.0], 0, [1], include_gold=True,
                               min_prob=0, relative_to_gold=0)
    assert candidates.ids == (1, 0)
    assert sample_candidate(candidates, seed=7, temperature=1e-9, minimum_mass=0) == 1


def test_sparse_greedy_gold_and_illegal_argmax_stay_clean():
    gold = sparse_substitution([3.0, 1.0], gold=0, legal_ids=[0, 1],
                               history_seed=1, branch_seed=1, sample_seed=1,
                               model_history_probability=1, sampling_probability=0)
    assert not gold.changed and gold.reason == "argmax_gold"
    illegal = sparse_substitution([1.0, 3.0], gold=0, legal_ids=[0],
                                  history_seed=1, branch_seed=1, sample_seed=1,
                                  model_history_probability=1, sampling_probability=0)
    assert not illegal.changed and illegal.reason == "illegal_argmax"


def test_sparse_temperature_distribution_keeps_gold_and_probability_zero_is_identity():
    decision = sparse_substitution([2.0, 1.9, 1.8], gold=0, legal_ids=[1, 2],
                                   history_seed=1, branch_seed=1, sample_seed=3,
                                   model_history_probability=0, sampling_probability=1)
    assert not decision.changed and decision.reason == "model_history_not_triggered"
    sampled = sparse_substitution([2.0, 1.9, 1.8], gold=0, legal_ids=[1, 2],
                                  history_seed=1, branch_seed=1, sample_seed=3,
                                  model_history_probability=1, sampling_probability=1,
                                  minimum_candidate_mass=0)
    assert 0 in sampled.candidates.ids


def test_numeric_body_spans_and_invalid_reverse_time():
    target = "[3.57][S01]Uh 'kay.[3.87][9.69][S02]So.[10.64][12.0][S3]bad[11.0]"
    spans = body_spans(target)
    assert [target[a:b] for a, b in spans] == ["Uh 'kay.", "So."]


class FakeTokenizer:
    all_special_ids = [99]
    def decode(self, ids, **kwargs): return chr(96 + ids[0])
    def encode(self, text, add_special_tokens=False): return [ord(text) - 96]


def test_anchor_pool_excludes_both_adjacent_repetitions():
    tok = FakeTokenizer()
    offsets = [(i, i + 1) for i in range(8)]
    spans = [(0, 8)]
    labels = list(range(1, 9))
    assert 2 not in eligible_anchors(z=[1, 2, 3, 3, 4, 5, 6, 7], labels=labels,
                                    target_offsets=offsets, target_start=0, spans=spans,
                                    tokenizer=tok, k=4, min_body_labels=4)
    assert 3 not in eligible_anchors(z=[1, 2, 3, 3, 4, 5, 6, 7], labels=labels,
                                    target_offsets=offsets, target_start=0, spans=spans,
                                    tokenizer=tok, k=4, min_body_labels=4)


def test_periodic_diagnostics():
    ids = [1, 2] + [7] * 32 + [3, 4] * 32
    runs = periodic_runs(ids)
    assert any(x["period"] == 1 and x["length"] == 32 for x in runs)
    assert any(x["period"] == 2 and x["length"] >= 64 for x in runs)
