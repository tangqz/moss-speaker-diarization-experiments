"""Pure token-coordinate, sampling, weighting, and diagnostic primitives."""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_right
import hashlib
import math
import random
import re
import unicodedata
from typing import Iterable, Mapping, Sequence

IGNORE_INDEX = -100
# Canonical targets use numeric timestamps.  The second alternative keeps the
# abstract notation used by the protocol document usable in synthetic tests.
BODY_RE = re.compile(
    r"\[(\d+(?:\.\d+)?)\]\s*\[(S\d+)\]([^\[\]]*)\[(\d+(?:\.\d+)?)\]"
    r"|\[start\]\[S\d+\](.*?)\[end\]",
    re.DOTALL,
)


@dataclass(frozen=True)
class CandidateSet:
    ids: tuple[int, ...]
    probs: tuple[float, ...]
    gold_prob: float
    error_mass: float
    candidate_mass: float


@dataclass(frozen=True)
class Budget:
    nc: int
    na: int
    total: int
    lambda_value: float
    clean_scale: float
    aux_scale: float


@dataclass(frozen=True)
class TransformResult:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    loss_scale: tuple[float, ...]
    recovery_start: int
    recovery_end: int
    insert_count: int


@dataclass(frozen=True)
class SparseDecision:
    replacement: int | None
    changed: bool
    history_triggered: bool
    temperature_branch: bool
    argmax_id: int | None
    reason: str | None
    candidates: CandidateSet | None
    sample_equals_gold: bool


def stable_seed(*parts: object) -> int:
    payload = "\x1f".join(str(x) for x in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lambda_at(update: int, maximum: float = 0.1, warmup_updates: int = 5) -> float:
    if update < 0 or warmup_updates < 0 or maximum < 0:
        raise ValueError("invalid lambda schedule")
    if warmup_updates == 0:
        return maximum
    return maximum * min((update + 1) / warmup_updates, 1.0)


def loss_budget(nc: int, na: int, update: int, *, ce_only: bool = False,
                maximum: float = 0.1, warmup_updates: int = 5) -> Budget:
    if nc <= 0 or na < 0 or (not ce_only and na <= 0):
        raise ValueError("Nc must be positive and auxiliary arms require Na > 0")
    lam = 0.0 if ce_only else lambda_at(update, maximum, warmup_updates)
    total = nc + (0 if ce_only else na)
    return Budget(nc, 0 if ce_only else na, total, lam,
                  total / nc, 0.0 if ce_only else lam * total / na)


def weighted_loss(clean_losses: Sequence[float], aux_losses: Sequence[float], budget: Budget) -> float:
    if len(clean_losses) != budget.nc or len(aux_losses) != budget.na:
        raise ValueError("loss vector counts do not match budget")
    return sum(clean_losses) / budget.nc + budget.lambda_value * (
        sum(aux_losses) / budget.na if budget.na else 0.0)


def scaled_framework_loss(clean_losses: Sequence[float], aux_losses: Sequence[float], budget: Budget) -> float:
    return (sum(x * budget.clean_scale for x in clean_losses)
            + sum(x * budget.aux_scale for x in aux_losses)) / budget.total


def body_spans(target: str) -> tuple[tuple[int, int], ...]:
    return tuple((m.start(3), m.end(3)) if m.group(3) is not None else (m.start(5), m.end(5))
                 for m in BODY_RE.finditer(target)
                 if (m.group(3) is not None and float(m.group(4)) >= float(m.group(1)))
                 or m.group(5) is not None)


def _clean_decoded_token(text: str) -> bool:
    if not text or text.isspace() or "\ufffd" in text or "[" in text or "]" in text:
        return False
    return not any(unicodedata.category(ch) in {"Cc", "Cf", "Cs"} for ch in text)


def legal_token_id(tokenizer, token_id: int, special_ids: Iterable[int] = ()) -> bool:
    if token_id in set(int(x) for x in special_ids):
        return False
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    if not _clean_decoded_token(text):
        return False
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return list(encoded) == [int(token_id)]


def legal_body_ids(tokenizer) -> tuple[int, ...]:
    special = set(getattr(tokenizer, "all_special_ids", ()) or ())
    size = len(tokenizer)
    return tuple(i for i in range(size) if legal_token_id(tokenizer, i, special))


def eligible_anchors(*, z: Sequence[int], labels: Sequence[int], target_offsets: Sequence[tuple[int, int]],
                     target_start: int, spans: Sequence[tuple[int, int]], tokenizer,
                     k: int, min_body_labels: int = 4,
                     precomputed_legal_ids: Iterable[int] | None = None) -> tuple[int, ...]:
    """Return absolute z positions whose target token lies wholly in body text."""
    if len(z) != len(labels) or len(target_offsets) > len(z) - target_start:
        raise ValueError("coordinate lengths are inconsistent")
    special = set(getattr(tokenizer, "all_special_ids", ()) or ())
    legal = set(map(int, precomputed_legal_ids)) if precomputed_legal_ids is not None else None
    ordered_spans = sorted((int(lo), int(hi)) for lo, hi in spans)
    span_starts = [lo for lo, _ in ordered_spans]
    body_token = []
    for offset in target_offsets:
        start, end = map(int, offset)
        span_index = bisect_right(span_starts, start) - 1
        body_token.append(start < end and span_index >= 0 and end <= ordered_spans[span_index][1])
    result = []
    for rel, is_body in enumerate(body_token):
        a = target_start + rel
        token_is_legal = int(z[a]) in legal if legal is not None else legal_token_id(tokenizer, int(z[a]), special)
        if (not is_body or not token_is_legal
                or a == 0 or a + 1 >= len(z)
                or int(z[a - 1]) == int(z[a]) or int(z[a + 1]) == int(z[a])):
            continue
        if a + 1 + k > len(z):
            continue
        within = 0
        for j in range(rel + 1, min(rel + 1 + k, len(body_token))):
            within += int(body_token[j])
        if within >= min_body_labels and all(int(labels[j]) != IGNORE_INDEX for j in range(a + 1, a + 1 + k)):
            result.append(a)
    return tuple(result)


def choose_quartile_anchor(anchors: Sequence[int], *, target_start: int, target_token_count: int,
                           seed: int, source_key: str, occurrence: int) -> tuple[int, int, bool]:
    if not anchors:
        raise ValueError("no eligible anchors")
    wanted = stable_seed(seed, source_key, occurrence, "quartile") % 4
    grouped = [[] for _ in range(4)]
    for a in anchors:
        progress = (a - target_start) / max(target_token_count, 1)
        grouped[min(3, max(0, int(progress * 4)))].append(int(a))
    fallback = not bool(grouped[wanted])
    pool = grouped[wanted] if not fallback else [a for group in grouped for a in group]
    rng = random.Random(stable_seed(seed, source_key, occurrence, wanted, "anchor"))
    return pool[rng.randrange(len(pool))], int(wanted), fallback


def candidate_set(logits: Sequence[float], gold: int, legal_ids: Iterable[int], *,
                  min_prob: float = 1e-4, relative_to_gold: float = 0.01,
                  top_k: int = 32, include_gold: bool = True) -> CandidateSet:
    if not logits:
        raise ValueError("empty vocabulary logits")
    values = [float(x) for x in logits]
    if any(math.isnan(x) or x == math.inf for x in values) or not any(math.isfinite(x) for x in values):
        raise ValueError("logits contain NaN/+inf or have no finite value")
    if not 0 <= gold < len(values):
        raise ValueError("gold token outside vocabulary")
    max_logit = max(values)
    exp_values = [math.exp(x - max_logit) for x in values]
    denom = math.fsum(exp_values)
    probs = [x / denom for x in exp_values]
    gold_prob = probs[gold]
    legal = set(int(x) for x in legal_ids)
    ids = [i for i, p in enumerate(probs)
           if (i != gold or include_gold) and (i in legal or include_gold and i == gold)
           and p >= min_prob and p >= relative_to_gold * gold_prob]
    ids.sort(key=lambda i: (-probs[i], i))
    ids = ids[:top_k]
    if include_gold and gold not in ids:
        ids.append(gold)
        ids.sort(key=lambda i: (-probs[i], i))
    ps = tuple(probs[i] for i in ids)
    return CandidateSet(tuple(ids), ps, gold_prob, 1.0 - gold_prob, math.fsum(ps))


def sample_candidate(candidates: CandidateSet, seed: int, *, temperature: float = 1.0,
                     minimum_mass: float = 0.01) -> int | None:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not candidates.ids or candidates.candidate_mass < minimum_mass:
        return None
    weights = tempered_weights(candidates.probs, temperature)
    if not weights:
        return None
    threshold = random.Random(seed).random() * math.fsum(weights)
    acc = 0.0
    for token_id, weight in zip(candidates.ids, weights):
        acc += weight
        if threshold <= acc:
            return token_id
    return candidates.ids[-1]


def tempered_weights(probs: Sequence[float], temperature: float) -> tuple[float, ...]:
    """Stable unnormalized temperature weights, accepting zero-mass forced gold."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logp = [math.log(float(p)) if p > 0 else -math.inf for p in probs]
    finite = [value for value in logp if math.isfinite(value)]
    if not finite:
        return ()
    maximum = max(finite)
    return tuple(0.0 if not math.isfinite(value)
                 else math.exp((value - maximum) / temperature) for value in logp)


def bernoulli(seed: int, probability: float) -> bool:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    return random.Random(seed).random() < probability


def argmax_smallest(logits: Sequence[float]) -> int:
    values = [float(x) for x in logits]
    if not values or any(math.isnan(x) or x == math.inf for x in values) or not any(math.isfinite(x) for x in values):
        raise ValueError("invalid logits for argmax")
    maximum = max(values)
    return next(i for i, value in enumerate(values) if value == maximum)


def sparse_substitution(logits: Sequence[float], *, gold: int, legal_ids: Iterable[int],
                        history_seed: int, branch_seed: int, sample_seed: int,
                        model_history_probability: float = 1.0,
                        sampling_probability: float = 0.02,
                        temperature: float = 1.0,
                        minimum_candidate_mass: float = 0.5) -> SparseDecision:
    legal = set(map(int, legal_ids))
    greedy = argmax_smallest(logits)
    triggered = bernoulli(history_seed, model_history_probability)
    if not triggered:
        return SparseDecision(None, False, False, False, greedy, "model_history_not_triggered", None, False)
    use_temperature = bernoulli(branch_seed, sampling_probability)
    if not use_temperature:
        if greedy == gold:
            return SparseDecision(None, False, True, False, greedy, "argmax_gold", None, False)
        if greedy not in legal:
            return SparseDecision(None, False, True, False, greedy, "illegal_argmax", None, False)
        return SparseDecision(greedy, True, True, False, greedy, None, None, False)
    candidates = candidate_set(logits, gold, legal, include_gold=True)
    if candidates.candidate_mass < minimum_candidate_mass:
        return SparseDecision(None, False, True, True, greedy, "low_legal_mass", candidates, False)
    sampled = sample_candidate(candidates, sample_seed, temperature=temperature,
                               minimum_mass=minimum_candidate_mass)
    if sampled == gold:
        return SparseDecision(None, False, True, True, greedy, "sample_gold", candidates, True)
    return SparseDecision(sampled, True, True, True, greedy, None, candidates, False)


def apply_event(z: Sequence[int], *, anchor: int, k: int, kind: str,
                clean_scale: float, aux_scale: float, replacement: int | None = None,
                repeat_count: int = 0) -> TransformResult:
    """Apply one event and place exactly K recovery labels after the event."""
    z = [int(x) for x in z]
    if not (0 <= anchor < len(z)) or anchor + 1 + k > len(z) or k <= 0:
        raise ValueError("invalid anchor/recovery window")
    if kind not in {"clean_aux", "substitute", "repeat_previous", "substitute_then_repeat"}:
        raise ValueError(f"unknown event kind: {kind}")
    if kind in {"substitute", "substitute_then_repeat"}:
        if replacement is None or int(replacement) == z[anchor]:
            raise ValueError("substitution requires a different token")
    if kind in {"repeat_previous", "substitute_then_repeat"}:
        if repeat_count not in {1, 2, 3}:
            raise ValueError("repeat_count must be 1, 2, or 3")
        if z[anchor + 1] == z[anchor] and kind == "repeat_previous":
            raise ValueError("repeat event would overlap a real adjacent repetition")
    else:
        repeat_count = 0

    out = list(z)
    event_token = z[anchor]
    if kind in {"substitute", "substitute_then_repeat"}:
        event_token = int(replacement)
        out[anchor] = event_token
    if repeat_count:
        out[anchor + 1:anchor + 1] = [event_token] * repeat_count
    recovery_start = anchor + 1 + repeat_count
    labels = [IGNORE_INDEX] * len(out)
    labels[recovery_start:recovery_start + k] = z[anchor + 1:anchor + 1 + k]
    scales = [0.0] * len(out)
    scales[recovery_start:recovery_start + k] = [float(aux_scale)] * k
    return TransformResult(tuple(out), tuple(labels), tuple(scales), recovery_start,
                           recovery_start + k, repeat_count)


def clean_view(z: Sequence[int], labels: Sequence[int], scale: float) -> TransformResult:
    if len(z) != len(labels):
        raise ValueError("clean inputs and labels differ in length")
    scales = tuple(float(scale) if int(y) != IGNORE_INDEX else 0.0 for y in labels)
    return TransformResult(tuple(map(int, z)), tuple(map(int, labels)), scales, -1, -1, 0)


def periodic_runs(ids: Sequence[int], *, min_single: int = 32, max_period: int = 16,
                  min_periodic: int = 64) -> tuple[Mapping[str, int], ...]:
    """Find de-duplicated single-token and period 2..16 suspicious runs."""
    values = list(map(int, ids))
    found: list[dict[str, int]] = []
    n = len(values)
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[j] == values[i]:
            j += 1
        if j - i >= min_single:
            found.append({"start": i, "end": j, "period": 1, "length": j - i})
        i = j
    for period in range(2, max_period + 1):
        i = 0
        while i + min_periodic <= n:
            j = i + period
            while j < n and values[j] == values[i + (j - i) % period]:
                j += 1
            if j - i >= min_periodic:
                found.append({"start": i, "end": j, "period": period, "length": j - i})
                i = j
            else:
                i += 1
    found.sort(key=lambda x: (x["start"], x["period"], -x["length"]))
    dedup = []
    for item in found:
        if any(item["start"] >= prev["start"] and item["end"] <= prev["end"] for prev in dedup):
            continue
        dedup.append(item)
    return tuple(dedup)
