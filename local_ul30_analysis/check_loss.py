"""CPU audit for the local UL30 boundary loss and SP normalization.

This is deliberately a small deterministic audit.  It extracts the production
``RecoveryTrainer._ul_value_from_row`` helper from ``trainer.py`` and uses a
CPU-only four-shard model to emulate the value/gradient contract of
``GatherLoss``.  It does not claim GPU, NCCL, FSDP, or end-to-end Trainer
parity.

Run from the repository root::

    python dkucc/local_ul30_analysis/check_loss.py

The script writes ``check_loss_result.json`` beside itself.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRAINER = ROOT / "dkucc" / "local_ul30" / "trainer.py"
DESIGN = ROOT / "dkucc" / "reports" / "local_ul30_design_20260917.md"
GATHER = (ROOT / "dkucc" / "ms_swift" / "upstream"
          / "ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076"
          / "swift" / "sequence_parallel" / "utils.py")
SEQ2SEQ = (ROOT / "dkucc" / "ms_swift" / "upstream"
           / "ms-swift-0673cf75dca7d0b9b608b4a76632fb508ead5076"
           / "swift" / "trainers" / "seq2seq_trainer.py")
PREPARE = ROOT / "dkucc" / "local_ul30" / "prepare.py"
CORE = ROOT / "dkucc" / "local_ul30" / "core.py"
OUT = Path(__file__).with_name("check_loss_result.json")

WORLD = 4
VOCAB = 11
LENGTH = 16
SHARD = LENGTH // WORLD
LAMBDA = 0.1
TOKEN = 3


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_ul_helper(path: Path) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """Extract the exact production value helper without importing Trainer."""
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = next(
        n for n in ast.walk(module)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "_ul_value_from_row"
    )
    # The helper has one staticmethod decorator, but compiling the function
    # itself is sufficient and avoids importing CUDA/Swift/Transformers.
    node.decorator_list = []
    code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
    namespace = {"torch": torch, "F": F}
    exec(code, namespace)
    return namespace["_ul_value_from_row"]


UL = extract_ul_helper(TRAINER)


def close(a: float, b: float, atol: float = 2e-6, rtol: float = 2e-6) -> bool:
    return math.isclose(float(a), float(b), abs_tol=atol, rel_tol=rtol)


def test_stable_ul() -> dict[str, Any]:
    # The repeated class is overwhelmingly likely (p rounds to 1 in fp32),
    # which is exactly where a ``clamp(1-p)`` implementation can lose signal.
    row = torch.zeros(VOCAB, dtype=torch.float32, requires_grad=True)
    row.data[TOKEN] = 80.0
    value = UL(row, TOKEN)
    value.backward()
    probability = float((-torch.expm1(-value.detach())).item())
    grad_repeat = float(row.grad[TOKEN].item())
    grad_norm = float(row.grad.norm().item())

    # A moderate row checks that the extracted helper still agrees with the
    # mathematically equivalent expression away from cancellation.
    moderate = torch.tensor([1.2, -0.7, 0.4, 0.1], dtype=torch.float32)
    got = float(UL(moderate, 0).item())
    p64 = torch.softmax(moderate.double(), dim=0)[0]
    expected = float((-torch.log1p(-p64)).item())
    passed = (
        probability == 1.0
        and value.isfinite().item()
        and grad_repeat > 0.9
        and grad_norm > 0.9
        and close(got, expected, atol=2e-6, rtol=2e-6)
    )
    return {
        "passed": bool(passed),
        "ul_at_p_near_1": float(value.item()),
        "repeat_probability_fp32": probability,
        "repeat_gradient": grad_repeat,
        "gradient_l2": grad_norm,
        "moderate_ul": got,
        "moderate_reference": expected,
        "interpretation": "finite UL and nonzero gradient remain when p rounds to 1 in fp32",
    }


def test_query_boundary() -> dict[str, Any]:
    # Match core.apply_event: anchor + repeat_count copies are repeated, and
    # recovery_start is the first correct target.  trainer.py queries
    # recovery_start - 1, the final repeated-token position.
    from dkucc.local_ul30.core import apply_event

    source = [10, 11, 12, TOKEN] + list(range(20, 36)) + [99]
    anchor = 3
    repeat_count = 3
    result = apply_event(
        source, anchor=anchor, k=16, kind="repeat_previous",
        clean_scale=1.0, aux_scale=1.0, repeat_count=repeat_count,
    )
    query = result.recovery_start - 1
    expected_query = anchor + repeat_count
    repeated_positions = list(range(anchor, query + 1))
    eos_position = len(source) - 1 + repeat_count
    passed = (
        query == expected_query
        and all(result.input_ids[p] == TOKEN for p in repeated_positions)
        and result.input_ids[eos_position] == 99
        and all(x == -100 for x in result.labels[anchor + 1:result.recovery_start])
        and list(result.labels[result.recovery_start:result.recovery_start + 16])
        == source[anchor + 1:anchor + 17]
    )
    return {
        "passed": bool(passed),
        "anchor": anchor,
        "repeat_count_inserted": repeat_count,
        "recovery_start": int(result.recovery_start),
        "ul_query_abs": int(query),
        "expected_last_repeated_position": expected_query,
        "repeated_positions": repeated_positions,
        "original_eos_position_after_insert": eos_position,
        "interpretation": "UL is evaluated at the last repeated token, whose next-token logits see the complete repeated prefix",
    }


def event_normalization() -> dict[str, Any]:
    # Four auxiliary rows each carry one exit CE and 15 tail CE terms.  The
    # metadata scales in prepare.py are the token-level realization of
    # lambda/4 * (exit + mean(tail)); average_tokens_across_devices then
    # cancels the SP/data-world denominator.
    exits = torch.tensor([0.4, 0.7, 1.1, 1.6], dtype=torch.float64)
    tails = torch.tensor([
        [0.20 + 0.01 * j for j in range(15)],
        [0.30 + 0.01 * j for j in range(15)],
        [0.40 + 0.01 * j for j in range(15)],
        [0.50 + 0.01 * j for j in range(15)],
    ], dtype=torch.float64)
    ul = torch.tensor([0.8, 1.2, 0.6, 1.5], dtype=torch.float64)
    recovery_expected = float((exits + tails.mean(dim=1)).mean().item())
    ul_expected = float(ul.mean().item())

    n = 100
    world = WORLD
    aux_exit_scale = LAMBDA * n / 4.0
    aux_tail_scale = LAMBDA * n / (4.0 * 15.0)
    # Each local parent loss divides by world*n.  Native average-tokens
    # compensation multiplies by world, leaving the following event values.
    recovery_rows = []
    for e, t in zip(exits, tails):
        weighted_sum = aux_exit_scale * e + aux_tail_scale * t.sum()
        recovery_rows.append(float((weighted_sum / (world * n) * world).item()))
    recovery_after_four_rows = sum(recovery_rows)

    # compute_loss adds UL * lambda / _ul_event_count (4) per auxiliary row.
    ul_rows = [float((u * LAMBDA / 4.0).item()) for u in ul]
    ul_after_four_rows = sum(ul_rows)
    passed = close(recovery_after_four_rows, LAMBDA * recovery_expected) and close(
        ul_after_four_rows, LAMBDA * ul_expected
    )
    return {
        "passed": bool(passed),
        "event_count": 4,
        "tail_terms_per_event": 15,
        "recovery_expected_unweighted": recovery_expected,
        "recovery_after_four_rows": recovery_after_four_rows,
        "recovery_lambda_weighted_target": LAMBDA * recovery_expected,
        "ul_expected_event_mean": ul_expected,
        "ul_after_four_rows": ul_after_four_rows,
        "ul_lambda_weighted_target": LAMBDA * ul_expected,
        "aux_exit_scale": aux_exit_scale,
        "aux_tail_scale": aux_tail_scale,
        "world_denominator": world * n,
        "interpretation": "four rows produce a mean over events; the exit is one term and the remaining 15 CE terms are averaged",
    }


def direct_target(weights: torch.Tensor, labels: torch.Tensor, query: int) -> tuple[float, torch.Tensor]:
    w = weights.detach().clone().requires_grad_(True)
    valid = labels != -100
    ce = F.cross_entropy(w[valid], labels[valid], reduction="sum") / valid.sum()
    ul = UL(w[query], TOKEN)
    loss = ce + LAMBDA * ul / 4.0
    loss.backward()
    return float(loss.detach().item()), w.grad.detach().clone()


def distributed_sp_target(weights: torch.Tensor, labels: torch.Tensor, query: int) -> tuple[float, torch.Tensor, dict[str, Any]]:
    """Simulate four GatherLoss outputs and its world-size backward factor.

    Every rank sees the gathered scalar value.  Only the owner has a graph for
    the UL term; other ranks retain a graph-connected zero, as trainer.py does.
    Local CE is allowed to be empty on rank 3.  Multiplying the local graph
    delta by WORLD models GatherLoss.backward's ``grad_output * world_size``;
    averaging the four rank losses models FSDP/DDP gradient averaging.
    """
    w = weights.detach().clone().requires_grad_(True)
    valid = labels != -100
    ce_all = F.cross_entropy(w[valid], labels[valid], reduction="sum")
    count = int(valid.sum().item())
    owner = query // SHARD
    rank_losses = []
    local_valid_counts = []
    nonowner_ul_graph_abs = []
    for rank in range(WORLD):
        start, end = rank * SHARD, (rank + 1) * SHARD
        idx = torch.arange(start, end)
        local_valid = labels[idx] != -100
        local_valid_counts.append(int(local_valid.sum().item()))
        if local_valid.any():
            local_ce = F.cross_entropy(w[idx][local_valid], labels[idx][local_valid], reduction="sum")
        else:
            # The production path keeps a graph-connected zero on a rank with
            # no valid labels rather than introducing a local denominator.
            local_ce = w[idx, 0].sum() * 0.0
        ce_global = ce_all.detach() + WORLD * (local_ce - local_ce.detach())

        local_zero = w[idx, 0].sum() * 0.0
        u = UL(w[query], TOKEN)
        if rank == owner:
            # GatherLoss backward multiplies this owner's incoming gradient by
            # WORLD before the simulated FSDP average below.
            ul_global = u.detach() + WORLD * (u - u.detach())
        else:
            ul_global = u.detach() + local_zero
            nonowner_ul_graph_abs.append(float(torch.autograd.grad(
                ul_global, w, retain_graph=True, allow_unused=True
            )[0].abs().sum().item()))
        rank_losses.append(ce_global / count + LAMBDA * ul_global / 4.0)

    distributed = sum(rank_losses) / WORLD
    distributed.backward()
    details = {
        "owner_rank": owner,
        "local_valid_label_counts": local_valid_counts,
        "nonowner_ul_graph_l1": nonowner_ul_graph_abs,
        "zero_label_ranks": [r for r, n in enumerate(local_valid_counts) if n == 0],
        "gatherloss_world_factor": WORLD,
        "simulated_fsdp_average": True,
    }
    return float(distributed.detach().item()), w.grad.detach().clone(), details


def test_sp_gradient_parity() -> dict[str, Any]:
    torch.manual_seed(20260917)
    weights = torch.randn(LENGTH, VOCAB, dtype=torch.float32)
    # Ranks 0--2 have valid CE; rank 3 is an explicit zero-label shard.  Query
    # positions cover both sides of shard boundaries, including rank 3.
    labels = torch.full((LENGTH,), -100, dtype=torch.long)
    labels[0:3] = torch.tensor([1, 2, 4])
    labels[4:7] = torch.tensor([2, 5, 1])
    labels[8:11] = torch.tensor([3, 6, 0])
    queries = [3, 4, 7, 8, 11, 12, 15]
    rows = []
    for query in queries:
        direct_value, direct_grad = direct_target(weights, labels, query)
        distributed_value, distributed_grad, details = distributed_sp_target(weights, labels, query)
        diff = (direct_grad - distributed_grad).abs()
        row = {
            "query": query,
            "owner_rank": details["owner_rank"],
            "direct_loss": direct_value,
            "sp4_fsdp_avg_loss": distributed_value,
            "loss_abs_error": abs(direct_value - distributed_value),
            "grad_max_abs_error": float(diff.max().item()),
            "grad_l2_error": float(diff.norm().item()),
            "local_valid_label_counts": details["local_valid_label_counts"],
            "zero_label_ranks": details["zero_label_ranks"],
            "nonowner_ul_graph_l1": details["nonowner_ul_graph_l1"],
            "passed": close(direct_value, distributed_value, atol=3e-6, rtol=3e-6)
            and float(diff.max().item()) < 3e-6
            and all(x == 0.0 for x in details["nonowner_ul_graph_l1"]),
        }
        rows.append(row)
    return {
        "passed": all(r["passed"] for r in rows),
        "world_size": WORLD,
        "shard_ranges": [[r * SHARD, (r + 1) * SHARD - 1] for r in range(WORLD)],
        "queries_tested": queries,
        "rows": rows,
        "interpretation": "CPU deterministic shard simulation: GatherLoss world-size backward compensation followed by four-rank gradient average matches the SP1 direct target",
    }


def source_evidence() -> dict[str, Any]:
    return {
        "design_sha256": sha256(DESIGN),
        "trainer_sha256": sha256(TRAINER),
        "gatherloss_utils_sha256": sha256(GATHER),
        "seq2seq_trainer_sha256": sha256(SEQ2SEQ),
        "prepare_sha256": sha256(PREPARE),
        "core_sha256": sha256(CORE),
        "production_helper": "dkucc/local_ul30/trainer.py:_ul_value_from_row (AST extracted)",
        "gatherloss_contract": "dkucc/ms_swift/.../swift/sequence_parallel/utils.py:55-62 (source-read; mock uses the same world-size factor)",
        "token_denominator_contract": "dkucc/ms_swift/.../swift/trainers/seq2seq_trainer.py:219-245 (source-read)",
        "query_contract": "dkucc/local_ul30/trainer.py:489-493 (source-read)",
        "cpu_only": True,
        "real_gpu_or_nccl_parity": False,
    }


def main() -> int:
    torch.set_num_threads(1)
    results = {
        "schema_version": "local-ul30-loss-audit-v1",
        "status": "pass",
        "tests": {
            "stable_ul": test_stable_ul(),
            "causal_query_boundary": test_query_boundary(),
            "event_normalization": event_normalization(),
            "sp4_fsdp_average_gradient_parity": test_sp_gradient_parity(),
        },
        "source_evidence": source_evidence(),
        "limitations": [
            "The SP4/FSDP check is a deterministic CPU autograd simulation, not a four-process NCCL or GPU run.",
            "It validates the loss/gradient normalization contract with synthetic logits and labels; it does not validate model forward equivalence, padding kernels, FSDP parameter sharding, or AMP behavior.",
            "End-to-end smoke and real GPU evidence remain the execution agent's responsibility.",
        ],
    }
    failed = [name for name, test in results["tests"].items() if not test["passed"]]
    if failed:
        results["status"] = "fail"
        results["failed_tests"] = failed
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": results["status"], "out": str(OUT), "failed_tests": failed}, ensure_ascii=False))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
