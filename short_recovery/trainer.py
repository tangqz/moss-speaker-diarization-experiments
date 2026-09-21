"""Lightweight online-probe Trainer adapter; native Swift owns loss/backward/optimizer."""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from swift.sequence_parallel import sequence_parallel
from swift.dataloader import DataLoaderShard
from swift.sequence_parallel.utils import SequenceParallelSampler
from swift.trainers import Seq2SeqTrainer
from swift.utils import seed_worker

from .core import (IGNORE_INDEX, apply_event, bernoulli, candidate_set,
                   sparse_substitution, stable_seed)


def _probe_view(inputs):
    """Clone mutable text tensors, while sharing read-only audio tensors."""
    mutable = {"input_ids", "labels", "loss_scale", "position_ids", "attention_mask"}
    return {key: value.clone() if key in mutable and isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()}


def _tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def preserved_probe_state(model):
    """Set eval/no-grad while preserving every module flag and all RNG streams."""
    module_flags = [(module, module.training) for module in model.modules()]
    py_state = random.getstate()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, flag in module_flags:
            module.training = flag
        random.setstate(py_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def _distributed_query_logits(logits: torch.Tensor, full_position_ids: torch.Tensor,
                              query: int) -> torch.Tensor:
    local_position_ids = sequence_parallel.split(
        full_position_ids, dim=-1, position_ids=full_position_ids)
    if logits.ndim != 3 or local_position_ids.ndim != 2 or logits.shape[:2] != local_position_ids.shape:
        raise RuntimeError("probe logits/position_ids shapes are inconsistent")
    positions = (local_position_ids[0] == int(query)).nonzero(as_tuple=True)[0]
    owner = torch.tensor([len(positions)], device=logits.device, dtype=torch.long)
    row = torch.zeros(logits.shape[-1], device=logits.device, dtype=torch.float32)
    if len(positions) == 1:
        row.copy_(logits[0, positions[0]].float())
    elif len(positions) > 1:
        raise RuntimeError("query position occurs more than once on a rank")
    if dist.is_initialized():
        group = sequence_parallel.sp_group
        dist.all_reduce(owner, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(row, op=dist.ReduceOp.SUM, group=group)
    if int(owner.item()) != 1:
        raise RuntimeError(f"expected exactly one SP owner for query={query}, got {owner.item()}")
    return row.cpu()


class RecoveryBoundaryCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        # Native Trainer already clipped immediately before this callback.  Log
        # the post-clip norm without changing gradients or optimizer state.
        total = torch.zeros((), device=args.device, dtype=torch.float64)
        for parameter in self.trainer.model.parameters():
            if parameter.grad is not None:
                # Reduce each native-dtype shard to one scalar; never materialize
                # an FP64 copy of a full parameter/gradient.
                norm = torch.linalg.vector_norm(parameter.grad.detach())
                total += norm.double().square()
        if dist.is_initialized():
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
        self.trainer._post_clip_grad_norm = float(total.sqrt().item())
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.trainer._optimizer_lr = float(self.trainer.optimizer.param_groups[0]["lr"])
        return control

    def on_step_end(self, args, state, control, **kwargs):
        state_path = self.trainer.audit_dir / "recovery_state.json"
        payload = {"schema_version": "short-recovery-state-v1", "global_step": state.global_step,
                   "next_row_index": state.global_step * args.gradient_accumulation_steps,
                   "plan_sha256": os.environ.get("RECOVERY_PLAN_SHA256"),
                   "config_sha256": os.environ.get("RECOVERY_CONFIG_SHA256")}
        if self.trainer.is_world_process_zero():
            state_path.write_text(json.dumps(payload, indent=2) + "\n")
        self.trainer._flush_update(state.global_step - 1)
        if state.global_step >= self.trainer.recovery_stop_updates:
            control.should_training_stop = True
        return control

    def on_save(self, args, state, control, **kwargs):
        payload = {"schema_version": "short-recovery-state-v1", "global_step": state.global_step,
                   "next_row_index": state.global_step * args.gradient_accumulation_steps,
                   "plan_sha256": os.environ.get("RECOVERY_PLAN_SHA256"),
                   "config_sha256": os.environ.get("RECOVERY_CONFIG_SHA256")}
        if self.trainer.is_world_process_zero():
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            if not checkpoint.is_dir():
                raise RuntimeError(f"native checkpoint directory missing at save callback: {checkpoint}")
            (checkpoint / "recovery_state.json").write_text(json.dumps(payload, indent=2) + "\n")
        return control


class RecoveryTrainer(Seq2SeqTrainer):
    """Only changes ordered input delivery and the pre-parent training input."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recovery_stop_updates = int(os.environ.get("RECOVERY_STOP_UPDATES", "30"))
        self.audit_dir = Path(self.args.output_dir) / "recovery_audit"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        legal_path = Path(os.environ["RECOVERY_LEGAL_IDS"])
        self.legal_ids = frozenset(json.loads(legal_path.read_text()))
        self._component_buffer = None
        self._post_clip_grad_norm = None
        self._optimizer_lr = None
        self.add_callback(RecoveryBoundaryCallback(self))

    def get_sp_dataloader(self, dataset, batch_size, skip_batches=0):
        """Independent ordered SP loader; upstream hardcodes shuffle=True."""
        import datasets
        data_collator = self.data_collator
        if isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        if not hasattr(dataset, "__len__"):
            raise ValueError("v1 requires a sized, ordered dataset")
        sampler = SequenceParallelSampler(sequence_parallel, dataset, shuffle=False,
                                          seed=int(os.environ.get("RECOVERY_SEED", "0")), round_up=False)
        params = {
            "batch_size": batch_size, "collate_fn": data_collator, "sampler": sampler,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
        }
        params.update(self._maybe_multiprocessing_context(self.args))
        if skip_batches > 0:
            from accelerate.data_loader import SkipBatchSampler
            params["sampler"] = SkipBatchSampler(
                sampler, skip_batches=skip_batches * batch_size)
        params["worker_init_fn"] = partial(
            seed_worker, num_workers=self.args.dataloader_num_workers,
            rank=sequence_parallel.dp_rank)
        return DataLoaderShard(dataset, device=self.accelerator.device, **params)

    def _write_event(self, record: dict[str, Any]):
        rank = dist.get_rank() if dist.is_initialized() else 0
        path = self.audit_dir / f"events.rank{rank}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def _validate_and_start_row(self, meta):
        update = int(self.state.global_step)
        if int(meta["optimizer_step"]) != update:
            raise RuntimeError(f"manifest optimizer_step {meta['optimizer_step']} != runtime {update}")
        if self._component_buffer is None:
            self._component_buffer = {"optimizer_step": update, "row_count": 0,
                                      "clean_contribution": 0.0, "aux_weighted_contribution": 0.0,
                                      "events": 0, "fallbacks": {}, "probe_seconds": 0.0,
                                      "body_labels": 0, "structure_labels": 0, "meta": meta,
                                      "history_triggered": 0, "temperature_branch": 0,
                                      "probed_anchor_count": 0, "sub_slots": 0,
                                      "argmax_wrong": 0, "argmax_illegal": 0, "sample_gold": 0,
                                      "sub_changed": 0, "repeat_inserted": 0,
                                      "modified_tokens": 0, "inserted_tokens": 0}
        buffer = self._component_buffer
        if buffer["optimizer_step"] != update:
            raise RuntimeError("component buffer crossed an optimizer boundary")
        row_index = buffer["row_count"]
        if meta["arm"] == "ce":
            expected_slot, expected_view = row_index, "clean"
        else:
            expected_slot, expected_view = row_index // 2, "clean" if row_index % 2 == 0 else "aux"
        if int(meta["source_slot"]) != expected_slot or meta["view"] != expected_view:
            raise RuntimeError(
                f"row order mismatch at update {update}: expected slot/view "
                f"{expected_slot}/{expected_view}, got {meta['source_slot']}/{meta['view']}")

    def _finish_row(self, meta, event_record, loss):
        buffer = self._component_buffer
        value = float(loss.detach().float().cpu().item())
        if meta["view"] == "clean":
            buffer["clean_contribution"] += value
        else:
            buffer["aux_weighted_contribution"] += value
            buffer["body_labels"] += int(meta["body_label_count"])
            buffer["structure_labels"] += int(meta["k"]) - int(meta["body_label_count"])
            buffer["probe_seconds"] += float(event_record.get("probe_seconds", 0.0))
            if event_record.get("actual_event_kind") not in {None, "clean_aux"}:
                buffer["events"] += 1
            for key in ("history_triggered", "temperature_branch", "argmax_wrong",
                        "argmax_illegal", "sample_gold", "sub_changed", "repeat_inserted",
                        "probed_anchor", "sub_slot"):
                target = {"probed_anchor": "probed_anchor_count", "sub_slot": "sub_slots"}.get(key, key)
                buffer[target] += int(bool(event_record.get(key)))
            buffer["modified_tokens"] += int(event_record.get("modified_tokens", 0))
            buffer["inserted_tokens"] += int(event_record.get("inserted_tokens", 0))
            fallback = event_record.get("fallback_reason")
            if fallback:
                buffer["fallbacks"][fallback] = buffer["fallbacks"].get(fallback, 0) + 1
        buffer["row_count"] += 1

    def _flush_update(self, optimizer_step):
        buffer = self._component_buffer
        if buffer is None or buffer["optimizer_step"] != optimizer_step:
            raise RuntimeError("optimizer boundary has no matching component buffer")
        meta = buffer["meta"]
        expected_rows = 4 if meta["arm"] == "ce" else 8
        if buffer["row_count"] != expected_rows:
            raise RuntimeError(f"optimizer update has {buffer['row_count']} rows, expected {expected_rows}")
        lam = float(meta["lambda"])
        aux_mean = (buffer["aux_weighted_contribution"] / lam
                    if lam > 0 else None)
        record = {"schema_version": "short-recovery-update-v1", "optimizer_step": optimizer_step,
                  "nc": int(meta["planned_nc"]), "na": int(meta["planned_na"]),
                  "n": int(meta["planned_n"]), "lambda": lam,
                  "clean_token_ce": buffer["clean_contribution"], "aux_token_ce": aux_mean,
                  "aux_weighted_ce": buffer["aux_weighted_contribution"],
                  "weighted_total_ce": buffer["clean_contribution"] + buffer["aux_weighted_contribution"],
                  "actual_error_events": buffer["events"], "fallbacks": buffer["fallbacks"],
                  "aux_slots": 4 if meta["arm"] != "ce" else 0,
                  "probed_anchor_count": buffer["probed_anchor_count"],
                  "sub_slots": buffer["sub_slots"],
                  "model_history_triggered": buffer["history_triggered"],
                  "temperature_branches": buffer["temperature_branch"],
                  "argmax_wrong": buffer["argmax_wrong"], "argmax_illegal": buffer["argmax_illegal"],
                  "sample_equals_gold": buffer["sample_gold"], "sub_changed": buffer["sub_changed"],
                  "repeat_inserted": buffer["repeat_inserted"],
                  "modified_tokens": buffer["modified_tokens"], "inserted_tokens": buffer["inserted_tokens"],
                  "original_body_tokens": sum(int(meta["original_body_token_count"])
                                              for meta in buffer.get("aux_meta", [])),
                  "body_recovery_labels": buffer["body_labels"],
                  "structure_recovery_labels": buffer["structure_labels"],
                  "probe_seconds": buffer["probe_seconds"],
                  "lr_before_scheduler_step": self._optimizer_lr,
                  "post_clip_grad_norm": self._post_clip_grad_norm,
                  "peak_allocated_gib": (torch.cuda.max_memory_allocated() / 2**30
                                         if torch.cuda.is_available() else None),
                  "peak_reserved_gib": (torch.cuda.max_memory_reserved() / 2**30
                                        if torch.cuda.is_available() else None)}
        if self.is_world_process_zero():
            with (self.audit_dir / "updates.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        self._component_buffer = None
        self._post_clip_grad_norm = None
        self._optimizer_lr = None

    def _probe(self, model, clean_inputs: dict[str, Any], meta: dict[str, Any]):
        probe_inputs = _probe_view(clean_inputs)
        probe_inputs.pop("labels", None)
        probe_inputs.pop("loss_scale", None)
        saved_sp = dict(sequence_parallel.extra_kwargs)
        debug = os.environ.get("RECOVERY_DEBUG_CONTRACTS") == "1"
        before_grad = [(None if p.grad is None else
                        (_tensor_digest(p.grad) if debug else (id(p.grad), p.grad._version)))
                       for p in model.parameters()]
        started = time.perf_counter()
        try:
            with preserved_probe_state(model):
                prepared = self._prepare_inputs(probe_inputs)
                prepared.pop("compute_loss_func", None)
                with self.template.forward_context(self.model, prepared), self.compute_loss_context_manager():
                    outputs = model(**prepared)
                position_ids = prepared.get("position_ids")
                if position_ids is None:
                    position_ids = sequence_parallel.extra_kwargs.get("text_position_ids")
                if position_ids is None:
                    raise RuntimeError("probe lacks global position_ids")
                anchor_logits = _distributed_query_logits(outputs.logits, position_ids, int(meta["query_abs"]))
                diag_logits = None
                if meta["event_kind"] == "repeat_previous":
                    diag_logits = _distributed_query_logits(outputs.logits, position_ids, int(meta["anchor_abs"]))
                del outputs
        finally:
            sequence_parallel.extra_kwargs = saved_sp
        after_grad = [(None if p.grad is None else
                       (_tensor_digest(p.grad) if debug else (id(p.grad), p.grad._version)))
                      for p in model.parameters()]
        if before_grad != after_grad:
            raise RuntimeError("no_grad probe changed pre-existing gradients")
        return anchor_logits, diag_logits, time.perf_counter() - started

    @staticmethod
    def _num_items_value(num_items_in_batch):
        if num_items_in_batch is None:
            raise RuntimeError("HF did not precompute the accumulation-window denominator")
        if isinstance(num_items_in_batch, torch.Tensor):
            if num_items_in_batch.numel() != 1:
                raise RuntimeError("num_items_in_batch must be scalar")
            return int(num_items_in_batch.item())
        return int(num_items_in_batch)

    def training_step(self, model, inputs, *args, **kwargs):
        metadata = inputs.pop("recovery_meta_json", None)
        unpadded = inputs.pop("recovery_unpadded_length", None)
        if not isinstance(metadata, list) or len(metadata) != 1:
            raise RuntimeError("ordered microbatch lost recovery metadata")
        meta = json.loads(metadata[0])
        self._validate_and_start_row(meta)
        if unpadded is None or int(unpadded[0]) != int(meta["raw_length"]):
            raise RuntimeError("collator length metadata mismatch")
        num_items = kwargs.get("num_items_in_batch")
        if num_items is None and args:
            num_items = args[0]
        received_n = self._num_items_value(num_items)
        world = int(getattr(self.args, "world_size", 1))
        expected_raw_n = world * int(meta["planned_n"])
        if received_n != expected_raw_n:
            raise RuntimeError(
                f"HF raw denominator {received_n} != world_size*unique_N {expected_raw_n}; "
                "Accelerate non-data-parallel normalization is incompatible until G2 parity is re-derived")
        valid_here = int((inputs["labels"] != IGNORE_INDEX).sum().item())
        expected_here = int(meta["clean_count"] if meta["view"] == "clean" else meta["k"])
        if valid_here != expected_here:
            raise RuntimeError(f"collated label count {valid_here} != {expected_here}")
        audio_keys = ["input_features", "audio_feature_lengths", "audio_chunk_mapping"]
        debug = os.environ.get("RECOVERY_DEBUG_CONTRACTS") == "1"
        audio_identity = {key: (_tensor_digest(inputs[key]) if debug else
                                (id(inputs[key]), inputs[key]._version, tuple(inputs[key].shape), inputs[key].dtype))
                          for key in audio_keys}
        event_record = dict(meta)
        event_record.update({"hf_raw_num_items_in_batch": received_n,
                             "native_world_compensation": world,
                             "effective_unique_denominator": received_n / world,
                             "collated_valid_labels": valid_here,
                             "run_id": os.environ.get("RECOVERY_RUN_ID"),
                             "tokenizer_sha256": os.environ.get("RECOVERY_TOKENIZER_SHA256"),
                             "clean_raw_length": int(meta["raw_length"]),
                             "clean_padded_length": int(inputs["input_ids"].shape[-1]),
                             "protocol_seed": os.environ.get("RECOVERY_SEED"),
                             "original_anchor_id": int(inputs["input_ids"][0, int(meta["anchor_abs"])].item()),
                             "sub_slot": meta["view"] == "aux" and meta["event_kind"] in {
                                 "substitute", "substitute_then_repeat"}})

        kind = meta["event_kind"]
        should_probe = meta["view"] == "aux" and (kind != "clean_aux" or
                         os.environ.get("RECOVERY_PROBE_CONTROLS", "1") == "1")
        if should_probe:
            logits, diag_logits, elapsed = self._probe(model, inputs, meta)
            gold = int(inputs["input_ids"][0, int(meta["anchor_abs"])].item())
            candidates = candidate_set(logits.tolist(), gold, self.legal_ids, include_gold=True)
            candidate_digest = stable_seed(candidates.ids, candidates.probs, candidates.gold_prob) % (2**63 - 1)
            if dist.is_initialized():
                comm_device = torch.device("cuda", torch.cuda.current_device())
                digest_tensor = torch.tensor([candidate_digest], device=comm_device, dtype=torch.long)
                gathered = [torch.zeros_like(digest_tensor) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered, digest_tensor)
                if len({int(x.item()) for x in gathered}) != 1:
                    raise RuntimeError("SP ranks produced different candidate sets")
            event_record.update({"gold_prob": candidates.gold_prob,
                                 "error_mass": candidates.error_mass,
                                 "candidate_mass": candidates.candidate_mass,
                                 "candidate_ids": list(candidates.ids),
                                 "candidate_probs": list(candidates.probs),
                                 "probe_seconds": elapsed, "probed_anchor": True})
            if diag_logits is not None:
                event_record["repeat_token_prob_at_query_a"] = float(torch.softmax(diag_logits, -1)[gold])
        else:
            candidates = None

        if meta["view"] == "aux" and kind in {
                "substitute", "repeat_previous", "substitute_then_repeat"}:
            raw_len = int(meta["raw_length"])
            z = inputs["input_ids"][0, :raw_len].tolist()
            replacement = None
            fallback = None
            if kind in {"substitute", "substitute_then_repeat"}:
                base_seed = (os.environ.get("RECOVERY_SEED", "0"), meta["source_key"],
                             meta["occurrence"], meta["anchor_abs"])
                decision = sparse_substitution(
                    logits.tolist(), gold=gold, legal_ids=self.legal_ids,
                    history_seed=stable_seed(*base_seed, "model-history"),
                    branch_seed=stable_seed(*base_seed, "temperature-branch"),
                    sample_seed=stable_seed(*base_seed, "temperature-sample"),
                    model_history_probability=float(meta["model_history_probability"]),
                    sampling_probability=float(meta["sampling_probability"]),
                    temperature=float(meta["sampling_temperature"]), minimum_candidate_mass=0.5)
                replacement = decision.replacement
                fallback = decision.reason
                event_record.update({
                    "history_triggered": decision.history_triggered,
                    "temperature_branch": decision.temperature_branch,
                    "argmax_id": decision.argmax_id,
                    "argmax_wrong": decision.argmax_id is not None and decision.argmax_id != gold,
                    "argmax_illegal": decision.argmax_id is not None and decision.argmax_id not in self.legal_ids,
                    "sample_gold": decision.sample_equals_gold,
                    "sub_changed": decision.changed,
                    "modified_tokens": int(decision.changed),
                    "inserted_tokens": (int(meta["repeat_count"])
                                        if decision.changed and kind == "substitute_then_repeat" else 0),
                    "history_seed": stable_seed(*base_seed, "model-history"),
                    "branch_seed": stable_seed(*base_seed, "temperature-branch"),
                    "sample_seed": stable_seed(*base_seed, "temperature-sample")})
                if dist.is_initialized():
                    shared = torch.tensor([-1 if replacement is None else replacement],
                                          device=torch.device("cuda", torch.cuda.current_device()), dtype=torch.long)
                    dist.broadcast(shared, src=0)
                    replacement = None if int(shared.item()) < 0 else int(shared.item())
            repeat_count = int(meta["repeat_count"])
            if kind == "repeat_previous":
                repeat_seed = stable_seed(os.environ.get("RECOVERY_SEED", "0"), meta["source_key"],
                                          meta["occurrence"], meta["anchor_abs"], "repeat-trigger")
                repeat_triggered = bernoulli(repeat_seed, float(meta["repeat_probability"]))
                if not repeat_triggered:
                    fallback = "repeat_not_triggered"
                event_record.update({"repeat_seed": repeat_seed, "repeat_inserted": repeat_triggered,
                                     "modified_tokens": 0,
                                     "inserted_tokens": repeat_count if repeat_triggered else 0})
            if kind == "repeat_previous" and raw_len + repeat_count > int(self.template.max_length):
                fallback = "max_length_after_insert"
                event_record.update({"repeat_inserted": False, "inserted_tokens": 0})
            if kind == "substitute_then_repeat":
                event_record["repeat_inserted"] = replacement is not None and fallback is None
                if fallback is None and raw_len + repeat_count > int(self.template.max_length):
                    fallback = "max_length_after_insert"
                    event_record.update({"repeat_inserted": False, "inserted_tokens": 0})
            actual_kind = "clean_aux" if fallback else kind
            result = apply_event(z, anchor=int(meta["anchor_abs"]), k=int(meta["k"]),
                                 kind=actual_kind, clean_scale=float(meta["clean_scale"]),
                                 aux_scale=float(meta["aux_scale"]), replacement=replacement,
                                 repeat_count=0 if fallback else repeat_count)
            sp = int(self.template.sequence_parallel_size)
            padded_len = len(result.input_ids) + (-len(result.input_ids)) % sp
            pad_id = int(self.template.tokenizer.pad_token_id)
            inputs["input_ids"] = torch.tensor([list(result.input_ids) + [pad_id] * (padded_len - len(result.input_ids))],
                                               dtype=inputs["input_ids"].dtype)
            inputs["labels"] = torch.tensor([list(result.labels) + [IGNORE_INDEX] * (padded_len - len(result.labels))],
                                            dtype=inputs["labels"].dtype)
            inputs["loss_scale"] = torch.tensor([list(result.loss_scale) + [0.0] * (padded_len - len(result.loss_scale))],
                                                dtype=inputs["loss_scale"].dtype)
            inputs["position_ids"] = torch.arange(padded_len).unsqueeze(0)
            if "attention_mask" in inputs:
                inputs["attention_mask"] = torch.tensor(
                    [[1] * len(result.input_ids) + [0] * (padded_len - len(result.input_ids))],
                    dtype=inputs["attention_mask"].dtype)
            event_record.update({"actual_event_kind": actual_kind, "replacement_id": replacement,
                                 "fallback_reason": fallback, "insert_count": result.insert_count,
                                 "new_length": len(result.input_ids),
                                 "new_anchor_id": int(result.input_ids[int(meta["anchor_abs"])]),
                                 "original_recovery_range": [int(meta["anchor_abs"]) + 1,
                                                             int(meta["anchor_abs"]) + 1 + int(meta["k"])],
                                 "new_recovery_range": [result.recovery_start, result.recovery_end],
                                 "recovery_start": result.recovery_start,
                                 "recovery_end": result.recovery_end})
        else:
            event_record.update({"actual_event_kind": kind, "fallback_reason": None,
                                 "insert_count": 0, "new_length": int(meta["raw_length"]),
                                 "modified_tokens": 0, "inserted_tokens": 0,
                                 "new_anchor_id": event_record["original_anchor_id"],
                                 "original_recovery_range": ([int(meta["anchor_abs"]) + 1,
                                                               int(meta["anchor_abs"]) + 1 + int(meta["k"])]
                                                              if meta["view"] == "aux" else None),
                                 "new_recovery_range": ([int(meta["anchor_abs"]) + 1,
                                                          int(meta["anchor_abs"]) + 1 + int(meta["k"])]
                                                         if meta["view"] == "aux" else None)})
        event_record["actual_padded_length"] = int(inputs["input_ids"].shape[-1])
        event_record["sp_rank_count"] = dist.get_world_size() if dist.is_initialized() else 1
        event_contract = {
            "kind": event_record.get("actual_event_kind"),
            "replacement": event_record.get("replacement_id"),
            "insert_count": event_record.get("insert_count"),
            "recovery_start": event_record.get("recovery_start"),
            "recovery_end": event_record.get("recovery_end"),
            "valid_labels": int((inputs["labels"] != IGNORE_INDEX).sum().item()),
            "input_length": int(inputs["input_ids"].shape[-1]),
        }
        event_digest = stable_seed(json.dumps(event_contract, sort_keys=True)) % (2**63 - 1)
        event_record["sp_event_digest"] = event_digest
        if dist.is_initialized():
            comm_device = torch.device("cuda", torch.cuda.current_device())
            digest_tensor = torch.tensor([event_digest], device=comm_device, dtype=torch.long)
            gathered = [torch.zeros_like(digest_tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, digest_tensor)
            if len({int(x.item()) for x in gathered}) != 1:
                raise RuntimeError("SP ranks disagree on the actual event contract")
        for key, identity in audio_identity.items():
            current = (_tensor_digest(inputs[key]) if debug else
                       (id(inputs[key]), inputs[key]._version, tuple(inputs[key].shape), inputs[key].dtype))
            if current != identity:
                raise RuntimeError(f"online transform changed {key}")
        result = super().training_step(model, inputs, *args, **kwargs)
        if meta["view"] == "aux":
            self._component_buffer.setdefault("aux_meta", []).append(meta)
        self._write_event(event_record)
        self._finish_row(meta, event_record, result)
        return result
