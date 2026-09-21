#!/usr/bin/env python3
"""Guarded MS-Swift training entry. Defaults to a non-launching dry run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

from .config import load_config
from .preflight import expected_bindings, experiment_source_hash, tree_hash, verify_gate
from .prepare import file_sha256


def build_swift_args(config, manifest: Path, output_dir: Path, resume: str | None) -> list[str]:
    args = [
        "--model", config.model_path, "--model_type", "moss_transcribe_diarize",
        "--template", "moss_short_recovery_v2_sparse", "--tuner_type", "full",
        "--freeze_llm", "false", "--freeze_vit", "false", "--freeze_aligner", "false",
        "--dataset", str(manifest), "--split_dataset_ratio", "0", "--load_from_cache_file", "false",
        "--dataset_shuffle", "false",
        "--lazy_tokenize", "true", "--strict", "true", "--remove_unused_columns", "false",
        "--max_length", str(config.max_length), "--truncation_strategy", "delete",
        "--packing", "false", "--padding_free", "false",
        "--sequence_parallel_size", str(config.sequence_parallel_size),
        "--torch_dtype", "float32", "--bf16", "true", "--fp16", "false", "--attn_impl", "sdpa",
        "--learning_rate", str(config.learning_rate), "--lr_scheduler_type", "linear",
        "--warmup_ratio", "0", "--weight_decay", "0", "--adam_beta1", "0.9",
        "--adam_beta2", "0.999", "--adam_epsilon", "1e-8", "--max_grad_norm", "1",
        "--optim", "adamw_torch_fused", "--max_steps", str(config.scheduler_horizon),
        "--per_device_train_batch_size", "1", "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
        "--gradient_checkpointing", "true", "--vit_gradient_checkpointing", "true",
        "--gradient_checkpointing_kwargs", '{"use_reentrant":false}',
        "--fsdp", str(Path(__file__).with_name("fsdp1_offload.json")),
        "--average_tokens_across_devices", "true", "--use_logits_to_keep", "false",
        "--dataloader_num_workers", "0", "--dataloader_persistent_workers", "false",
        "--train_dataloader_shuffle", "false", "--ddp_find_unused_parameters", "false",
        "--logging_steps", "1", "--save_steps", "30", "--eval_strategy", "no",
        "--prediction_loss_only", "true", "--report_to", "tensorboard",
        "--seed", str(config.seed), "--data_seed", str(config.seed), "--add_version", "false",
        "--output_dir", str(output_dir),
    ]
    if resume:
        args += ["--resume_from_checkpoint", resume]
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest-summary", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop-at", type=int, choices=[1, 2, 5, 15, 30, 60, 90, 120, 150])
    parser.add_argument("--engineering-stage", choices=["G1", "G2", "G3"],
                        help="Run a bounded evidence job; never records a passing gate stage.")
    parser.add_argument("--authorized-pending-gates", action="store_true",
                        help="Formal user-authorized run with a bound gate whose GPU stages remain pending.")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    engineering = args.engineering_stage is not None
    if engineering and args.authorized_pending_gates:
        raise RuntimeError("engineering and formal pending-gate modes are mutually exclusive")
    stop_at = args.stop_at if args.stop_at is not None else (1 if engineering else 30)
    if engineering and stop_at not in {1, 2}:
        raise RuntimeError("engineering G1-G3 runs are limited to one or two optimizer updates")
    formal_stops = ({5, 15, 30} if config.optimizer_updates == 30
                    else {30, 60, 90, 120, 150})
    if not engineering and stop_at not in formal_stops:
        raise RuntimeError(f"formal stop {stop_at} is not valid for a {config.optimizer_updates}-update run")
    if not engineering and stop_at > config.optimizer_updates:
        raise RuntimeError("formal stop exceeds configured optimizer updates")
    if engineering and args.resume:
        raise RuntimeError("bounded G1-G3 engineering runs start from common weights; resume is a separate G4 check")
    if args.engineering_stage == "G1" and config.sequence_parallel_size != 1:
        raise RuntimeError("G1 engineering requires an SP1 config")
    if args.engineering_stage in {"G2", "G3"} and config.sequence_parallel_size != 4:
        raise RuntimeError(f"{args.engineering_stage} engineering requires an SP4 config")
    swift_source = Path(config.swift_source)
    if not swift_source.is_dir() or not swift_source.name.endswith(config.upstream_swift_commit):
        raise RuntimeError("configured pinned Swift source path is missing or names another commit")
    summary = json.loads(args.manifest_summary.read_text())
    manifest = Path(summary["manifest"])
    if summary.get("status") != "pass" or summary.get("config_sha256") != config.digest:
        raise RuntimeError("manifest summary does not match the validated config")
    if file_sha256(manifest) != summary.get("manifest_sha256"):
        raise RuntimeError("ordered manifest hash mismatch")
    if engineering:
        output_dir = (Path(config.output_root) / "short-recovery-engineering" / args.run_id /
                      args.engineering_stage / config.arm / f"seed-{config.seed}")
    else:
        output_dir = Path(config.output_root) / f"short-recovery-{args.run_id}" / config.arm / f"seed-{config.seed}"
    plan = build_swift_args(config, manifest, output_dir, args.resume)
    if engineering:
        plan[plan.index("--save_steps") + 1] = "1"
    run_mode = ("engineering_evidence_only" if engineering else
                "formal_authorized_pending_gates" if args.authorized_pending_gates else "formal")
    receipt = {"schema_version": "short-recovery-launch-plan-v1", "launch": args.launch,
               "mode": run_mode,
               "engineering_stage": args.engineering_stage, "stop_at": stop_at,
               "config_sha256": config.digest, "manifest_sha256": summary["manifest_sha256"],
               "experiment_source_sha256": experiment_source_hash(),
               "swift_source": str(swift_source.resolve()),
               "model_tree_sha256": tree_hash(Path(config.model_path)) if Path(config.model_path).exists() else None,
               "output_dir": str(output_dir), "swift_args": plan}
    print(json.dumps(receipt, indent=2))
    if not args.launch:
        return
    gate = Path(config.gate_file)
    current_bindings = expected_bindings(config, summary)
    verify_gate(gate, config.digest,
                require_all=not (engineering or args.authorized_pending_gates),
                bindings=current_bindings)
    if not torch.cuda.is_available() or torch.cuda.device_count() != config.sequence_parallel_size:
        raise RuntimeError("launch requires exactly the configured SP GPU count")
    calibration_path = Path(config.calibration_file)
    if config.arm in {"sub", "mix", "sub_repeat"}:
        if not calibration_path.is_file():
            raise RuntimeError("sparse perturbation arm requires a real Dev calibration artifact")
        calibration = json.loads(calibration_path.read_text())
        rec = calibration.get("recommendation", {})
        if (calibration.get("schema_version") != "short-recovery-dev-calibration-v2"
                or calibration.get("status") != "pass"
                or calibration.get("dev_sha256") != config.dev_sha256
                or calibration.get("initial_checkpoint") != config.model_path
                or calibration.get("tokenizer_sha256") != summary["tokenizer_sha256"]
                or calibration.get("model_tree_sha256") != current_bindings["model_tree_sha256"]
                or calibration.get("experiment_source_sha256") != current_bindings["experiment_source_sha256"]
                or config.model_history_probability > rec.get("model_history_probability_max", -1)
                or config.sampling_probability > rec.get("sampling_probability_max", -1)
                or config.repeat_probability > rec.get("repeat_probability_max", -1)):
            raise RuntimeError("config probabilities exceed or do not match the bound Dev calibration")
    # torchrun executes this entrypoint independently on every rank.  Only rank
    # zero owns filesystem-level launch validation and receipt creation; doing
    # these on all ranks creates a race where rank zero's receipt makes the
    # fresh directory look like a pre-existing run to the other ranks.
    process_rank = int(os.environ.get("RANK", "0"))
    if (process_rank == 0 and output_dir.exists() and any(output_dir.iterdir())
            and not args.resume):
        raise RuntimeError("refusing to overwrite a non-empty experiment output")
    if args.resume:
        resume_path = Path(args.resume).resolve()
        try:
            resume_path.relative_to(output_dir.resolve())
        except ValueError as exc:
            raise RuntimeError("resume checkpoint must belong to this experiment output") from exc
        recovery_state = resume_path / "recovery_state.json"
        if not recovery_state.is_file():
            raise RuntimeError("resume checkpoint lacks recovery_state.json")
        saved = json.loads(recovery_state.read_text())
        checkpoint_step = int(resume_path.name.rsplit("-", 1)[-1])
        if (saved.get("config_sha256") != config.digest
                or saved.get("plan_sha256") != summary["manifest_sha256"]
                or saved.get("global_step") != checkpoint_step
                or saved.get("next_row_index") != checkpoint_step * config.gradient_accumulation_steps):
            raise RuntimeError("resume state does not match config/plan/checkpoint boundary")
    if process_rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "launch_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    os.environ.update({
        "MOSS_ROOT": config.official_source, "MOSS_BASELINE_PLUGIN": config.baseline_plugin,
        "MOSS_IMPLICIT_CAUSAL": "1", "RECOVERY_LEGAL_IDS": str(args.manifest_summary.parent / "legal_body_ids.json"),
        "RECOVERY_SEED": str(config.seed), "RECOVERY_STOP_UPDATES": str(stop_at),
        "RECOVERY_PROBE_CONTROLS": "1" if config.probe_controls else "0",
        "RECOVERY_PLAN_SHA256": summary["manifest_sha256"], "RECOVERY_CONFIG_SHA256": config.digest,
        "RECOVERY_RUN_ID": args.run_id, "RECOVERY_TOKENIZER_SHA256": summary["tokenizer_sha256"],
        "CELOSS_PARALLEL_SIZE": "512",
    })
    if engineering:
        os.environ["RECOVERY_DEBUG_CONTRACTS"] = "1"
    package_parent = str(Path(__file__).resolve().parent.parent)
    for entry in (str(swift_source.resolve()), package_parent):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from . import plugin  # registers baseline model and independent template
    from swift.trainers.trainer_factory import TrainerFactory
    original = TrainerFactory.TRAINER_MAPPING["causal_lm"]
    package_root = __package__.split(".")[0]
    TrainerFactory.TRAINER_MAPPING["causal_lm"] = f"{package_root}.trainer.RecoveryTrainer"
    try:
        from swift.pipelines import sft_main
        sft_main(plan)
    finally:
        TrainerFactory.TRAINER_MAPPING["causal_lm"] = original


if __name__ == "__main__":
    main()
