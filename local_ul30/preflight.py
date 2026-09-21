#!/usr/bin/env python3
"""Fail-closed local/GPU gate management. It never submits a Slurm job."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .config import load_config
from .prepare import file_sha256

STAGES = ("G0", "G1", "G2", "G3", "G4", "G5")
REQUIRED_CHECKS = {
    "G0": {"canonical_hashes", "official_collator", "tokenizer_identity", "eos_preserved",
           "split_isolation", "ordered_manifest"},
    "G1": {"sp1_native_ce_parity", "full_prefix_cache_probe_parity", "clean_aux_update",
           "probe_grad_unchanged", "module_flags_restored", "rng_restored"},
    "G2": {"sp4_loss_parity", "sp4_gradient_parity", "cross_shard_substitution",
           "cross_shard_insertion", "zero_label_shard", "raw_world_denominator"},
    "G3": {"longest_plus_two_update", "native_checkpoint", "eight_microbatch_prefetch",
           "peak_memory_within_42gib"},
    "G4": {"continuous_vs_resume_order", "continuous_vs_resume_events", "optimizer_step_equal",
           "lr_equal", "counts_equal", "weights_within_tolerance", "hf_vllm_prefix_parity"},
    "G5": {"initial_weights_equal", "train_order_equal", "anchor_budget_equal",
           "clean_dev_unaugmented", "fixed_dev_library_ready", "full_dev_ready"},
}
SOURCE_BINDING_FILES = (
    "__init__.py", "core.py", "config.py", "prepare.py", "plugin.py", "trainer.py",
    "train.py", "preflight.py", "gpu_checks.py", "evaluate.py", "injected_eval.py",
    "calibrate.py", "finalize_calibration.py", "fsdp1_offload.json",
)
MANIFEST_BINDING_KEYS = (
    "manifest_sha256", "tokenizer_sha256", "special_ids_sha256",
    "all_train_targets_sha256", "legal_body_ids_sha256",
)


def tree_hash(path: Path) -> str:
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for file in files:
        h.update(str(file.relative_to(path) if path.is_dir() else file.name).encode())
        h.update(bytes.fromhex(file_sha256(file)))
    return h.hexdigest()


def experiment_source_hash(root: Path | None = None) -> str:
    """Hash only runtime definitions; omit artifacts, evidence, logs, tests and environments."""
    root = root or Path(__file__).resolve().parent
    missing = [name for name in SOURCE_BINDING_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"runtime source binding files missing: {missing}")
    h = hashlib.sha256()
    for name in SOURCE_BINDING_FILES:
        h.update(name.encode("utf-8"))
        h.update(bytes.fromhex(file_sha256(root / name)))
    return h.hexdigest()


def expected_bindings(config, manifest_summary: dict) -> dict:
    if not isinstance(manifest_summary, dict) or manifest_summary.get("status") != "pass":
        raise ValueError("a passing manifest summary is required to bind a gate")
    if (manifest_summary.get("config_sha256") != config.digest
            or manifest_summary.get("arm") != config.arm):
        raise ValueError("manifest summary config/arm identity mismatch")
    missing = [key for key in MANIFEST_BINDING_KEYS if not manifest_summary.get(key)]
    if missing:
        raise ValueError(f"manifest summary lacks required bindings: {missing}")
    manifest = Path(manifest_summary.get("manifest", ""))
    if not manifest.is_file() or file_sha256(manifest) != manifest_summary["manifest_sha256"]:
        raise ValueError("manifest summary points to a missing or hash-mismatched manifest")
    model = Path(config.model_path)
    if not model.exists():
        raise FileNotFoundError(f"initial model is unavailable for binding: {model}")
    bindings = {
        "config_sha256": config.digest, "train_sha256": config.train_sha256,
        "dev_sha256": config.dev_sha256, "upstream_swift_commit": config.upstream_swift_commit,
        "official_moss_commit": config.official_moss_commit,
        "experiment_source_sha256": experiment_source_hash(),
        "model_tree_sha256": tree_hash(model),
    }
    bindings.update({key: manifest_summary[key] for key in MANIFEST_BINDING_KEYS})
    # local-ul30 events are deterministic legal-token insertions.  No model
    # sampling calibration or extra Dev forward is part of this protocol.
    return bindings


def validate_stage_result(result: dict, stage: str, bindings: dict) -> None:
    if stage not in STAGES or result.get("schema_version") != "short-recovery-stage-result-v1":
        raise RuntimeError("stage result schema/stage is invalid")
    required = {"stage", "status", "checks", "command", "artifacts", "bindings"}
    if not required <= set(result) or result["stage"] != stage or result["status"] != "pass":
        raise RuntimeError("stage-specific result is incomplete or non-passing")
    if not result["command"] or not isinstance(result["checks"], dict):
        raise RuntimeError("stage result lacks an executed command or check mapping")
    missing = REQUIRED_CHECKS[stage] - {key for key, value in result["checks"].items() if value is True}
    if missing:
        raise RuntimeError(f"stage result lacks passing mandatory checks: {sorted(missing)}")
    if result["bindings"] != bindings:
        raise RuntimeError("stage result source bindings differ from gate")
    artifacts = result["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise RuntimeError("stage result requires named artifact paths and SHA-256 digests")
    for name, item in artifacts.items():
        if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
            raise RuntimeError(f"stage artifact {name} lacks path/hash")
        artifact_path = Path(item["path"])
        if not artifact_path.is_file() or file_sha256(artifact_path) != item["sha256"]:
            raise RuntimeError(f"stage artifact {name} is missing or hash-mismatched")
    if stage == "G3" and float(result.get("peak_memory_gib", float("inf"))) > 42:
        raise RuntimeError("G3 peak memory exceeds 42 GiB")


def init_gate(path: Path, config, manifest_summary: dict) -> dict:
    bindings = expected_bindings(config, manifest_summary)
    payload = {"schema_version": "short-recovery-gate-v1", "config_sha256": config.digest,
               "bindings": bindings, "bound": True,
               "stages": {stage: {"status": "pending", "evidence": None} for stage in STAGES}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def verify_gate(path: Path, config_digest: str, require_all: bool = True,
                bindings: dict | None = None) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"gate file missing: {path}")
    gate = json.loads(path.read_text())
    if gate.get("schema_version") != "short-recovery-gate-v1" or gate.get("config_sha256") != config_digest:
        raise ValueError("gate schema/config hash mismatch")
    if not gate.get("bound") or bindings is not None and gate.get("bindings") != bindings:
        raise ValueError("gate is unbound or source/plan/model/tokenizer bindings differ")
    for stage in STAGES:
        item = gate.get("stages", {}).get(stage, {})
        if item.get("status") != "pass":
            if require_all:
                raise RuntimeError(f"{stage} gate is {item.get('status', 'missing')}")
            continue
        evidence = Path(item.get("evidence", ""))
        if not evidence.is_file() or file_sha256(evidence) != item.get("evidence_sha256"):
            raise RuntimeError(f"{stage} evidence missing or hash mismatch")
        validate_stage_result(json.loads(evidence.read_text()), stage, gate["bindings"])
    return gate


def dry_run(config_path: Path, output: Path) -> dict:
    config = load_config(config_path)
    checks = {}
    for name, raw in [("train_manifest", config.train_manifest), ("dev_manifest", config.dev_manifest),
                      ("model_path", config.model_path), ("processor_path", config.processor_path),
                      ("official_source", config.official_source), ("swift_source", config.swift_source),
                      ("baseline_plugin", config.baseline_plugin)]:
        path = Path(raw)
        checks[name] = {"path": raw, "exists": path.exists()}
    swift = Path(config.swift_source)
    checks["pinned_swift_commit_path"] = {"path": str(swift), "exists": swift.is_dir(),
                                           "name_matches_commit": swift.name.endswith(config.upstream_swift_commit)}
    result = {"schema_version": "short-recovery-dry-run-v1", "status": "pending",
              "reason": "GPU G0-G5 results have not been executed", "config_sha256": config.digest,
              "experiment_source_sha256": experiment_source_hash(), "checks": checks,
              "required_pythonpath": [str(Path(__file__).resolve().parent.parent), str(swift)],
              "launch_allowed": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def record_stage(gate_path: Path, stage: str, evidence: Path, config_digest: str) -> dict:
    gate = verify_gate(gate_path, config_digest, require_all=False)
    result = json.loads(evidence.read_text())
    validate_stage_result(result, stage, gate["bindings"])
    gate["stages"][stage] = {"status": "pass", "evidence": str(evidence.resolve()),
                              "evidence_sha256": file_sha256(evidence)}
    gate_path.write_text(json.dumps(gate, indent=2) + "\n")
    return gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init-gate")
    p_init.add_argument("--output", required=True, type=Path)
    p_init.add_argument("--manifest-summary", required=True, type=Path)
    p_verify = sub.add_parser("verify-gate")
    p_verify.add_argument("--gate", required=True, type=Path)
    p_dry = sub.add_parser("dry-run")
    p_dry.add_argument("--output", required=True, type=Path)
    p_record = sub.add_parser("record-stage")
    p_record.add_argument("--gate", required=True, type=Path)
    p_record.add_argument("--stage", required=True, choices=STAGES)
    p_record.add_argument("--evidence", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "init-gate":
        result = init_gate(args.output, config, json.loads(args.manifest_summary.read_text()))
    elif args.command == "verify-gate":
        result = verify_gate(args.gate, config.digest)
    elif args.command == "dry-run":
        result = dry_run(args.config, args.output)
    else:
        result = record_stage(args.gate, args.stage, args.evidence, config.digest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
